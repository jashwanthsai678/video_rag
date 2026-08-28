"""Hybrid (vector + keyword) index over Segments, backed by FAISS and BM25."""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import Config
from .retrieval import BM25, reciprocal_rank_fusion, tokenize
from .segment import Segment
from .store import SegmentStore

logger = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
STORE_FILE = "segments.db"
# Written last, after both the FAISS and SQLite files have been renamed into
# place - the actual commit point of save(). Its presence (and a matching
# `generation`) is what load() uses to tell "fully saved" apart from
# "interrupted partway through", which the FAISS-count/segment-count check
# alone can't always catch (see save()/load()).
MANIFEST_FILE = "manifest.json"
SCHEMA_VERSION = 4


@dataclass
class SearchHit:
    """One retrieved moment, with both scores that matter kept separate."""

    segment: Segment
    score: float                       # semantic cosine similarity, [-1, 1]
    fused: float                       # rank-fusion score, used for ordering
    found_by: Tuple[str, ...] = field(default_factory=tuple)


class IndexCompatibilityError(RuntimeError):
    """Saved index was built with settings incompatible with the current ones."""


class SemanticIndex:
    """Embeds segment text with a sentence-transformers model and indexes it
    with FAISS (cosine similarity via normalized inner product), fused with a
    BM25 keyword index.

    Both retrievers are needed: dense embeddings handle paraphrase ("add a
    patient" vs "New > Patient Record"), while BM25 handles the exact UI
    tokens — button labels, menu names — that embeddings blur together.
    """

    def __init__(self, config: Config):
        self.config = config
        self._model = None
        self._faiss = None
        self._bm25: Optional[BM25] = None
        self.index = None
        self.segments: List[Segment] = []
        # Tokenized in lockstep with `segments` (same indices) so `add()` only
        # has to tokenize the segments actually being added, instead of the
        # `bm25` property re-tokenizing the whole corpus on every mutation.
        self._tokens: List[List[str]] = []
        # How many of `self.segments` are already durably written to
        # segments.db as of the last save()/load(). Lets save() append only
        # the new rows instead of rewriting the whole metadata table, as long
        # as nothing was removed in between (see save()).
        self._store_synced_count: int = 0
        self._removed_since_save: bool = False

    # --- Lazy heavy dependencies --------------------------------------------

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %r", self.config.embedding_model)
            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model

    @property
    def faiss(self):
        if self._faiss is None:
            import faiss

            self._faiss = faiss
        return self._faiss

    @property
    def bm25(self) -> BM25:
        """Built on demand, invalidated whenever segments change.

        Rebuilt from the `_tokens` cache rather than re-tokenizing every
        segment's text: BM25 itself has no incremental-IDF form (adding a
        document changes every existing document's length-normalization
        term), so a rebuild is unavoidable, but the tokenization work for
        already-seen segments is not.
        """
        if self._bm25 is None:
            self._bm25 = BM25(self._tokens)
        return self._bm25

    # --- Embedding -----------------------------------------------------------

    def _embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if is_query and self.config.query_instruction:
            texts = [self.config.query_instruction + t for t in texts]
        emb = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.config.embedding_batch_size,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)

    # --- Mutation ------------------------------------------------------------

    def add(self, segments: List[Segment]) -> None:
        if not segments:
            return
        vectors = self._embed([s.text for s in segments])
        if self.index is None:
            self.index = self.faiss.IndexFlatIP(vectors.shape[1])
        elif vectors.shape[1] != self.index.d:
            raise IndexCompatibilityError(
                f"Embedding dimension {vectors.shape[1]} does not match the "
                f"existing index ({self.index.d}). The embedding model changed; "
                f"re-index from scratch."
            )
        self.index.add(vectors)
        self.segments.extend(segments)
        self._tokens.extend(tokenize(s.text) for s in segments)
        self._bm25 = None

    def remove_video(self, video_name: str) -> int:
        """Drop every segment belonging to `video_name`. Returns how many.

        Needed for re-indexing: without it, indexing a video twice leaves both
        copies in the index and every hit is silently duplicated.
        """
        keep = [i for i, s in enumerate(self.segments) if s.video_name != video_name]
        removed = len(self.segments) - len(keep)
        if not removed:
            return 0

        if keep and self.index is not None:
            # IndexFlat stores raw vectors, so rebuild from the kept rows
            # instead of re-embedding text we've already embedded.
            all_vectors = np.asarray(
                self.index.reconstruct_n(0, self.index.ntotal), dtype=np.float32
            )
            rebuilt = self.faiss.IndexFlatIP(self.index.d)
            rebuilt.add(all_vectors[keep])
            self.index = rebuilt
        else:
            self.index = None

        self.segments = [self.segments[i] for i in keep]
        self._tokens = [self._tokens[i] for i in keep]
        self._bm25 = None
        self._removed_since_save = True
        logger.info("Removed %d segment(s) for %r", removed, video_name)
        return removed

    # --- Search --------------------------------------------------------------

    def search(self, query: str, top_k: int = 8) -> List[SearchHit]:
        """Hybrid search: vector + BM25 candidates fused by reciprocal rank."""
        if self.index is None or not self.segments:
            return []

        depth = min(len(self.segments), max(top_k * self.config.candidate_multiplier, top_k))
        query_vector = self._embed([query], is_query=True)

        scores, indices = self.index.search(query_vector, depth)
        vector_ranking = [int(i) for i in indices[0].tolist() if i != -1]
        cosine = {int(i): float(s) for s, i in zip(scores[0], indices[0]) if i != -1}

        keyword_ranking: List[int] = []
        if self.config.keyword_weight > 0:
            keyword_ranking = [i for i, _ in self.bm25.top_n(tokenize(query), depth)]

        fused = reciprocal_rank_fusion(
            [vector_ranking, keyword_ranking],
            k=self.config.rrf_k,
            weights=[self.config.vector_weight, self.config.keyword_weight],
        )
        if not fused:
            return []

        # Keyword-only candidates have no cosine score yet, but thresholding
        # and reporting both need one. Reconstruct the stored vector rather
        # than re-embedding the text.
        for doc_id in fused:
            if doc_id not in cosine:
                stored = np.asarray(self.index.reconstruct(int(doc_id)), dtype=np.float32)
                cosine[doc_id] = float(stored @ query_vector[0])

        vector_set, keyword_set = set(vector_ranking), set(keyword_ranking)
        hits = [
            SearchHit(
                segment=self.segments[doc_id],
                score=cosine[doc_id],
                fused=fused_score,
                found_by=tuple(
                    name
                    for name, member in (("vector", doc_id in vector_set),
                                         ("keyword", doc_id in keyword_set))
                    if member
                ),
            )
            for doc_id, fused_score in fused.items()
        ]
        # Order by fusion, break ties on raw similarity.
        hits.sort(key=lambda h: (-h.fused, -h.score))
        return hits[:top_k]

    # --- Persistence -----------------------------------------------------
    #
    # Metadata (which video, what second, what text) is stored in SQLite
    # rather than a JSON array. The practical difference shows up in
    # `video_rag.store.inspect()`: answering "what's indexed here" needs
    # neither FAISS nor torch, just the standard library's sqlite3 module.
    #
    # The FAISS half of every save() still serializes the *entire* vector
    # index — that's inherent to FAISS's file format, which has no
    # incremental/append form. The SQLite half doesn't have that
    # constraint, so save() takes one of two paths for it:
    #
    #   - nothing removed since the last save: append only the segments
    #     added since then (their faiss_position values are still valid,
    #     since remove_video() is the only thing that renumbers them) —
    #     the common case when a batch just keeps adding more videos.
    #   - a remove_video() happened since the last save: full compact(),
    #     since that call already renumbered every faiss_position and the
    #     on-disk rows must follow.
    #
    # The full-rewrite path still goes through a tmp file + os.replace() so
    # an interrupted save can't leave a half-written file. The append path
    # writes directly to the live file: copying the whole existing file
    # before every append would reintroduce the O(n) cost this exists to
    # avoid. It relies on SQLite's own transaction (a rollback-journal that
    # auto-recovers the pre-append state on the next open if the process
    # dies mid-write) plus the integrity check in load().

    def save(self, path: str) -> None:
        if self.index is None:
            logger.warning("Nothing to save: index is empty")
            return

        # Identifies this specific save. Written into segments.db's meta
        # table below and cross-checked against the manifest on load(): if
        # they don't match, the store and the manifest belong to different
        # save() calls, which only happens when one was interrupted.
        generation = uuid.uuid4().hex

        os.makedirs(path, exist_ok=True)
        # Write to a temporary file and swap, so an interrupted save can't
        # leave a half-written index that fails to load next run.
        tmp_index = os.path.join(path, INDEX_FILE + ".tmp")
        self.faiss.write_index(self.index, tmp_index)
        os.replace(tmp_index, os.path.join(path, INDEX_FILE))

        store_path = os.path.join(path, STORE_FILE)
        if self._removed_since_save:
            tmp_store = store_path + ".tmp"
            Path(tmp_store).unlink(missing_ok=True)
            with SegmentStore(tmp_store) as store:
                store.compact(self.segments)
                self._set_store_meta(store, generation)
            os.replace(tmp_store, store_path)
        else:
            new_segments = self.segments[self._store_synced_count :]
            with SegmentStore(store_path) as store:
                store.append(new_segments, start_position=self._store_synced_count)
                self._set_store_meta(store, generation)

        # The true commit point: written last, after both files above are
        # already durably in place. A crash before this line means the
        # manifest doesn't reflect this save, which load() treats as
        # incomplete rather than trusting whatever's on disk.
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generation": generation,
            "segment_count": len(self.segments),
            "embedding_model": self.config.embedding_model,
            "dimension": self.index.d,
            "complete": True,
        }
        tmp_manifest = os.path.join(path, MANIFEST_FILE + ".tmp")
        Path(tmp_manifest).write_text(json.dumps(manifest), encoding="utf-8")
        os.replace(tmp_manifest, os.path.join(path, MANIFEST_FILE))

        self._store_synced_count = len(self.segments)
        self._removed_since_save = False
        logger.info("Saved %d segment(s) to %s", len(self.segments), path)

    def _set_store_meta(self, store: SegmentStore, generation: str) -> None:
        store.set_meta("schema_version", str(SCHEMA_VERSION))
        store.set_meta("embedding_model", self.config.embedding_model)
        store.set_meta("dimension", str(self.index.d))
        store.set_meta("generation", generation)

    def load(self, path: str) -> bool:
        index_file = os.path.join(path, INDEX_FILE)
        store_file = os.path.join(path, STORE_FILE)
        if not (os.path.exists(index_file) and os.path.exists(store_file)):
            return False

        with SegmentStore(store_file) as store:
            saved_model = store.get_meta("embedding_model")
            saved_schema_version = store.get_meta("schema_version")
            saved_generation = store.get_meta("generation")
            segments = store.all_segments()

        if saved_model and saved_model != self.config.embedding_model:
            raise IndexCompatibilityError(
                f"Index at {path} was built with embedding model {saved_model!r}, "
                f"but the configured model is {self.config.embedding_model!r}. "
                f"Vectors from different models are not comparable - delete the "
                f"index directory and re-index, or set "
                f"VIDEORAG_EMBEDDING_MODEL={saved_model!r}."
            )

        index = self.faiss.read_index(index_file)
        if index.ntotal != len(segments):
            raise IndexCompatibilityError(
                f"Index at {path} is inconsistent: {index.ntotal} vector(s) but "
                f"{len(segments)} segment(s). Delete the index directory and re-index."
            )

        self._check_manifest(path, index, segments, saved_schema_version, saved_generation)

        self.index = index
        self.segments = segments
        self._tokens = [tokenize(s.text) for s in segments]
        self._bm25 = None
        # Everything just loaded is, by definition, already durable.
        self._store_synced_count = len(segments)
        self._removed_since_save = False
        logger.info("Loaded %d segment(s) from %s", len(segments), path)
        return True

    def _check_manifest(
        self,
        path: str,
        index,
        segments: List[Segment],
        saved_schema_version: Optional[str],
        saved_generation: Optional[str],
    ) -> None:
        """Refuse to load a save() that was interrupted before completion.

        The ntotal/segment-count check in load() catches most incomplete
        saves, but not the narrow case of a crash between the SQLite and
        manifest writes followed by a later, unrelated save whose count
        happens to coincide - only the generation check below catches that.
        """
        manifest_path = os.path.join(path, MANIFEST_FILE)
        version = int(saved_schema_version) if saved_schema_version else 0

        if not os.path.exists(manifest_path):
            if version < 4:
                # Saved by a version of this code that predates the
                # manifest; nothing more to check than what load() already
                # verified above.
                return
            raise IndexCompatibilityError(
                f"Index at {path} has no manifest but claims schema version "
                f"{version} - the save that produced it was interrupted before "
                f"completion. Delete the index directory and re-index."
            )

        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        mismatches = []
        if not manifest.get("complete"):
            mismatches.append("manifest is marked incomplete")
        if manifest.get("generation") != saved_generation:
            mismatches.append("manifest generation does not match the saved store")
        if manifest.get("segment_count") != len(segments):
            mismatches.append("manifest segment count does not match the saved store")
        if manifest.get("dimension") != index.d:
            mismatches.append("manifest dimension does not match the saved index")

        if mismatches:
            raise IndexCompatibilityError(
                f"Index at {path} failed its integrity check ({'; '.join(mismatches)}) "
                f"- the save that produced it was interrupted before completion. "
                f"Delete the index directory and re-index."
            )
