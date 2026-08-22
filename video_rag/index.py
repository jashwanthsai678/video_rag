"""Hybrid (vector + keyword) index over Segments, backed by FAISS and BM25."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .config import Config
from .retrieval import BM25, reciprocal_rank_fusion, tokenize
from .segment import Segment

logger = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
SEGMENTS_FILE = "segments.json"
META_FILE = "meta.json"
SCHEMA_VERSION = 2


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
        """Built on demand, invalidated whenever segments change."""
        if self._bm25 is None:
            self._bm25 = BM25([tokenize(s.text) for s in self.segments])
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
        self._bm25 = None
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

    # --- Persistence ---------------------------------------------------------

    def _metadata(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "embedding_model": self.config.embedding_model,
            "dimension": int(self.index.d) if self.index is not None else 0,
            "segment_count": len(self.segments),
        }

    def save(self, path: str) -> None:
        if self.index is None:
            logger.warning("Nothing to save: index is empty")
            return

        os.makedirs(path, exist_ok=True)
        # Write to temporary files and swap, so an interrupted save can't leave
        # a half-written index that fails to load next run.
        tmp_index = os.path.join(path, INDEX_FILE + ".tmp")
        self.faiss.write_index(self.index, tmp_index)
        os.replace(tmp_index, os.path.join(path, INDEX_FILE))

        for name, payload in (
            (SEGMENTS_FILE, [s.to_dict() for s in self.segments]),
            (META_FILE, self._metadata()),
        ):
            tmp = os.path.join(path, name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, os.path.join(path, name))

        logger.info("Saved %d segment(s) to %s", len(self.segments), path)

    def load(self, path: str) -> bool:
        index_file = os.path.join(path, INDEX_FILE)
        segments_file = os.path.join(path, SEGMENTS_FILE)
        if not (os.path.exists(index_file) and os.path.exists(segments_file)):
            return False

        meta_file = os.path.join(path, META_FILE)
        meta = {}
        if os.path.exists(meta_file):
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)

        saved_model = meta.get("embedding_model")
        if saved_model and saved_model != self.config.embedding_model:
            raise IndexCompatibilityError(
                f"Index at {path} was built with embedding model {saved_model!r}, "
                f"but the configured model is {self.config.embedding_model!r}. "
                f"Vectors from different models are not comparable - delete the "
                f"index directory and re-index, or set "
                f"VIDEORAG_EMBEDDING_MODEL={saved_model!r}."
            )

        index = self.faiss.read_index(index_file)
        with open(segments_file, encoding="utf-8") as f:
            segments = [Segment.from_dict(d) for d in json.load(f)]

        if index.ntotal != len(segments):
            raise IndexCompatibilityError(
                f"Index at {path} is inconsistent: {index.ntotal} vector(s) but "
                f"{len(segments)} segment(s). Delete the index directory and re-index."
            )

        self.index = index
        self.segments = segments
        self._bm25 = None
        logger.info("Loaded %d segment(s) from %s", len(segments), path)
        return True
