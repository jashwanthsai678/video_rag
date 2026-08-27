"""Hybrid index integration: real FAISS, stubbed embedding model.

Skipped when faiss/numpy are absent so the core suite stays dependency-free.
The embedding model is faked — a deterministic bag-of-words encoder — because
the goal is to test *our* wiring (fusion, rebuild-on-remove, persistence,
compatibility guards), not the quality of somebody else's weights.
"""

import re

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("faiss")

from video_rag.config import Config  # noqa: E402
from video_rag.index import IndexCompatibilityError, SemanticIndex  # noqa: E402
from video_rag.retrieval import tokenize  # noqa: E402
from video_rag.segment import Segment  # noqa: E402
from video_rag.store import SegmentStore, inspect  # noqa: E402

VOCAB = [
    "patient", "new", "record", "export", "report",
    "pdf", "billing", "save", "window", "insurance",
]


class FakeEncoder:
    """Bag-of-words over a fixed vocabulary, L2-normalized.

    Cosine similarity then equals normalized token overlap: predictable
    enough to assert on, and it deliberately assigns *zero* similarity to
    any out-of-vocabulary term — which is how we prove BM25 is pulling its
    weight in the fusion.
    """

    def __init__(self):
        self.seen = []

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        self.seen.extend(texts)
        rows = []
        for text in texts:
            vector = np.zeros(len(VOCAB), dtype=np.float32)
            tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
            for i, word in enumerate(VOCAB):
                if word in tokens:
                    vector[i] = 1.0
            norm = float(np.linalg.norm(vector))
            if norm:
                vector /= norm
            rows.append(vector)
        return np.vstack(rows)


SEGMENT_TEXTS = [
    ("patients", 10.0, "Click New then Patient Record"),
    ("patients", 40.0, "Patient billing and insurance"),
    ("patients", 900.0, "Save and close the window"),
    ("reports", 20.0, "Export the report as PDF"),
    ("reports", 50.0, "Click the Reconcile button"),  # no VOCAB terms at all
]


def make_index(**config_overrides):
    config = Config(**config_overrides)
    index = SemanticIndex(config)
    index._model = FakeEncoder()  # bypass the lazy real model
    index.add([
        Segment(
            video_name=video, video_path=f"videos/{video}.mp4",
            timestamp=at, text=text, source="ocr",
        )
        for video, at, text in SEGMENT_TEXTS
    ])
    return index


class TestSearch:
    def test_finds_the_semantically_matching_segment(self):
        hits = make_index().search("new patient record", top_k=3)
        assert hits[0].segment.text == "Click New then Patient Record"
        assert hits[0].score > 0.5

    def test_applies_the_query_instruction_prefix(self):
        """BGE is asymmetric: the prefix goes on queries, never on passages."""
        index = make_index()
        prefix = index.config.query_instruction
        assert not any(t.startswith(prefix) for t in index._model.seen)  # indexing
        index.search("new patient", top_k=2)
        assert index._model.seen[-1] == prefix + "new patient"

    def test_no_prefix_when_configured_empty(self):
        index = make_index(query_instruction="")
        index.search("new patient", top_k=2)
        assert index._model.seen[-1] == "new patient"

    def test_keyword_retriever_finds_what_the_vector_cannot(self):
        """'Reconcile' is outside the fake vocabulary, so its embedding is a
        zero vector — only BM25 can surface it. This is the whole reason
        hybrid retrieval exists for UI text."""
        hits = make_index().search("reconcile", top_k=5)
        reconcile = [h for h in hits if "Reconcile" in h.segment.text]
        assert reconcile, "keyword retriever failed to surface an exact match"
        assert "keyword" in reconcile[0].found_by

    def test_keyword_weight_zero_disables_bm25(self):
        hits = make_index(keyword_weight=0.0).search("reconcile", top_k=5)
        assert all("keyword" not in h.found_by for h in hits)

    def test_every_hit_carries_a_cosine_score(self):
        """Keyword-only candidates must still get a real similarity, or
        thresholding and reporting silently break."""
        hits = make_index().search("reconcile", top_k=5)
        assert all(isinstance(h.score, float) and -1.0 <= h.score <= 1.0 for h in hits)

    def test_results_are_ordered_by_fusion_score(self):
        hits = make_index().search("patient record", top_k=5)
        assert [h.fused for h in hits] == sorted((h.fused for h in hits), reverse=True)

    def test_respects_top_k(self):
        assert len(make_index().search("patient", top_k=2)) == 2

    def test_empty_index_returns_nothing(self):
        empty = SemanticIndex(Config())
        empty._model = FakeEncoder()
        assert empty.search("anything") == []


class TestRemoveVideo:
    def test_removes_only_the_named_video(self):
        index = make_index()
        removed = index.remove_video("reports")
        assert removed == 2
        assert {s.video_name for s in index.segments} == {"patients"}
        assert index.index.ntotal == 3

    def test_removed_segments_no_longer_surface(self):
        index = make_index()
        index.remove_video("reports")
        hits = index.search("export report pdf", top_k=5)
        assert all(h.segment.video_name == "patients" for h in hits)

    def test_surviving_vectors_are_preserved_not_re_embedded(self):
        """Rebuilding must reuse the stored vectors: a segment's similarity
        should be identical before and after an unrelated removal."""
        index = make_index()
        before = index.search("new patient record", top_k=1)[0].score
        index.remove_video("reports")
        after = index.search("new patient record", top_k=1)[0].score
        assert after == pytest.approx(before)

    def test_removing_everything_empties_the_index(self):
        index = make_index()
        index.remove_video("patients")
        index.remove_video("reports")
        assert index.segments == []
        assert index.index is None
        assert index.search("patient") == []

    def test_unknown_video_is_a_no_op(self):
        index = make_index()
        assert index.remove_video("nope") == 0
        assert len(index.segments) == len(SEGMENT_TEXTS)

    def test_reindexing_a_video_does_not_duplicate_it(self):
        """Regression: --force used to append, leaving two copies of every
        segment and silently doubling each hit."""
        index = make_index()
        index.remove_video("reports")
        index.add([
            Segment(
                video_name="reports", video_path="videos/reports.mp4",
                timestamp=20.0, text="Export the report as PDF", source="ocr",
            )
        ])
        assert sum(s.video_name == "reports" for s in index.segments) == 1
        assert index.index.ntotal == len(index.segments)


class TestPersistence:
    def test_roundtrip(self, tmp_path):
        index = make_index(index_path=str(tmp_path / "idx"))
        index.save(str(tmp_path / "idx"))

        reloaded = SemanticIndex(index.config)
        reloaded._model = FakeEncoder()
        assert reloaded.load(str(tmp_path / "idx")) is True
        assert len(reloaded.segments) == len(SEGMENT_TEXTS)
        assert reloaded.search("new patient record", top_k=1)[0].segment.timestamp == 10.0

    def test_load_returns_false_when_nothing_is_saved(self, tmp_path):
        index = SemanticIndex(Config())
        assert index.load(str(tmp_path / "missing")) is False

    def test_save_leaves_no_temporary_files(self, tmp_path):
        index = make_index()
        index.save(str(tmp_path / "idx"))
        assert not list((tmp_path / "idx").glob("*.tmp"))

    def test_metadata_records_the_embedding_model(self, tmp_path):
        """Metadata is queryable via `inspect()` without loading FAISS/torch."""
        index = make_index()
        index.save(str(tmp_path / "idx"))
        report = inspect(str(tmp_path / "idx"))
        assert report["embedding_model"] == index.config.embedding_model
        assert report["total_segments"] == len(SEGMENT_TEXTS)
        assert report["videos"] == {"patients": 3, "reports": 2}

    def test_changing_the_embedding_model_is_refused(self, tmp_path):
        """Regression: this used to load happily and then either assert deep
        inside FAISS or return meaningless neighbours."""
        index = make_index()
        index.save(str(tmp_path / "idx"))

        other = SemanticIndex(Config(embedding_model="some/other-model"))
        with pytest.raises(IndexCompatibilityError, match="embedding model"):
            other.load(str(tmp_path / "idx"))

    def test_vector_and_segment_count_mismatch_is_refused(self, tmp_path):
        index = make_index()
        index.save(str(tmp_path / "idx"))
        # Simulate a truncated store: fewer metadata rows than FAISS vectors.
        with SegmentStore(str(tmp_path / "idx" / "segments.db")) as store:
            store.compact(store.all_segments()[:2])

        reloaded = SemanticIndex(index.config)
        with pytest.raises(IndexCompatibilityError, match="inconsistent"):
            reloaded.load(str(tmp_path / "idx"))

    def test_saving_an_empty_index_is_harmless(self, tmp_path):
        SemanticIndex(Config()).save(str(tmp_path / "idx"))
        assert not (tmp_path / "idx" / "index.faiss").exists()

    def test_non_ascii_text_survives_a_roundtrip(self, tmp_path):
        index = make_index()
        index.add([
            Segment(
                video_name="patients", video_path="v.mp4", timestamp=5.0,
                text="Guardar paciente — Über", source="ocr",
            )
        ])
        index.save(str(tmp_path / "idx"))

        reloaded = SemanticIndex(index.config)
        reloaded._model = FakeEncoder()
        reloaded.load(str(tmp_path / "idx"))
        assert any("Über" in s.text for s in reloaded.segments)


class TestAddGuards:
    def test_adding_nothing_is_a_no_op(self):
        index = make_index()
        before = index.index.ntotal
        index.add([])
        assert index.index.ntotal == before

    def test_dimension_change_raises_instead_of_corrupting(self):
        index = make_index()

        class WiderEncoder(FakeEncoder):
            def encode(self, texts, **kwargs):
                return np.zeros((len(texts), len(VOCAB) + 5), dtype=np.float32)

        index._model = WiderEncoder()
        with pytest.raises(IndexCompatibilityError, match="dimension"):
            index.add([
                Segment(
                    video_name="x", video_path="x.mp4", timestamp=0.0,
                    text="text", source="ocr",
                )
            ])

    def test_bm25_is_rebuilt_after_a_mutation(self):
        index = make_index()
        index.search("reconcile", top_k=5)  # forces BM25 construction
        index.add([
            Segment(
                video_name="reports", video_path="videos/reports.mp4",
                timestamp=70.0, text="Reconcile ledger entries", source="ocr",
            )
        ])
        hits = index.search("reconcile ledger", top_k=5)
        assert any("ledger" in h.segment.text for h in hits)


class TestTokenCache:
    """`_tokens` is kept parallel to `segments` so `bm25` never has to
    re-tokenize already-seen text on every mutation."""

    def test_tokens_stay_aligned_with_segments_after_add(self):
        index = make_index()
        assert len(index._tokens) == len(index.segments)
        assert index._tokens[0] == tokenize(index.segments[0].text)

    def test_tokens_stay_aligned_with_segments_after_remove(self):
        index = make_index()
        index.remove_video("reports")
        assert len(index._tokens) == len(index.segments)
        assert index._tokens == [tokenize(s.text) for s in index.segments]

    def test_tokens_survive_a_save_load_roundtrip(self, tmp_path):
        index = make_index()
        index.save(str(tmp_path / "idx"))

        reloaded = SemanticIndex(index.config)
        reloaded._model = FakeEncoder()
        reloaded.load(str(tmp_path / "idx"))
        assert reloaded._tokens == [tokenize(s.text) for s in reloaded.segments]

    def test_removing_everything_empties_the_token_cache(self):
        index = make_index()
        index.remove_video("patients")
        index.remove_video("reports")
        assert index._tokens == []
