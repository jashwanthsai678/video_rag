"""Clip selection: which video, which moment, and how wide a window.

The index is stubbed with canned hits so these tests pin the *selection*
logic — the part that decides what the user actually receives.
"""

import pytest

pytest.importorskip("cv2")
pytest.importorskip("faiss")

from video_rag.config import Config  # noqa: E402
from video_rag.index import SearchHit  # noqa: E402
from video_rag.pipeline import VideoRAGPipeline  # noqa: E402
from video_rag.segment import Segment  # noqa: E402


class StubIndex:
    """Stands in for SemanticIndex: returns canned hits, records nothing."""

    def __init__(self, hits):
        self.hits = hits
        self.segments = [h.segment for h in hits]

    def search(self, query, top_k=8):
        return self.hits[:top_k]

    def load(self, path):
        return False


def hit(video, at, score, duration=0.0, text="text"):
    return SearchHit(
        segment=Segment(
            video_name=video,
            video_path=f"videos/{video}.mp4",
            timestamp=at,
            end_timestamp=(at + duration) if duration else None,
            text=text,
            source="ocr",
        ),
        score=score,
        fused=score,
        found_by=("vector",),
    )


def build(hits, tmp_path, **overrides):
    config = Config(
        index_path=str(tmp_path / "idx"),
        clips_dir=str(tmp_path / "clips"),
        data_dir=str(tmp_path / "data"),
        frames_dir=str(tmp_path / "frames"),
        llm_enabled=False,
        **overrides,
    )
    pipeline = VideoRAGPipeline(config)
    pipeline.index = StubIndex(hits)
    return pipeline


class TestClipWindow:
    def test_distant_matches_do_not_produce_one_enormous_clip(self, tmp_path):
        """Regression: min-to-max over every match turned a ~10s answer into
        a 14-minute clip whenever the query also hit somewhere unrelated."""
        pipeline = build(
            [
                hit("patients", 12.0, 0.88),
                hit("patients", 18.0, 0.85),
                hit("patients", 870.0, 0.72),  # unrelated later mention
            ],
            tmp_path,
        )
        found = pipeline.locate("how do i add a patient")

        assert found.clusters_found == 2
        assert found.end_seconds - found.start_seconds < 40
        assert found.start_seconds <= 12.0 and found.end_seconds >= 18.0
        assert all(s.timestamp < 100 for s in found.matched_segments)

    def test_picks_the_strongest_cluster_not_the_earliest(self, tmp_path):
        pipeline = build(
            [
                hit("patients", 5.0, 0.60),
                hit("patients", 600.0, 0.92),
                hit("patients", 604.0, 0.90),
            ],
            tmp_path,
        )
        found = pipeline.locate("q")
        assert found.start_seconds > 500

    def test_a_contiguous_run_stays_whole(self, tmp_path):
        pipeline = build(
            [hit("patients", t, 0.8) for t in (10.0, 25.0, 40.0)],
            tmp_path,
            cluster_max_gap_seconds=30.0,
        )
        found = pipeline.locate("q")
        assert found.clusters_found == 1
        assert len(found.matched_segments) == 3

    def test_buffer_pads_both_ends(self, tmp_path):
        pipeline = build([hit("patients", 60.0, 0.9)], tmp_path, clip_buffer_seconds=4.0)
        found = pipeline.locate("q")
        assert found.start_seconds == pytest.approx(56.0)
        assert found.end_seconds == pytest.approx(64.0)

    def test_start_never_goes_negative(self, tmp_path):
        pipeline = build([hit("patients", 1.0, 0.9)], tmp_path, clip_buffer_seconds=10.0)
        assert build([hit("patients", 1.0, 0.9)], tmp_path).locate("q").start_seconds >= 0.0
        assert pipeline.locate("q").start_seconds == 0.0

    def test_window_is_capped_by_max_clip_seconds(self, tmp_path):
        pipeline = build(
            [hit("patients", 0.0, 0.9), hit("patients", 500.0, 0.9)],
            tmp_path,
            cluster_max_gap_seconds=10_000.0,  # force them into one cluster
            max_clip_seconds=60.0,
        )
        found = pipeline.locate("q")
        assert found.end_seconds - found.start_seconds == pytest.approx(60.0)

    def test_speech_segment_duration_is_included(self, tmp_path):
        pipeline = build(
            [hit("patients", 30.0, 0.9, duration=20.0)], tmp_path, clip_buffer_seconds=0.0
        )
        found = pipeline.locate("q")
        assert found.end_seconds == pytest.approx(50.0)


class TestVideoSelection:
    def test_concentrated_evidence_wins(self, tmp_path):
        """Regression: averaging per video meant three strong hits lost to a
        single weaker one."""
        pipeline = build(
            [
                hit("correct", 10.0, 0.85),
                hit("correct", 12.0, 0.80),
                hit("correct", 14.0, 0.75),
                hit("correct", 16.0, 0.40),
                hit("correct", 18.0, 0.35),
                hit("wrong", 10.0, 0.70),
            ],
            tmp_path,
        )
        assert pipeline.locate("q").source_video == "correct"

    def test_confidence_is_the_best_single_hit(self, tmp_path):
        pipeline = build(
            [hit("patients", 10.0, 0.91), hit("patients", 12.0, 0.40)], tmp_path
        )
        assert pipeline.locate("q").score == pytest.approx(0.91)

    def test_matched_segments_come_from_one_video_only(self, tmp_path):
        pipeline = build(
            [hit("a", 10.0, 0.9), hit("a", 12.0, 0.9), hit("b", 11.0, 0.5)], tmp_path
        )
        found = pipeline.locate("q")
        assert {s.video_name for s in found.matched_segments} == {"a"}


class TestThreshold:
    def test_weak_matches_are_rejected(self, tmp_path):
        pipeline = build(
            [hit("patients", 10.0, 0.30)], tmp_path, relevance_threshold=0.55
        )
        found = pipeline.locate("q")
        assert found.status == "no_match"
        assert found.score == pytest.approx(0.30)
        assert found.hits, "hits should still be reported for debugging"

    def test_a_match_at_the_threshold_is_accepted(self, tmp_path):
        pipeline = build(
            [hit("patients", 10.0, 0.55)], tmp_path, relevance_threshold=0.55
        )
        assert pipeline.locate("q").status == "success"

    def test_no_hits_at_all(self, tmp_path):
        pipeline = build([], tmp_path)
        assert pipeline.locate("q").status == "no_match"


class TestQuery:
    def test_query_reuses_locate_and_adds_an_explanation(self, tmp_path):
        pipeline = build([hit("patients", 10.0, 0.9, text="Click New")], tmp_path)
        result = pipeline.query("q")
        assert result.status == "success"
        # llm_enabled=False, so this is the deterministic fallback listing.
        assert "Click New" in result.explanation

    def test_query_short_circuits_on_no_match(self, tmp_path):
        pipeline = build([hit("patients", 10.0, 0.1)], tmp_path, relevance_threshold=0.9)
        result = pipeline.query("q")
        assert result.status == "no_match"
        assert result.explanation == ""
        assert result.clip_path is None

    def test_missing_source_video_skips_the_clip_without_failing(self, tmp_path):
        pipeline = build([hit("patients", 10.0, 0.9)], tmp_path)
        result = pipeline.query("How do I add a new patient?")
        assert result.status == "success"
        assert result.clip_path is None  # videos/patients.mp4 does not exist
