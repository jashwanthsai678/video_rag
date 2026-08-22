"""The evaluation harness itself — the numbers have to be trustworthy."""

import json
from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from video_rag.evaluate import Label, Report, evaluate, load_labels
from video_rag.segment import Segment

# --- Stubs: the harness only needs `.config.top_k` and `.locate()` -----------


@dataclass
class FakeHit:
    segment: Segment
    score: float = 0.8


@dataclass
class FakeLocated:
    status: str = "success"
    source_video: Optional[str] = None
    score: float = 0.8
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    hits: List[FakeHit] = field(default_factory=list)


class FakeConfig:
    top_k = 8


class FakePipeline:
    """Returns a canned result per question."""

    def __init__(self, responses):
        self.config = FakeConfig()
        self.responses = responses
        self.videos = {"patients": "videos/patients.mp4"}
        self.calls = []

    def locate(self, question, top_k=None):
        self.calls.append((question, top_k))
        return self.responses[question]


def moment(video, at, duration=0.0, score=0.8):
    return FakeHit(
        segment=Segment(
            video_name=video,
            video_path=f"videos/{video}.mp4",
            timestamp=at,
            end_timestamp=at + duration if duration else None,
            text="text",
            source="ocr",
        ),
        score=score,
    )


# --- Label parsing -----------------------------------------------------------


class TestLabel:
    def test_positive_label(self):
        label = Label.from_dict({"question": "q", "video": "patients", "at": 72})
        assert label.at == 72.0
        assert not label.is_negative

    def test_negative_label_has_no_video(self):
        assert Label.from_dict({"question": "q", "video": None}).is_negative

    def test_missing_video_key_is_treated_as_negative(self):
        assert Label.from_dict({"question": "q"}).is_negative

    def test_tolerance_defaults_and_overrides(self):
        assert Label.from_dict({"question": "q", "video": "v"}).tolerance == 15.0
        assert Label.from_dict({"question": "q", "video": "v", "tolerance": 3}).tolerance == 3.0

    def test_load_labels_rejects_a_non_list(self, tmp_path):
        path = tmp_path / "labels.json"
        path.write_text(json.dumps({"question": "q"}), encoding="utf-8")
        with pytest.raises(ValueError, match="expected a JSON list"):
            load_labels(str(path))

    def test_load_labels_reads_a_list(self, tmp_path):
        path = tmp_path / "labels.json"
        path.write_text(
            json.dumps([{"question": "q", "video": "patients", "at": 1}]), encoding="utf-8"
        )
        assert len(load_labels(str(path))) == 1


# --- Metrics -----------------------------------------------------------------


class TestEvaluate:
    def test_counts_a_hit_within_tolerance(self):
        pipeline = FakePipeline({
            "q": FakeLocated(
                source_video="patients", start_seconds=60, end_seconds=90,
                hits=[moment("patients", 72.0)],
            )
        })
        report = evaluate(pipeline, [Label("q", "patients", 72.0)])
        assert report.recall_at_k == 1.0
        assert report.video_accuracy == 1.0
        assert report.clip_accuracy == 1.0
        assert report.mrr == pytest.approx(1.0)

    def test_a_hit_outside_tolerance_is_a_miss(self):
        pipeline = FakePipeline({
            "q": FakeLocated(
                source_video="patients", start_seconds=0, end_seconds=10,
                hits=[moment("patients", 5.0)],
            )
        })
        report = evaluate(pipeline, [Label("q", "patients", 500.0, tolerance=15.0)])
        assert report.recall_at_k == 0.0
        assert report.missed == ["q"]

    def test_mrr_reflects_the_rank_of_the_first_correct_hit(self):
        pipeline = FakePipeline({
            "q": FakeLocated(
                source_video="patients", start_seconds=60, end_seconds=90,
                hits=[moment("other", 1.0), moment("patients", 72.0)],
            )
        })
        report = evaluate(pipeline, [Label("q", "patients", 72.0)])
        assert report.mrr == pytest.approx(0.5)

    def test_right_video_but_wrong_window_fails_clip_accuracy_only(self):
        """The distinction that matters: retrieval found it, clipping lost it."""
        pipeline = FakePipeline({
            "q": FakeLocated(
                source_video="patients", start_seconds=0, end_seconds=20,
                hits=[moment("patients", 72.0)],
            )
        })
        report = evaluate(pipeline, [Label("q", "patients", 72.0)])
        assert report.recall_at_k == 1.0
        assert report.video_accuracy == 1.0
        assert report.clip_accuracy == 0.0

    def test_long_segments_count_as_hits_across_their_span(self):
        pipeline = FakePipeline({
            "q": FakeLocated(
                source_video="patients", start_seconds=0, end_seconds=200,
                hits=[moment("patients", 60.0, duration=60.0)],
            )
        })
        report = evaluate(pipeline, [Label("q", "patients", 100.0, tolerance=0.0)])
        assert report.recall_at_k == 1.0

    def test_negative_label_answered_is_a_false_positive(self):
        pipeline = FakePipeline({
            "q": FakeLocated(status="success", source_video="patients", score=0.9)
        })
        report = evaluate(pipeline, [Label("q", None)])
        assert report.negatives == 1
        assert report.false_positives == 1
        assert report.specificity == 0.0

    def test_negative_label_rejected_is_correct(self):
        pipeline = FakePipeline({"q": FakeLocated(status="no_match", score=0.31)})
        report = evaluate(pipeline, [Label("q", None)])
        assert report.specificity == 1.0
        assert report.negative_scores == [0.31]

    def test_negatives_are_excluded_from_recall(self):
        pipeline = FakePipeline({
            "pos": FakeLocated(
                source_video="patients", start_seconds=0, end_seconds=10,
                hits=[moment("patients", 5.0)],
            ),
            "neg": FakeLocated(status="no_match", score=0.2),
        })
        report = evaluate(pipeline, [Label("pos", "patients", 5.0), Label("neg", None)])
        assert report.positives == 1
        assert report.negatives == 1
        assert report.recall_at_k == 1.0

    def test_top_k_is_forwarded(self):
        pipeline = FakePipeline({"q": FakeLocated(status="no_match")})
        evaluate(pipeline, [Label("q", None)], top_k=3)
        assert pipeline.calls == [("q", 3)]


class TestReportFormatting:
    def test_empty_report_does_not_divide_by_zero(self):
        report = Report()
        assert report.recall_at_k == 0.0
        assert report.mrr == 0.0
        assert report.specificity == 0.0
        assert "Recall@8" in report.format(8)

    def test_threshold_suggestion_sits_between_the_distributions(self):
        report = Report(
            positives=2, negatives=1,
            positive_scores=[0.70, 0.80], negative_scores=[0.40],
        )
        text = report.format(8)
        assert "suggested threshold: 0.55" in text

    def test_missed_questions_are_truncated_in_the_output(self):
        report = Report(positives=40, missed=[f"q{i}" for i in range(40)])
        text = report.format(8)
        assert "and 25 more" in text
