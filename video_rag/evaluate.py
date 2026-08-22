"""Measure retrieval quality against hand-labelled question -> moment pairs.

Without this you cannot tell an improvement from a regression: every knob in
`Config` (threshold, fusion weights, scene detection, chunking) is a guess
until it moves a number. Labels are cheap — twenty questions and the
timestamps where they are answered is an afternoon's work.

Label file format (JSON list):

    [
      {"question": "How do I add a new patient?", "video": "patients", "at": 72.0},
      {"question": "How do I export a report?", "video": "reports", "at": 310.5,
       "tolerance": 20},
      {"question": "What is the airspeed of a swallow?", "video": null}
    ]

`video: null` marks a question the library should *not* answer — the only way
to calibrate `relevance_threshold` without guessing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    # Type-only: importing the pipeline for real would pull in torch/faiss/cv2,
    # and this module's scoring logic is testable without any of them.
    from .pipeline import VideoRAGPipeline

logger = logging.getLogger(__name__)

DEFAULT_TOLERANCE_SECONDS = 15.0


@dataclass
class Label:
    question: str
    video: Optional[str]  # None => should return no_match
    at: Optional[float] = None
    tolerance: float = DEFAULT_TOLERANCE_SECONDS

    @classmethod
    def from_dict(cls, d: dict) -> "Label":
        return cls(
            question=d["question"],
            video=d.get("video"),
            at=None if d.get("at") is None else float(d["at"]),
            tolerance=float(d.get("tolerance", DEFAULT_TOLERANCE_SECONDS)),
        )

    @property
    def is_negative(self) -> bool:
        return self.video is None


@dataclass
class Report:
    positives: int = 0
    negatives: int = 0
    hit_at_k: int = 0
    video_correct: int = 0
    clip_contains: int = 0
    reciprocal_ranks: List[float] = field(default_factory=list)
    positive_scores: List[float] = field(default_factory=list)
    negative_scores: List[float] = field(default_factory=list)
    false_positives: int = 0
    missed: List[str] = field(default_factory=list)

    def _ratio(self, numerator: int, denominator: int) -> float:
        return (numerator / denominator) if denominator else 0.0

    @property
    def recall_at_k(self) -> float:
        return self._ratio(self.hit_at_k, self.positives)

    @property
    def video_accuracy(self) -> float:
        return self._ratio(self.video_correct, self.positives)

    @property
    def clip_accuracy(self) -> float:
        return self._ratio(self.clip_contains, self.positives)

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank. Misses count as 0, so it divides by positives."""
        if not self.positives:
            return 0.0
        return sum(self.reciprocal_ranks) / self.positives

    @property
    def specificity(self) -> float:
        """Share of should-not-answer questions correctly rejected."""
        return self._ratio(self.negatives - self.false_positives, self.negatives)

    def format(self, top_k: int) -> str:
        lines = [
            f"Labels: {self.positives} positive, {self.negatives} negative",
            "",
            f"Recall@{top_k}      {self.recall_at_k:6.1%}   "
            f"(a correct moment appears anywhere in the top {top_k})",
            f"MRR             {self.mrr:6.3f}   (1.0 = correct moment always ranked first)",
            f"Video accuracy  {self.video_accuracy:6.1%}   (picked the right video)",
            f"Clip accuracy   {self.clip_accuracy:6.1%}   "
            f"(the returned clip actually contains the moment)",
        ]
        if self.negatives:
            lines.append(
                f"Specificity     {self.specificity:6.1%}   "
                f"(correctly refused to answer)"
            )

        if self.positive_scores or self.negative_scores:
            # ASCII only: this is printed to a terminal, and a Windows console
            # on a legacy codepage will mangle or reject non-ASCII output.
            lines += ["", "Score distribution - use this to set VIDEORAG_RELEVANCE_THRESHOLD:"]
            if self.positive_scores:
                lines.append(_distribution("  answerable    ", self.positive_scores))
            if self.negative_scores:
                lines.append(_distribution("  unanswerable  ", self.negative_scores))
            if self.positive_scores and self.negative_scores:
                lines.append(
                    f"  suggested threshold: "
                    f"{_suggest_threshold(self.positive_scores, self.negative_scores):.2f}"
                )

        if self.missed:
            lines += ["", "Missed:"]
            lines += [f"  - {q}" for q in self.missed[:15]]
            if len(self.missed) > 15:
                lines.append(f"  ... and {len(self.missed) - 15} more")
        return "\n".join(lines)


def _distribution(prefix: str, scores: List[float]) -> str:
    ordered = sorted(scores)
    lowest, highest = ordered[0], ordered[-1]
    median = ordered[len(ordered) // 2]
    return f"{prefix}min {lowest:.3f}  median {median:.3f}  max {highest:.3f}"


def _suggest_threshold(positives: List[float], negatives: List[float]) -> float:
    """Midpoint between the weakest true positive and the strongest negative.

    Crude on purpose: the point is to show the two distributions overlap (or
    don't), not to pretend a handful of labels supports anything fancier.
    """
    return max(0.0, (min(positives) + max(negatives)) / 2.0)


def load_labels(path: str) -> List[Label]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of label objects")
    return [Label.from_dict(d) for d in raw]


def evaluate(
    pipeline: "VideoRAGPipeline", labels: List[Label], top_k: Optional[int] = None
) -> Report:
    """Run every label through retrieval only — no LLM calls, no clips written."""
    k = top_k or pipeline.config.top_k
    report = Report()

    for label in labels:
        found = pipeline.locate(label.question, top_k=k)

        if label.is_negative:
            report.negatives += 1
            report.negative_scores.append(found.score)
            if found.status == "success":
                report.false_positives += 1
                logger.debug("False positive: %r -> %s", label.question, found.source_video)
            continue

        report.positives += 1
        report.positive_scores.append(found.score)

        # Rank of the first hit landing on the labelled moment.
        rank_of_hit = None
        for position, hit in enumerate(found.hits, 1):
            if _matches(hit.segment.video_name, hit.segment.span, label):
                rank_of_hit = position
                break

        if rank_of_hit is not None:
            report.hit_at_k += 1
            report.reciprocal_ranks.append(1.0 / rank_of_hit)
        else:
            report.missed.append(label.question)

        if found.source_video == label.video:
            report.video_correct += 1
            if label.at is not None and found.start_seconds <= label.at <= found.end_seconds:
                report.clip_contains += 1

    return report


def _matches(video_name: str, span, label: Label) -> bool:
    if video_name != label.video:
        return False
    if label.at is None:
        return True
    start, end = span
    return (start - label.tolerance) <= label.at <= (end + label.tolerance)


def write_template(path: str, pipeline: "VideoRAGPipeline", count: int = 5) -> str:
    """Emit a starter label file listing the indexed videos, to fill in."""
    videos = sorted(pipeline.videos) or ["<video-name>"]
    template = [
        {"question": "<a question this video answers>", "video": videos[i % len(videos)], "at": 0.0}
        for i in range(count)
    ]
    template.append({"question": "<something the videos do NOT cover>", "video": None})
    Path(path).write_text(json.dumps(template, indent=2), encoding="utf-8")
    return path
