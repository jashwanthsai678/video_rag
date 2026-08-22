"""Shared data model for anything that can be retrieved and clipped."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Optional, Tuple


@dataclass
class Segment:
    """A single retrievable unit: either on-screen text (OCR) or spoken
    words (transcription), anchored to a timestamp in a source video."""

    video_name: str
    video_path: str
    timestamp: float           # seconds into the video
    text: str
    source: str                # "ocr" | "speech"
    confidence: float = 1.0
    end_timestamp: Optional[float] = None  # for speech segments with duration

    @property
    def span(self) -> Tuple[float, float]:
        """(start, end) in seconds. OCR segments are instants, so end == start."""
        end = self.timestamp if self.end_timestamp is None else self.end_timestamp
        return (self.timestamp, max(self.timestamp, end))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        # Ignore unknown keys so an index written by a newer version with extra
        # fields still loads instead of raising TypeError.
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
