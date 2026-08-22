"""The shared data model."""

import pytest

from video_rag.segment import Segment


def make_segment(**overrides):
    defaults = {
        "video_name": "patients",
        "video_path": "videos/patients.mp4",
        "timestamp": 72.0,
        "text": "New | Patient Record",
        "source": "ocr",
    }
    return Segment(**{**defaults, **overrides})


class TestSpan:
    def test_ocr_segments_are_instants(self):
        assert make_segment(end_timestamp=None).span == (72.0, 72.0)

    def test_speech_segments_carry_duration(self):
        assert make_segment(source="speech", end_timestamp=80.5).span == (72.0, 80.5)

    def test_an_end_before_the_start_never_inverts_the_span(self):
        start, end = make_segment(end_timestamp=10.0).span
        assert start <= end


class TestSerialization:
    def test_roundtrip_preserves_every_field(self):
        original = make_segment(source="speech", end_timestamp=80.5, confidence=0.42)
        restored = Segment.from_dict(original.to_dict())
        assert restored == original

    def test_unknown_keys_are_ignored(self):
        """An index written by a newer version must still load."""
        payload = make_segment().to_dict()
        payload["some_future_field"] = "surprise"
        assert Segment.from_dict(payload).video_name == "patients"

    def test_missing_required_key_still_fails_loudly(self):
        payload = make_segment().to_dict()
        del payload["timestamp"]
        with pytest.raises(TypeError):
            Segment.from_dict(payload)
