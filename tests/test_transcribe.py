"""A video with no audio track must index cleanly, not crash.

These tests only exercise the audio-track probe, never the actual Whisper
model, so they stay fast and don't download model weights.
"""

import numpy as np
import pytest

av = pytest.importorskip("av")

from video_rag.config import Config  # noqa: E402
from video_rag.transcribe import SpeechTranscriber  # noqa: E402


def _write_video_only_mp4(path, frames=5):
    """A valid mp4 with a video stream and no audio stream at all."""
    container = av.open(str(path), mode="w")
    stream = container.add_stream("h264", rate=24)
    stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"

    for _ in range(frames):
        frame = av.VideoFrame.from_ndarray(
            np.zeros((64, 64, 3), dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


class TestSilentVideo:
    def test_has_audio_track_is_false_for_a_video_only_file(self, tmp_path):
        video_path = tmp_path / "silent.mp4"
        _write_video_only_mp4(video_path)

        assert SpeechTranscriber._has_audio_track(str(video_path)) is False

    def test_transcribe_returns_no_segments_without_loading_whisper(self, tmp_path):
        video_path = tmp_path / "silent.mp4"
        _write_video_only_mp4(video_path)

        transcriber = SpeechTranscriber(Config())
        segments = transcriber.transcribe(str(video_path), "silent")

        assert segments == []
        # The whole point: a silent video must never trigger a model load
        # (weights download / GPU init) it has no use for.
        assert transcriber._model is None
