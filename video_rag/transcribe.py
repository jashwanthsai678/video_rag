"""Speech-to-text transcription using faster-whisper (local, no API key)."""

from __future__ import annotations

import logging
from typing import List

from .config import Config
from .segment import Segment

logger = logging.getLogger(__name__)


class SpeechTranscriber:
    """Wraps faster-whisper. Loaded lazily since it downloads model weights
    on first use and isn't needed if a video has no narration."""

    def __init__(self, config: Config):
        self.config = config
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import torch
            from faster_whisper import WhisperModel

            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            logger.info(
                "Loading Whisper model %r (device=%s)", self.config.whisper_model, device
            )
            self._model = WhisperModel(
                self.config.whisper_model, device=device, compute_type=compute_type
            )
        return self._model

    def transcribe(self, video_path: str, video_name: str) -> List[Segment]:
        import av

        if not self._has_audio_track(video_path):
            logger.info("%s: no audio track; skipping transcription", video_name)
            return []

        try:
            segments_gen, info = self.model.transcribe(
                video_path,
                beam_size=5,
                # Empty config value => auto-detect rather than forcing English.
                language=self.config.whisper_language or None,
            )
            logger.info(
                "Transcribing %s (detected language: %s)",
                video_name,
                getattr(info, "language", "unknown"),
            )

            segments = []
            for seg in segments_gen:
                text = seg.text.strip()
                if not text:
                    continue
                segments.append(
                    Segment(
                        video_name=video_name,
                        video_path=video_path,
                        timestamp=float(seg.start),
                        end_timestamp=float(seg.end),
                        text=text,
                        source="speech",
                        confidence=1.0,
                    )
                )
        except (IndexError, av.error.FFmpegError):
            # An audio stream was present but couldn't actually be decoded
            # (corrupt/empty track). Same outcome as "no audio": OCR segments
            # for this video should still be indexed, not lost.
            logger.warning(
                "%s: audio track present but undecodable; skipping transcription", video_name
            )
            return []

        logger.info("%s: %d speech segment(s)", video_name, len(segments))
        return segments

    @staticmethod
    def _has_audio_track(video_path: str) -> bool:
        """Cheap container probe so a silent video never triggers a Whisper
        model load (weights download / GPU init) it has no use for."""
        import av

        try:
            with av.open(video_path, mode="r", metadata_errors="ignore") as container:
                return len(container.streams.audio) > 0
        except av.error.FFmpegError:
            logger.warning("%s: could not probe for an audio stream; assuming none", video_path)
            return False
