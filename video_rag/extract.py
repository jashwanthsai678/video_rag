"""Extract representative frames from a video, skipping near-duplicates."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterator, List, Tuple

import cv2
import numpy as np

from .config import Config

logger = logging.getLogger(__name__)

DEFAULT_FPS = 30.0
THUMBNAIL_WIDTH = 320


class FrameExtractor:
    """Pulls ~`fps_extract` frames/sec from a video, keeping only frames that
    differ meaningfully from the last kept frame, so OCR isn't wasted on
    hundreds of identical frames of a static screen.
    """

    def __init__(self, config: Config):
        self.config = config

    def iter_frames(self, video_path: str) -> Iterator[Tuple[np.ndarray, float]]:
        """Yields (frame_bgr, timestamp_seconds) without touching the disk.

        Streaming rather than writing every frame out: a 1 fps pass over a long
        screen recording is thousands of full-resolution images, and every one
        of them is dead weight the moment OCR has read it.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0 or math.isnan(fps):
                logger.warning(
                    "Could not read frame rate from %s; assuming %.0f fps",
                    video_path,
                    DEFAULT_FPS,
                )
                fps = DEFAULT_FPS

            interval = max(1, int(round(fps / max(self.config.fps_extract, 1e-6))))
            prev_thumb = None
            examined = kept = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if examined % interval == 0:
                    thumb = self._thumbnail(frame)
                    if prev_thumb is None or self._is_scene_change(prev_thumb, thumb):
                        prev_thumb = thumb
                        kept += 1
                        yield self._enhance(frame), examined / fps

                examined += 1
        finally:
            cap.release()

        logger.info(
            "%s: kept %d frame(s) from %d decoded (sampling every %d)",
            Path(video_path).name,
            kept,
            examined,
            interval,
        )

    def extract(self, video_path: str, video_name: str) -> List[Tuple[str, float]]:
        """Same sampling as `iter_frames`, but persists frames to disk.

        Only needed for debugging/inspection — the pipeline streams instead.
        """
        frame_dir = Path(self.config.frames_dir) / video_name
        frame_dir.mkdir(parents=True, exist_ok=True)

        saved: List[Tuple[str, float]] = []
        for frame, timestamp in self.iter_frames(video_path):
            # Millisecond-resolution names: truncating to whole seconds makes
            # any fps_extract > 1 overwrite the previous frame and mistime it.
            path = frame_dir / f"frame_{int(round(timestamp * 1000)):09d}.jpg"
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            saved.append((str(path), timestamp))
        return saved

    # --- internals -----------------------------------------------------------

    @staticmethod
    def _thumbnail(frame: np.ndarray, width: int = THUMBNAIL_WIDTH) -> np.ndarray:
        """Downscaled grayscale copy — comparison is cheaper and less noisy."""
        height, full_width = frame.shape[:2]
        if full_width > width:
            scale = width / float(full_width)
            frame = cv2.resize(
                frame,
                (max(1, int(full_width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _is_scene_change(self, prev_thumb: np.ndarray, thumb: np.ndarray) -> bool:
        """True when enough pixels moved to suggest the screen actually changed."""
        if prev_thumb.shape != thumb.shape:
            return True
        diff = cv2.absdiff(prev_thumb, thumb)
        moved = float(np.count_nonzero(diff > self.config.scene_pixel_delta))
        return (moved / diff.size) >= self.config.scene_change_min_fraction

    @staticmethod
    def _enhance(frame: np.ndarray) -> np.ndarray:
        """Mild contrast/brightness lift — helps OCR on low-contrast UI text."""
        return cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
