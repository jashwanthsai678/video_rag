"""Cut short clips out of a source video around a matched timestamp."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Optional

from .config import Config
from .util import slugify

logger = logging.getLogger(__name__)


class ClipCutter:
    def __init__(self, config: Config):
        self.config = config

    def available(self) -> bool:
        return shutil.which(self.config.ffmpeg_binary) is not None

    def cut(
        self, video_path: str, start_seconds: float, end_seconds: float, out_name: str = "clip"
    ) -> Optional[str]:
        """Cuts [start_seconds, end_seconds] out of video_path.

        Takes an explicit, already-padded range: the caller decides the window
        (it is the only thing that knows which cluster of matches it picked),
        so padding logic lives in exactly one place.
        Returns the output path, or None if ffmpeg failed.
        """
        if not self.available():
            logger.warning(
                "%r not found on PATH; cannot cut clips", self.config.ffmpeg_binary
            )
            return None

        start = max(0.0, float(start_seconds))
        duration = max(1.0, float(end_seconds) - start)

        os.makedirs(self.config.clips_dir, exist_ok=True)
        # Sanitize again here — cut() is public, and a caller-supplied name
        # containing "?" or "/" would silently fail on Windows.
        out_path = os.path.join(self.config.clips_dir, f"{slugify(out_name, max_length=80)}.mp4")

        cmd = [
            self.config.ffmpeg_binary, "-y",
            "-ss", f"{start:.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-profile:v", "baseline",
            "-level", "3.0",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-movflags", "+faststart",
            out_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(out_path):
            logger.info("Wrote clip %s (%.1fs from %.1fs)", out_path, duration, start)
            return out_path

        # Never swallow this: a failed cut used to look identical to "no clip
        # requested", which hid the illegal-filename bug completely.
        logger.error(
            "ffmpeg failed (exit %s) writing %s\n%s",
            result.returncode,
            out_path,
            (result.stderr or "").strip()[-2000:],
        )
        return None
