"""On-screen text extraction (OCR) with light preprocessing and dedup."""

from __future__ import annotations

import logging
from typing import Tuple, Union

import cv2
import numpy as np

from .config import Config
from .retrieval import tokenize

logger = logging.getLogger(__name__)

ImageInput = Union[str, np.ndarray]


class ScreenTextReader:
    """Wraps EasyOCR with confidence filtering and dedup. Loaded lazily so
    importing this module doesn't require a GPU / model download unless OCR
    is actually used.
    """

    def __init__(self, config: Config):
        self.config = config
        self._reader = None

    @property
    def reader(self):
        if self._reader is None:
            import easyocr
            import torch

            gpu = torch.cuda.is_available()
            logger.info("Loading EasyOCR (gpu=%s)", gpu)
            self._reader = easyocr.Reader(self.config.ocr_language_list, gpu=gpu)
        return self._reader

    def read(self, image: ImageInput) -> Tuple[str, float]:
        """Returns (combined_text, average_confidence).

        Accepts a path or an in-memory BGR frame so the pipeline can OCR
        streamed frames without a disk round-trip.
        """
        results = list(self.reader.readtext(image))

        if self.config.ocr_dual_pass:
            try:
                results += list(self.reader.readtext(self._preprocess(image)))
            except Exception:  # pragma: no cover - OCR backends vary
                logger.debug("Second OCR pass failed; using first pass only", exc_info=True)

        detections = [
            (text.strip(), float(conf))
            for (_box, text, conf) in results
            if conf >= self.config.ocr_confidence and text.strip()
        ]
        if not detections:
            return "", 0.0

        # Deduplicate on normalized tokens rather than the raw string, so the
        # two passes (and OCR's own near-miss variants) don't both survive.
        seen = set()
        unique = []
        kept_confidences = []
        for text, conf in detections:
            key = " ".join(tokenize(text)) or text.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(text)
            kept_confidences.append(conf)

        combined = " | ".join(unique)
        return combined, sum(kept_confidences) / len(kept_confidences)

    @staticmethod
    def _preprocess(image: ImageInput) -> np.ndarray:
        """Binarize for a second pass — recovers low-contrast text."""
        img = cv2.imread(image) if isinstance(image, str) else image
        if img is None:
            raise ValueError(f"Could not read image: {image!r}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    @staticmethod
    def is_near_duplicate(text: str, prev_text: str, threshold: float) -> bool:
        """Jaccard similarity of word sets — cheap way to skip frames whose
        on-screen text barely changed from the previous kept frame."""
        if not prev_text or not text:
            return False
        w1, w2 = set(text.lower().split()), set(prev_text.lower().split())
        if not w1 or not w2:
            return False
        return (len(w1 & w2) / len(w1 | w2)) >= threshold
