"""Small shared helpers. Deliberately dependency-free."""

from __future__ import annotations

import re

_UNSAFE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_length: int = 48, fallback: str = "clip") -> str:
    """Collapse arbitrary text into a filesystem-safe slug.

    Necessary because output names are derived from user questions, and
    Windows rejects ``? : * " < > | / \\`` in filenames — so a raw question
    like "How do I add a new patient?" cannot be used as a filename.
    """
    slug = _UNSAFE.sub("_", text.lower()).strip("_")
    return slug[:max_length].strip("_") or fallback


def format_timestamp(seconds: float) -> str:
    """Seconds -> ``MM:SS`` (or ``H:MM:SS`` past an hour)."""
    total = max(0, int(round(float(seconds))))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
