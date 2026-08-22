"""Filename and formatting helpers."""

import pytest

from video_rag.util import format_timestamp, slugify


class TestSlugify:
    @pytest.mark.parametrize(
        "raw",
        [
            "How do I add a new patient?",
            'Where is the "export" button?',
            "Reports / invoices: how?",
            "path\\with\\backslashes",
            "wild*cards|and<brackets>",
        ],
    )
    def test_strips_every_character_windows_rejects(self, raw):
        """Regression: the old code only replaced spaces, so any question
        ending in '?' produced an illegal filename and ffmpeg failed silently."""
        slug = slugify(raw)
        assert not (set(slug) & set('?:*"<>|/\\ ')), slug

    def test_produces_a_writable_filename(self, tmp_path):
        target = tmp_path / f"{slugify('How do I add a new patient?')}.mp4"
        target.write_bytes(b"")  # would raise OSError(EINVAL) on Windows before
        assert target.exists()

    def test_collapses_runs_and_trims_edges(self):
        assert slugify("  Hello --- World!!  ") == "hello_world"

    def test_respects_max_length_without_trailing_separator(self):
        slug = slugify("a" * 10 + " " + "b" * 100, max_length=12)
        assert len(slug) <= 12
        assert not slug.endswith("_")

    def test_falls_back_when_nothing_survives(self):
        assert slugify("???") == "clip"
        assert slugify("", fallback="unnamed") == "unnamed"

    def test_is_deterministic(self):
        assert slugify("Same Question?") == slugify("Same Question?")


class TestFormatTimestamp:
    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0, "00:00"),
            (9.4, "00:09"),
            (72, "01:12"),
            (599, "09:59"),
            (3600, "1:00:00"),
            (3725, "1:02:05"),
        ],
    )
    def test_formats(self, seconds, expected):
        assert format_timestamp(seconds) == expected

    def test_clamps_negative_to_zero(self):
        assert format_timestamp(-5) == "00:00"
