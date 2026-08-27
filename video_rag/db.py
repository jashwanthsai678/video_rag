"""An embedded, video-native database façade over VideoRAGPipeline.

This is packaging, not a new engine: `Database` wraps the same FAISS +
BM25 + SQLite stack `VideoRAGPipeline` already assembles, and gives it a
shape people expect from an embedded database (`chromadb`, `lancedb`) —
`.ingest_directory()` / `.search()` — rather than a bespoke pipeline API.

`search()` is deliberately pure data: no ffmpeg call, no LLM call. A
database's query method should return rows, not render media or make a
network request — the same reason `SELECT` doesn't produce a PDF. Cutting a
clip and writing a guide are separate, explicit, opt-in calls that take the
`Match` `search()` already found, so a caller who only wants retrieval never
pays for ffmpeg or an LLM round-trip.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from .config import Config
from .index import INDEX_FILE, STORE_FILE
from .pipeline import VideoRAGPipeline
from .segment import Segment


@dataclass
class Match:
    """One retrieved moment: the "row" a database query returns.

    Deliberately carries no clip path and no generated text — those are
    products of `Database.render_clip()` / `Database.explain()`, made on
    request, not implied by having found something.
    """

    video_name: str
    video_path: Optional[str]
    start_seconds: float
    end_seconds: float
    score: float
    segments: List[Segment] = field(default_factory=list)
    clusters_found: int = 0


class Database:
    """Embedded database for searchable software-walkthrough videos.

    >>> db = Database("./my_video_vault")
    >>> db.ingest_directory("./company_onboarding_videos")
    >>> match = db.search("How do I invite a new teammate?")
    >>> match.video_name, match.start_seconds, match.end_seconds
    >>> db.render_clip(match)             # -> "./my_video_vault/clips/....mp4"
    >>> db.explain(match, "How do I invite a new teammate?")
    """

    def __init__(self, path: str, **config_overrides):
        config = Config(
            data_dir=path,
            index_path=f"{path}/index",
            frames_dir=f"{path}/frames",
            clips_dir=f"{path}/clips",
            **config_overrides,
        )
        self._pipeline = VideoRAGPipeline(config)

    @property
    def config(self) -> Config:
        return self._pipeline.config

    # --- ingest ------------------------------------------------------------

    def ingest_directory(self, videos_dir: str, force: bool = False) -> int:
        """Index every video in `videos_dir`. Returns segments added."""
        return self._pipeline.index_videos(videos_dir=videos_dir, force=force)

    def add(self, video_path: str, name: Optional[str] = None) -> int:
        """Index one video file, replacing any prior copy of the same name."""
        return self._pipeline.index_video(video_path, video_name=name, replace=True)

    def remove(self, name: str) -> int:
        """Drop every segment belonging to video `name`. Returns how many."""
        removed = self._pipeline.index.remove_video(name)
        if not removed:
            return 0

        if self._pipeline.index.segments:
            self._pipeline.index.save(self._pipeline.config.index_path)
        else:
            # SemanticIndex.save() no-ops on a fully-emptied index (it can't
            # tell "never populated" from "just emptied" apart), which would
            # leave the pre-removal index files on disk claiming stale
            # content. Clear them explicitly instead.
            index_dir = self._pipeline.config.index_path
            for name_ in (INDEX_FILE, STORE_FILE):
                path = os.path.join(index_dir, name_)
                if os.path.exists(path):
                    os.remove(path)

        self._pipeline.videos.pop(name, None)
        return removed

    # --- query: pure data, no side effects ----------------------------------

    def search(self, question: str, top_k: Optional[int] = None) -> Optional[Match]:
        """Find the best matching moment. No clip cut, no LLM call."""
        found = self._pipeline.locate(question, top_k=top_k)
        if found.status != "success":
            return None
        return Match(
            video_name=found.source_video,
            video_path=self._pipeline.videos.get(found.source_video),
            start_seconds=found.start_seconds,
            end_seconds=found.end_seconds,
            score=found.score,
            segments=found.matched_segments,
            clusters_found=found.clusters_found,
        )

    # --- optional, explicit side effects ------------------------------------

    def render_clip(self, match: Match) -> Optional[str]:
        """Cut the clip for a `Match` with ffmpeg. Not implied by search()."""
        if not match.video_path:
            return None
        return self._pipeline.clipper.cut(
            match.video_path,
            match.start_seconds,
            match.end_seconds,
            out_name=f"{match.video_name}_{int(match.start_seconds)}",
        )

    def explain(self, match: Match, question: str) -> str:
        """Generate a step-by-step guide for a `Match` via the configured LLM."""
        return self._pipeline.explainer.generate(question, match.segments)

    # --- introspection: no FAISS, no torch ----------------------------------

    @property
    def videos(self) -> List[str]:
        return sorted(self._pipeline.videos)

    def __len__(self) -> int:
        return len(self._pipeline.index.segments)
