"""VideoRAGPipeline: the main entry point tying every stage together."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .clipper import ClipCutter
from .config import Config
from .explain import ExplanationGenerator
from .extract import FrameExtractor
from .index import SearchHit, SemanticIndex
from .ocr import ScreenTextReader
from .retrieval import cluster_by_time, confidence, group_by, rank_score, span_of
from .segment import Segment
from .transcribe import SpeechTranscriber
from .util import slugify

logger = logging.getLogger(__name__)

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


@dataclass
class Located:
    """Where the answer is — retrieval only, no LLM call and no ffmpeg.

    Split out from QueryResult so retrieval quality can be measured (see
    `videorag eval`) without paying for generation or writing files.
    """

    query: str
    status: str  # "success" | "no_match"
    source_video: Optional[str] = None
    score: float = 0.0            # similarity of the single best hit
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    matched_segments: List[Segment] = field(default_factory=list)
    hits: List[SearchHit] = field(default_factory=list)
    clusters_found: int = 0


@dataclass
class QueryResult:
    query: str
    status: str  # "success" | "no_match" | "error"
    source_video: Optional[str] = None
    score: float = 0.0            # similarity of the single best hit
    explanation: str = ""
    clip_path: Optional[str] = None
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    matched_segments: List[Segment] = field(default_factory=list)
    hits: List[SearchHit] = field(default_factory=list)
    clusters_found: int = 0
    error: Optional[str] = None


class VideoRAGPipeline:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.extractor = FrameExtractor(self.config)
        self.ocr = ScreenTextReader(self.config)
        self.transcriber = SpeechTranscriber(self.config)
        self.index = SemanticIndex(self.config)
        self.explainer = ExplanationGenerator(self.config)
        self.clipper = ClipCutter(self.config)
        self.videos: Dict[str, str] = {}  # name -> path

        self.index.load(self.config.index_path)  # no-op if nothing saved yet
        for seg in self.index.segments:
            self.videos.setdefault(seg.video_name, seg.video_path)

    # --- Indexing ----------------------------------------------------

    def discover_videos(self, videos_dir: Optional[str] = None) -> Dict[str, str]:
        folder = Path(videos_dir or self.config.videos_dir)
        found = {}
        if folder.exists():
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() in VIDEO_EXTS:
                    found[f.stem] = str(f)
        return found

    def index_video(
        self, video_path: str, video_name: Optional[str] = None, replace: bool = False
    ) -> int:
        """Extracts frames/audio from a single video, embeds them, and adds
        them to the index. Returns the number of segments added."""
        name = video_name or Path(video_path).stem
        self.videos[name] = video_path

        if replace:
            self.index.remove_video(name)

        segments: List[Segment] = []
        if self.config.ocr_enabled:
            segments.extend(self._read_screen_text(video_path, name))
        if self.config.transcription_enabled:
            segments.extend(self.transcriber.transcribe(video_path, name))

        self.index.add(segments)
        logger.info("%s: indexed %d segment(s)", name, len(segments))
        return len(segments)

    def _read_screen_text(self, video_path: str, name: str) -> List[Segment]:
        """OCR every kept frame, streaming so frames never hit the disk."""
        segments: List[Segment] = []
        previous_text = ""
        frames_read = 0

        for frame, timestamp in self.extractor.iter_frames(video_path):
            frames_read += 1
            if self.config.keep_frames:
                self._persist_frame(frame, name, timestamp)

            text, conf = self.ocr.read(frame)
            if len(text) < self.config.min_text_length:
                continue
            if self.ocr.is_near_duplicate(
                text, previous_text, self.config.ocr_similarity_threshold
            ):
                continue

            previous_text = text
            segments.append(
                Segment(
                    video_name=name,
                    video_path=video_path,
                    timestamp=timestamp,
                    text=text,
                    source="ocr",
                    confidence=conf,
                )
            )
            if len(segments) % 25 == 0:
                logger.info("%s: %d OCR segment(s) so far", name, len(segments))

        logger.info("%s: %d OCR segment(s) from %d frame(s)", name, len(segments), frames_read)
        return segments

    def _persist_frame(self, frame, name: str, timestamp: float) -> None:
        import cv2

        frame_dir = Path(self.config.frames_dir) / name
        frame_dir.mkdir(parents=True, exist_ok=True)
        path = frame_dir / f"frame_{int(round(timestamp * 1000)):09d}.jpg"
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    def index_videos(self, videos_dir: Optional[str] = None, force: bool = False) -> int:
        """Indexes every video in a directory, skipping ones already in the
        index unless force=True. Returns total segments added."""
        already = {s.video_name for s in self.index.segments}
        videos = self.discover_videos(videos_dir)
        if not videos:
            logger.warning("No videos found in %s", videos_dir or self.config.videos_dir)
            return 0

        added = 0
        for position, (name, path) in enumerate(videos.items(), 1):
            if name in already and not force:
                logger.info("Skipping %s (already indexed)", name)
                continue
            logger.info("Indexing %s (%d/%d)", name, position, len(videos))
            try:
                added += self.index_video(path, name, replace=name in already)
            except Exception:
                logger.exception("Failed to index %s; continuing", name)

        if added:
            self.index.save(self.config.index_path)
        return added

    # --- Querying ------------------------------------------------------

    def locate(self, question: str, top_k: Optional[int] = None) -> Located:
        """Find the best moment. Pure retrieval: no LLM call, no files written."""
        hits = self.index.search(question, top_k or self.config.top_k)
        if not hits:
            return Located(query=question, status="no_match")

        best_video, video_hits = self._best_video(hits)
        best_confidence = confidence(h.score for h in video_hits)

        if best_confidence < self.config.relevance_threshold:
            logger.info(
                "Best similarity %.3f is below threshold %.3f",
                best_confidence,
                self.config.relevance_threshold,
            )
            return Located(
                query=question, status="no_match", score=best_confidence, hits=hits
            )

        cluster_hits, cluster_count = self._best_cluster(video_hits)
        span = span_of([h.segment.span for h in cluster_hits], range(len(cluster_hits)))
        start, end = self._clip_bounds(span.start, span.end)

        return Located(
            query=question,
            status="success",
            source_video=best_video,
            score=best_confidence,
            start_seconds=start,
            end_seconds=end,
            matched_segments=[h.segment for h in cluster_hits],
            hits=hits,
            clusters_found=cluster_count,
        )

    def query(self, question: str, top_k: Optional[int] = None) -> QueryResult:
        """Locate the answer, write the clip, and generate the how-to guide."""
        found = self.locate(question, top_k=top_k)
        if found.status != "success":
            return QueryResult(
                query=question, status=found.status, score=found.score, hits=found.hits
            )

        explanation = self.explainer.generate(question, found.matched_segments)
        clip_path = self._cut_clip(
            found.source_video, question, found.start_seconds, found.end_seconds
        )

        return QueryResult(
            query=question,
            status="success",
            source_video=found.source_video,
            score=found.score,
            explanation=explanation,
            clip_path=clip_path,
            start_seconds=found.start_seconds,
            end_seconds=found.end_seconds,
            matched_segments=found.matched_segments,
            hits=found.hits,
            clusters_found=found.clusters_found,
        )

    def _best_video(self, hits: List[SearchHit]):
        """Pick the video with the strongest concentration of evidence."""
        by_video = group_by(hits, lambda h: h.segment.video_name)
        name = max(
            by_video,
            key=lambda v: (
                rank_score((h.score for h in by_video[v]), self.config.aggregate_top_n),
                confidence(h.score for h in by_video[v]),
            ),
        )
        return name, by_video[name]

    def _best_cluster(self, video_hits: List[SearchHit]):
        """Split matches into contiguous runs and keep the strongest run.

        Returning min-to-max over every match in the video is what turns a
        10-second answer into a 14-minute "clip" whenever the query also hits
        somewhere unrelated.
        """
        spans = [h.segment.span for h in video_hits]
        groups = cluster_by_time(spans, self.config.cluster_max_gap_seconds)
        best = max(
            groups,
            key=lambda g: rank_score(
                (video_hits[i].score for i in g), self.config.aggregate_top_n
            ),
        )
        chosen = sorted((video_hits[i] for i in best), key=lambda h: h.segment.timestamp)
        logger.info(
            "Selected 1 of %d cluster(s): %d hit(s) in the winning run",
            len(groups),
            len(chosen),
        )
        return chosen, len(groups)

    def _clip_bounds(self, start: float, end: float):
        """Pad the matched span, then cap it so exports stay bounded."""
        buffer = self.config.clip_buffer_seconds
        padded_start = max(0.0, start - buffer)
        padded_end = max(padded_start + 1.0, end + buffer)
        if padded_end - padded_start > self.config.max_clip_seconds:
            logger.info(
                "Clip span %.1fs exceeds max %.1fs; truncating",
                padded_end - padded_start,
                self.config.max_clip_seconds,
            )
            padded_end = padded_start + self.config.max_clip_seconds
        return padded_start, padded_end

    def _cut_clip(self, video_name: str, question: str, start: float, end: float):
        video_path = self.videos.get(video_name)
        if not video_path or not os.path.exists(video_path):
            logger.warning("Source video for %r is unavailable; skipping clip", video_name)
            return None
        # slugify, not a bare space-replace: an unsanitized question ends in
        # "?", which is an illegal filename character on Windows.
        return self.clipper.cut(
            video_path, start, end, out_name=f"{slugify(video_name)}_{slugify(question)}"
        )
