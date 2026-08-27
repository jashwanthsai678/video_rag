"""SQLite-backed segment metadata store.

Metadata (which video, what second, what text) used to live in a JSON array
that got fully deserialized and rewritten on every save. Backing it with
SQLite instead means asking "what's indexed here, and how much of it belongs
to video X" no longer requires reading every segment into memory — and,
because this module imports nothing beyond the standard library, answering
that question needs neither FAISS nor torch. `inspect()` below is the
concrete payoff: it reports what's in an index without loading the ML stack.

Segments are keyed by `faiss_position` — their row index in the FAISS
vector index — since that's the only handle a FAISS `IndexFlatIP` gives you.
Positions must stay dense (0..N-1) for FAISS reconstruction to work, so a
deletion is followed by `compact()`, which renumbers the survivors.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

from .segment import Segment

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    faiss_position INTEGER PRIMARY KEY,
    video_name     TEXT NOT NULL,
    video_path     TEXT NOT NULL,
    timestamp      REAL NOT NULL,
    end_timestamp  REAL,
    text           TEXT NOT NULL,
    source         TEXT NOT NULL,
    confidence     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_segments_video ON segments(video_name);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_SELECT_COLUMNS = (
    "video_name, video_path, timestamp, end_timestamp, text, source, confidence"
)


def _row_to_segment(row: tuple) -> Segment:
    return Segment(
        video_name=row[0],
        video_path=row[1],
        timestamp=row[2],
        end_timestamp=row[3],
        text=row[4],
        source=row[5],
        confidence=row[6],
    )


class SegmentStore:
    """Durable segment metadata for one index, backed by a single SQLite file."""

    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        # Default rollback-journal mode, not WAL: WAL leaves persistent
        # `-wal`/`-shm` sidecar files that `save()`'s atomic rename (of the
        # main .db file only) would not move along with it, which risks
        # pairing a fresh file with a stale sidecar on next open. This store
        # is single-writer and closed right after use, so WAL's concurrent-
        # reader benefit doesn't apply here anyway.
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SegmentStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # --- reads ---------------------------------------------------------------

    def all_segments(self) -> List[Segment]:
        """Every segment, ordered to match the FAISS vector index exactly."""
        rows = self._conn.execute(
            f"SELECT {_SELECT_COLUMNS} FROM segments ORDER BY faiss_position"
        ).fetchall()
        return [_row_to_segment(r) for r in rows]

    def video_names(self) -> List[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT video_name FROM segments ORDER BY video_name"
        ).fetchall()
        return [r[0] for r in rows]

    def segment_count(self, video_name: Optional[str] = None) -> int:
        if video_name is None:
            row = self._conn.execute("SELECT COUNT(*) FROM segments").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM segments WHERE video_name = ?", (video_name,)
            ).fetchone()
        return row[0]

    def get_meta(self, key: str) -> Optional[str]:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    # --- writes ----------------------------------------------------------

    def append(self, segments: Iterable[Segment], start_position: int) -> None:
        """Insert segments at consecutive positions starting at `start_position`.

        Caller's responsibility: `start_position` must equal the current
        FAISS `ntotal` at the time of the matching `faiss_index.add()` call,
        or positions will no longer line up with vector rows.
        """
        rows = [
            (
                start_position + i,
                s.video_name,
                s.video_path,
                s.timestamp,
                s.end_timestamp,
                s.text,
                s.source,
                s.confidence,
            )
            for i, s in enumerate(segments)
        ]
        if not rows:
            return
        self._conn.executemany(
            "INSERT INTO segments (faiss_position, video_name, video_path, "
            "timestamp, end_timestamp, text, source, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def delete_video(self, video_name: str) -> int:
        """Remove every segment for `video_name`. Returns how many were removed.

        Leaves the remaining rows' positions non-contiguous — call `compact()`
        with the rebuilt, in-order segment list afterward.
        """
        cursor = self._conn.execute("DELETE FROM segments WHERE video_name = ?", (video_name,))
        self._conn.commit()
        return cursor.rowcount

    def compact(self, ordered_segments: List[Segment]) -> None:
        """Replace every row with `ordered_segments`, renumbered from 0.

        Needed after a deletion: FAISS's `IndexFlatIP` has no notion of a
        gap, so once the vector index is rebuilt dense, the metadata must be
        renumbered to match it exactly.
        """
        self._conn.execute("DELETE FROM segments")
        self.append(ordered_segments, start_position=0)

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._conn.commit()


def inspect(index_path: str) -> dict:
    """Report what's indexed at `index_path` without touching FAISS or torch.

    This is the reason the metadata store matters on its own: a JSON blob
    would need full deserialization to answer "what videos are in here", and
    a health check or a `videorag inspect` CLI command would have paid the
    startup cost of every heavy ML dependency just to print a summary.
    """
    db_path = str(Path(index_path) / "segments.db")
    if not Path(db_path).exists():
        return {"videos": {}, "embedding_model": None, "total_segments": 0}

    with SegmentStore(db_path) as store:
        videos = {name: store.segment_count(name) for name in store.video_names()}
        return {
            "videos": videos,
            "embedding_model": store.get_meta("embedding_model"),
            "total_segments": store.segment_count(),
        }
