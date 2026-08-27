"""SegmentStore: pure stdlib (sqlite3), no FAISS/torch involved."""

import pytest

from video_rag.segment import Segment
from video_rag.store import SegmentStore, inspect


def seg(video, at, text="text", **overrides):
    defaults = {"video_path": f"videos/{video}.mp4", "source": "ocr", "confidence": 1.0}
    return Segment(video_name=video, timestamp=at, text=text, **{**defaults, **overrides})


class TestAppendAndRead:
    def test_round_trips_every_field(self, tmp_path):
        original = seg("patients", 72.0, text="Click New", source="speech",
                        end_timestamp=80.0, confidence=0.42)
        with SegmentStore(str(tmp_path / "s.db")) as store:
            store.append([original], start_position=0)
            restored = store.all_segments()[0]
        assert restored == original

    def test_all_segments_are_ordered_by_faiss_position_not_insertion_order(self, tmp_path):
        with SegmentStore(str(tmp_path / "s.db")) as store:
            store.append([seg("b", 5.0)], start_position=1)
            store.append([seg("a", 1.0)], start_position=0)
            names = [s.video_name for s in store.all_segments()]
        assert names == ["a", "b"]

    def test_appending_nothing_is_a_no_op(self, tmp_path):
        with SegmentStore(str(tmp_path / "s.db")) as store:
            store.append([], start_position=0)
            assert store.segment_count() == 0

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "s.db"
        with SegmentStore(str(path)):
            pass
        assert path.exists()


class TestVideoQueries:
    @pytest.fixture
    def populated(self, tmp_path):
        store = SegmentStore(str(tmp_path / "s.db"))
        store.append(
            [seg("patients", 1.0), seg("patients", 2.0), seg("reports", 5.0)],
            start_position=0,
        )
        yield store
        store.close()

    def test_video_names_are_distinct_and_sorted(self, populated):
        assert populated.video_names() == ["patients", "reports"]

    def test_segment_count_overall(self, populated):
        assert populated.segment_count() == 3

    def test_segment_count_per_video(self, populated):
        assert populated.segment_count("patients") == 2
        assert populated.segment_count("reports") == 1

    def test_segment_count_for_unknown_video_is_zero(self, populated):
        assert populated.segment_count("nope") == 0


class TestDeleteAndCompact:
    def test_delete_video_returns_the_count_removed(self, tmp_path):
        with SegmentStore(str(tmp_path / "s.db")) as store:
            store.append([seg("a", 1.0), seg("b", 2.0), seg("a", 3.0)], start_position=0)
            assert store.delete_video("a") == 2
            assert store.video_names() == ["b"]

    def test_delete_unknown_video_removes_nothing(self, tmp_path):
        with SegmentStore(str(tmp_path / "s.db")) as store:
            store.append([seg("a", 1.0)], start_position=0)
            assert store.delete_video("nope") == 0
            assert store.segment_count() == 1

    def test_compact_renumbers_from_zero(self, tmp_path):
        """This is the step that keeps metadata aligned with a rebuilt FAISS
        index after a deletion leaves the position sequence with a gap."""
        with SegmentStore(str(tmp_path / "s.db")) as store:
            store.append([seg("a", 1.0), seg("b", 2.0), seg("c", 3.0)], start_position=0)
            store.delete_video("b")
            survivors = store.all_segments()  # positions [0, 2] - has a gap
            store.compact(survivors)

            rows = store._conn.execute(
                "SELECT faiss_position FROM segments ORDER BY faiss_position"
            ).fetchall()
        assert [r[0] for r in rows] == [0, 1]

    def test_compact_with_an_empty_list_clears_the_store(self, tmp_path):
        with SegmentStore(str(tmp_path / "s.db")) as store:
            store.append([seg("a", 1.0)], start_position=0)
            store.compact([])
            assert store.segment_count() == 0


class TestMeta:
    def test_set_and_get(self, tmp_path):
        with SegmentStore(str(tmp_path / "s.db")) as store:
            store.set_meta("embedding_model", "BAAI/bge-base-en-v1.5")
            assert store.get_meta("embedding_model") == "BAAI/bge-base-en-v1.5"

    def test_missing_key_returns_none(self, tmp_path):
        with SegmentStore(str(tmp_path / "s.db")) as store:
            assert store.get_meta("nope") is None

    def test_set_meta_overwrites(self, tmp_path):
        with SegmentStore(str(tmp_path / "s.db")) as store:
            store.set_meta("k", "v1")
            store.set_meta("k", "v2")
            assert store.get_meta("k") == "v2"


class TestPersistsAcrossReopen:
    def test_data_survives_closing_and_reopening_the_file(self, tmp_path):
        path = str(tmp_path / "s.db")
        with SegmentStore(path) as store:
            store.append([seg("patients", 72.0)], start_position=0)
            store.set_meta("embedding_model", "some-model")

        with SegmentStore(path) as reopened:
            assert reopened.segment_count() == 1
            assert reopened.get_meta("embedding_model") == "some-model"

    def test_no_wal_sidecar_files_are_left_behind(self, tmp_path):
        """Regression: WAL mode leaves -wal/-shm files that a rename-based
        atomic save (of the main .db file only) would not move together,
        risking a fresh file paired with a stale sidecar on next open."""
        path = tmp_path / "s.db"
        with SegmentStore(str(path)) as store:
            store.append([seg("patients", 1.0)], start_position=0)
        assert not path.with_name(path.name + "-wal").exists()
        assert not path.with_name(path.name + "-shm").exists()


class TestInspect:
    def test_missing_index_reports_empty(self, tmp_path):
        result = inspect(str(tmp_path / "does-not-exist"))
        assert result == {"videos": {}, "embedding_model": None, "total_segments": 0}

    def test_reports_per_video_counts_and_model_without_touching_faiss(self, tmp_path):
        index_dir = tmp_path / "idx"
        index_dir.mkdir()
        with SegmentStore(str(index_dir / "segments.db")) as store:
            store.append([seg("patients", 1.0), seg("patients", 2.0), seg("reports", 5.0)], 0)
            store.set_meta("embedding_model", "BAAI/bge-base-en-v1.5")

        result = inspect(str(index_dir))
        assert result == {
            "videos": {"patients": 2, "reports": 1},
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "total_segments": 3,
        }
