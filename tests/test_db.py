"""The Database facade: search() stays pure data; render/explain are opt-in.

`db.py` imports `pipeline.py`, which transitively imports cv2/faiss (via
extract.py/index.py), so this suite needs the same import guards as
test_pipeline.py even though it never touches real video files.
"""

import pytest

pytest.importorskip("cv2")
pytest.importorskip("faiss")

from video_rag.db import Database, Match  # noqa: E402
from video_rag.index import SearchHit  # noqa: E402
from video_rag.pipeline import Located  # noqa: E402
from video_rag.segment import Segment  # noqa: E402


def hit(video, at, score, text="text"):
    return SearchHit(
        segment=Segment(
            video_name=video, video_path=f"videos/{video}.mp4",
            timestamp=at, text=text, source="ocr",
        ),
        score=score, fused=score, found_by=("vector",),
    )


class FakePipeline:
    """Stands in for VideoRAGPipeline so these tests exercise only the
    facade's own logic (the search/render/explain boundary), not indexing."""

    def __init__(self):
        self.videos = {"patients": "videos/patients.mp4"}
        self.locate_calls = []
        self.index_videos_calls = []
        self.index_video_calls = []
        self._located = None
        self.clipper = self
        self.explainer = self
        self.cut_calls = []
        self.explain_calls = []

        class FakeIndex:
            def __init__(self):
                self.segments = []

            def remove_video(self, name):
                before = len(self.segments)
                self.segments = [s for s in self.segments if s.video_name != name]
                return before - len(self.segments)

            def save(self, path):
                pass

        self.index = FakeIndex()
        self.config = self

    index_path = "unused"

    def locate(self, question, top_k=None):
        self.locate_calls.append((question, top_k))
        return self._located

    def index_videos(self, videos_dir, force):
        self.index_videos_calls.append((videos_dir, force))
        return 3

    def index_video(self, video_path, video_name=None, replace=False):
        self.index_video_calls.append((video_path, video_name, replace))
        return 1

    def cut(self, video_path, start, end, out_name):
        self.cut_calls.append((video_path, start, end, out_name))
        return f"/clips/{out_name}.mp4"

    def generate(self, question, segments):
        self.explain_calls.append((question, tuple(s.text for s in segments)))
        return "1. Do the thing."


def build(fake_pipeline):
    db = Database.__new__(Database)  # skip __init__: no real Config/pipeline needed
    db._pipeline = fake_pipeline
    return db


class TestSearchIsPureData:
    def test_returns_none_on_no_match(self):
        fake = FakePipeline()
        fake._located = Located(query="q", status="no_match", score=0.2)
        db = build(fake)
        assert db.search("q") is None

    def test_returns_a_match_with_no_clip_or_explanation_attached(self):
        fake = FakePipeline()
        fake._located = Located(
            query="q", status="success", source_video="patients",
            score=0.9, start_seconds=10.0, end_seconds=20.0,
            matched_segments=[hit("patients", 12.0, 0.9).segment], clusters_found=2,
        )
        db = build(fake)
        match = db.search("q")

        assert isinstance(match, Match)
        assert match.video_name == "patients"
        assert match.video_path == "videos/patients.mp4"
        assert match.score == pytest.approx(0.9)
        assert match.clusters_found == 2
        assert not hasattr(match, "clip_path")
        assert not fake.cut_calls
        assert not fake.explain_calls

    def test_forwards_top_k(self):
        fake = FakePipeline()
        fake._located = Located(query="q", status="no_match")
        build(fake).search("q", top_k=3)
        assert fake.locate_calls == [("q", 3)]

    def test_match_with_a_video_not_on_disk_has_no_path(self):
        fake = FakePipeline()
        fake.videos = {}
        fake._located = Located(
            query="q", status="success", source_video="ghost",
            score=0.9, start_seconds=0.0, end_seconds=1.0,
        )
        match = build(fake).search("q")
        assert match.video_path is None


class TestRenderClipIsExplicit:
    def test_cuts_using_the_matchs_window(self):
        fake = FakePipeline()
        db = build(fake)
        match = Match(
            video_name="patients", video_path="videos/patients.mp4",
            start_seconds=56.0, end_seconds=64.0, score=0.9,
        )
        clip = db.render_clip(match)
        assert clip == "/clips/patients_56.mp4"
        assert fake.cut_calls == [("videos/patients.mp4", 56.0, 64.0, "patients_56")]

    def test_missing_video_path_skips_the_cut_without_raising(self):
        fake = FakePipeline()
        db = build(fake)
        match = Match(
            video_name="ghost", video_path=None, start_seconds=0, end_seconds=1, score=0.5
        )
        assert db.render_clip(match) is None
        assert not fake.cut_calls


class TestExplainIsExplicit:
    def test_passes_the_matchs_segments_to_the_explainer(self):
        fake = FakePipeline()
        db = build(fake)
        segment = hit("patients", 10.0, 0.9, text="Click New").segment
        match = Match(
            video_name="patients", video_path="v.mp4", start_seconds=0, end_seconds=1,
            score=0.9, segments=[segment],
        )
        guide = db.explain(match, "how do i add a patient")
        assert guide == "1. Do the thing."
        assert fake.explain_calls == [("how do i add a patient", ("Click New",))]


class TestIngestAndAdd:
    def test_ingest_directory_forwards_to_the_pipeline(self):
        fake = FakePipeline()
        db = build(fake)
        assert db.ingest_directory("./videos", force=True) == 3
        assert fake.index_videos_calls == [("./videos", True)]

    def test_add_replaces_by_default(self):
        fake = FakePipeline()
        db = build(fake)
        assert db.add("./videos/v.mp4", name="v") == 1
        assert fake.index_video_calls == [("./videos/v.mp4", "v", True)]


class TestRemove:
    def test_removes_and_persists_when_segments_remain(self):
        fake = FakePipeline()
        fake.index.segments = [
            Segment(video_name="a", video_path="a.mp4", timestamp=1.0, text="t", source="ocr"),
            Segment(video_name="b", video_path="b.mp4", timestamp=1.0, text="t", source="ocr"),
        ]
        saved = []
        fake.index.save = lambda path: saved.append(path)
        db = build(fake)

        assert db.remove("a") == 1
        assert saved == ["unused"]  # segments remain ("b"), so save() ran
        assert [s.video_name for s in fake.index.segments] == ["b"]

    def test_removing_everything_clears_persisted_files_instead_of_leaving_them_stale(
        self, tmp_path
    ):
        """Regression: SemanticIndex.save() no-ops on a fully emptied index,
        which would leave the pre-removal index.faiss/segments.db on disk
        claiming content that no longer exists."""
        fake = FakePipeline()
        fake.index.segments = [
            Segment(
                video_name="patients", video_path="p.mp4",
                timestamp=1.0, text="t", source="ocr",
            ),
        ]
        fake.config.index_path = str(tmp_path)
        (tmp_path / "index.faiss").write_bytes(b"stale")
        (tmp_path / "segments.db").write_bytes(b"stale")
        db = build(fake)

        removed = db.remove("patients")

        assert removed == 1
        assert not (tmp_path / "index.faiss").exists()
        assert not (tmp_path / "segments.db").exists()

    def test_removing_an_unknown_video_does_nothing(self):
        fake = FakePipeline()
        db = build(fake)
        assert db.remove("nope") == 0


class TestIntrospection:
    def test_videos_is_sorted(self):
        fake = FakePipeline()
        fake.videos = {"reports": "r.mp4", "onboarding": "o.mp4"}
        assert build(fake).videos == ["onboarding", "reports"]

    def test_len_reflects_segment_count(self):
        fake = FakePipeline()
        fake.index.segments = [
            Segment(video_name="a", video_path="a.mp4", timestamp=1.0, text="t", source="ocr")
        ]
        assert len(build(fake)) == 1
