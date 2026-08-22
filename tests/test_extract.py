"""Frame sampling against real synthesized video.

Uses MJPG/AVI rather than mp4v: it is the most consistently available writer
across OpenCV builds, and the container is irrelevant to what's being tested.
"""

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from video_rag.config import Config  # noqa: E402
from video_rag.extract import FrameExtractor  # noqa: E402

WIDTH, HEIGHT, FPS = 160, 120, 10.0


def write_video(path, screens, seconds_per_screen=1.0):
    """Render a video that holds each 'screen' (a flat colour) for a while.

    Approximates a screen recording: long static stretches punctuated by
    abrupt full-frame changes.
    """
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened(), "OpenCV has no usable MJPG writer"
    try:
        for value in screens:
            frame = np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)
            for _ in range(int(FPS * seconds_per_screen)):
                writer.write(frame)
    finally:
        writer.release()
    return str(path)


def write_changing_video(path, frame_count=30, step=8):
    """One distinct frame per frame, so sub-second samples all survive gating.

    Needed to exercise sampling faster than 1 fps: with long static screens the
    gate keeps at most one frame per screen, and multiple kept frames never
    share a second.
    """
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened(), "OpenCV has no usable MJPG writer"
    try:
        for i in range(frame_count):
            writer.write(np.full((HEIGHT, WIDTH, 3), (i * step) % 256, dtype=np.uint8))
    finally:
        writer.release()
    return str(path)


def config_for(tmp_path, **overrides):
    return Config(
        data_dir=str(tmp_path / "data"),
        frames_dir=str(tmp_path / "data" / "frames"),
        clips_dir=str(tmp_path / "data" / "clips"),
        **overrides,
    )


class TestSceneGating:
    def test_keeps_one_frame_per_distinct_screen(self, tmp_path):
        video = write_video(tmp_path / "v.avi", [10, 120, 230])
        extractor = FrameExtractor(config_for(tmp_path))
        frames = list(extractor.iter_frames(video))
        # Three static screens, sampled at 1 fps: one kept per screen change.
        assert len(frames) == 3

    def test_a_static_screen_yields_a_single_frame(self, tmp_path):
        video = write_video(tmp_path / "v.avi", [60], seconds_per_screen=5.0)
        frames = list(FrameExtractor(config_for(tmp_path)).iter_frames(video))
        assert len(frames) == 1

    def test_timestamps_increase_and_land_in_the_right_screen(self, tmp_path):
        video = write_video(tmp_path / "v.avi", [10, 120, 230])
        frames = list(FrameExtractor(config_for(tmp_path)).iter_frames(video))
        times = [ts for _, ts in frames]
        assert times == sorted(times)
        # Each screen lasts 1s, so the three kept frames straddle the boundaries.
        assert times[0] < 1.0 <= times[1] < 2.0 <= times[2]

    def test_a_high_threshold_discards_subtle_changes(self, tmp_path):
        video = write_video(tmp_path / "v.avi", [100, 104])  # near-identical
        extractor = FrameExtractor(config_for(tmp_path, scene_pixel_delta=25))
        assert len(list(extractor.iter_frames(video))) == 1

    def test_lowering_the_pixel_delta_recovers_subtle_changes(self, tmp_path):
        video = write_video(tmp_path / "v.avi", [100, 104])
        extractor = FrameExtractor(config_for(tmp_path, scene_pixel_delta=2))
        assert len(list(extractor.iter_frames(video))) == 2

    def test_frames_are_returned_as_full_size_bgr_arrays(self, tmp_path):
        video = write_video(tmp_path / "v.avi", [10, 200])
        frame, _ = next(iter(FrameExtractor(config_for(tmp_path)).iter_frames(video)))
        assert frame.shape == (HEIGHT, WIDTH, 3)
        assert frame.dtype == np.uint8

    def test_missing_video_raises(self, tmp_path):
        extractor = FrameExtractor(config_for(tmp_path))
        with pytest.raises(FileNotFoundError):
            list(extractor.iter_frames(str(tmp_path / "nope.avi")))


class TestDiskExtraction:
    def test_higher_sampling_rate_does_not_overwrite_frames(self, tmp_path):
        """Regression: filenames were truncated to whole seconds, so any
        fps_extract > 1 wrote two frames to one path and mistimed both."""
        video = write_changing_video(tmp_path / "v.avi")
        extractor = FrameExtractor(config_for(tmp_path, fps_extract=4.0, scene_pixel_delta=2))
        saved = extractor.extract(video, "v")

        # Guard against a vacuous assertion: the old naming only broke when two
        # kept frames fell inside the same second, so require that to happen.
        seconds = [int(ts) for _, ts in saved]
        assert len(seconds) > len(set(seconds)), "test did not exercise sub-second sampling"

        paths = [path for path, _ in saved]
        assert len(set(paths)) == len(paths), "two frames shared one filename"
        for path, _ in saved:
            assert cv2.imread(path) is not None

    def test_written_files_match_returned_timestamps(self, tmp_path):
        video = write_video(tmp_path / "v.avi", [10, 120])
        saved = FrameExtractor(config_for(tmp_path)).extract(video, "v")
        for path, timestamp in saved:
            assert f"{int(round(timestamp * 1000)):09d}" in path

    def test_streaming_leaves_no_files_behind(self, tmp_path):
        """The pipeline path must not accumulate frames on disk."""
        video = write_video(tmp_path / "v.avi", [10, 120, 230])
        config = config_for(tmp_path)
        list(FrameExtractor(config).iter_frames(video))
        frames_dir = tmp_path / "data" / "frames"
        assert not frames_dir.exists() or not list(frames_dir.rglob("*.jpg"))
