import numpy as np
import pytest

from frames import (VideoReadError, center_crop_resize, extract, frame_subset, timestamps,
                    video_duration)


def test_timestamps_cover_video_and_skip_ends():
    ts = timestamps(100.0, n=16)
    assert len(ts) == 16
    assert np.all(np.diff(ts) > 0)
    assert ts[0] >= 5.0 and ts[-1] <= 95.0
    assert ts[-1] - ts[0] > 80.0


def test_timestamps_reject_nonpositive_duration():
    with pytest.raises(ValueError):
        timestamps(0.0)


def test_frame_subset_spreads_over_all_frames():
    assert frame_subset(16, 4).tolist() == [0, 5, 10, 15]
    assert frame_subset(16, 16).tolist() == list(range(16))


def test_portrait_frame_is_center_cropped_without_distortion():
    bgr = np.zeros((426, 240, 3), np.uint8)
    bgr[93:333, :] = 255  # the centered 240x240 square is white
    out = center_crop_resize(bgr, 256)
    assert out.shape == (256, 256, 3)
    assert out.min() == 255


class FakeCap:
    def __init__(self, fps, count, end_ms):
        self.props = {"fps": fps, "count": count}
        self.end_ms, self.pos_ms = end_ms, 0.0

    def get(self, prop):
        import cv2
        return {cv2.CAP_PROP_FPS: self.props["fps"], cv2.CAP_PROP_FRAME_COUNT: self.props["count"],
                cv2.CAP_PROP_POS_MSEC: self.pos_ms}[prop]

    def set(self, prop, value):
        self.pos_ms = self.end_ms * value
        return True


def test_duration_falls_back_to_seek_when_frame_count_missing():
    assert video_duration(FakeCap(25.0, 0.0, 12_000.0)) == pytest.approx(12.0)


def test_duration_raises_when_unknowable():
    with pytest.raises(VideoReadError):
        video_duration(FakeCap(0.0, 0.0, 0.0))


def test_extract_shapes(make_video):
    s = extract(make_video(n=250))  # 10 s at 25 fps
    assert s.rgb.shape == (16, 256, 256, 3) and s.rgb.dtype == np.uint8
    assert s.gray.shape == (16, 8, 64, 64)
    assert s.duration_s == pytest.approx(10.0, abs=0.1)
    assert s.decode_ms > 0


def test_extract_very_short_clip_pads_bursts(make_video):
    s = extract(make_video(n=25))  # 1 s, shorter than the burst span
    assert s.rgb.shape[0] == 16 and s.gray.shape == (16, 8, 64, 64)


def test_extract_keep_subset(make_video):
    s = extract(make_video(n=250), keep=frame_subset(16, 4))
    assert s.rgb.shape[0] == 4


def test_extract_rejects_non_video(tmp_path):
    p = tmp_path / "bad.mp4"
    p.write_text("not a video")
    with pytest.raises(VideoReadError):
        extract(p)
