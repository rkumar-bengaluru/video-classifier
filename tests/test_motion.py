import numpy as np

from motion import FEATURE_NAMES, video_features

N = 4


def idx(name):
    return FEATURE_NAMES.index(name)


def rgb(value=0):
    return np.full((N, 256, 256, 3), value, np.uint8)


def test_static_video_has_no_motion():
    gray = np.full((N, 8, 64, 64), 128, np.uint8)
    f = video_features(rgb(), gray)
    assert f.shape == (len(FEATURE_NAMES),) and f.dtype == np.float32
    assert not np.isnan(f).any()
    assert f[idx("motion_mean")] == 0.0
    assert f[idx("cut_rate")] == 0.0
    assert f[idx("static_frac")] == 1.0


def test_black_to_white_cut_is_detected():
    gray = np.zeros((N, 8, 64, 64), np.uint8)
    gray[:, 4:] = 255
    f = video_features(rgb(), gray)
    assert f[idx("cut_rate")] > 0
    assert f[idx("motion_max")] > 0.1


def test_translating_texture_registers_as_shake():
    rng = np.random.default_rng(0)
    tex = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    burst = np.stack([np.roll(tex, 3 * i, axis=1) for i in range(8)])
    gray = np.stack([burst] * N)
    f = video_features(rgb(), gray)
    assert f[idx("shake_mean")] > 1.5
    assert f[idx("cut_rate")] == 0.0
