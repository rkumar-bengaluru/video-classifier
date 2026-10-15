"""Model-free motion/appearance features: the temporal signals a per-frame image encoder cannot see."""
import cv2
import numpy as np

import config as C

CUT_THRESH = 0.5      # half-L1 histogram distance between consecutive burst frames
STATIC_THRESH = 0.01  # mean abs diff (0..1) below which a burst counts as static

FEATURE_NAMES = [
    "motion_mean", "motion_std", "motion_max", "cut_rate", "shake_mean", "shake_std",
    "static_frac", "scene_div", "brightness", "brightness_std", "saturation",
    "edge_density", "colorfulness",
]


def _hist(gray):
    h = cv2.calcHist([gray], [0], None, [32], [0, 256]).ravel()
    return h / max(h.sum(), 1.0)


def _hist_dist(a, b):
    return 0.5 * float(np.abs(a - b).sum())


def burst_stats(burst):
    f = burst.astype(np.float32) / 255.0
    diffs = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
    hists = [_hist(g) for g in burst]
    cuts = np.array([_hist_dist(hists[i], hists[i + 1]) > CUT_THRESH for i in range(len(burst) - 1)])
    window = cv2.createHanningWindow((burst.shape[2], burst.shape[1]), cv2.CV_32F)
    shifts = []
    for i in np.flatnonzero(~cuts):
        (dx, dy), _ = cv2.phaseCorrelate(f[i], f[i + 1], window)
        shifts.append(float(np.hypot(dx, dy)))
    return float(diffs.mean()), int(cuts.sum()), shifts


def _colorfulness(rgb):
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return (np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())) / 100.0


def video_features(rgb, gray):
    stats = [burst_stats(b) for b in gray]
    motion = np.array([s[0] for s in stats])
    n_cuts = sum(s[1] for s in stats)
    shifts = np.array([x for s in stats for x in s[2]] or [0.0])
    span_s = len(gray) * (gray.shape[1] - 1) * C.BURST_STEP_S
    firsts = [_hist(b[0]) for b in gray]
    scene_div = np.mean([_hist_dist(firsts[i], firsts[i + 1]) for i in range(len(firsts) - 1)]) if len(firsts) > 1 else 0.0
    hsv = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2HSV) for f in rgb]).astype(np.float32)
    value = hsv[..., 2].mean(axis=(1, 2)) / 255.0
    small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (128, 128)) for f in rgb]
    edges = np.mean([(cv2.Canny(g, 100, 200) > 0).mean() for g in small])
    feats = [
        motion.mean(), motion.std(), motion.max(), n_cuts / span_s, shifts.mean(), shifts.std(),
        (motion < STATIC_THRESH).mean(), scene_div, value.mean(), value.std(),
        hsv[..., 1].mean() / 255.0, edges, np.mean([_colorfulness(f) for f in rgb]),
    ]
    return np.nan_to_num(np.asarray(feats, dtype=np.float32))
