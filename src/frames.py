"""Uniform-timestamp sampling: RGB frames for the encoder, gray bursts for motion features."""
import time
from dataclasses import dataclass

import cv2
import numpy as np

import config as C


class VideoReadError(RuntimeError):
    pass


@dataclass
class VideoSample:
    rgb: np.ndarray    # (n, FRAME_SIZE, FRAME_SIZE, 3) uint8 RGB
    gray: np.ndarray   # (n, BURST_LEN, BURST_SIZE, BURST_SIZE) uint8
    duration_s: float
    decode_ms: float


def timestamps(duration_s, n=C.N_FRAMES, edge=C.EDGE_SKIP):
    if duration_s <= 0:
        raise ValueError(f"duration must be positive, got {duration_s}")
    start, end = duration_s * edge, duration_s * (1 - edge)
    return start + (np.arange(n) + 0.5) * (end - start) / n


def frame_subset(n_total, n):
    return np.linspace(0, n_total - 1, n).round().astype(int)


def center_crop_resize(bgr, size=C.FRAME_SIZE):
    h, w = bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = cv2.resize(bgr[y0:y0 + s, x0:x0 + s], (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _burst_gray(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (C.BURST_SIZE, C.BURST_SIZE), interpolation=cv2.INTER_AREA)


def video_duration(cap):
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    if fps > 0 and count > 0:
        return count / fps
    # Container has no usable frame count (some webm/VFR files): seek to the end instead.
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
    ms = cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.0)
    if ms > 0:
        return ms / 1000.0
    raise VideoReadError("cannot determine video duration")


def extract(path, n_total=C.N_FRAMES, keep=None):
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoReadError(f"cannot open {path}")
    try:
        duration = video_duration(cap)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, round(fps * C.BURST_STEP_S))
        ts = timestamps(duration, n_total)
        if keep is not None:
            ts = ts[keep]
        rgb, gray, last = [], [], None
        for t in ts:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            frame = frame if ok else last
            if frame is None:
                continue
            last = frame
            burst = [_burst_gray(frame)]
            for _ in range(C.BURST_LEN - 1):
                for _ in range(step - 1):
                    cap.grab()
                ok, f = cap.read()
                if ok:
                    last = f
                burst.append(_burst_gray(last))  # past the end: repeat the last good frame
            rgb.append(center_crop_resize(frame))
            gray.append(np.stack(burst))
    finally:
        cap.release()
    if not rgb:
        raise VideoReadError(f"no decodable frames in {path}")
    while len(rgb) < len(ts):  # leading seeks failed: pad with the first good frame
        rgb.insert(0, rgb[0])
        gray.insert(0, gray[0])
    return VideoSample(np.stack(rgb), np.stack(gray), float(duration), (time.perf_counter() - t0) * 1000)
