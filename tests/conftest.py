import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def write_video(path, frames, fps=25):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    return path


@pytest.fixture
def make_video(tmp_path):
    """kind: 'noise' (random frames), 'static' (all black), 'cut' (black then white)."""
    def _make(name="v.avi", n=100, size=(320, 240), fps=25, kind="noise"):
        rng = np.random.default_rng(0)
        w, h = size
        frames = []
        for i in range(n):
            if kind == "noise":
                f = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            else:
                f = np.full((h, w, 3), 255 if (kind == "cut" and i >= n // 2) else 0, np.uint8)
            frames.append(f)
        return write_video(tmp_path / name, frames, fps)
    return _make


@pytest.fixture
def fake_artifacts(tmp_path):
    """Separable synthetic features for 60 videos; the last class deliberately has only 2 videos."""
    import json
    import pandas as pd
    import config as C
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(6), [12, 12, 12, 12, 10, 2])
    n = len(y)
    ids = [f"{i:03d}" for i in range(n)]
    art = tmp_path / "artifacts"
    feat = art / "features"
    feat.mkdir(parents=True)
    centers = rng.normal(size=(6, 8))
    emb = centers[y][:, None, :] + 0.3 * rng.normal(size=(n, 16, 8))
    emb /= np.linalg.norm(emb, axis=2, keepdims=True)
    np.save(feat / "small_f16.npy", emb.astype(np.float16))
    np.save(feat / "small_text.npy", (centers / np.linalg.norm(centers, axis=1, keepdims=True)).astype(np.float32))
    for k in C.FRAME_SUBSETS:
        np.save(feat / f"motion_n{k}.npy", (y[:, None] + 0.5 * rng.normal(size=(n, 13))).astype(np.float32))
    (feat / "video_ids.txt").write_text("\n".join(ids))
    (feat / "encoders.json").write_text(json.dumps({"small": ["FakeEnc", "none"]}))
    (feat / "timings.json").write_text(json.dumps({
        "decode_ms_per_timestamp": 5.0, "motion_ms_per_timestamp": 0.5,
        "encoder_ms_per_frame": {"small": 10.0}, "encoder_mb_fp32": {"small": 45.0}}))
    labels = pd.DataFrame({
        "video_id": ids, "youtube_id": ids, "duration_bucket": "short", "domain": "d", "sub_category": "s",
        "label_auto": [C.CLASSES[i] for i in y], "label_final": [C.CLASSES[i] for i in y],
        "label_changed": False, "note": ""})
    labels.loc[0, ["label_auto", "label_changed"]] = [C.CLASSES[1], True]
    labels.to_csv(art / "labels.csv", index=False)
    return art
