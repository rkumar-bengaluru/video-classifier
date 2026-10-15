"""Rebuild-pipeline batch steps: decode videos to frame caches, then compute all cached features."""
import json
import time

import numpy as np

import config as C
from download import find_video
from frames import VideoReadError, extract, frame_subset
from motion import video_features


def extract_all(sample, videos_dir=C.VIDEOS, frames_dir=C.FRAMES):
    frames_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for vid in sample.video_id:
        out = frames_dir / f"{vid}.npz"
        if out.exists():
            continue
        path = find_video(videos_dir, vid)
        if path is None:
            failed.append((vid, "video file missing"))
            continue
        try:
            s = extract(path)
        except VideoReadError as e:
            failed.append((vid, str(e)))
            continue
        np.savez_compressed(out, rgb=s.rgb, gray=s.gray, duration_s=s.duration_s, decode_ms=s.decode_ms)
        print(f"{vid}: {s.duration_s:.0f}s video decoded in {s.decode_ms:.0f} ms", flush=True)
    return failed


def featurize_all(ids, frames_dir=C.FRAMES, out_dir=C.FEATURES):
    out_dir.mkdir(parents=True, exist_ok=True)
    caches = [np.load(frames_dir / f"{v}.npz") for v in ids]
    rgb = [c["rgb"] for c in caches]
    gray = [c["gray"] for c in caches]
    timings = {"decode_ms_per_timestamp": float(np.median([float(c["decode_ms"]) / C.N_FRAMES for c in caches]))}

    motion_ms = []
    for n in C.FRAME_SUBSETS:
        idx = frame_subset(C.N_FRAMES, n)
        rows = []
        for r, g in zip(rgb, gray):
            t0 = time.perf_counter()
            rows.append(video_features(r[idx], g[idx]))
            if n == C.N_FRAMES:
                motion_ms.append((time.perf_counter() - t0) * 1000 / n)
        np.save(out_dir / f"motion_n{n}.npy", np.stack(rows))
    timings["motion_ms_per_timestamp"] = float(np.median(motion_ms))

    from encoders import load, ms_per_frame, resolve
    chosen, enc_ms, enc_mb = {}, {}, {}
    for kind in ("small", "large"):
        choice = resolve(kind)
        if choice is None:
            print(f"no {kind} encoder available in open_clip; skipping")
            continue
        enc = load(*choice)
        emb = np.stack([enc.embed(r) for r in rgb])
        np.save(out_dir / f"{kind}_f16.npy", emb.astype(np.float16))
        np.save(out_dir / f"{kind}_text.npy", enc.class_text_embeddings())
        chosen[kind] = list(choice)
        enc_ms[kind] = ms_per_frame(enc, rgb[0])
        enc_mb[kind] = enc.image_params * 4 / 1e6
        print(f"{kind}: {choice[0]} D={emb.shape[2]} {enc_ms[kind]:.1f} ms/frame {enc_mb[kind]:.0f} MB fp32", flush=True)
    timings.update(encoder_ms_per_frame=enc_ms, encoder_mb_fp32=enc_mb)

    (out_dir / "video_ids.txt").write_text("\n".join(ids))
    (out_dir / "encoders.json").write_text(json.dumps(chosen, indent=2))
    (out_dir / "timings.json").write_text(json.dumps(timings, indent=2))
