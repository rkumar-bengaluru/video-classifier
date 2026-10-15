"""Fetch sampled videos with yt-dlp (<=240p, video only); dead ids fall back to same-stratum spares."""
import subprocess
import sys
from pathlib import Path

import pandas as pd

import config as C

VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".avi"}


def find_video(videos_dir, video_id):
    hits = sorted(p for p in Path(videos_dir).glob(f"{video_id}.*") if p.suffix.lower() in VIDEO_EXTS)
    return hits[0] if hits else None


def ytdlp_cmd(youtube_id, out_dir, video_id):
    return [
        sys.executable, "-m", "yt_dlp",
        "-f", "bv*[height<=240][vcodec^=avc1]/bv*[height<=240][vcodec^=vp09]/bv*[height<=240]/wv*",
        "--no-playlist", "--quiet", "--no-warnings",
        "-o", str(Path(out_dir) / f"{video_id}.%(ext)s"),
        f"https://www.youtube.com/watch?v={youtube_id}",
    ]


def _run_ytdlp(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=1800).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def download_sample(plan, out_dir=C.VIDEOS, fetch=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fetch = fetch or (lambda r: _run_ytdlp(ytdlp_cmd(r.youtube_id, out_dir, r.video_id)))

    def ensure(r):
        if find_video(out_dir, r.video_id) is not None:
            return True
        ok = fetch(r)
        print(f"{r.video_id} ({r.youtube_id}): {'ok' if ok else 'FAILED'}", flush=True)
        return ok and find_video(out_dir, r.video_id) is not None

    rows, missing = [], []
    for _, g in plan.groupby("stratum", sort=True):
        spares = [r for r in g.itertuples(index=False) if r.role == "spare"]
        for r in (r for r in g.itertuples(index=False) if r.role == "primary"):
            got = r if ensure(r) else None
            while got is None and spares:
                s = spares.pop(0)
                if ensure(s):
                    got = s
            if got is None:
                missing.append(r.video_id)
                continue
            d = got._asdict()
            d["role"] = "primary"
            d["replaced"] = "" if got is r else r.video_id
            rows.append(d)
    return pd.DataFrame(rows), missing
