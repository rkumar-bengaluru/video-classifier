"""Stratified 10% sample of VideoMME over duration x domain, plus spares per stratum."""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import config as C


def fetch_parquet(url, dest):
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def load_videomme(path):
    df = pd.read_parquet(path).drop_duplicates("video_id")
    df = df.rename(columns={"videoID": "youtube_id"})
    return df[["video_id", "youtube_id", "duration", "domain", "sub_category"]].reset_index(drop=True)


def stratified_sample(videos, frac=0.1, n_spares=2, seed=C.SEED):
    rng = np.random.default_rng(seed)
    parts = []
    for (duration, domain), g in videos.groupby(["duration", "domain"], sort=True):
        g = g.sort_values("video_id")
        n = int(round(len(g) * frac))
        perm = g.iloc[rng.permutation(len(g))]
        chosen = perm.iloc[: n + n_spares].copy()
        chosen["role"] = ["primary"] * n + ["spare"] * (len(chosen) - n)
        chosen["stratum"] = f"{duration}|{domain}"
        parts.append(chosen)
    return pd.concat(parts, ignore_index=True)
