"""labels.csv: auto label from the sub-category mapping, final label after contact-sheet review."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import config as C

COLUMNS = ["video_id", "youtube_id", "duration_bucket", "domain", "sub_category",
           "label_auto", "label_final", "label_changed", "note"]


def load_overrides(path):
    if not Path(path).exists():
        return pd.DataFrame(columns=["video_id", "label_final", "note"])
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def build_labels(sample, overrides=None):
    df = sample.rename(columns={"duration": "duration_bucket"}).copy()
    df["label_auto"] = df.sub_category.map(C.SUBCATEGORY_TO_CLASS)
    if df.label_auto.isna().any():
        raise ValueError(f"unmapped sub-categories: {sorted(df[df.label_auto.isna()].sub_category.unique())}")
    df["label_final"], df["note"] = df.label_auto, ""
    if overrides is not None and len(overrides):
        bad = set(overrides.label_final) - set(C.CLASSES)
        unknown = set(overrides.video_id) - set(df.video_id)
        if bad or unknown:
            raise ValueError(f"bad override classes {bad} or unknown video ids {unknown}")
        o = overrides.set_index("video_id")
        hit = df.video_id.isin(o.index)
        df.loc[hit, "label_final"] = df.loc[hit, "video_id"].map(o.label_final)
        df.loc[hit, "note"] = df.loc[hit, "video_id"].map(o.note)
    df["label_changed"] = df.label_final != df.label_auto
    return df[COLUMNS].reset_index(drop=True)


def contact_sheet(rgb, title, cell=160):
    cols = int(np.ceil(np.sqrt(len(rgb))))
    rows = int(np.ceil(len(rgb) / cols))
    grid = np.zeros((rows * cell, cols * cell, 3), np.uint8)
    for i, f in enumerate(rgb):
        r, c = divmod(i, cols)
        grid[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = cv2.resize(f, (cell, cell))
    bar = np.full((28, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return cv2.cvtColor(np.vstack([bar, grid]), cv2.COLOR_RGB2BGR)


def sheet_pages(sheets, per_page=6, cols=3, width=480):
    pages = []
    for start in range(0, len(sheets), per_page):
        chunk = [cv2.resize(s, (width, int(s.shape[0] * width / s.shape[1]))) for s in sheets[start:start + per_page]]
        while len(chunk) % cols:
            chunk.append(np.full_like(chunk[0], 255))
        pages.append(np.vstack([np.hstack(chunk[i:i + cols]) for i in range(0, len(chunk), cols)]))
    return pages
