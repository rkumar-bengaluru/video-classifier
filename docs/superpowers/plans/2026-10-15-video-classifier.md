# Video Classifier for Chunking Routing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CPU-only pipeline that classifies a video into one of six chunking-relevant classes. It is benchmarked on a stratified 10% sample of VideoMME with a measured accuracy/latency/size trade-off sweep, and runs end to end from `run.yaml`.

**Architecture:**
- Videos are sampled at 16 uniform timestamps. Each timestamp yields a 256px RGB frame for a frozen `open_clip` image encoder and an 8-frame 64px grayscale burst for 13 handcrafted motion/appearance features.
- Pooled embeddings and/or motion features feed a `StandardScaler` + `LogisticRegression` head, exported to ONNX.
- A sweep over {feature set} × {4, 8, 16 frames} on cached features produces `eval_results.json` and a Pareto plot.
- The cheapest configuration within 0.03 macro-F1 of the best one becomes the production config, used by `predict` and `benchmark`.

**Tech Stack:** Python 3.12, numpy, pandas/pyarrow, opencv-python-headless, open_clip_torch 3.3.0 (+ timm, torch CPU), scikit-learn, skl2onnx, onnxruntime, matplotlib, yt-dlp, pytest.

**Spec:** `docs/superpowers/specs/2026-10-15-video-classifier-design.md`

## Global Constraints

- Six classes, in this exact order everywhere: `gameplay_animation, sports_broadcast, vlog_egocentric, talking_head, presentation_screen, cinematic_documentary`.
- Seed **42** for every random process.
- 16 uniform timestamps, skipping 5% at each end; frame subsets {4, 8, 16} are taken with `np.linspace(0, 15, n).round()`.
- Bursts are 8 grayscale frames of 64×64, 0.25 s apart.
- Encoders: large = `ViT-B-32` / `laion2b_s34b_b79k`. Small = first available of `MobileCLIP2-S0`/`dfndr2b`, `MobileCLIP-S1`/`datacompdr`, `TinyCLIP-ViT-39M-16-Text-19M`/`yfcc15m`. Never a same-size model.
- Embeddings are L2-normalized and stored fp16; pooling is mean ⊕ std.
- The head is `StandardScaler` → `LogisticRegression` (`class_weight="balanced"`, C from inner CV over [0.01, 0.1, 1, 10]).
- Evaluation: stratified 5-fold CV × 5 repeats; 95% bootstrap CIs; macro-F1 is the primary metric.
- The utility matrix U is exactly the 6×6 matrix in spec §5.3.
- Committed `artifacts/` stays under 10 MB; `data/` is gitignored.
- `run.yaml` runs non-interactively on CPU from a fresh checkout without downloading any video. `evaluate` must not import torch.
- All commands are run from the repo root through `python src/main.py <cmd>`.
- Wherever a step says `python`, use the repo venv: `.venv/Scripts/python` on Windows (Git Bash), `.venv/bin/python` on Linux/macOS, or plain `python` after activating the venv (`source .venv/Scripts/activate` or `source .venv/bin/activate`). The steps below spell out the Windows path; substitute the right one on other systems.
- The small-encoder fallback order is fixed: `MobileCLIP2-S0` → `MobileCLIP-S1` → `TinyCLIP-ViT-39M-16-Text-19M`. If all three fail to load, the sweep drops the small rows, which the spec allows. A candidate that is listed but fails to build is removed from `config.py`, with the reason in the commit message.

## Review Focus

1. **Containers that report frame count or FPS as 0** (common with yt-dlp webm/VFR files). The duration must fall back to seeking to the end, and fail with a clear `VideoReadError` instead of returning garbage. Covered by the test in Task 4.
2. **Very short clips** (≈1 s, shorter than the ~1.75 s burst span). Extraction still returns 16 RGB frames and full bursts by repeating the last frame. Covered by the test in Task 4.
3. **Portrait or letterboxed video** (for example 240×426 shorts). The center square is cropped without distortion. Covered by the test in Task 4.
4. **A class with only 1–2 labeled videos.** CV must not crash, and the probability columns must stay aligned to the global class order even when a training fold lacks that class. Covered by the test in Task 9.
5. **A corrupt or non-video file passed to `predict`**, or a directory argument. A bad file yields an `{"error": ...}` record without aborting the other videos, the CLI exits 1, and directories expand to the video files they contain. Covered by the tests in Task 11.

---

### Task 1: Scaffold, config, dependencies

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `src/config.py`, `src/main.py`, `tests/conftest.py`, `tests/test_config.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces:
  - `config` constants: `ROOT, DATA, VIDEOS, FRAMES, SHEETS, ART, FEATURES, MODELS, SAMPLES, SEED, N_FRAMES, FRAME_SUBSETS, FRAME_SIZE, BURST_LEN, BURST_STEP_S, BURST_SIZE, EDGE_SKIP, VIDEOMME_PARQUET_URL, CLASSES, CHUNKERS, FALLBACK_CHUNKER, SUBCATEGORY_TO_CLASS, UTILITY, ENCODER_CANDIDATES, ZERO_SHOT_PROMPTS`.
  - `main.main(argv) -> int`, with subcommands registered through `SUBCOMMANDS`.
  - The conftest fixture `make_video`.

- [ ] **Step 1: Create the venv and install the dependencies**

`requirements.txt`:
```
--extra-index-url https://download.pytorch.org/whl/cpu
torch
numpy
pandas
pyarrow
opencv-python-headless
scikit-learn
skl2onnx
onnx
onnxruntime
open_clip_torch==3.3.0
timm
pillow
matplotlib
yt-dlp
pytest
```
Run: `python -m venv .venv && .venv/Scripts/python -m pip install -q -r requirements.txt`
Then pin exact versions: `.venv/Scripts/python -m pip freeze | grep -iE "^(torch|numpy|pandas|pyarrow|opencv-python-headless|scikit-learn|skl2onnx|onnx|onnxruntime|open_clip_torch|timm|pillow|matplotlib|yt-dlp|pytest)=="`. Rewrite each line of `requirements.txt` as `name==version`, keeping the `--extra-index-url` line first. Write torch's version without its `+cpu` suffix, so that `==X.Y.Z` still resolves to the CPU wheel through the extra index.

- [ ] **Step 2: `.gitignore`, `pytest.ini`, conftest**

Append to `.gitignore`:
```
data/
.pytest_cache/
```
`pytest.ini`:
```ini
[pytest]
testpaths = tests
markers =
    slow: needs network access or model weights
addopts = -m "not slow"
```
`tests/conftest.py`:
```python
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
```

- [ ] **Step 3: Write the failing config test**

`tests/test_config.py`:
```python
import numpy as np

import config as C

VIDEOMME_SUBCATEGORIES = {
    "Humanity & History", "Literature & Art", "Biology & Medicine", "Finance & Commerce",
    "Astronomy", "Geography", "Law", "Life Tip", "Technology", "Animation", "Movie & TV Show",
    "Documentary", "News Report", "Esports", "Basketball", "Football", "Athletics",
    "Other Sports", "Stage Play", "Magic Show", "Variety Show", "Acrobatics", "Handicraft",
    "Food", "Fashion", "Daily Life", "Travel", "Pet & Animal", "Exercise", "Multilingual",
}


def test_every_videomme_subcategory_maps_to_a_class():
    assert set(C.SUBCATEGORY_TO_CLASS) == VIDEOMME_SUBCATEGORIES
    assert set(C.SUBCATEGORY_TO_CLASS.values()) <= set(C.CLASSES)


def test_classes_chunkers_prompts_consistent():
    assert len(C.CLASSES) == 6
    assert list(C.CHUNKERS) == C.CLASSES
    assert list(C.ZERO_SHOT_PROMPTS) == C.CLASSES


def test_utility_matrix_shape_and_values():
    U = np.array(C.UTILITY)
    assert U.shape == (6, 6)
    assert np.allclose(U, U.T)
    assert np.allclose(np.diag(U), 1.0)
    assert ((U >= 0) & (U <= 1)).all()
```

- [ ] **Step 4: Run it and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'config'`.

- [ ] **Step 5: Write `src/config.py`**

```python
"""Single source of truth for classes, chunkers, paths and pipeline constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VIDEOS = DATA / "videos"
FRAMES = DATA / "frames"
SHEETS = DATA / "sheets"
ART = ROOT / "artifacts"
FEATURES = ART / "features"
MODELS = ART / "models"
SAMPLES = ROOT / "samples"

SEED = 42
N_FRAMES = 16
FRAME_SUBSETS = (4, 8, 16)
FRAME_SIZE = 256          # encoder transforms resize to their own input size
BURST_LEN = 8
BURST_STEP_S = 0.25
BURST_SIZE = 64
EDGE_SKIP = 0.05

VIDEOMME_PARQUET_URL = (
    "https://huggingface.co/datasets/lmms-lab/Video-MME/resolve/main/"
    "videomme/test-00000-of-00001.parquet"
)

CLASSES = [
    "gameplay_animation",
    "sports_broadcast",
    "vlog_egocentric",
    "talking_head",
    "presentation_screen",
    "cinematic_documentary",
]

CHUNKERS = {
    "gameplay_animation": "shot_boundary",
    "sports_broadcast": "motion_segment",
    "vlog_egocentric": "activity",
    "talking_head": "speaker_turn",
    "presentation_screen": "slide_change",
    "cinematic_documentary": "semantic_scene",
}
FALLBACK_CHUNKER = "fixed_window"

# Default (auto) labels; corrected per video in the contact-sheet review.
SUBCATEGORY_TO_CLASS = {
    "Esports": "gameplay_animation",
    "Animation": "gameplay_animation",
    "Basketball": "sports_broadcast",
    "Football": "sports_broadcast",
    "Athletics": "sports_broadcast",
    "Other Sports": "sports_broadcast",
    "Daily Life": "vlog_egocentric",
    "Travel": "vlog_egocentric",
    "Food": "vlog_egocentric",
    "Handicraft": "vlog_egocentric",
    "Exercise": "vlog_egocentric",
    "Fashion": "vlog_egocentric",
    "News Report": "talking_head",
    "Movie & TV Show": "cinematic_documentary",
    "Documentary": "cinematic_documentary",
    "Stage Play": "cinematic_documentary",
    "Variety Show": "cinematic_documentary",
    "Magic Show": "cinematic_documentary",
    "Acrobatics": "cinematic_documentary",
    "Pet & Animal": "cinematic_documentary",
    "Humanity & History": "cinematic_documentary",
    "Literature & Art": "cinematic_documentary",
    "Biology & Medicine": "cinematic_documentary",
    "Finance & Commerce": "cinematic_documentary",
    "Astronomy": "cinematic_documentary",
    "Geography": "cinematic_documentary",
    "Law": "cinematic_documentary",
    "Life Tip": "cinematic_documentary",
    "Technology": "cinematic_documentary",
    "Multilingual": "cinematic_documentary",
}

# U[true][pred]: share of downstream chunking quality kept when routing `true` to `pred`'s chunker.
# An assumption to validate against retrieval recall; see spec section 5.3.
UTILITY = [
    [1.0, 0.7, 0.3, 0.2, 0.2, 0.4],
    [0.7, 1.0, 0.3, 0.2, 0.2, 0.3],
    [0.3, 0.3, 1.0, 0.3, 0.2, 0.5],
    [0.2, 0.2, 0.3, 1.0, 0.6, 0.3],
    [0.2, 0.2, 0.2, 0.6, 1.0, 0.2],
    [0.4, 0.3, 0.5, 0.3, 0.2, 1.0],
]

ENCODER_CANDIDATES = {
    "small": [
        ("MobileCLIP2-S0", "dfndr2b"),
        ("MobileCLIP-S1", "datacompdr"),
        ("TinyCLIP-ViT-39M-16-Text-19M", "yfcc15m"),
    ],
    "large": [("ViT-B-32", "laion2b_s34b_b79k")],
}

ZERO_SHOT_PROMPTS = {
    "gameplay_animation": ["a screenshot of a video game", "esports gameplay with a HUD overlay",
                           "a frame from an animated cartoon"],
    "sports_broadcast": ["a TV broadcast of a sports match", "athletes competing in a stadium",
                         "a basketball or football game on television"],
    "vlog_egocentric": ["a first-person handheld vlog", "a person filming their daily life with a phone",
                        "a travel or cooking vlog"],
    "talking_head": ["a news anchor speaking to the camera", "a person talking in an interview",
                     "a podcast host talking"],
    "presentation_screen": ["a presentation slide with text", "a screen recording of a computer",
                            "a lecture with slides and diagrams"],
    "cinematic_documentary": ["a scene from a movie", "a nature documentary shot", "a cinematic film still"],
}
```

- [ ] **Step 6: Write `src/main.py` with an empty subcommand registry**

```python
"""CLI entry point: python src/main.py <command>."""
import argparse
import sys

# Each entry: name -> (help, configure(parser), handler(args) -> int)
SUBCOMMANDS = {}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="main.py", description="Video type classifier for chunking routing")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, (help_text, configure, handler) in SUBCOMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        configure(p)
        p.set_defaults(func=handler)
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
```
Later tasks add a `cmd_*` handler plus a `SUBCOMMANDS["name"] = (...)` line **above** `def main`. Imports go inside the handlers so that `evaluate` never imports torch.

- [ ] **Step 7: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest -q && .venv/Scripts/python src/main.py --help`
Expected: 3 passed; help text printed.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt pytest.ini .gitignore src/config.py src/main.py tests/conftest.py tests/test_config.py
git commit -m "Scaffold: config, CLI skeleton, pinned CPU dependencies"
```

---

### Task 2: Stratified sampling

**Files:**
- Create: `src/sampling.py`, `tests/test_sampling.py`
- Modify: `src/main.py` (add `sample`)

**Interfaces:**
- Consumes: `config.SEED, DATA, ART, VIDEOMME_PARQUET_URL`.
- Produces:
  - `fetch_parquet(url, dest: Path) -> Path`
  - `load_videomme(path) -> DataFrame[video_id, youtube_id, duration, domain, sub_category]`
  - `stratified_sample(videos, frac=0.1, n_spares=2, seed=SEED) -> DataFrame` with the added columns `stratum` (`"{duration}|{domain}"`) and `role` (`"primary"` or `"spare"`).
  - CLI `sample` writes `artifacts/sample_plan.csv`.

- [ ] **Step 1: Write the failing test**

`tests/test_sampling.py`:
```python
import pandas as pd

from sampling import stratified_sample

DOMAIN_SIZES = {"Knowledge": 90, "Film & Television": 40, "Sports Competition": 50,
                "Artistic Performance": 40, "Life Record": 70, "Multilingual": 10}


def fake_videomme():
    rows, i = [], 0
    for duration in ("short", "medium", "long"):
        for domain, size in DOMAIN_SIZES.items():
            for _ in range(size):
                rows.append({"video_id": f"{i:03d}", "youtube_id": f"yt{i}", "duration": duration,
                             "domain": domain, "sub_category": f"{domain}-sub"})
                i += 1
    return pd.DataFrame(rows)


def test_sample_is_ten_percent_per_stratum_with_disjoint_spares():
    s = stratified_sample(fake_videomme())
    prim = s[s.role == "primary"]
    assert len(prim) == 90
    per = prim.groupby(["duration", "domain"]).size()
    for (duration, domain), n in per.items():
        assert n == round(DOMAIN_SIZES[domain] * 0.1)
    spares = s[s.role == "spare"]
    assert len(spares) == 36
    assert set(spares.video_id).isdisjoint(prim.video_id)


def test_sample_is_deterministic_for_seed():
    a = stratified_sample(fake_videomme(), seed=42)
    b = stratified_sample(fake_videomme(), seed=42)
    c = stratified_sample(fake_videomme(), seed=7)
    assert a.video_id.tolist() == b.video_id.tolist()
    assert a.video_id.tolist() != c.video_id.tolist()
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_sampling.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'sampling'`.

- [ ] **Step 3: Implement `src/sampling.py`**

```python
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
```

- [ ] **Step 4: Add the `sample` subcommand to `src/main.py`** (above `def main`)

```python
def _read(path):
    import pandas as pd
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def cmd_sample(args):
    import config as C
    from sampling import fetch_parquet, load_videomme, stratified_sample
    parquet = fetch_parquet(C.VIDEOMME_PARQUET_URL, C.DATA / "videomme.parquet")
    plan = stratified_sample(load_videomme(parquet))
    C.ART.mkdir(parents=True, exist_ok=True)
    plan.to_csv(C.ART / "sample_plan.csv", index=False)
    print(plan.groupby("role").size().to_string())


SUBCOMMANDS["sample"] = ("draw the stratified 10% VideoMME sample", lambda p: None, cmd_sample)
```

- [ ] **Step 5: Run the tests and the real command**

Run: `.venv/Scripts/python -m pytest tests/test_sampling.py -q && .venv/Scripts/python src/main.py sample`
Expected: 2 passed; output shows `primary 90`, `spare 36`.

- [ ] **Step 6: Commit**

```bash
git add src/sampling.py src/main.py tests/test_sampling.py artifacts/sample_plan.csv
git commit -m "Stratified 10% VideoMME sample with spares"
```

---

### Task 3: Download with spare replacement (then start the real download in the background)

**Files:**
- Create: `src/download.py`, `tests/test_download.py`
- Modify: `src/main.py` (add `download`)

**Interfaces:**
- Consumes: the `artifacts/sample_plan.csv` columns from Task 2.
- Produces:
  - `VIDEO_EXTS`
  - `find_video(videos_dir, video_id) -> Path | None`
  - `ytdlp_cmd(youtube_id, out_dir, video_id) -> list[str]`
  - `download_sample(plan, out_dir=VIDEOS, fetch=None) -> (DataFrame final, list[str] missing)`. `final` has the plan's columns plus `replaced` (the original video_id, or `""`).
  - CLI `download` writes `artifacts/sample.csv`.

- [ ] **Step 1: Write the failing test**

`tests/test_download.py`:
```python
import pandas as pd

from download import download_sample, find_video, ytdlp_cmd


def plan():
    return pd.DataFrame([
        {"video_id": "001", "youtube_id": "a", "stratum": "s|A", "role": "primary"},
        {"video_id": "002", "youtube_id": "b", "stratum": "s|A", "role": "primary"},
        {"video_id": "003", "youtube_id": "c", "stratum": "s|A", "role": "spare"},
        {"video_id": "004", "youtube_id": "d", "stratum": "s|A", "role": "spare"},
        {"video_id": "005", "youtube_id": "e", "stratum": "s|B", "role": "primary"},
    ])


def fake_fetch(out_dir, dead):
    def fetch(row):
        if row.video_id in dead:
            return False
        (out_dir / f"{row.video_id}.mp4").write_bytes(b"x")
        return True
    return fetch


def test_dead_primary_is_replaced_by_spare_from_same_stratum(tmp_path):
    final, missing = download_sample(plan(), tmp_path, fetch=fake_fetch(tmp_path, dead={"002"}))
    assert final.video_id.tolist() == ["001", "003", "005"]
    assert final.set_index("video_id").loc["003", "replaced"] == "002"
    assert missing == []


def test_missing_when_spares_exhausted(tmp_path):
    final, missing = download_sample(plan(), tmp_path, fetch=fake_fetch(tmp_path, dead={"002", "003", "004"}))
    assert missing == ["002"]
    assert len(final) == 2


def test_existing_file_is_not_refetched(tmp_path):
    (tmp_path / "001.webm").write_bytes(b"x")
    calls = []
    def fetch(row):
        calls.append(row.video_id)
        (tmp_path / f"{row.video_id}.mp4").write_bytes(b"x")
        return True
    download_sample(plan(), tmp_path, fetch=fetch)
    assert "001" not in calls
    assert find_video(tmp_path, "001").suffix == ".webm"


def test_ytdlp_cmd_is_240p_video_only():
    cmd = ytdlp_cmd("abc", "out", "007")
    assert "https://www.youtube.com/watch?v=abc" in cmd
    assert any("height<=240" in c for c in cmd)
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_download.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'download'`.

- [ ] **Step 3: Implement `src/download.py`**

```python
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
```

- [ ] **Step 4: Add the `download` subcommand** (above `def main`)

```python
def cmd_download(args):
    import config as C
    from download import download_sample
    final, missing = download_sample(_read(C.ART / "sample_plan.csv"))
    final.to_csv(C.ART / "sample.csv", index=False)
    print(f"{len(final)} videos ready, {int((final.replaced != '').sum())} replaced by spares, "
          f"unfilled slots: {missing or 'none'}")


SUBCOMMANDS["download"] = ("download sampled videos (yt-dlp, <=240p)", lambda p: None, cmd_download)
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_download.py -q`
Expected: 4 passed.

- [ ] **Step 6: Commit, then start the real download in the background**

```bash
git add src/download.py src/main.py tests/test_download.py
git commit -m "yt-dlp download with same-stratum spare replacement"
```
Start the download with Bash `run_in_background: true`: `.venv/Scripts/python src/main.py download > data/download.log 2>&1`. Carry on with Tasks 4–11 while it runs (≈2.5 GB). If the user has supplied videos instead, put them in `data/videos/{video_id}.<ext>` and the command will skip them.

---

### Task 4: Frame sampling

**Files:**
- Create: `src/frames.py`, `tests/test_frames.py`

**Interfaces:**
- Consumes: `config.N_FRAMES, FRAME_SIZE, BURST_LEN, BURST_STEP_S, BURST_SIZE, EDGE_SKIP`.
- Produces:
  - `VideoReadError(RuntimeError)`
  - `VideoSample(rgb (n,256,256,3) uint8 RGB, gray (n,8,64,64) uint8, duration_s float, decode_ms float)`
  - `timestamps(duration_s, n=16, edge=0.05) -> np.ndarray`
  - `frame_subset(n_total, n) -> np.ndarray[int]`
  - `center_crop_resize(bgr, size=256) -> RGB ndarray`
  - `video_duration(cap) -> float`
  - `extract(path, n_total=16, keep=None) -> VideoSample`. `keep` is an index array into the n_total timestamps.

- [ ] **Step 1: Write the failing tests**

`tests/test_frames.py`:
```python
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
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/Scripts/python -m pytest tests/test_frames.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'frames'`.

- [ ] **Step 3: Implement `src/frames.py`**

```python
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
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_frames.py -q`
Expected: 10 passed. If `test_extract_rejects_non_video` fails because OpenCV "opens" the text file, it means `video_duration` returned a value: add `if count <= 0 and fps <= 0` handling, so the empty result raises from `video_duration` or from the "no decodable frames" branch. Do not weaken the test.

- [ ] **Step 5: Commit**

```bash
git add src/frames.py tests/test_frames.py
git commit -m "Uniform-timestamp frame sampling with gray motion bursts"
```

---

### Task 5: Motion and appearance features

**Files:**
- Create: `src/motion.py`, `tests/test_motion.py`

**Interfaces:**
- Consumes: `config.BURST_STEP_S`; `VideoSample.rgb` and `.gray` shapes from Task 4.
- Produces:
  - `FEATURE_NAMES` (13 names)
  - `burst_stats(burst) -> (float mean_diff, int n_cuts, list[float] shifts)`
  - `video_features(rgb, gray) -> np.ndarray (13,) float32`, free of NaN.

- [ ] **Step 1: Write the failing tests**

`tests/test_motion.py`:
```python
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
    assert f[idx("motion_max")] > 0.5


def test_translating_texture_registers_as_shake():
    rng = np.random.default_rng(0)
    tex = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    burst = np.stack([np.roll(tex, 3 * i, axis=1) for i in range(8)])
    gray = np.stack([burst] * N)
    f = video_features(rgb(), gray)
    assert f[idx("shake_mean")] > 1.5
    assert f[idx("cut_rate")] == 0.0
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/Scripts/python -m pytest tests/test_motion.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'motion'`.

- [ ] **Step 3: Implement `src/motion.py`**

```python
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
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_motion.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/motion.py tests/test_motion.py
git commit -m "Handcrafted motion/cut/shake/appearance features"
```

---

### Task 6: Frozen encoders

**Files:**
- Create: `src/encoders.py`, `tests/test_encoders.py`

**Interfaces:**
- Consumes: `config.ENCODER_CANDIDATES, ZERO_SHOT_PROMPTS, CLASSES`.
- Produces:
  - `resolve(kind) -> (name, pretrained) | None`
  - `Encoder(name, pretrained)` with `.embed(frames uint8 (N,H,W,3), batch_size=32) -> (N, D) float32 L2-normalized`, `.embed_text(texts) -> (T, D)`, `.class_text_embeddings() -> (6, D)`, `.image_params`.
  - `load(name, pretrained) -> Encoder`
  - `ms_per_frame(enc, frames) -> float`

- [ ] **Step 1: Write the tests**

`tests/test_encoders.py`:
```python
import numpy as np
import pytest

import config as C


def test_resolve_returns_a_listed_candidate():
    from encoders import resolve
    small = resolve("small")
    assert small is None or small in C.ENCODER_CANDIDATES["small"]
    assert resolve("large") == ("ViT-B-32", "laion2b_s34b_b79k")


@pytest.mark.slow
def test_small_encoder_embeds_normalized_vectors():
    from encoders import load, resolve
    enc = load(*resolve("small"))
    frames = np.random.default_rng(0).integers(0, 256, (3, 256, 256, 3), dtype=np.uint8)
    e = enc.embed(frames)
    assert e.shape[0] == 3 and e.dtype == np.float32
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-4)
    t = enc.class_text_embeddings()
    assert t.shape == (6, e.shape[1])
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/Scripts/python -m pytest tests/test_encoders.py -q -m "slow or not slow"`
Expected: FAIL with `ModuleNotFoundError: No module named 'encoders'`.

- [ ] **Step 3: Implement `src/encoders.py`**

```python
"""Frozen open_clip encoders: frames -> L2-normalized embeddings (no training)."""
import time

import numpy as np
import open_clip
import torch
from PIL import Image

import config as C


def resolve(kind):
    available = set(open_clip.list_pretrained())
    for name, tag in C.ENCODER_CANDIDATES[kind]:
        if (name, tag) in available:
            return name, tag
    return None


class Encoder:
    def __init__(self, name, pretrained):
        self.name, self.pretrained = name, pretrained
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(name)

    @property
    def image_params(self):
        return sum(p.numel() for p in self.model.visual.parameters())

    @torch.inference_mode()
    def embed(self, frames, batch_size=32):
        out = []
        for i in range(0, len(frames), batch_size):
            x = torch.stack([self.preprocess(Image.fromarray(f)) for f in frames[i:i + batch_size]])
            out.append(self.model.encode_image(x, normalize=True).float().numpy())
        return np.concatenate(out)

    @torch.inference_mode()
    def embed_text(self, texts):
        return self.model.encode_text(self.tokenizer(texts), normalize=True).float().numpy()

    def class_text_embeddings(self):
        rows = []
        for cls in C.CLASSES:
            m = self.embed_text(C.ZERO_SHOT_PROMPTS[cls]).mean(0)
            rows.append(m / np.linalg.norm(m))
        return np.stack(rows).astype(np.float32)


def load(name, pretrained):
    return Encoder(name, pretrained)


def ms_per_frame(enc, frames, repeats=5):
    enc.embed(frames[:1])  # warm-up
    times = []
    for i in range(repeats):
        t0 = time.perf_counter()
        enc.embed(frames[i % len(frames):i % len(frames) + 1])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))
```

- [ ] **Step 4: Run all encoder tests, including the slow one (this downloads about 300 MB once)**

`MobileCLIP2-S0` was confirmed in `open_clip`'s `pretrained.py` on main, but the 3.3.0 wheel may predate it. In that case `resolve()` simply returns the next candidate.

Run: `.venv/Scripts/python -m pytest tests/test_encoders.py -q -m "slow or not slow"`
Expected: 2 passed. Record which small encoder `resolve("small")` returned.
- If `MobileCLIP2-S0` fails to build (for example because of a timm version error), remove it from `ENCODER_CANDIDATES["small"]` in `config.py`, note the reason in the commit message, and re-run so that the next candidate is used.
- If no small candidate works at all, the sweep drops the small rows, as the spec allows.

- [ ] **Step 5: Commit**

```bash
git add src/encoders.py tests/test_encoders.py src/config.py
git commit -m "Frozen open_clip encoder wrapper with candidate resolution"
```

---

### Task 7: Labels, contact sheets, extract/featurize commands

**Files:**
- Create: `src/labels.py`, `src/featurize.py`, `tests/test_labels.py`
- Modify: `src/main.py` (add `extract`, `label-sheets`, `labels`, `featurize`)

**Interfaces:**
- Consumes:
  - `frames.extract, frame_subset, VideoReadError`
  - `motion.video_features`
  - `download.find_video`
  - `encoders.resolve, load, ms_per_frame`
  - the `artifacts/sample.csv` columns: `video_id, youtube_id, duration, domain, sub_category, stratum, role, replaced`.
- Produces:
  - `labels.build_labels(sample, overrides=None) -> DataFrame[video_id, youtube_id, duration_bucket, domain, sub_category, label_auto, label_final, label_changed, note]`
  - `labels.load_overrides(path) -> DataFrame[video_id, label_final, note]`
  - `labels.contact_sheet(rgb, title) -> BGR ndarray`
  - `labels.sheet_pages(sheets, per_page=6) -> list[ndarray]`
  - `featurize.extract_all(sample, videos_dir, frames_dir) -> list[(video_id, reason)]`
  - `featurize.featurize_all(ids, frames_dir, out_dir)`, which writes into `artifacts/features/`:
    - `motion_n{4,8,16}.npy` (V,13)
    - `{small,large}_f16.npy` (V,16,D) fp16
    - `{small,large}_text.npy` (6,D)
    - `video_ids.txt`
    - `encoders.json` `{kind: [name, pretrained]}`
    - `timings.json` `{decode_ms_per_timestamp, motion_ms_per_timestamp, encoder_ms_per_frame{kind}, encoder_mb_fp32{kind}}`

- [ ] **Step 1: Write the failing tests**

`tests/test_labels.py`:
```python
import numpy as np
import pandas as pd
import pytest

from labels import build_labels, contact_sheet, load_overrides, sheet_pages


def sample():
    return pd.DataFrame({
        "video_id": ["001", "002"], "youtube_id": ["a", "b"], "duration": ["short", "long"],
        "domain": ["Sports Competition", "Knowledge"], "sub_category": ["Football", "Technology"],
    })


def test_auto_labels_from_mapping_without_overrides():
    df = build_labels(sample())
    assert df.label_auto.tolist() == ["sports_broadcast", "cinematic_documentary"]
    assert (df.label_final == df.label_auto).all()
    assert not df.label_changed.any()


def test_overrides_change_final_label_only():
    o = pd.DataFrame({"video_id": ["002"], "label_final": ["presentation_screen"], "note": ["slides"]})
    df = build_labels(sample(), o).set_index("video_id")
    assert df.loc["002", "label_auto"] == "cinematic_documentary"
    assert df.loc["002", "label_final"] == "presentation_screen"
    assert bool(df.loc["002", "label_changed"]) and df.loc["002", "note"] == "slides"


def test_override_with_unknown_class_is_rejected():
    o = pd.DataFrame({"video_id": ["001"], "label_final": ["podcast"], "note": [""]})
    with pytest.raises(ValueError):
        build_labels(sample(), o)


def test_missing_overrides_file_is_empty(tmp_path):
    assert len(load_overrides(tmp_path / "none.csv")) == 0


def test_contact_sheet_and_pages():
    rgb = np.zeros((16, 256, 256, 3), np.uint8)
    sheet = contact_sheet(rgb, "001 | Football -> sports_broadcast")
    assert sheet.ndim == 3 and sheet.shape[1] == 4 * 160
    pages = sheet_pages([sheet] * 7, per_page=6)
    assert len(pages) == 2
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/Scripts/python -m pytest tests/test_labels.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'labels'`.

- [ ] **Step 3: Implement `src/labels.py`**

```python
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
```

- [ ] **Step 4: Implement `src/featurize.py`**

```python
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
```

- [ ] **Step 5: Add the subcommands** (above `def main`)

```python
def _ready_ids():
    import config as C
    sample = _read(C.ART / "sample.csv")
    return sample, [v for v in sample.video_id if (C.FRAMES / f"{v}.npz").exists()]


def cmd_extract(args):
    import config as C
    from featurize import extract_all
    failed = extract_all(_read(C.ART / "sample.csv"))
    for vid, reason in failed:
        print(f"FAILED {vid}: {reason}")
    return 1 if failed else 0


def cmd_label_sheets(args):
    import cv2
    import numpy as np
    import config as C
    from labels import contact_sheet, sheet_pages
    sample, ids = _ready_ids()
    info = sample.set_index("video_id")
    C.SHEETS.mkdir(parents=True, exist_ok=True)
    sheets = []
    for vid in ids:
        row = info.loc[vid]
        title = f"{vid} | {row.sub_category} -> {C.SUBCATEGORY_TO_CLASS[row.sub_category]}"
        sheet = contact_sheet(np.load(C.FRAMES / f"{vid}.npz")["rgb"], title)
        cv2.imwrite(str(C.SHEETS / f"{vid}.jpg"), sheet)
        sheets.append(sheet)
    for k, page in enumerate(sheet_pages(sheets)):
        cv2.imwrite(str(C.SHEETS / f"page_{k:02d}.jpg"), page)
    print(f"wrote {len(sheets)} sheets to {C.SHEETS}")


def cmd_labels(args):
    import config as C
    from labels import build_labels, load_overrides
    df = build_labels(_read(C.ART / "sample.csv"), load_overrides(C.ART / "label_overrides.csv"))
    df.to_csv(C.ART / "labels.csv", index=False)
    print(df.label_final.value_counts().to_string())
    print(f"changed by review: {int(df.label_changed.sum())}/{len(df)}")


def cmd_featurize(args):
    from featurize import featurize_all
    _, ids = _ready_ids()
    featurize_all(ids)


SUBCOMMANDS["extract"] = ("decode frames + bursts for the sample", lambda p: None, cmd_extract)
SUBCOMMANDS["label-sheets"] = ("render contact sheets for label review", lambda p: None, cmd_label_sheets)
SUBCOMMANDS["labels"] = ("build labels.csv from mapping + overrides", lambda p: None, cmd_labels)
SUBCOMMANDS["featurize"] = ("compute motion features and encoder embeddings", lambda p: None, cmd_featurize)
```

- [ ] **Step 6: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_labels.py -q`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add src/labels.py src/featurize.py src/main.py tests/test_labels.py
git commit -m "Labels with review overrides, contact sheets, extract/featurize commands"
```

---

### Task 8: Metrics (CUS, Bayes rule, bootstrap, risk–coverage)

**Files:**
- Create: `src/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: `config.SEED, CLASSES`.
- Produces:
  - `confusion(y, p, k) -> (k,k) int`
  - `macro_f1(y, p, k) -> (per_class ndarray, float)`. The mean is over classes with support > 0.
  - `accuracy(y, p) -> float`
  - `cus(y, p, U) -> float`
  - `bayes_decide(proba, U) -> ndarray[int]`
  - `bootstrap_ci(metric, y, preds, n_boot=200, seed=42) -> {"mean","lo","hi"}`. `preds` has shape (R, N) or (N,).
  - `risk_coverage(y, proba) -> list[{"tau","coverage","accuracy"}]`
  - `choose_tau(rows, target_acc=0.9, min_coverage=0.5) -> float`
  - `summarize(y, p, U) -> {"accuracy","macro_f1","cus","per_class_f1","confusion"}`

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:
```python
import numpy as np
import pytest

from metrics import (accuracy, bayes_decide, bootstrap_ci, choose_tau, cus, macro_f1,
                     risk_coverage, summarize)

U3 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.9], [0.0, 0.9, 1.0]])


def test_cus_hand_computed():
    assert cus(np.array([0, 1, 2]), np.array([0, 2, 2]), U3) == pytest.approx((1 + 0.9 + 1) / 3)


def test_bayes_rule_prefers_expected_utility_over_argmax():
    proba = np.array([[0.40, 0.35, 0.25]])
    # EU(k) = sum_j p_j U[j,k] = [0.40, 0.575, 0.565]
    assert proba.argmax(1)[0] == 0
    assert bayes_decide(proba, U3)[0] == 1


def test_macro_f1_ignores_classes_without_support():
    per, m = macro_f1(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 3)
    assert per[0] == pytest.approx(2 / 3) and per[1] == pytest.approx(0.8)
    assert m == pytest.approx((2 / 3 + 0.8) / 2)


def test_bootstrap_ci_on_perfect_predictions():
    y = np.arange(20) % 3
    ci = bootstrap_ci(accuracy, y, np.stack([y, y]), n_boot=50)
    assert ci == {"mean": 1.0, "lo": 1.0, "hi": 1.0}


def test_risk_coverage_and_tau():
    y = np.array([0, 1, 0, 1])
    proba = np.array([[0.95, 0.05], [0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])
    rows = {r["tau"]: r for r in risk_coverage(y, proba)}
    assert rows[0.0]["coverage"] == 1.0 and rows[0.0]["accuracy"] == 0.75
    assert rows[0.95]["coverage"] == 0.25 and rows[0.95]["accuracy"] == 1.0
    assert choose_tau(list(rows.values()), target_acc=0.99, min_coverage=0.9) == 0.0


def test_summarize_keys():
    s = summarize(np.array([0, 1, 2]), np.array([0, 1, 1]), U3)
    assert set(s) == {"accuracy", "macro_f1", "cus", "per_class_f1", "confusion"}
    assert np.array(s["confusion"]).sum() == 3
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/Scripts/python -m pytest tests/test_metrics.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'metrics'`.

- [ ] **Step 3: Implement `src/metrics.py`**

```python
"""Classification metrics plus the Chunking Utility Score and its expected-utility decision rule."""
import numpy as np

import config as C


def confusion(y, p, k):
    return np.bincount(np.asarray(y) * k + np.asarray(p), minlength=k * k).reshape(k, k)


def macro_f1(y, p, k):
    cm = confusion(y, p, k)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    denom = (support + predicted).astype(float)
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return f1, float(f1[support > 0].mean())


def accuracy(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def cus(y, p, U):
    return float(np.asarray(U)[y, p].mean())


def bayes_decide(proba, U):
    # Expected utility of routing to k: sum_j p(j) * U[j, k]; pick the max.
    return (proba @ np.asarray(U)).argmax(axis=1)


def bootstrap_ci(metric, y, preds, n_boot=200, seed=C.SEED):
    y, preds = np.asarray(y), np.atleast_2d(preds)
    point = float(np.mean([metric(y, p) for p in preds]))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        boots.append(np.mean([metric(y[idx], p[idx]) for p in preds]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": point, "lo": float(lo), "hi": float(hi)}


def risk_coverage(y, proba, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.0, 1.0, 0.05), 2)
    conf, pred = proba.max(1), proba.argmax(1)
    rows = []
    for t in thresholds:
        m = conf >= t
        rows.append({"tau": float(t), "coverage": float(m.mean()),
                     "accuracy": float((pred[m] == y[m]).mean()) if m.any() else None})
    return rows


def choose_tau(rows, target_acc=0.9, min_coverage=0.5):
    ok = [r for r in rows if r["accuracy"] is not None and r["accuracy"] >= target_acc and r["coverage"] >= min_coverage]
    return ok[0]["tau"] if ok else 0.0


def summarize(y, p, U):
    k = len(U)
    per, m = macro_f1(y, p, k)
    return {"accuracy": accuracy(y, p), "macro_f1": m, "cus": cus(y, p, U),
            "per_class_f1": dict(zip(C.CLASSES, per.round(4).tolist())),
            "confusion": confusion(y, p, k).tolist()}
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_metrics.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "Metrics: macro-F1, CUS, Bayes decision rule, bootstrap CI, risk-coverage"
```

---

### Task 9: Sweep and evaluation

**Files:**
- Create: `src/train_eval.py`, `tests/test_train_eval.py`
- Modify: `tests/conftest.py` (add the `fake_artifacts` fixture), `src/main.py` (add `evaluate`)

**Interfaces:**
- Consumes:
  - `metrics.*`
  - `frames.frame_subset`
  - the feature files and their formats from Task 7
  - `artifacts/labels.csv`
- Produces:
  - `FEATURE_SETS`
  - `Config(features: str, n_frames: int)` with `.name` (`"small+motion@8"`), `.encoder` (`"small"`, `"large"` or `None`), `.uses_motion`, and `Config.parse(name)`
  - `pool(emb (V,F,D)) -> (V,2D)`
  - `build_X(cfg, feats: dict) -> (V, d) float32`. `feats` keys are `"small"`/`"large"` (V,F,D) and `"motion_n{n}"` (V,13).
  - `make_head() -> Pipeline`
  - `cross_val_proba(X, y, k, n_splits=5, n_repeats=5, seed=42) -> (R,N,k)`
  - `zero_shot_proba(emb, text, n) -> (N,k)`
  - `load_features(feat_dir) -> (feats, encoders)`
  - `run_evaluation(art=ART, n_repeats=5, n_boot=200) -> dict`. This writes `art/eval_results.json`, `art/pareto.png`, `art/mapping.csv` and `art/utility.csv`. The result includes `out["operating_point"] = {"config", "tau", "risk_coverage"}`.

- [ ] **Step 1: Add the `fake_artifacts` fixture to `tests/conftest.py`**

```python
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
```

- [ ] **Step 2: Write the failing tests**

`tests/test_train_eval.py`:
```python
import json
import subprocess
import sys

import numpy as np

from train_eval import Config, build_X, cross_val_proba, pool, run_evaluation


def test_config_parsing_and_parts():
    c = Config.parse("small+motion@8")
    assert (c.features, c.n_frames, c.encoder, c.uses_motion) == ("small+motion", 8, "small", True)
    assert Config.parse("motion@4").encoder is None


def test_pool_is_mean_and_std():
    emb = np.random.default_rng(0).normal(size=(3, 4, 5))
    p = pool(emb)
    assert p.shape == (3, 10)
    assert np.allclose(p[:, :5], emb.mean(1), atol=1e-6)


def test_build_X_uses_frame_subset():
    feats = {"small": np.zeros((2, 16, 4)), "motion_n4": np.ones((2, 13))}
    assert build_X(Config("small+motion", 4), feats).shape == (2, 8 + 13)


def test_cv_with_tiny_class_keeps_columns_aligned():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(6), [12, 12, 12, 12, 10, 2])
    X = np.eye(6)[y] * 3 + 0.3 * rng.normal(size=(len(y), 6))
    proba = cross_val_proba(X, y, 6, n_repeats=2)
    assert proba.shape == (2, len(y), 6)
    assert np.allclose(proba.sum(2), 1.0)
    assert (proba[0].argmax(1) == y).mean() > 0.9


def test_run_evaluation_writes_results(fake_artifacts):
    out = run_evaluation(fake_artifacts, n_repeats=1, n_boot=20)
    saved = json.loads((fake_artifacts / "eval_results.json").read_text())
    assert saved["operating_point"]["config"] in saved["configs"]
    assert "zeroshot-small@16" in saved["configs"]
    best = out["configs"][out["operating_point"]["config"]]
    assert best["argmax"]["macro_f1"]["mean"] > 0.8
    assert {"total", "decode", "encoder", "motion", "head"} <= set(best["latency_ms"])
    assert saved["label_noise"]["n_changed"] == 1
    assert (fake_artifacts / "pareto.png").stat().st_size > 0
    assert (fake_artifacts / "utility.csv").exists() and (fake_artifacts / "mapping.csv").exists()


def test_evaluate_does_not_import_torch():
    code = "import sys; sys.path.insert(0, 'src'); import train_eval; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
```

- [ ] **Step 3: Run them and confirm they fail**

Run: `.venv/Scripts/python -m pytest tests/test_train_eval.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'train_eval'`.

- [ ] **Step 4: Implement `src/train_eval.py`**

```python
"""Sweep {feature set} x {frames} on cached features: repeated CV, CIs, CUS, latency/size, Pareto plot."""
import json
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from frames import frame_subset
from metrics import (accuracy, bayes_decide, bootstrap_ci, choose_tau, cus, macro_f1,
                     risk_coverage, summarize)

FEATURE_SETS = ("motion", "small", "large", "small+motion", "large+motion")
CS = [0.01, 0.1, 1.0, 10.0]

warnings.filterwarnings("ignore", message="The least populated class")


@dataclass(frozen=True)
class Config:
    features: str
    n_frames: int

    @property
    def name(self):
        return f"{self.features}@{self.n_frames}"

    @property
    def encoder(self):
        return next((p for p in self.features.split("+") if p in ("small", "large")), None)

    @property
    def uses_motion(self):
        return "motion" in self.features.split("+")

    @classmethod
    def parse(cls, name):
        features, n = name.split("@")
        return cls(features, int(n))


def pool(emb):
    emb = emb.astype(np.float32)
    return np.concatenate([emb.mean(1), emb.std(1)], axis=1)


def build_X(cfg, feats):
    parts = []
    if cfg.encoder:
        e = feats[cfg.encoder]
        parts.append(pool(e[:, frame_subset(e.shape[1], cfg.n_frames)]))
    if cfg.uses_motion:
        parts.append(feats[f"motion_n{cfg.n_frames}"])
    return np.concatenate(parts, axis=1).astype(np.float32)


def make_head():
    return make_pipeline(StandardScaler(),
                         LogisticRegressionCV(Cs=CS, cv=3, class_weight="balanced", max_iter=5000))


def cross_val_proba(X, y, k, n_splits=5, n_repeats=5, seed=C.SEED):
    out = np.zeros((n_repeats, len(y), k))
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for tr, te in skf.split(X, y):
            head = make_head().fit(X[tr], y[tr])
            # A training fold may lack a tiny class: place columns by the head's own class ids.
            out[r][np.ix_(te, head.classes_)] = head.predict_proba(X[te])
    return out


def zero_shot_proba(emb, text, n):
    e = emb[:, frame_subset(emb.shape[1], n)].astype(np.float32).mean(1)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    logits = 100.0 * e @ text.astype(np.float32).T
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def load_features(feat_dir):
    encs = json.loads((feat_dir / "encoders.json").read_text())
    feats = {f"motion_n{n}": np.load(feat_dir / f"motion_n{n}.npy") for n in C.FRAME_SUBSETS}
    for kind in encs:
        feats[kind] = np.load(feat_dir / f"{kind}_f16.npy")
        feats[f"{kind}_text"] = np.load(feat_dir / f"{kind}_text.npy")
    return feats, encs


def configs(feats):
    return [Config(fs, n) for fs in FEATURE_SETS for n in C.FRAME_SUBSETS
            if all(p == "motion" or p in feats for p in fs.split("+"))]


def latency_ms(cfg, t, head_ms):
    n = cfg.n_frames
    lat = {"decode": t["decode_ms_per_timestamp"] * n,
           "motion": t["motion_ms_per_timestamp"] * n if cfg.uses_motion else 0.0,
           "encoder": t["encoder_ms_per_frame"][cfg.encoder] * n if cfg.encoder else 0.0,
           "head": head_ms}
    lat["total"] = sum(lat.values())
    return lat


def size_mb(cfg, t, n_features, k):
    enc = t["encoder_mb_fp32"][cfg.encoder] if cfg.encoder else 0.0
    head = ((n_features + 1) * k + 2 * n_features) * 4 / 1e6
    return {"encoder_fp32": enc, "encoder_fp16": enc / 2, "head": head, "total_fp32": enc + head}


def head_latency_ms(X, y, reps=50):
    head = make_head().fit(X, y)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        head.predict_proba(X[:1])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def evaluate_proba(proba, y, U, n_boot):
    k = proba.shape[2]
    preds = {"argmax": proba.argmax(2), "bayes": np.stack([bayes_decide(p, U) for p in proba])}
    out = {}
    for rule, p in preds.items():
        out[rule] = {"accuracy": bootstrap_ci(accuracy, y, p, n_boot),
                     "macro_f1": bootstrap_ci(lambda a, b: macro_f1(a, b, k)[1], y, p, n_boot),
                     "cus": bootstrap_ci(lambda a, b: cus(a, b, U), y, p, n_boot)}
    out["detail"] = summarize(y, preds["argmax"][0], U)
    return out


def choose_operating_point(results, tol=0.03):
    sup = {k: v for k, v in results.items() if v["kind"] == "supervised"}
    best = max(v["argmax"]["macro_f1"]["mean"] for v in sup.values())
    near = [k for k, v in sup.items() if v["argmax"]["macro_f1"]["mean"] >= best - tol]
    return min(near, key=lambda k: sup[k]["latency_ms"]["total"])


def cascade_sim(p1, p2, y, lat1, lat2):
    """Stage 1 (motion-only) routes when confident enough to match stage-2 accuracy; the rest escalate."""
    acc2 = float((p2.argmax(1) == y).mean())
    conf1, pred1 = p1.max(1), p1.argmax(1)
    tau1, stop = 1.01, np.zeros(len(y), bool)
    for t in np.round(np.arange(0.3, 1.0, 0.05), 2):
        m = conf1 >= t
        if m.any() and (pred1[m] == y[m]).mean() >= acc2:
            tau1, stop = float(t), m
            break
    pred = np.where(stop, pred1, p2.argmax(1))
    return {"tau1": tau1, "stage1_stop_fraction": float(stop.mean()), "cascade_accuracy": float((pred == y).mean()),
            "stage2_accuracy": acc2, "expected_ms": lat1 + (1 - stop.mean()) * max(lat2 - lat1, 0.0)}


def plot_pareto(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, metric in zip(axes, ("macro_f1", "cus")):
        for name, r in results.items():
            m, x = r["argmax"][metric], r["latency_ms"]["total"]
            ax.errorbar(x, m["mean"], yerr=[[m["mean"] - m["lo"]], [m["hi"] - m["mean"]]], fmt="o",
                        ms=4 + 2 * np.sqrt(r["size_mb"]["total_fp32"]), alpha=0.75,
                        mfc="none" if r["kind"] == "zero_shot" else None)
            ax.annotate(name, (x, m["mean"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("CPU ms / video (sum of measured components)")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    fig.suptitle("Accuracy vs latency — marker size ~ model MB, hollow = zero-shot, bars = 95% CI")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_tables(art):
    pd.DataFrame([{"sub_category": s, "class": c, "chunker": C.CHUNKERS[c]}
                  for s, c in C.SUBCATEGORY_TO_CLASS.items()]).to_csv(art / "mapping.csv", index=False)
    pd.DataFrame(C.UTILITY, index=C.CLASSES, columns=C.CLASSES).to_csv(art / "utility.csv")


def run_evaluation(art=C.ART, n_repeats=5, n_boot=200):
    labels = pd.read_csv(art / "labels.csv", dtype={"video_id": str}, keep_default_na=False).set_index("video_id")
    ids = (art / "features" / "video_ids.txt").read_text().split()
    labels = labels.loc[ids]
    y = labels.label_final.map(C.CLASSES.index).to_numpy()
    y_auto = labels.label_auto.map(C.CLASSES.index).to_numpy()
    feats, encs = load_features(art / "features")
    timings = json.loads((art / "features" / "timings.json").read_text())
    U, k = np.array(C.UTILITY), len(C.CLASSES)

    results, oof = {}, {}
    for cfg in configs(feats):
        X = build_X(cfg, feats)
        proba = cross_val_proba(X, y, k, n_repeats=n_repeats)
        r = evaluate_proba(proba, y, U, n_boot)
        r.update(kind="supervised", n_features=int(X.shape[1]),
                 latency_ms=latency_ms(cfg, timings, head_latency_ms(X, y)), size_mb=size_mb(cfg, timings, X.shape[1], k))
        results[cfg.name], oof[cfg.name] = r, proba[0]
        print(f"{cfg.name:>18}  macroF1={r['argmax']['macro_f1']['mean']:.3f}  "
              f"CUS={r['argmax']['cus']['mean']:.3f}  ms={r['latency_ms']['total']:.1f}", flush=True)
    for kind in encs:
        cfg = Config(kind, C.N_FRAMES)
        proba = zero_shot_proba(feats[kind], feats[f"{kind}_text"], C.N_FRAMES)[None]
        r = evaluate_proba(proba, y, U, n_boot)
        r.update(kind="zero_shot", n_features=0, latency_ms=latency_ms(cfg, timings, 0.0), size_mb=size_mb(cfg, timings, 0, k))
        results[f"zeroshot-{kind}@{C.N_FRAMES}"] = r

    op = choose_operating_point(results)
    op_cfg = Config.parse(op)
    rc = risk_coverage(y, oof[op])
    stage1 = f"motion@{op_cfg.n_frames}"
    cascade = (cascade_sim(oof[stage1], oof[op], y, results[stage1]["latency_ms"]["total"],
                           results[op]["latency_ms"]["total"]) if op_cfg.encoder else None)
    p_auto = cross_val_proba(build_X(op_cfg, feats), y_auto, k, n_repeats=n_repeats)
    support = {c: int((y == i).sum()) for i, c in enumerate(C.CLASSES)}
    out = {
        "meta": {"n_videos": len(y), "class_support": support,
                 "small_classes": [c for c, s in support.items() if s < 5], "encoders": encs,
                 "cv": f"stratified 5-fold x {n_repeats} repeats", "bootstrap": n_boot, "seed": C.SEED,
                 "latency_note": "sum of components measured on the featurization machine"},
        "configs": results,
        "operating_point": {"config": op, "tau": choose_tau(rc), "risk_coverage": rc,
                            "rule": "fastest config within 0.03 macro-F1 of the best"},
        "cascade": cascade,
        "label_noise": {"agreement_auto_vs_final": float((labels.label_auto == labels.label_final).mean()),
                        "n_changed": int((labels.label_auto != labels.label_final).sum()),
                        "op_macro_f1_on_label_auto": bootstrap_ci(lambda a, b: macro_f1(a, b, k)[1],
                                                                  y_auto, p_auto.argmax(2), n_boot)},
    }
    (art / "eval_results.json").write_text(json.dumps(out, indent=2))
    plot_pareto(results, art / "pareto.png")
    write_tables(art)
    print(f"operating point: {op}  tau={out['operating_point']['tau']}")
    return out
```

- [ ] **Step 5: Add the `evaluate` subcommand** (above `def main`)

```python
def cmd_evaluate(args):
    from train_eval import run_evaluation
    run_evaluation(n_repeats=args.repeats)


SUBCOMMANDS["evaluate"] = ("run the CV sweep on committed features",
                           lambda p: p.add_argument("--repeats", type=int, default=5), cmd_evaluate)
```

- [ ] **Step 6: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_train_eval.py -q`
Expected: 6 passed.
- If `test_cv_with_tiny_class_keeps_columns_aligned` raises inside `LogisticRegressionCV` because an inner fold lacks a class, change `make_head` to pass `cv=StratifiedKFold(3, shuffle=True, random_state=C.SEED)`.
- If it still fails, wrap the head in a fallback: on `ValueError`, fit `LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)`. Keep the test unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/train_eval.py src/main.py tests/conftest.py tests/test_train_eval.py
git commit -m "Trade-off sweep: repeated CV, CIs, CUS/Bayes, cascade sim, Pareto plot"
```

---

### Task 10: ONNX export of the production head

**Files:**
- Create: `src/export.py`, `tests/test_export.py`
- Modify: `src/main.py` (add `export`)

**Interfaces:**
- Consumes:
  - `train_eval.Config, build_X, load_features, CS`
  - the `eval_results.json` key `operating_point`
  - `encoders.json`
- Produces:
  - `fit_final(X, y) -> Pipeline`
  - `to_onnx(pipe, n_features, path)`
  - `onnx_proba(path, X) -> ndarray`
  - `export_production(art=ART) -> dict`. It writes `art/models/head_{features}_{n}.onnx` and `art/models/production.json` with the keys `{config, features, n_frames, encoder: [name, pretrained] | null, head, tau, classes, n_features}`.

- [ ] **Step 1: Write the failing test**

`tests/test_export.py`:
```python
import json

import numpy as np

from export import export_production, fit_final, onnx_proba
from train_eval import Config, build_X, load_features, run_evaluation


def test_export_writes_onnx_matching_sklearn(fake_artifacts):
    run_evaluation(fake_artifacts, n_repeats=1, n_boot=10)
    prod = export_production(fake_artifacts)
    saved = json.loads((fake_artifacts / "models" / "production.json").read_text())
    assert saved == prod
    feats, _ = load_features(fake_artifacts / "features")
    X = build_X(Config.parse(prod["config"]), feats)
    assert X.shape[1] == prod["n_features"]
    y = np.repeat(np.arange(6), [12, 12, 12, 12, 10, 2])
    ref = fit_final(X, y).predict_proba(X)
    got = onnx_proba(fake_artifacts / "models" / prod["head"], X)
    assert np.allclose(got, ref, atol=1e-5)
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_export.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'export'`.

- [ ] **Step 3: Implement `src/export.py`**

```python
"""Refit the chosen operating point on all labeled videos and export the head to ONNX."""
import json

import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from train_eval import CS, Config, build_X, load_features


def fit_final(X, y):
    cv = LogisticRegressionCV(Cs=CS, cv=3, class_weight="balanced", max_iter=5000)
    cv.fit(StandardScaler().fit_transform(X), y)
    return make_pipeline(StandardScaler(), LogisticRegression(C=float(cv.C_[0]), class_weight="balanced",
                                                              max_iter=5000)).fit(X, y)


def to_onnx(pipe, n_features, path):
    model = convert_sklearn(pipe, initial_types=[("input", FloatTensorType([None, n_features]))],
                            options={LogisticRegression: {"zipmap": False}}, target_opset=17)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.SerializeToString())
    return path


def onnx_proba(path, X):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(None, {"input": X.astype(np.float32)})[1]


def export_production(art=C.ART):
    results = json.loads((art / "eval_results.json").read_text())
    cfg = Config.parse(results["operating_point"]["config"])
    feats, encs = load_features(art / "features")
    ids = (art / "features" / "video_ids.txt").read_text().split()
    labels = pd.read_csv(art / "labels.csv", dtype={"video_id": str}, keep_default_na=False).set_index("video_id").loc[ids]
    y = labels.label_final.map(C.CLASSES.index).to_numpy()
    X = build_X(cfg, feats)
    pipe = fit_final(X, y)
    path = to_onnx(pipe, X.shape[1], art / "models" / f"head_{cfg.features.replace('+', '_')}_{cfg.n_frames}.onnx")
    if not np.allclose(onnx_proba(path, X), pipe.predict_proba(X), atol=1e-5):
        raise RuntimeError("ONNX head disagrees with sklearn")
    prod = {"config": cfg.name, "features": cfg.features, "n_frames": cfg.n_frames,
            "encoder": encs[cfg.encoder] if cfg.encoder else None, "head": path.name,
            "tau": results["operating_point"]["tau"],
            "classes": [C.CLASSES[i] for i in pipe.classes_], "n_features": int(X.shape[1])}
    (art / "models" / "production.json").write_text(json.dumps(prod, indent=2))
    print(f"exported {path.name} ({path.stat().st_size / 1024:.1f} KB) for {cfg.name}")
    return prod
```

- [ ] **Step 4: Add the `export` subcommand** (above `def main`)

```python
def cmd_export(args):
    from export import export_production
    export_production()


SUBCOMMANDS["export"] = ("export the production head to ONNX", lambda p: None, cmd_export)
```

- [ ] **Step 5: Run the test and confirm it passes**

Run: `.venv/Scripts/python -m pytest tests/test_export.py -q`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add src/export.py src/main.py tests/test_export.py
git commit -m "ONNX export of production head with sklearn parity check"
```

---

### Task 11: predict, benchmark, fetch-encoder

**Files:**
- Create: `src/predict.py`, `src/benchmark.py`, `tests/test_predict.py`
- Modify: `src/main.py` (add `predict`, `benchmark`, `fetch-encoder`)

**Interfaces:**
- Consumes:
  - `production.json` from Task 10
  - `frames.extract, frame_subset`
  - `motion.video_features`
  - `train_eval.Config, build_X`
  - an encoder object with `.embed(frames, batch_size)` (the real `encoders.Encoder` or a test fake)
- Produces:
  - `collect_videos(paths) -> list[Path]`
  - `Predictor(models_dir=MODELS, encoder=None)` with `.predict(path) -> dict` and `.predict_many(paths) -> list[dict]`. Each result is `{video, class, confidence, recommended_chunker, fallback, probabilities, latency_ms{decode, motion, encoder, head, total}}` or `{video, error}`.
  - `benchmark.run(predictor, paths, repeats=3) -> dict`, which is also written to `artifacts/benchmark.json`.

- [ ] **Step 1: Write the failing tests**

`tests/test_predict.py`:
```python
import numpy as np

import config as C
from benchmark import run as run_benchmark
from export import export_production
from predict import Predictor, collect_videos
from train_eval import run_evaluation


class FakeEncoder:
    def embed(self, frames, batch_size=32):
        v = np.zeros((len(frames), 8), np.float32)
        v[:, 0] = 1.0
        return v


def predictor(fake_artifacts):
    run_evaluation(fake_artifacts, n_repeats=1, n_boot=10)
    export_production(fake_artifacts)
    return Predictor(fake_artifacts / "models", encoder=FakeEncoder())


def test_predict_returns_routing_decision(fake_artifacts, make_video):
    r = predictor(fake_artifacts).predict(make_video(n=100))
    assert r["class"] in C.CLASSES
    assert 0.0 <= r["confidence"] <= 1.0
    assert abs(sum(r["probabilities"].values()) - 1.0) < 1e-4
    expected = C.FALLBACK_CHUNKER if r["fallback"] else C.CHUNKERS[r["class"]]
    assert r["recommended_chunker"] == expected
    assert r["latency_ms"]["total"] > 0


def test_corrupt_file_reports_error_without_stopping_batch(fake_artifacts, make_video, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_text("nope")
    results = predictor(fake_artifacts).predict_many([bad, make_video(n=100)])
    assert "error" in results[0] and "class" in results[1]


def test_collect_videos_expands_directories(tmp_path, make_video):
    make_video("a.avi")
    make_video("b.avi")
    (tmp_path / "notes.txt").write_text("x")
    assert [p.name for p in collect_videos([tmp_path])] == ["a.avi", "b.avi"]


def test_benchmark_reports_batch1_and_batch32(fake_artifacts, make_video):
    out = run_benchmark(predictor(fake_artifacts), [make_video(n=100)], repeats=1)
    assert out["batch1_ms"]["p50"] > 0 and out["batch32_throughput_vps"] > 0
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/Scripts/python -m pytest tests/test_predict.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'predict'`.

- [ ] **Step 3: Implement `src/predict.py`**

```python
"""Production inference: video -> {class, confidence, chunker, fallback, latency breakdown}."""
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

import config as C
from download import VIDEO_EXTS
from frames import extract, frame_subset
from motion import video_features
from train_eval import Config, build_X


def collect_videos(paths):
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out.extend(sorted(q for q in p.iterdir() if q.suffix.lower() in VIDEO_EXTS))
        else:
            out.append(p)
    return out


def _ms(t0):
    return (time.perf_counter() - t0) * 1000


class Predictor:
    def __init__(self, models_dir=C.MODELS, encoder=None):
        self.prod = json.loads((Path(models_dir) / "production.json").read_text())
        self.cfg = Config.parse(self.prod["config"])
        self.keep = frame_subset(C.N_FRAMES, self.cfg.n_frames)
        self.session = ort.InferenceSession(str(Path(models_dir) / self.prod["head"]),
                                            providers=["CPUExecutionProvider"])
        self.encoder = encoder
        if self.cfg.encoder and self.encoder is None:
            from encoders import load
            self.encoder = load(*self.prod["encoder"])

    def predict(self, path):
        return self.predict_many([path])[0]

    def predict_many(self, paths):
        results, ok = [], []
        for p in map(Path, paths):
            try:
                s = extract(p, C.N_FRAMES, keep=self.keep)
            except Exception as e:  # corrupt/unsupported input must not abort the batch
                results.append({"video": p.name, "error": str(e)})
                continue
            lat, feats = {"decode": s.decode_ms, "motion": 0.0, "encoder": 0.0}, {}
            if self.cfg.uses_motion:
                t0 = time.perf_counter()
                feats[f"motion_n{self.cfg.n_frames}"] = video_features(s.rgb, s.gray)[None]
                lat["motion"] = _ms(t0)
            ok.append((len(results), p, s, feats, lat))
            results.append(None)
        if ok and self.cfg.encoder:
            t0 = time.perf_counter()
            emb = self.encoder.embed(np.concatenate([s.rgb for _, _, s, _, _ in ok]), batch_size=32)
            per, n = _ms(t0) / len(ok), self.cfg.n_frames
            for j, (_, _, _, feats, lat) in enumerate(ok):
                feats[self.cfg.encoder] = emb[j * n:(j + 1) * n][None]
                lat["encoder"] = per
        if ok:
            X = np.concatenate([build_X(self.cfg, feats) for _, _, _, feats, _ in ok])
            t0 = time.perf_counter()
            proba = self.session.run(None, {"input": X})[1]
            per = _ms(t0) / len(ok)
            for j, (i, p, _, _, lat) in enumerate(ok):
                lat["head"] = per
                lat["total"] = sum(lat.values())
                results[i] = self._result(p, proba[j], lat)
        return results

    def _result(self, path, proba, lat):
        k = int(proba.argmax())
        cls, conf = self.prod["classes"][k], float(proba[k])
        fallback = conf < self.prod["tau"]
        return {"video": path.name, "class": cls, "confidence": round(conf, 4),
                "recommended_chunker": C.FALLBACK_CHUNKER if fallback else C.CHUNKERS[cls],
                "fallback": fallback,
                "probabilities": {c: round(float(q), 4) for c, q in zip(self.prod["classes"], proba)},
                "latency_ms": {k_: round(v, 2) for k_, v in lat.items()}}
```

- [ ] **Step 4: Implement `src/benchmark.py`**

```python
"""End-to-end CPU timing of the production pipeline at batch 1 and batch 32."""
import os
import platform
import time

import numpy as np


def run(predictor, paths, repeats=3):
    predictor.predict(paths[0])  # warm-up
    times, comps = [], []
    for _ in range(repeats):
        for p in paths:
            t0 = time.perf_counter()
            r = predictor.predict(p)
            times.append((time.perf_counter() - t0) * 1000)
            comps.append(r.get("latency_ms", {}))
    batch = [paths[i % len(paths)] for i in range(32)]
    t0 = time.perf_counter()
    predictor.predict_many(batch)
    wall = (time.perf_counter() - t0) * 1000
    keys = ("decode", "motion", "encoder", "head")
    return {
        "config": predictor.cfg.name,
        "videos": [str(p) for p in paths],
        "batch1_ms": {"p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95)),
                      "mean": float(np.mean(times))},
        "batch1_throughput_vps": 1000.0 / float(np.mean(times)),
        "batch32_wall_ms": wall,
        "batch32_throughput_vps": 32000.0 / wall,
        "components_ms_median": {k: float(np.median([c.get(k, 0.0) for c in comps])) for k in keys},
        "machine": {"cpu_count": os.cpu_count(), "platform": platform.platform(), "processor": platform.processor()},
    }
```

- [ ] **Step 5: Add the subcommands** (above `def main`)

```python
def cmd_predict(args):
    import json
    from predict import Predictor, collect_videos
    results = Predictor().predict_many(collect_videos(args.paths))
    for r in results:
        print(json.dumps(r))
    return 1 if any("error" in r for r in results) else 0


def cmd_benchmark(args):
    import json
    import config as C
    from benchmark import run
    from predict import Predictor, collect_videos
    out = run(Predictor(), collect_videos(args.paths))
    (C.ART / "benchmark.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def cmd_fetch_encoder(args):
    import json
    import config as C
    from encoders import load
    if args.all:
        wanted = list(json.loads((C.FEATURES / "encoders.json").read_text()).values())
    else:
        enc = json.loads((C.MODELS / "production.json").read_text())["encoder"]
        wanted = [enc] if enc else []
    for name, pretrained in wanted:
        load(name, pretrained)
        print(f"encoder ready: {name}/{pretrained}")
    if not wanted:
        print("production config uses no encoder; nothing to fetch")


SUBCOMMANDS["predict"] = ("classify videos (files or directories)",
                          lambda p: p.add_argument("paths", nargs="+"), cmd_predict)
SUBCOMMANDS["benchmark"] = ("time the production pipeline at batch 1 and 32",
                            lambda p: p.add_argument("paths", nargs="+"), cmd_benchmark)
SUBCOMMANDS["fetch-encoder"] = ("download encoder weights (production, or --all)",
                                lambda p: p.add_argument("--all", action="store_true"), cmd_fetch_encoder)
```

- [ ] **Step 6: Run the whole suite and confirm it passes**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all tests pass (the slow test is deselected).

- [ ] **Step 7: Commit**

```bash
git add src/predict.py src/benchmark.py src/main.py tests/test_predict.py
git commit -m "Production predict/benchmark CLI with ONNX head and fallback routing"
```

---

### Task 12: Data run: extract, review labels with the user, featurize, evaluate, export

This is an operational task: no new code, only real artifacts. Every step's output must be checked before moving on.

**Files:**
- Create (committed): `artifacts/sample.csv`, `artifacts/label_overrides.csv`, `artifacts/labels.csv`, `artifacts/features/*`, `artifacts/eval_results.json`, `artifacts/pareto.png`, `artifacts/mapping.csv`, `artifacts/utility.csv`, `artifacts/models/*`

- [ ] **Step 1: Confirm the download finished**

Run: `tail -5 data/download.log && ls data/videos | wc -l`
Expected: the final line reads `90 videos ready ...`. If there are unfilled slots, report them to the user and continue with what is available, noting the gap in the write-up.

- [ ] **Step 2: Extract frames**

Run: `.venv/Scripts/python src/main.py extract`
Expected: one line per video and no `FAILED` lines. Any failures are listed; carry on without those videos and record them.

- [ ] **Step 3: Render the contact sheets**

Run: `.venv/Scripts/python src/main.py label-sheets`
Expected: `wrote 90 sheets` and `data/sheets/page_00.jpg` … `page_14.jpg`.

- [ ] **Step 4: Review the labels together with the user (checkpoint: the user must confirm)**

1. Read each `data/sheets/page_XX.jpg` with the Read tool. The title of each sheet shows `video_id | sub_category -> auto label`.
2. Wherever the frames clearly contradict the auto label, propose an override using the class definitions in spec §2. Examples: slides or screen content → `presentation_screen`; a person talking to the camera → `talking_head`; first-person or handheld footage → `vlog_egocentric`.
3. Show the user the proposed list as `video_id: auto -> proposed (reason)` and wait for them to confirm or edit it.
4. Write the confirmed list to `artifacts/label_overrides.csv` with the header `video_id,label_final,note`.

- [ ] **Step 5: Build the labels**

Run: `.venv/Scripts/python src/main.py labels`
Expected: per-class counts and `changed by review: N/90`. Note any class with fewer than 5 videos for the write-up.

- [ ] **Step 6: Featurize (about 2–5 minutes on 16 cores)**

Run: `.venv/Scripts/python src/main.py featurize`
Expected:
- the per-encoder lines print D, ms/frame and MB;
- `artifacts/features/` contains `motion_n{4,8,16}.npy`, `{small,large}_f16.npy`, `{small,large}_text.npy`, `video_ids.txt`, `encoders.json` and `timings.json`;
- `du -sh artifacts` reports under 10 MB.

- [ ] **Step 7: Evaluate**

Run: `.venv/Scripts/python src/main.py evaluate`
Expected: 15 configuration lines, the `operating point:` line, `eval_results.json` and `pareto.png`. Open `pareto.png` with Read and sanity-check it. Motion-only should be the fastest point, and the encoder configurations should score higher macro-F1.

- [ ] **Step 8: Export**

Run: `.venv/Scripts/python src/main.py export`
Expected: `exported head_*.onnx (… KB)` and `artifacts/models/production.json` written.

- [ ] **Step 9: Commit**

```bash
git add artifacts/
git commit -m "Benchmark artifacts: labels, cached features, sweep results, production head"
```

---

### Task 13: Demo samples, run.yaml, fresh-checkout verification

**Files:**
- Create: `samples/*.mp4` (2–3 clips, each under 2 MB)
- Modify: `run.yaml`

- [ ] **Step 1: Decide the samples with the user (checkpoint)**

The proposal: 15-second, 144p excerpts from 3 sample videos of different classes (for example sports, talking head, gameplay). Ask the user to confirm, since committing short YouTube excerpts is their call. The alternative is clips they supply.

- [ ] **Step 2: Cut the clips**

For each chosen `<id>` and `<start_s>`:
`ffmpeg -y -ss <start_s> -i data/videos/<id>.mp4 -t 15 -vf scale=-2:144 -an -c:v libx264 -crf 30 samples/<class>_<id>.mp4`
Check: `ls -la samples/` shows every file under 2 MB.

- [ ] **Step 3: Write `run.yaml`**

```yaml
# Commands run EXACTLY as written, in order, from a fresh checkout. All non-interactive and idempotent.

setup_commands:
  - "pip install -r requirements.txt"
  - "python src/main.py fetch-encoder"

run_commands:
  - "python src/main.py evaluate"
  - "python src/main.py predict samples/"
  - "python src/main.py benchmark samples/"

test_commands:
  - "python -m pytest -q"

notes: "CPU only, no GPU. Takes about 5 minutes. fetch-encoder downloads the production encoder's weights once (about 300 MB for MobileCLIP2-S0, 605 MB for ViT-B-32) from the Hugging Face Hub. evaluate re-runs the full trade-off sweep from the committed VideoMME 10% features in artifacts/ (no video download needed). The full rebuild from YouTube (sample, download, extract, labels, featurize, evaluate, export) is documented in docs/README.md."
```
Fill in the weight size of the encoder that `production.json` actually names, and delete the other one from the note.

- [ ] **Step 4: Verify from a fresh checkout**

```bash
FRESH="$TMPDIR/fresh-check" && rm -rf "$FRESH" && git clone -q . "$FRESH" && cd "$FRESH" \
 && python -m venv .v && source .v/Scripts/activate \
 && pip install -q -r requirements.txt && python src/main.py fetch-encoder \
 && python src/main.py evaluate && python src/main.py predict samples/ \
 && python src/main.py benchmark samples/ && python -m pytest -q; echo "exit=$?"
```
Here `$TMPDIR` means the session scratchpad directory. Expected: `exit=0`, sensible predictions for the samples and benchmark JSON printed. If anything fails, fix it in the repo, commit, and re-run this step.

- [ ] **Step 5: Commit**

```bash
git add samples/ run.yaml artifacts/benchmark.json
git commit -m "Demo clips and run.yaml verified from a fresh checkout"
```

---

### Task 14: Write-up (≤2 pages) and push

**Files:**
- Modify: `docs/README.md`

- [ ] **Step 1: Write `docs/README.md`**

Use these sections, in order. Take every number from `artifacts/eval_results.json` and `artifacts/benchmark.json`; nothing is estimated by hand.

1. **Problem and taxonomy.** A table of the 6 classes → chunker → typical visual signature, plus why these classes (they occur in VideoMME and each drives a different chunker).
2. **Data.**
   - The stratified 90 videos (seed 42) and the number replaced by spares.
   - Labeling: `meta.class_support`, `label_noise.agreement_auto_vs_final` and `n_changed`.
3. **Feature engineering.**
   - Uniform 16-timestamp sampling and the reason for it.
   - The 13 motion/appearance features (name the signals) and why an image encoder misses them.
   - Frozen encoders: the names from `meta.encoders`, with mean ⊕ std pooling.
4. **Results.**
   - Embed `../artifacts/pareto.png`.
   - A table of the configurations: macro-F1 [CI], CUS [CI], ms/video, MB.
   - Operating point and τ.
   - Per-class F1 and the confusion matrix of the operating point.
   - Bayes rule vs argmax on CUS.
   - Zero-shot baselines.
5. **Custom metric (CUS).** The definition, the U matrix, the expected-utility decision rule, and the fact that U is an assumption to validate against retrieval recall.
6. **Deployment (on paper).** A Mermaid diagram of queue → stateless CPU workers (seek-decode → stage-1 motion head → stage-2 encoder via ONNX Runtime int8, batched → fallback) → chunker router.
   - Stage-1 stop fraction and expected ms taken from `cascade`.
   - Cost at 100M videos = `expected_ms` × 1e8 converted to CPU-hours, and the worker count needed for a 24-hour turnaround.
   - Monitoring: class-mix drift and the low-confidence rate feeding active learning.
7. **Limitations and next steps.** n = 90 with small classes listed from `meta.small_classes`, proxy labels, the assumed U, no audio (CLAP for talking head vs presentation), latency measured on one machine (give its spec from `benchmark.json`), and how to reproduce (the full rebuild command sequence).

Keep it to two printed pages; the tables carry the detail.

- [ ] **Step 2: Final full check**

Run: `.venv/Scripts/python -m pytest -q && git status --short`
Expected: everything passes, and only intended files are modified.

- [ ] **Step 3: Commit and push (checkpoint: confirm with the user before pushing)**

```bash
git add docs/README.md
git commit -m "Write-up: taxonomy, features, trade-off results, CUS, deployment"
```
Ask the user how the branch should reach the fork's default branch: merge `feat/classifier` into `main` and push, or something else. The grader reads only the default branch. Push only after they confirm.
