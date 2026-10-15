# Video Classifier for Chunking Routing — Design Spec

- **Date:** 2026-10-15
- **Scope:** AIntropy Part 1, Task 1 (Video Classifier) only. Task 2 (User Intent Classifier) is out of scope for this repo.
- **Time budget:** ~3.5 hours, CPU-only machine.
- **Status:** Approved in brainstorming; pending written-spec review.

## 1. Problem and success criteria

Build a classifier that routes each incoming video to the chunking strategy best suited to its type, as the first stage of an ingestion pipeline that builds a video index. Requirements from the problem statement:

1. Benchmark on the **10% sample of VideoMME**.
2. Pipeline (including models) must be **lightweight and fast at scale**.
3. Demonstrate the **trade-off between accuracy, latency and model size** (the key skill assessed).
4. Clearly define **evaluation metrics and criteria**; a well-founded custom metric earns extra credit.
5. Describe **deployment (on paper)** and the **feature-engineering step**.
6. `run.yaml` is executed literally on a fresh checkout and must work.

Success means: `run.yaml` runs cleanly on CPU in a few minutes; `eval_results.json` and a Pareto plot report measured accuracy/F1/CUS against measured latency and size for several operating points; the write-up (≤2 pages in `docs/`) recommends one operating point and describes deployment.

Out of scope: the chunkers themselves, captioning, retrieval, Task 2.

## 2. Class taxonomy

Six classes, chosen because each one implies a different chunker and each occurs in VideoMME.

| Class | Recommended chunker | Default VideoMME sub-categories |
|---|---|---|
| `gameplay_animation` | shot-boundary | Esports, Animation |
| `sports_broadcast` | motion / play-segment | Basketball, Football, Athletics, Other Sports |
| `vlog_egocentric` | activity-based | Daily Life, Travel, Food, Handicraft, Exercise, Fashion |
| `talking_head` | speaker-turn / ASR transcript | News Report; interviews and lecture-to-camera found in review |
| `presentation_screen` | slide-change + transcript | Knowledge videos dominated by slides/screens (found in review) |
| `cinematic_documentary` | semantic scene | Movie & TV Show, Documentary, Stage Play, Variety Show, Magic Show, Acrobatics, nature/animal and other Knowledge B-roll, Multilingual (default; corrected in review) |

The full `sub_category → class` table is committed as `artifacts/mapping.csv`. Every VideoMME sub-category must map to exactly one class.

## 3. Data pipeline

### 3.1 Sampling (`sample`)
- Load the `lmms-lab/Video-MME` metadata parquet and dedupe to the 900 unique videos.
- Draw a stratified 10% (90 videos) over duration bucket (short/medium/long) × domain, seed **42**. Class support is reported as it falls; it is not rebalanced.
- Draw 1–2 spare videos per stratum. A spare replaces a video whose YouTube id is unavailable; replacements are logged in `artifacts/sample.csv`.

### 3.2 Download (`download`)
- `yt-dlp`, video-only, ≤240p, to `data/videos/{video_id}.mp4` (≈2.5 GB total, gitignored).
- Any pre-downloaded file with the same naming is accepted, and the step skips files that already exist.

### 3.3 Frame extraction (`extract`)
- 16 timestamps spaced uniformly over the whole video, skipping the first and last 5%.
- At each timestamp: one 224px RGB frame (short side resized, center crop) for the encoder, and a burst of 8 grayscale 64×64 frames ~0.25 s apart for motion features.
- Decoding uses OpenCV seeking. Decode time per video is recorded as its own latency component.

### 3.4 Labels (`label-sheets`, `artifacts/labels.csv`)
- Columns: `video_id, youtube_id, duration_bucket, domain, sub_category, label_auto, label_final, label_changed, note`.
- `label_auto` comes from `mapping.csv`.
- `label_final` comes from a manual review of 4×4 contact sheets: Claude proposes corrections and the user confirms them.
- The agreement rate between `label_auto` and `label_final` is **measured and reported**, never assumed.

## 4. Features

### 4.1 Motion and appearance features (`motion.py`, no model, ~14 dims)
Computed per burst, then aggregated to a fixed-size vector per video:
- Motion energy (mean absolute frame difference): mean and std across bursts.
- Hard-cut rate: a grayscale-histogram jump above a threshold, normalized per second.
- Camera shake: variance of the `cv2.phaseCorrelate` translation (a handheld/egocentric proxy).
- Static fraction: the share of bursts with near-zero motion (slides, talking heads).
- Scene diversity: mean histogram distance between the 16 timestamps.
- Appearance: mean brightness, saturation, edge density and colorfulness.

### 4.2 Frozen encoders (`encoders.py`)
- Image tower only, loaded through `open_clip`, in inference mode.
- **Large:** `ViT-B-32` (~88M params, 512-d).
- **Small:** the smallest MobileCLIP listed by `open_clip.list_pretrained()` (S0, else S1). If none is available, use the smallest TinyCLIP; if that is also unavailable, drop the small-encoder rows from the sweep. A same-size model such as `ViT-B-32-quickgelu` is **not** a valid small fallback. The encoder actually used is logged in `eval_results.json`.
- Per-frame embeddings are L2-normalized and stored as fp16 in `artifacts/features/{encoder}_f16.npy` with shape (90, 16, D), about 1.5 MB per encoder.
- Pooling: mean ⊕ std over the frames (2×D). The std captures visual variety.

### 4.3 Head
- `StandardScaler` → multinomial `LogisticRegression` with `class_weight="balanced"` and C chosen by inner cross-validation.
- Exported to ONNX with `skl2onnx` as `artifacts/models/head_{config}.onnx` (KB-sized).

## 5. Experiments and evaluation

### 5.1 Sweep (`evaluate`)
- Feature sets: {motion only, small encoder, large encoder, small + motion, large + motion} × frames {4, 8, 16}, giving 15 configurations. Fewer frames are taken as an evenly spaced subset of the 16 cached frames.
- Baselines: zero-shot CLIP text prompts with the small and large encoders (no training).
- Everything runs from cached features in seconds.

### 5.2 Protocol
- Stratified 5-fold cross-validation repeated 5 times (seed 42) on `label_final`; classes with fewer than 5 videos are flagged.
- Metrics are reported as mean ± 95% bootstrap confidence interval over out-of-fold predictions.
- The best configuration is also evaluated against `label_auto` to show the effect of label noise.

### 5.3 Metrics
- **Standard:** accuracy, **macro-F1 (primary)**, per-class F1, and the out-of-fold confusion matrix.
- **Efficiency** (measured on this CPU): decode ms/video, motion-feature ms/video, encoder ms/frame, head ms, end-to-end p50/p95 ms/video at batch 1, throughput (videos/s) at batch 32, and size in MB (encoder fp32 and fp16, head in KB).
- **Chunking Utility Score (CUS)**, the custom metric:
  - U[true, pred] gives the fraction of downstream chunking quality kept when the router predicts `pred` for a video of class `true`. CUS = mean U over predictions.
  - U is committed as `artifacts/utility.csv`. It is an **assumption**, to be validated later against downstream retrieval recall.

    ```
            G    S    V    T    P    C
    G     1.0  0.7  0.3  0.2  0.2  0.4
    S     0.7  1.0  0.3  0.2  0.2  0.3
    V     0.3  0.3  1.0  0.3  0.2  0.5
    T     0.2  0.2  0.3  1.0  0.6  0.3
    P     0.2  0.2  0.2  0.6  1.0  0.2
    C     0.4  0.3  0.5  0.3  0.2  1.0
    ```
    G = gameplay_animation, S = sports_broadcast, V = vlog_egocentric, T = talking_head, P = presentation_screen, C = cinematic_documentary.
- **Bayes decision rule:** predict argmax_k Σ_j p(j)·U[j,k], which maximizes expected utility, instead of argmax p. Report the CUS and macro-F1 of both rules.
- **Selective prediction:** a risk–coverage curve. Below confidence τ the video is routed to a generic fixed-window chunker; τ is chosen from this curve.

### 5.4 Outputs
- `artifacts/eval_results.json`: per-configuration metrics with confidence intervals, efficiency numbers, encoder names, and the per-class support.
- `artifacts/pareto.png`: macro-F1 and CUS against ms/video, with marker size showing MB.

## 6. Code layout

```
src/
  main.py        # argparse subcommands; wires modules
  config.py      # classes, chunkers, paths, seed, frame counts, encoder names
  sampling.py    # VideoMME parquet -> stratified 90 (+ spares)
  download.py    # yt-dlp 240p video-only
  frames.py      # uniform timestamps; 224 RGB + 8x64 gray bursts
  motion.py      # handcrafted features
  encoders.py    # open_clip image-tower wrapper: embed(frames) -> (N, D)
  labels.py      # mapping, contact sheets, labels.csv I/O
  train_eval.py  # sweep, repeated CV, metrics, CUS, Bayes rule, risk-coverage
  export.py      # sklearn head -> ONNX
  predict.py     # video -> JSON result
  benchmark.py   # timed end-to-end runs, batch 1 and 32
artifacts/       # committed, <10 MB total
samples/         # 2–3 committed clips, <2 MB each
tests/           # pytest
data/            # gitignored
```

**Subcommands.**
- Full rebuild: `sample → download → extract → label-sheets → featurize → evaluate → export`.
- Grader run: `fetch-encoder → evaluate → predict → benchmark`. No video download.

**`predict` output** (one JSON object per video):
```json
{"video": "x.mp4", "class": "sports_broadcast", "confidence": 0.87,
 "recommended_chunker": "motion_segment", "fallback": false,
 "latency_ms": {"decode": 0.0, "motion": 0.0, "encoder": 0.0, "head": 0.0, "total": 0.0}}
```
When confidence < τ, `fallback` is true and `recommended_chunker` is `fixed_window`.

## 7. `run.yaml` and dependencies

- `requirements.txt` begins with `--extra-index-url https://download.pytorch.org/whl/cpu` so CPU-only torch wheels are installed.
- setup: `pip install -r requirements.txt`; `python src/main.py fetch-encoder`.
- run: `python src/main.py evaluate`; `python src/main.py predict samples/`; `python src/main.py benchmark samples/`.
- test: `python -m pytest -q tests`.
- notes: CPU only; about 3 minutes; about 400 MB of encoder weights are downloaded; the full rebuild from YouTube is documented in `docs/` and is not part of the grader run.
- All commands are non-interactive and idempotent, and paths are relative to the repo root.

## 8. Testing

- Timestamps cover the full duration, skip the 5% ends, and are strictly increasing.
- Motion features: a synthetic black→white cut clip yields a cut rate > 0; a static synthetic clip yields motion energy ≈ 0.
- CUS and the Bayes decision rule match hand-computed values on a toy example.
- The probabilities from the ONNX head match the sklearn probabilities within 1e-5.
- `predict` runs end-to-end on one sample clip and returns JSON with valid fields.

## 9. Deployment (write-up only, not built)

- An ingest queue feeds stateless CPU workers. Each worker seek-decodes 16 low-resolution frames plus bursts from object storage, without decoding the full video.
- **Stage 1:** motion features and the motion-only head. If confidence ≥ τ₁, route the video and stop.
- **Stage 2:** the image encoder through ONNX Runtime with int8 quantization, batching frames across videos, then the combined head.
- **Fallback:** if confidence is still < τ₂, use the generic fixed-window chunker.
- **Cost at 100M videos** is computed from measured per-stage latency and the measured fraction of videos that stop at stage 1. Workers scale horizontally with no shared state.
- **Monitoring:** predicted class mix (drift) and the low-confidence rate, which feed an active-learning labeling queue.

## 10. Write-up (`docs/`, ≤2 pages)

Problem framing and taxonomy; feature engineering; architecture and cascade diagram; Pareto plot and the chosen operating point; results table, confusion matrix, CUS and label-agreement rate; deployment; limitations (n = 90, proxy labels, assumed U matrix, image encoder without audio). Future work: audio embeddings for talking_head vs presentation, active learning.

## 11. Schedule and cut order

| Time | Work |
|---|---|
| 0:00–0:30 | Sampling; download running in the background |
| 0:30–1:15 | Frames, motion features, encoders, featurize |
| 1:15–1:40 | Contact-sheet label review (with the user) |
| 1:40–2:30 | Sweep, evaluation, export |
| 2:30–3:00 | Predict, benchmark, `run.yaml`, tests |
| 3:00–3:30 | Write-up, push |

**If we run behind, cut in this order:** zero-shot baselines, the 4-frame column, the Bayes decision rule, the risk–coverage curve. A working `run.yaml` and the write-up are never cut.

## 12. Risks

- **YouTube availability or rate limits:** use spare videos; the user may supply files directly.
- **Tiny classes** (fewer than 5 videos): flag them and report the support beside the F1 scores; do not over-interpret.
- **Encoder availability in `open_clip`:** follow the fallback chain in §4.2 and log the choice.
- **Latency varies by machine:** `benchmark` re-measures on the grader's machine; the write-up quotes our machine's numbers with its specs.
