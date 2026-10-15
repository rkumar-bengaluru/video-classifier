"""Sweep {feature set} x {frames} on cached features: repeated CV, CIs, CUS, latency/size, Pareto plot."""
import json
import time
import warnings
from contextlib import contextmanager
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


@contextmanager
def quiet_tiny_class_warning():
    """Tiny classes (fewer members than folds) are expected and handled by column alignment in
    cross_val_proba, so sklearn's warning about them is noise. Scoped here because pytest resets
    module-level filters per test."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The least populated class")
        yield


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
                         LogisticRegressionCV(Cs=CS, cv=3, class_weight="balanced", max_iter=5000,
                                              l1_ratios=(0,), scoring="accuracy", use_legacy_attributes=True))


def cross_val_proba(X, y, k, n_splits=5, n_repeats=5, seed=C.SEED):
    out = np.zeros((n_repeats, len(y), k))
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        with quiet_tiny_class_warning():
            folds = list(skf.split(X, y))
        for tr, te in folds:
            with quiet_tiny_class_warning():
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
    with quiet_tiny_class_warning():
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
