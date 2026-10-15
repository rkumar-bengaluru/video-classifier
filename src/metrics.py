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
