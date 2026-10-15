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
from train_eval import CS, Config, build_X, load_features, quiet_tiny_class_warning


def fit_final(X, y):
    cv = LogisticRegressionCV(Cs=CS, cv=3, class_weight="balanced", max_iter=5000,
                              l1_ratios=(0,), scoring="accuracy", use_legacy_attributes=True)
    with quiet_tiny_class_warning():
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
