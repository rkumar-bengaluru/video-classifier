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
