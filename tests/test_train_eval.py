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


def test_cv_alignment_when_training_fold_lacks_a_middle_class():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(6), [12, 12, 1, 12, 12, 12])
    X = np.eye(6)[y] * 3 + 0.3 * rng.normal(size=(len(y), 6))
    lone = int(np.flatnonzero(y == 2)[0])
    proba = cross_val_proba(X, y, 6, n_repeats=2)
    assert proba.shape == (2, len(y), 6)
    assert np.allclose(proba.sum(2), 1.0)
    # The fold testing the lone class-2 video trains without class 2, so the head has only 5 columns.
    # Correct placement via head.classes_ leaves column 2 at exactly 0 for that video.
    assert (proba[:, lone, 2] == 0).all()
    # A naive out[te, :len(classes_)] assignment would shift classes 3..5 one column left, so every
    # video of those classes in that fold would be mispredicted and accuracy on the others would drop.
    others = np.arange(len(y)) != lone
    for r in range(2):
        assert (proba[r].argmax(1)[others] == y[others]).mean() > 0.9
