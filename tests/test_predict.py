import json

import numpy as np
import pytest

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


def test_benchmark_raises_on_corrupt_file(fake_artifacts, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_text("nope")
    with pytest.raises(RuntimeError, match="bad.mp4"):
        run_benchmark(predictor(fake_artifacts), [bad], repeats=1)


def test_benchmark_raises_on_empty_paths():
    with pytest.raises(ValueError, match="no video files"):
        run_benchmark(None, [])


def test_predict_cli_empty_dir_returns_1(tmp_path, capsys, monkeypatch):
    import main as M
    import predict
    monkeypatch.setattr(predict, "Predictor", lambda: pytest.fail("should not load"))
    empty = tmp_path / "empty"
    empty.mkdir()
    assert M.main(["predict", str(empty)]) == 1
    assert "no video files" in capsys.readouterr().err


def test_low_confidence_falls_back(fake_artifacts, make_video):
    predictor(fake_artifacts)
    pj = fake_artifacts / "models" / "production.json"
    prod = json.loads(pj.read_text())
    prod["tau"] = 1.01
    pj.write_text(json.dumps(prod))
    r = Predictor(fake_artifacts / "models", encoder=FakeEncoder()).predict(make_video(n=100))
    assert r["fallback"] is True and r["recommended_chunker"] == C.FALLBACK_CHUNKER


class ContentEncoder:
    """Embedding depends on frame content: per-frame mean colour spread over 8 dims, L2-normalised."""
    def embed(self, frames, batch_size=32):
        f = frames.reshape(len(frames), -1, 3).mean(axis=1) / 255.0 + 0.05
        v = np.concatenate([f, f[:, ::-1], f[:, :2]], axis=1).astype(np.float32)
        v += np.arange(8, dtype=np.float32) * 0.01
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_batched_predictions_match_single(fake_artifacts, make_video):
    res = fake_artifacts / "eval_results.json"
    run_evaluation(fake_artifacts, n_repeats=1, n_boot=10)
    d = json.loads(res.read_text())
    d["operating_point"]["config"] = "small@4"
    res.write_text(json.dumps(d))
    export_production(fake_artifacts)
    pred = Predictor(fake_artifacts / "models", encoder=ContentEncoder())
    assert pred.cfg.encoder
    paths = [make_video("a.avi", kind="noise"), make_video("b.avi", kind="static"),
             make_video("c.avi", kind="cut")]
    batch = pred.predict_many(paths)
    for p, b in zip(paths, batch):
        s = pred.predict(p)
        assert b["class"] == s["class"]
        for c in C.CLASSES:
            assert abs(b["probabilities"][c] - s["probabilities"][c]) < 1e-5


def test_result_video_key_is_path_as_given(fake_artifacts, make_video):
    p = make_video(n=100)
    assert predictor(fake_artifacts).predict(p)["video"] == str(p)
    bad = p.parent / "missing.mp4"
    assert predictor(fake_artifacts).predict(bad)["video"] == str(bad)
