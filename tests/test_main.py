import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_help_lists_subcommands_without_crashing():
    r = subprocess.run([sys.executable, "src/main.py", "--help"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "sample" in r.stdout
    assert "download" in r.stdout


def test_cmd_predict_streams_in_chunks_of_32(monkeypatch, capsys):
    import argparse
    import main
    import predict
    chunks = []

    class FakePredictor:
        def predict_many(self, paths):
            chunks.append(len(paths))
            out = [{"video": str(p), "class": "x"} for p in paths]
            if len(chunks) == 2:
                out[0] = {"video": str(paths[0]), "error": "boom"}
            return out

    monkeypatch.setattr(predict, "Predictor", FakePredictor)
    monkeypatch.setattr(predict, "collect_videos", lambda paths: [Path(f"v{i}.mp4") for i in range(70)])
    rc = main.cmd_predict(argparse.Namespace(paths=["x"]))
    assert chunks == [32, 32, 6]
    assert len(capsys.readouterr().out.strip().splitlines()) == 70
    assert rc == 1
