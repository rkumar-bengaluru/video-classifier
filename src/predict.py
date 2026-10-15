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
                results.append({"video": str(p), "error": str(e)})
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
            # training embeddings are stored fp16; match that precision at serve time
            emb = emb.astype(np.float16).astype(np.float32)
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
        return {"video": str(path), "class": cls, "confidence": round(conf, 4),
                "recommended_chunker": C.FALLBACK_CHUNKER if fallback else C.CHUNKERS[cls],
                "fallback": fallback,
                "probabilities": {c: round(float(q), 4) for c, q in zip(self.prod["classes"], proba)},
                "latency_ms": {k_: round(v, 2) for k_, v in lat.items()}}
