"""End-to-end CPU timing of the production pipeline at batch 1 and batch 32."""
import os
import platform
import time

import numpy as np


def _check(results):
    for r in results:
        if "error" in r:
            raise RuntimeError(f"benchmark input failed: {r['video']}: {r['error']}")


def run(predictor, paths, repeats=3):
    if not paths:
        raise ValueError("no video files found in the given paths")
    _check([predictor.predict(paths[0])])  # warm-up
    times, comps = [], []
    for _ in range(repeats):
        for p in paths:
            t0 = time.perf_counter()
            r = predictor.predict(p)
            _check([r])
            times.append((time.perf_counter() - t0) * 1000)
            comps.append(r.get("latency_ms", {}))
    batch = [paths[i % len(paths)] for i in range(32)]
    t0 = time.perf_counter()
    _check(predictor.predict_many(batch))
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
