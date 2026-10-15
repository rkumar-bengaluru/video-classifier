import numpy as np
import pytest

import config as C


def test_resolve_returns_a_listed_candidate():
    from encoders import resolve
    small = resolve("small")
    assert small is None or small in C.ENCODER_CANDIDATES["small"]
    assert resolve("large") == ("ViT-B-32", "laion2b_s34b_b79k")


@pytest.mark.slow
def test_small_encoder_embeds_normalized_vectors():
    from encoders import load, resolve
    enc = load(*resolve("small"))
    frames = np.random.default_rng(0).integers(0, 256, (3, 256, 256, 3), dtype=np.uint8)
    e = enc.embed(frames)
    assert e.shape[0] == 3 and e.dtype == np.float32
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-4)
    t = enc.class_text_embeddings()
    assert t.shape == (6, e.shape[1])
