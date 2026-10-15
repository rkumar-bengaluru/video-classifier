"""Frozen open_clip encoders: frames -> L2-normalized embeddings (no training)."""
import time

import numpy as np
import open_clip
import torch
from timm.utils.model import reparameterize_model
from PIL import Image

import config as C


def resolve(kind):
    available = set(open_clip.list_pretrained())
    for name, tag in C.ENCODER_CANDIDATES[kind]:
        if (name, tag) in available:
            return name, tag
    return None


class Encoder:
    def __init__(self, name, pretrained):
        self.name, self.pretrained = name, pretrained
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        self.model.eval()
        # MobileCLIP/FastViT blocks are multi-branch at train time; fuse for inference (no-op for ViT-B-32)
        self.model = reparameterize_model(self.model)
        self.tokenizer = open_clip.get_tokenizer(name)

    @property
    def image_params(self):
        return sum(p.numel() for p in self.model.visual.parameters())

    @torch.inference_mode()
    def embed(self, frames, batch_size=32):
        out = []
        for i in range(0, len(frames), batch_size):
            x = torch.stack([self.preprocess(Image.fromarray(f)) for f in frames[i:i + batch_size]])
            out.append(self.model.encode_image(x, normalize=True).float().numpy())
        return np.concatenate(out)

    @torch.inference_mode()
    def embed_text(self, texts):
        return self.model.encode_text(self.tokenizer(texts), normalize=True).float().numpy()

    def class_text_embeddings(self):
        rows = []
        for cls in C.CLASSES:
            m = self.embed_text(C.ZERO_SHOT_PROMPTS[cls]).mean(0)
            rows.append(m / np.linalg.norm(m))
        return np.stack(rows).astype(np.float32)


def load(name, pretrained):
    return Encoder(name, pretrained)


def ms_per_frame(enc, frames, repeats=10):
    for _ in range(2):  # warm-up
        enc.embed(frames[:1])
    times = []
    for i in range(repeats):
        t0 = time.perf_counter()
        enc.embed(frames[i % len(frames):i % len(frames) + 1])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))
