"""Stand-ins for the parts you are not building.

Each workstream depends on the other two. Without these, nobody can start
until everybody has finished, which is the usual way a three-way split
turns into one person doing everything the night before.

Use these to develop and test your part alone:

    Workstream A (data)     needs a model to feed     -> FakeModel
    Workstream B (model)    needs data to train on    -> make_fake_dataset
    Workstream C (measure)  needs a checkpoint        -> save_fake_checkpoint

Every fake satisfies the same contract as the real thing, so code that
works against a fake works against the real implementation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "make_fake_dataset",
    "FakeModel",
    "save_fake_checkpoint",
    "synthetic_crack_mask",
]


def synthetic_crack_mask(h: int = 160, w: int = 160, width: int = 3, seed: int = 0):
    """A wandering crack across the image, as a binary mask and a matching image."""
    rng = np.random.default_rng(seed)
    img = rng.integers(70, 190, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), np.uint8)

    y = rng.integers(h // 4, 3 * h // 4)
    x0, x1 = int(w * 0.05), int(w * 0.95)
    drift = rng.integers(-1, 2, x1 - x0).cumsum()
    for j, x in enumerate(range(x0, x1)):
        yy = int(np.clip(y + drift[j], 1, h - width - 1))
        mask[yy:yy + width, x] = 255
        img[yy:yy + width, x] = rng.integers(15, 45)
    return img, mask


def make_fake_dataset(root: str | Path, n: int = 40, size: int = 160) -> Path:
    """Write a synthetic dataset in the layout the real loader expects.

    Roughly one image in five contains no crack, mirroring real datasets
    and exercising the empty-mask paths in the metrics.
    """
    import cv2

    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)

    for i in range(n):
        if i % 5 == 0:
            rng = np.random.default_rng(1000 + i)
            img = rng.integers(70, 190, (size, size, 3), dtype=np.uint8)
            mask = np.zeros((size, size), np.uint8)
        else:
            img, mask = synthetic_crack_mask(size, size, width=2 + (i % 3), seed=i)
        cv2.imwrite(str(root / "images" / f"{i:04d}.png"), img)
        cv2.imwrite(str(root / "masks" / f"{i:04d}.png"), mask)
    return root


class FakeModel:
    """A model-shaped object that returns plausible logits.

    Not trained and not learning — it exists so workstream A can verify
    its batches flow through something, and workstream C can build
    evaluation before a real checkpoint exists.

    Returns logits correlated with the input's darkness, since cracks are
    dark. That makes the metrics non-degenerate rather than random.
    """

    def __init__(self, noise: float = 1.0):
        self.noise = noise
        self._training = False

    def __call__(self, images):
        import torch

        darkness = -images.mean(dim=1, keepdim=True)
        logits = (darkness - darkness.mean()) * 6.0
        return logits + torch.randn_like(logits) * self.noise

    def forward(self, images):
        return self(images)

    def eval(self):
        self._training = False
        return self

    def train(self, mode: bool = True):
        self._training = mode
        return self

    def to(self, *_args, **_kwargs):
        return self

    def parameters(self):
        return iter(())

    def state_dict(self):
        return {}


def save_fake_checkpoint(path: str | Path, size: int = 160, base: int = 16) -> Path:
    """Write a real (untrained) checkpoint CrackSegmenter can actually load.

    Workstream C uses this to build and test the measurement layer before
    workstream B has trained anything.
    """
    import sys

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.models.unet import UNet

    cfg = {
        "data": {"image_size": size},
        "model": {"in_channels": 3, "base": base, "depth": 4, "bilinear": True},
    }
    model = UNet(base=base, depth=4)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg,
            "epoch": 0,
            "val_iou": 0.0,
            "best_threshold": 0.5,
        },
        path,
    )
    return path


if __name__ == "__main__":
    import tempfile

    import torch

    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_dataset(Path(tmp) / "data", n=10, size=128)
        n_images = len(list((root / "images").iterdir()))
        print(f"fake dataset: {n_images} images at {root}")

        model = FakeModel()
        images = torch.rand(2, 3, 128, 128)
        print(f"fake model: {tuple(images.shape)} -> {tuple(model(images).shape)}")

        ckpt = save_fake_checkpoint(Path(tmp) / "fake.pt", size=128)
        print(f"fake checkpoint: {ckpt.stat().st_size / 1e6:.1f} MB")

    print("\nfakes OK")
