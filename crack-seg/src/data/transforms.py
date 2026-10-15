"""Augmentation pipelines for crack segmentation.

DESIGN NOTES
------------
Cracks are thin, high-frequency, near-black structures on textured stone.
That constrains what augmentation is safe:

  Safe and useful
    flips and 90-degree rotations   cracks have no canonical orientation
    brightness / contrast / gamma   capture conditions vary enormously
    blur and noise                  phone cameras at distance
    shadows                         the single biggest nuisance class

  Use with care
    elastic / grid distortion       can thin a crack out of existence, but
                                    mild settings improve generalisation
    scale changes                   a crack 1px wide can vanish when
                                    downscaled; NEAREST interpolation on
                                    the mask is essential

  Avoid
    heavy JPEG compression          introduces artefacts the model learns
                                    to treat as cracks
    channel shuffle / hue shifts    stone colour carries real signal

INTERPOLATION
-------------
Every geometric transform must use INTER_NEAREST for the mask. Bilinear
interpolation produces fractional mask values along crack edges, which
after thresholding either thickens or erases a one-pixel crack.
"""
from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

# ImageNet statistics. Correct if you ever swap in a pretrained encoder,
# and harmless if you do not.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

__all__ = ["train_transform", "eval_transform", "build_transforms", "denormalise"]


def train_transform(size: int = 320, strength: str = "medium") -> A.Compose:
    """Augmentation for training.

    Args:
        size: output square size in pixels.
        strength: 'light' | 'medium' | 'heavy'. Start at medium; go heavy
            only if train and validation curves diverge.
    """
    if strength not in {"light", "medium", "heavy"}:
        raise ValueError("strength must be light, medium or heavy")

    geometric = [
        A.RandomResizedCrop(
            size=(size, size),
            scale=(0.5, 1.0) if strength != "light" else (0.75, 1.0),
            ratio=(0.85, 1.18),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]

    photometric = [
        A.RandomBrightnessContrast(
            brightness_limit=0.25, contrast_limit=0.25, p=0.7
        ),
        A.RandomGamma(gamma_limit=(70, 140), p=0.3),
    ]

    degradation = [
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ],
            p=0.10,
        ),
        A.GaussNoise(p=0.25),
    ]

    if strength == "light":
        extra = []
    elif strength == "medium":
        extra = [
            A.ElasticTransform(
                alpha=25.0,
                sigma=6.0,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.15,
            ),
        ]
    else:  # heavy
        extra = [
            A.ElasticTransform(
                alpha=40.0,
                sigma=7.0,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.25,
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.2,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.2,
            ),
            A.CoarseDropout(p=0.15),
        ]

    return A.Compose(
        geometric + photometric + degradation + extra
        + [A.Normalize(mean=MEAN, std=STD), ToTensorV2()]
    )


def eval_transform(size: int = 320) -> A.Compose:
    """Deterministic resize for validation, test and inference.

    No augmentation, and the same normalisation as training. Using a plain
    resize rather than a crop means the whole image is evaluated, which is
    what the reported metrics should reflect.
    """
    return A.Compose(
        [
            A.Resize(
                height=size,
                width=size,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )


def build_transforms(cfg: dict):
    """Return (train_tf, eval_tf) from a config dict."""
    d = cfg.get("data", {})
    size = int(d.get("image_size", 320))
    strength = str(d.get("aug_strength", "medium"))
    return train_transform(size, strength), eval_transform(size)


def denormalise(tensor):
    """Undo Normalize for visualisation. Accepts (3, H, W) or (N, 3, H, W)."""
    import torch

    mean = torch.tensor(MEAN, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(STD, device=tensor.device).view(-1, 1, 1)
    if tensor.ndim == 4:
        mean, std = mean.unsqueeze(0), std.unsqueeze(0)
    return (tensor * std + mean).clamp(0, 1)


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    image = rng.integers(40, 210, (240, 300, 3), dtype=np.uint8)
    mask = np.zeros((240, 300), np.float32)
    mask[118:121, 20:280] = 1.0          # a 3px crack
    image[118:121, 20:280] = 25

    for name, tf in (
        ("train/light", train_transform(320, "light")),
        ("train/medium", train_transform(320, "medium")),
        ("train/heavy", train_transform(320, "heavy")),
        ("eval", eval_transform(320)),
    ):
        kept = []
        for _ in range(12):
            out = tf(image=image, mask=mask)
            im, mk = out["image"], out["mask"]
            kept.append(float(mk.sum()))
        assert tuple(im.shape) == (3, 320, 320), f"{name}: bad image shape {im.shape}"
        assert mk.ndim == 2 and tuple(mk.shape) == (320, 320), f"{name}: bad mask shape"
        vals = set(np.unique(mk.numpy()).tolist())
        assert vals <= {0.0, 1.0}, f"{name}: mask not binary after transform: {vals}"
        survived = sum(1 for k in kept if k > 0)
        print(
            f"{name:<14} image {tuple(im.shape)}  mask {tuple(mk.shape)}  "
            f"binary OK  crack survived {survived}/12 draws"
        )

    print("\nall transform checks passed")
