"""Dataset for paired crack images and binary masks.

EXPECTED LAYOUT
---------------
    data/
      processed/
        images/   0001.jpg  0002.jpg  ...
        masks/    0001.png  0002.png  ...

Masks are single-channel; any non-zero pixel is treated as crack. Stems
must match between the two folders — scripts/prepare_data.py converts the
public datasets into exactly this layout and reports any unpaired files.

SPLITTING
---------
Splits are made by hashing the filename, not by shuffling with a seed.
That means a given image lands in the same split even if files are added
later, which stops results drifting between runs for reasons unrelated to
the model. This matters more than it sounds: a train/test leak caused by a
re-shuffle is very hard to notice from the metrics alone.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}

__all__ = ["CrackDataset", "discover_pairs", "split_pairs", "compute_pos_weight"]


def discover_pairs(root: str | Path) -> list[tuple[Path, Path]]:
    """Find matching image/mask pairs under root/images and root/masks.

    Raises:
        FileNotFoundError: if either folder is missing.
        RuntimeError: if no pairs match.
    """
    root = Path(root)
    img_dir, mask_dir = root / "images", root / "masks"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"expected {img_dir} and {mask_dir}. "
            "Run scripts/prepare_data.py first."
        )

    masks: dict[str, Path] = {}
    for p in sorted(mask_dir.iterdir()):
        if p.suffix.lower() in MASK_EXTS:
            masks[p.stem] = p

    pairs, unpaired = [], []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        m = masks.get(p.stem)
        if m is None:
            unpaired.append(p.name)
        else:
            pairs.append((p, m))

    if unpaired:
        preview = ", ".join(unpaired[:5])
        more = f" (+{len(unpaired) - 5} more)" if len(unpaired) > 5 else ""
        print(f"[dataset] warning: {len(unpaired)} images have no mask: {preview}{more}")
    if not pairs:
        raise RuntimeError(f"no image/mask pairs found under {root}")
    return pairs


def _bucket(name: str) -> float:
    """Deterministic value in [0, 1) derived from a filename."""
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_pairs(
    pairs: list[tuple[Path, Path]],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict[str, list[tuple[Path, Path]]]:
    """Deterministic train/val/test split keyed on filename hash."""
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")
    out: dict[str, list[tuple[Path, Path]]] = {"train": [], "val": [], "test": []}
    for img, mask in pairs:
        b = _bucket(img.stem)
        if b < test_frac:
            out["test"].append((img, mask))
        elif b < test_frac + val_frac:
            out["val"].append((img, mask))
        else:
            out["train"].append((img, mask))
    return out


class CrackDataset(Dataset):
    """Paired image/mask dataset.

    Returns:
        image: float32 tensor (3, H, W), normalised by the transform.
        mask:  float32 tensor (1, H, W) containing 0.0 or 1.0.

    Args:
        pairs: list of (image_path, mask_path).
        transform: an albumentations Compose expecting image= and mask=
            keyword arguments and returning tensors (see transforms.py).
            If None, images are returned at native size, scaled to [0, 1].
        mask_threshold: mask pixels strictly above this (after scaling to
            [0,1]) count as crack. Handles anti-aliased or JPEG-compressed
            masks, where values near the edges are not exactly 0 or 255.
    """

    def __init__(self, pairs, transform=None, mask_threshold: float = 0.5):
        self.pairs = list(pairs)
        self.transform = transform
        self.mask_threshold = mask_threshold

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read mask {mask_path}")
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        mask = (mask.astype(np.float32) / 255.0 > self.mask_threshold).astype(np.float32)
        return image, mask

    def __getitem__(self, idx: int):
        image, mask = self._load(idx)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image_t, mask_t = out["image"], out["mask"]
            if not torch.is_tensor(image_t):
                image_t = torch.from_numpy(image_t.transpose(2, 0, 1))
            if not torch.is_tensor(mask_t):
                mask_t = torch.from_numpy(mask_t)
        else:
            image_t = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0)
            mask_t = torch.from_numpy(mask)

        # Albumentations returns masks as (H, W); the model wants (1, H, W).
        mask_t = mask_t.float()
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)
        elif mask_t.ndim == 3 and mask_t.shape[-1] == 1:
            mask_t = mask_t.permute(2, 0, 1)

        return image_t.float(), mask_t


def compute_pos_weight(pairs, max_samples: int = 200) -> float:
    """Estimate (background pixels / crack pixels) over a sample of masks.

    Feed the result to BCEDiceLoss(pos_weight=...) to counteract the
    imbalance. Returns 1.0 if no crack pixels are found at all.
    """
    pos = neg = 0
    for _, mask_path in list(pairs)[:max_samples]:
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        b = m > 127
        pos += int(b.sum())
        neg += int(b.size - b.sum())
    if pos == 0:
        print("[dataset] warning: no crack pixels found; pos_weight defaulting to 1.0")
        return 1.0
    return neg / pos


if __name__ == "__main__":
    import tempfile

    # Build a small synthetic dataset so the loader can be exercised without
    # downloading anything.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "images").mkdir()
        (root / "masks").mkdir()
        rng = np.random.default_rng(0)
        for i in range(20):
            img = rng.integers(60, 200, (128, 128, 3), dtype=np.uint8)
            msk = np.zeros((128, 128), np.uint8)
            if i % 4:  # three quarters of images contain a crack
                msk[60:63, 10:118] = 255
                img[60:63, 10:118] = 30
            cv2.imwrite(str(root / "images" / f"{i:04d}.png"), img)
            cv2.imwrite(str(root / "masks" / f"{i:04d}.png"), msk)

        pairs = discover_pairs(root)
        splits = split_pairs(pairs, val_frac=0.2, test_frac=0.2)
        print(f"pairs found: {len(pairs)}")
        print({k: len(v) for k, v in splits.items()})

        ds = CrackDataset(splits["train"])
        img, mask = ds[0]
        print(f"image {tuple(img.shape)} {img.dtype}  range [{img.min():.2f}, {img.max():.2f}]")
        print(f"mask  {tuple(mask.shape)} {mask.dtype}  unique {sorted(set(mask.unique().tolist()))}")
        print(f"pos_weight estimate: {compute_pos_weight(pairs):.1f}")

        # Splits must be stable across calls.
        again = split_pairs(pairs, val_frac=0.2, test_frac=0.2)
        assert [p[0].name for p in splits["test"]] == [p[0].name for p in again["test"]]
        print("deterministic split: OK")
