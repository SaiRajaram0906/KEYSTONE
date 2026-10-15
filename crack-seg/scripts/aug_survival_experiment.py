"""Workstream A write-up experiment: does augmentation erase thin cracks from the IMAGE?

The self-test in src/data/transforms.py counts a draw as "survived" when the
MASK still has crack pixels. That misses the dangerous failure: blur,
downscaling or distortion fade the crack out of the image while
nearest-neighbour interpolation keeps it in the mask. The model is then taught
"plain wall = crack".

This script measures the image side. For every draw it compares the crack's
darkness against a 2-3 px ring of wall around it, before and after the
transform:

    ratio = contrast_after / contrast_before
    faded      ratio < 0.5  -> crack lost at least half its contrast
    no crack   output mask is empty: the random crop missed the crack, or the
               mask was erased (the only case transforms.py's self-test catches)

Albumentations draws its own random numbers, so exact percentages move by a
point or two between runs; the pattern does not.

It runs your real pipelines (train_transform light/medium/heavy, eval_transform)
and each augmentation on its own, on synthetic cracks 1, 2 and 3 px wide at
DeepCrack's image size (544x384), at random angles.

    python scripts/aug_survival_experiment.py
    python scripts/aug_survival_experiment.py --draws 50        # quicker

Writes runs/aug_experiment/results.csv and results.md.
Changes nothing in src/: read-only use of src/data/transforms.py.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import albumentations as A
import cv2
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

from src.data.transforms import MEAN, STD, eval_transform, train_transform

FADE_RATIO = 0.5


# ---------------------------------------------------------------------------
# Synthetic cracks of known width
# ---------------------------------------------------------------------------
def synthetic_crack(h: int, w: int, width: int, rng: np.random.Generator,
                    contrast: float = 60.0):
    """Textured stone-like background with one straight dark crack of `width` px."""
    gray = 140 + gaussian_filter(rng.normal(0, 1, (h, w)), 6) * 80 + rng.normal(0, 4, (h, w))
    angle = rng.uniform(0, np.pi)
    cy, cx, r = h / 2, w / 2, 0.45 * min(h, w)
    p0 = (int(cx - r * np.cos(angle)), int(cy - r * np.sin(angle)))
    p1 = (int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle)))
    mask = np.zeros((h, w), np.uint8)
    cv2.line(mask, p0, p1, 1, thickness=width, lineType=cv2.LINE_8)
    gray = gray - contrast * mask
    img = np.clip(np.stack([gray + 3, gray, gray - 5], -1), 0, 255).astype(np.uint8)
    return img, mask.astype(np.float32)


# ---------------------------------------------------------------------------
# Survival measurement
# ---------------------------------------------------------------------------
def crack_contrast(gray: np.ndarray, mask: np.ndarray) -> float:
    """mean(ring 2-3 px around the crack) - mean(crack pixels). NaN if no crack."""
    m = mask > 0.5
    ring = binary_dilation(m, iterations=3) & ~binary_dilation(m, iterations=1)
    if m.sum() == 0 or ring.sum() == 0:
        return float("nan")
    return float(gray[ring].mean() - gray[m].mean())


def to_gray01(img) -> np.ndarray:
    """Transform output (normalised CHW tensor, or HWC uint8) -> HxW gray in [0,1]."""
    if hasattr(img, "numpy"):
        x = img.numpy().astype(np.float32)
        if x.ndim == 3 and x.shape[0] == 3:
            x = x.transpose(1, 2, 0)
        x = x * np.array(STD, np.float32) + np.array(MEAN, np.float32)
        return np.clip(x, 0, 1).mean(-1)
    x = np.asarray(img)
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    return x.mean(-1) if x.ndim == 3 else x


def to_mask(mk) -> np.ndarray:
    return mk.numpy() if hasattr(mk, "numpy") else np.asarray(mk)


# ---------------------------------------------------------------------------
# What to test: your full pipelines, and each op from them on its own
# ---------------------------------------------------------------------------
def single_ops(size: int) -> dict[str, A.Compose]:
    """Each augmentation from transforms.py in isolation, with p=1 and the same
    parameters, so a fade can be attributed to one op."""
    nn = dict(interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST)
    ops = {
        "RandomResizedCrop scale 0.5-1": A.RandomResizedCrop(size=(size, size), scale=(0.5, 1.0),
                                                             ratio=(0.85, 1.18), p=1.0, **nn),
        "RandomResizedCrop scale 0.75-1": A.RandomResizedCrop(size=(size, size), scale=(0.75, 1.0),
                                                              ratio=(0.85, 1.18), p=1.0, **nn),
        "Resize (eval)": A.Resize(height=size, width=size, **nn),
        "Flip + Rotate90": A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                                      A.RandomRotate90(p=1.0)]),
        "BrightnessContrast 0.25": A.RandomBrightnessContrast(brightness_limit=0.25,
                                                              contrast_limit=0.25, p=1.0),
        "Gamma 70-140": A.RandomGamma(gamma_limit=(70, 140), p=1.0),
        "GaussianBlur 3-5": A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        "MotionBlur 5": A.MotionBlur(blur_limit=5, p=1.0),
        "GaussNoise (default)": A.GaussNoise(p=1.0),
        "Elastic (medium)": A.ElasticTransform(alpha=25.0, sigma=6.0, p=1.0, **nn),
        "Elastic (heavy)": A.ElasticTransform(alpha=40.0, sigma=7.0, p=1.0, **nn),
        "GridDistortion (heavy)": A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0, **nn),
    }
    return {k: (v if isinstance(v, A.Compose) else A.Compose([v])) for k, v in ops.items()}


def candidate_native_crop(size: int) -> A.Compose:
    """A candidate fix, NOT used by training: crop at native resolution instead of
    resizing, and drop blur. Kept here only so the write-up can compare it."""
    from albumentations.pytorch import ToTensorV2
    return A.Compose([
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_REFLECT_101),
        A.RandomCrop(height=size, width=size, p=1.0),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        A.RandomGamma(gamma_limit=(70, 140), p=0.3),
        A.GaussNoise(p=0.25),
        A.ElasticTransform(alpha=25.0, sigma=6.0, interpolation=cv2.INTER_LINEAR,
                           mask_interpolation=cv2.INTER_NEAREST, p=0.15),
        A.Normalize(mean=MEAN, std=STD), ToTensorV2(),
    ])


def pipelines(size: int) -> dict[str, A.Compose]:
    return {
        "CANDIDATE native crop, no blur": candidate_native_crop(size),
        "PIPELINE train/light": train_transform(size, "light"),
        "PIPELINE train/medium": train_transform(size, "medium"),
        "PIPELINE train/heavy": train_transform(size, "heavy"),
        "PIPELINE eval": eval_transform(size),
    }


def run(tests: dict, widths, draws, h, w, seed):
    rows = []
    for name, tf in tests.items():
        for width in widths:
            rng = np.random.default_rng([seed, zlib.crc32(name.encode()), width])
            ratios, vanished = [], 0
            for _ in range(draws):
                img, mask = synthetic_crack(h, w, width, rng)
                c0 = crack_contrast(to_gray01(img), mask)
                out = tf(image=img, mask=mask)
                mk = to_mask(out["mask"])
                if (mk > 0.5).sum() == 0:
                    # No crack pixels left in the output mask: either the crop missed the
                    # crack or interpolation erased it. Excluded from the fade ratio.
                    vanished += 1
                    continue
                c1 = crack_contrast(to_gray01(out["image"]), mk)
                if np.isfinite(c1) and c0 > 0:
                    ratios.append(c1 / c0)
            r = np.array(ratios) if ratios else np.array([np.nan])
            rows.append({
                "test": name, "width_px": width, "draws": draws,
                "faded_pct": round(100.0 * float(np.mean(r < FADE_RATIO)), 1) if ratios else float("nan"),
                "mask_vanished_pct": round(100.0 * vanished / draws, 1),
                "median_ratio": round(float(np.nanmedian(r)), 3),
                "p10_ratio": round(float(np.nanpercentile(r, 10)), 3),
            })
            print(f"  {name:<32} {width}px  faded {rows[-1]['faded_pct']:>5}%  "
                  f"median ratio {rows[-1]['median_ratio']:.2f}")
    return rows


def write(rows, out: Path, widths):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    lines = [
        f"Faded draws (%): crack keeps < {FADE_RATIO:.0%} of its contrast against the wall "
        "around it, while the mask still marks it.",
        "Median ratio in brackets (1.00 = crack as dark as before).",
        "",
        "| test | " + " | ".join(f"{w}px" for w in widths) + " |",
        "|" + "---|" * (1 + len(widths)),
    ]
    for name in dict.fromkeys(r["test"] for r in rows):
        cells = []
        for w in widths:
            r = next(x for x in rows if x["test"] == name and x["width_px"] == w)
            cells.append(f"{r['faded_pct']:.0f}% ({r['median_ratio']:.2f})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    (out / "results.md").write_text(text, encoding="utf-8")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--widths", default="1,2,3")
    ap.add_argument("--size", type=int, default=320, help="training size (configs: data.image_size)")
    ap.add_argument("--height", type=int, default=384, help="source image height (DeepCrack: 384)")
    ap.add_argument("--width", type=int, default=544, help="source image width (DeepCrack: 544)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["all", "pipelines", "ops", "candidate"], default="all")
    ap.add_argument("--out", type=Path, default=Path("runs/aug_experiment"))
    args = ap.parse_args(argv)

    widths = [int(x) for x in args.widths.split(",")]
    tests = {}
    if args.only == "candidate":
        tests["CANDIDATE native crop, no blur"] = candidate_native_crop(args.size)
    if args.only in ("all", "pipelines"):
        tests.update(pipelines(args.size))
    if args.only in ("all", "ops"):
        tests.update(single_ops(args.size))

    t0 = time.time()
    print(f"{len(tests)} tests x {len(widths)} widths x {args.draws} draws, "
          f"source {args.width}x{args.height} -> {args.size}x{args.size}\n")
    rows = run(tests, widths, args.draws, args.height, args.width, args.seed)
    print()
    print(write(rows, args.out, widths))
    print(f"{time.time() - t0:.0f}s -> {args.out}/results.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
