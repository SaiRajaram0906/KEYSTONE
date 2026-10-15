"""Convert public crack datasets into the layout the loader expects.

    data/processed/
      images/  0001.jpg ...
      masks/   0001.png ...

WHERE TO GET THE DATA
---------------------
None of these download automatically — they sit behind varying licences
and hosts, and silently pulling them would be the wrong default. Fetch one
by hand, unzip it under data/raw/, then point this script at it.

  DeepCrack        537 images with pixel-level labels. Best starting point:
                   clean annotations, varied surfaces.
                   github.com/yhlleo/DeepCrack

  CrackForest      118 images of road surface. Small; masks are .mat files,
                   handled below.
                   github.com/cuilimeng/CrackForest-dataset

  Crack500         3,368 images, larger and noisier. Good for a second
                   experiment once the pipeline works.
                   github.com/fyangneil/pavement-crack-detection

  CrackTree260 /   Thin cracks on concrete. Useful for testing whether the
  CRKWH100         model handles sub-3-pixel cracks.

A NOTE ON DOMAIN
----------------
Every dataset above is predominantly road and concrete. KEYSTONE's target
is stone masonry, where mortar joints look a great deal like cracks. Expect
the model to flag joints until you fine-tune on masonry images, and say so
in the report rather than presenting road-surface metrics as if they
transferred. This is the single most important limitation of this project.

USAGE
-----
    python scripts/prepare_data.py --src data/raw/DeepCrack --layout deepcrack
    python scripts/prepare_data.py --src data/raw/CrackForest --layout crackforest
    python scripts/prepare_data.py --src data/raw/custom --layout pairs \
        --images-dir photos --masks-dir labels
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _write_pair(idx: int, img_path: Path, mask: np.ndarray, out: Path) -> bool:
    """Write one image/mask pair, returning False if the image is unreadable."""
    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"  [skip] unreadable image: {img_path.name}")
        return False

    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(
            mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    # Normalise to a strict 0/255 binary mask.
    binary = ((mask > 127).astype(np.uint8)) * 255

    cv2.imwrite(str(out / "images" / f"{idx:04d}.png"), image)
    cv2.imwrite(str(out / "masks" / f"{idx:04d}.png"), binary)
    return True


def _find_mask(stem: str, mask_dir: Path) -> Path | None:
    """Match a mask to an image stem, tolerating common suffixes."""
    candidates = [stem, f"{stem}_mask", f"{stem}_label", f"{stem}_gt", stem.replace("_image", "")]
    for cand in candidates:
        for ext in (".png", ".bmp", ".jpg", ".jpeg", ".tif"):
            p = mask_dir / f"{cand}{ext}"
            if p.exists():
                return p
    return None


def convert_pairs(src: Path, out: Path, images_dir: str, masks_dir: str,
                  start: int = 0) -> int:
    """Convert matched pairs, numbering output files from `start`.

    Returns how many pairs were written. Callers converting several folders
    into one output must pass a running `start`, or later folders overwrite
    earlier ones (0000.png from test_img replacing 0000.png from train_img).
    """
    img_dir, mask_dir = src / images_dir, src / masks_dir
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"expected {img_dir} and {mask_dir}")

    n, missing = 0, 0
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        mp = _find_mask(p.stem, mask_dir)
        if mp is None:
            missing += 1
            continue
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            missing += 1
            continue
        if _write_pair(start + n, p, mask, out):
            n += 1
    if missing:
        print(f"  [warn] {missing} image(s) had no readable mask and were skipped")
    return n


def convert_crackforest(src: Path, out: Path) -> int:
    """CrackForest ships masks as MATLAB .mat files under groundTruth/."""
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("scipy is required for the crackforest layout") from exc

    img_dir = src / "image"
    gt_dir = src / "groundTruth"
    if not img_dir.is_dir() or not gt_dir.is_dir():
        raise FileNotFoundError(f"expected {img_dir} and {gt_dir}")

    n = 0
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        mat_path = gt_dir / f"{p.stem}.mat"
        if not mat_path.exists():
            continue
        mat = loadmat(str(mat_path))
        # groundTruth{0,0}['Segmentation'] holds labels where 2 == crack.
        seg = mat["groundTruth"][0, 0]["Segmentation"][0, 0]
        mask = ((np.asarray(seg) > 1).astype(np.uint8)) * 255
        if _write_pair(n, p, mask, out):
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="unzipped dataset directory")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument(
        "--layout",
        default="pairs",
        choices=["pairs", "deepcrack", "crackforest"],
    )
    ap.add_argument("--images-dir", default="images")
    ap.add_argument("--masks-dir", default="masks")
    ap.add_argument("--clean", action="store_true", help="empty the output first")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    print(f"converting {src} ({args.layout}) -> {out}")
    if args.layout == "crackforest":
        n = convert_crackforest(src, out)
    elif args.layout == "deepcrack":
        # DeepCrack uses train_img / train_lab and test_img / test_lab.
        n = 0
        for imgs, masks in (("train_img", "train_lab"), ("test_img", "test_lab")):
            if (src / imgs).is_dir():
                n += convert_pairs(src, out, imgs, masks, start=n)
        if n == 0:
            n = convert_pairs(src, out, args.images_dir, args.masks_dir)
    else:
        n = convert_pairs(src, out, args.images_dir, args.masks_dir)

    if n == 0:
        raise SystemExit("no pairs converted — check --src and --layout")

    on_disk = sum(1 for _ in (out / "images").iterdir())
    if on_disk != n:
        raise SystemExit(f"wrote {n} pairs but {on_disk} images are on disk — "
                         "files were overwritten or the output was not empty (use --clean)")

    # Report the class balance; feed this to loss.pos_weight if not using 'auto'.
    pos = neg = 0
    for mp in sorted((out / "masks").iterdir())[:300]:
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        b = m > 127
        pos += int(b.sum())
        neg += int(b.size - b.sum())

    print(f"\n{n} pairs written to {out}")
    if pos:
        print(f"crack pixels: {pos / (pos + neg):.3%} of total")
        print(f"suggested pos_weight: {neg / pos:.1f}")
    else:
        print("[warn] no crack pixels found — are the masks inverted?")


if __name__ == "__main__":
    main()
