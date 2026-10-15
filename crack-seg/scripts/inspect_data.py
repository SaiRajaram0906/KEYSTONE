"""Inspect a converted dataset before anyone trains on it.

    python scripts/inspect_data.py --root data/processed

This is workstream A's main quality gate. Every problem it looks for has
silently ruined somebody's segmentation project at some point:

    inverted masks        crack labelled as 0, background as 255. Training
                          proceeds, loss decreases, IoU stays near zero and
                          nobody can work out why.
    empty masks           images whose mask is entirely background. A few
                          are normal and useful; a lot means the conversion
                          matched the wrong files.
    full masks            masks that are almost entirely foreground —
                          usually an inverted mask caught the other way.
    size mismatch         mask and image different shapes. Silently resized
                          at load time, which shifts every label slightly.
    non-binary masks      anti-aliased or JPEG-compressed masks with values
                          between 0 and 255.
    duplicate images      the same photo under two names, which leaks across
                          the train/test split.

It also writes a contact sheet. Look at it. Five minutes of eyeballing
masks catches things no statistic will.
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_pairs(root: Path):
    img_dir, mask_dir = root / "images", root / "masks"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise SystemExit(
            f"expected {img_dir} and {mask_dir}\n"
            "Run scripts/prepare_data.py first."
        )
    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    pairs = []
    orphan_images = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        m = masks.get(p.stem)
        (pairs if m else orphan_images).append((p, m) if m else p)
    orphan_masks = sorted(set(masks) - {p.stem for p, _ in pairs})
    return pairs, orphan_images, orphan_masks


def inspect(root: Path, sheet_path: Path | None, sheet_n: int, thumb: int):
    pairs, orphan_images, orphan_masks = load_pairs(root)
    print(f"root: {root}")
    print(f"pairs: {len(pairs)}")
    if orphan_images:
        print(f"  [warn] {len(orphan_images)} image(s) with no mask")
        for p in orphan_images[:5]:
            print(f"         {p.name}")
    if orphan_masks:
        print(f"  [warn] {len(orphan_masks)} mask(s) with no image")
    if not pairs:
        raise SystemExit("nothing to inspect")

    pos_total = neg_total = 0
    fractions: list[float] = []
    empty, nearly_full, mismatched, non_binary = [], [], [], []
    hashes: dict[str, list[str]] = {}
    sizes = Counter()

    for img_path, mask_path in pairs:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"  [warn] unreadable: {img_path.name}")
            continue

        sizes[img.shape[:2]] += 1
        if mask.shape[:2] != img.shape[:2]:
            mismatched.append(img_path.name)

        vals = np.unique(mask)
        if not set(vals.tolist()) <= {0, 255}:
            mid = int(((mask > 10) & (mask < 245)).sum())
            if mid > 0.001 * mask.size:
                non_binary.append((img_path.name, mid / mask.size))

        binary = mask > 127
        frac = float(binary.mean())
        fractions.append(frac)
        pos_total += int(binary.sum())
        neg_total += int(binary.size - binary.sum())

        if frac == 0.0:
            empty.append(img_path.name)
        elif frac > 0.5:
            nearly_full.append((img_path.name, frac))

        h = hashlib.md5(cv2.resize(img, (32, 32)).tobytes()).hexdigest()
        hashes.setdefault(h, []).append(img_path.name)

    fr = np.array(fractions)
    print("\nCLASS BALANCE")
    print(f"  crack pixels overall : {pos_total / (pos_total + neg_total):.3%}")
    print(f"  per-image mean       : {fr.mean():.3%}")
    print(f"  per-image median     : {np.median(fr):.3%}")
    print(f"  min / max            : {fr.min():.3%} / {fr.max():.3%}")
    if pos_total:
        print(f"  suggested pos_weight : {neg_total / pos_total:.1f}")

    print("\nSANITY CHECKS")
    ok = True

    if pos_total == 0:
        print("  FAIL  no crack pixels at all — masks are empty or inverted")
        ok = False
    elif pos_total / (pos_total + neg_total) > 0.30:
        print(
            f"  FAIL  {pos_total / (pos_total + neg_total):.1%} of pixels are crack. "
            "That is far too high — the masks are almost certainly inverted."
        )
        ok = False
    else:
        print("  pass  crack fraction is in a plausible range")

    n_empty_pct = len(empty) / len(pairs)
    if n_empty_pct > 0.4:
        print(f"  FAIL  {n_empty_pct:.0%} of masks are entirely empty — check the conversion")
        ok = False
    elif empty:
        print(f"  pass  {len(empty)} empty mask(s) ({n_empty_pct:.0%}) — normal, and useful for evaluation")
    else:
        print("  pass  no empty masks")

    if nearly_full:
        print(f"  FAIL  {len(nearly_full)} mask(s) are over half foreground, e.g. {nearly_full[0][0]}")
        ok = False
    else:
        print("  pass  no suspiciously full masks")

    if mismatched:
        print(f"  FAIL  {len(mismatched)} mask(s) differ in size from their image, e.g. {mismatched[0]}")
        ok = False
    else:
        print("  pass  every mask matches its image size")

    if non_binary:
        worst = max(non_binary, key=lambda t: t[1])
        print(
            f"  warn  {len(non_binary)} mask(s) have grey values, worst {worst[0]} "
            f"({worst[1]:.2%} of pixels). Thresholding handles it, but check they are not JPEGs."
        )
    else:
        print("  pass  masks are binary")

    dupes = {h: names for h, names in hashes.items() if len(names) > 1}
    if dupes:
        first = next(iter(dupes.values()))
        print(f"  FAIL  {len(dupes)} duplicate image group(s), e.g. {first[:2]} — these leak across splits")
        ok = False
    else:
        print("  pass  no duplicate images")

    if len(sizes) > 1:
        common = sizes.most_common(3)
        print(f"  note  {len(sizes)} distinct image sizes; most common {common}")

    if sheet_path:
        write_contact_sheet(pairs, sheet_path, sheet_n, thumb)
        print(f"\ncontact sheet: {sheet_path}")
        print("Open it. Masks should sit exactly on the dark lines in the image.")

    print("\n" + ("DATASET LOOKS USABLE" if ok else "DATASET HAS PROBLEMS — fix before training"))
    return ok


def write_contact_sheet(pairs, out: Path, n: int, thumb: int, cols: int = 6):
    """Grid of samples with the mask painted over the image in red."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)
    tiles = []
    for i in idx:
        img_path, mask_path = pairs[int(i)]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (thumb, thumb))
        mask = cv2.resize(mask, (thumb, thumb), interpolation=cv2.INTER_NEAREST)
        painted = img.copy()
        painted[mask > 127] = (0.25 * painted[mask > 127] + 0.75 * np.array([60, 60, 235])).astype(np.uint8)
        cv2.putText(painted, img_path.stem[:12], (4, thumb - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(painted)

    if not tiles:
        return
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r : r + cols]
        while len(row) < cols:
            row.append(np.full((thumb, thumb, 3), 20, np.uint8))
        rows.append(np.hstack(row))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed")
    ap.add_argument("--sheet", default="data/contact_sheet.png",
                    help="where to write the visual check; '' to skip")
    ap.add_argument("--sheet-n", type=int, default=24)
    ap.add_argument("--thumb", type=int, default=180)
    args = ap.parse_args()

    ok = inspect(
        Path(args.root),
        Path(args.sheet) if args.sheet else None,
        args.sheet_n,
        args.thumb,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
