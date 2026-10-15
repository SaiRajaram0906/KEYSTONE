"""Run a trained model on images and write overlays plus measurements.

    # one image
    python -m src.predict --checkpoint runs/default/best.pt --input wall.jpg

    # a folder, with a scale so widths come out in millimetres
    python -m src.predict --checkpoint runs/default/best.pt \
        --input captures/2026-03/ --outdir preds/ --mm-per-px 0.8

    # compare two aligned captures of the same surface
    python -m src.predict --checkpoint runs/default/best.pt \
        --input captures/2026-09/north.jpg \
        --compare-to captures/2026-03/north.jpg --mm-per-px 0.8

Overlays are written as side-by-side panels: original, predicted mask,
and the mask painted over the original. Looking at these is worth more
than any single metric — systematic failures (shadows read as cracks,
mortar joints read as cracks) are obvious in the overlays and invisible
in the IoU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from keystone.interface import CrackSegmenter, compare_captures, measure_mask

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def make_overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Three-panel BGR image: original | mask | overlay."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    mask_vis = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    painted = bgr.copy()
    painted[mask > 0] = (0.35 * painted[mask > 0] + 0.65 * np.array([60, 60, 235])).astype(
        np.uint8
    )

    gap = np.full((bgr.shape[0], 8, 3), 30, np.uint8)
    return np.hstack([bgr, gap, mask_vis, gap, painted])


def gather_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise RuntimeError(f"no images found in {path}")
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="image file or directory")
    ap.add_argument("--outdir", default="predictions")
    ap.add_argument("--mm-per-px", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the checkpoint's tuned threshold")
    ap.add_argument("--compare-to", default=None,
                    help="an earlier, aligned capture of the same surface")
    ap.add_argument("--no-overlay", action="store_true")
    args = ap.parse_args()

    seg = CrackSegmenter(args.checkpoint)
    if args.threshold is not None:
        seg.threshold = args.threshold
    print(f"[model] threshold {seg.threshold:.2f}, input size {seg.image_size}")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- comparison mode -------------------------------------------------
    if args.compare_to:
        later_path, earlier_path = Path(args.input), Path(args.compare_to)
        m_later = seg.segment(later_path)
        m_earlier = seg.segment(earlier_path)

        if m_later.shape != m_earlier.shape:
            m_earlier = cv2.resize(
                m_earlier, (m_later.shape[1], m_later.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            print("[warn] captures differed in size; the earlier one was resized.")
            print("       In KEYSTONE both captures are registered to the same")
            print("       control points, so this path should not be needed.")

        delta = compare_captures(m_earlier, m_later, mm_per_px=args.mm_per_px)
        print(f"\nearlier : {delta.earlier}")
        print(f"later   : {delta.later}")
        print(f"change  : {delta}")

        with open(out_dir / "comparison.json", "w") as f:
            json.dump(delta.as_dict(), f, indent=2)

        # Green where crack was present before, red where it is new.
        vis = cv2.imread(str(later_path), cv2.IMREAD_COLOR)
        vis[(m_earlier > 0)] = (0, 200, 0)
        vis[(m_later > 0) & (m_earlier == 0)] = (0, 0, 235)
        cv2.imwrite(str(out_dir / "change_map.png"), vis)
        print(f"\nwritten to {out_dir}/comparison.json and change_map.png")
        return

    # ---- batch mode ------------------------------------------------------
    files = gather_inputs(Path(args.input))
    records = []
    for p in files:
        rgb = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = seg.segment(rgb)
        stats = measure_mask(mask, mm_per_px=args.mm_per_px)

        cv2.imwrite(str(out_dir / f"{p.stem}_mask.png"), (mask * 255).astype(np.uint8))
        if not args.no_overlay:
            cv2.imwrite(str(out_dir / f"{p.stem}_overlay.png"), make_overlay(rgb, mask))

        records.append({"image": p.name, **stats.as_dict()})
        print(f"{p.name:<32} {stats}")

    with open(out_dir / "measurements.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n{len(files)} image(s) processed; results in {out_dir}")


if __name__ == "__main__":
    main()
