"""Workstream C: is the width/length measurement right on REAL model output?

keystone/interface.py's self-test checks the measurement on synthetic masks
of known width. That proves the skeleton + distance-transform maths, but
never runs the model. This script closes that gap on the test split:

  for every test image
    GT        measure the ground-truth mask                (the truth)
    MODEL     segment with the checkpoint, measure the mask (what KEYSTONE gets)
    CEILING   ground truth pushed through the model's input grid and back
              (area-downsample to image_size x image_size, linear upsample,
              threshold 0.5) -- what a PERFECT model at this input size would
              give. MODEL vs GT error that CEILING also shows is caused by the
              resize, not by the network.

and reports, in pixels, how far MODEL and CEILING are from GT for crack width
(mean / p95 / max), total length and number of cracks.

    python scripts/measure_vs_gt.py --checkpoint runs/loss_focal_tversky/best.pt

Writes runs/measure_vs_gt/<run name>/per_image.csv and summary.md.
Read-only: changes nothing in src/ or keystone/.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from keystone.interface import CrackSegmenter, measure_mask
from src.data.dataset import discover_pairs, split_pairs

FIELDS = ("mean_width", "p95_width", "max_width", "total_length", "n_components")
THIN_PX = 3.0  # GT mean width below this counts as a "thin" crack


def through_grid(gt: np.ndarray, size: int) -> np.ndarray:
    """GT mask as a perfect model working at size x size would return it."""
    h, w = gt.shape
    small = cv2.resize(gt.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return (back > 0.5).astype(np.uint8)


def stats(mask: np.ndarray, min_px: int) -> dict:
    s = measure_mask(mask, mm_per_px=None, min_component_px=min_px)
    return {f: float(getattr(s, f)) for f in FIELDS}


def summarise(rows: list[dict], who: str) -> list[dict]:
    out = []
    for f in FIELDS:
        err = np.array([r[f"{who}_{f}"] - r[f"gt_{f}"] for r in rows], dtype=float)
        gt = np.array([r[f"gt_{f}"] for r in rows], dtype=float)
        rel = np.abs(err) / np.maximum(gt, 1e-9)
        out.append({
            "source": who, "quantity": f, "n": len(err),
            "median_signed_err": float(np.median(err)),
            "median_abs_err": float(np.median(np.abs(err))),
            "mean_abs_err": float(np.mean(np.abs(err))),
            "median_rel_err_pct": float(100 * np.median(rel)),
            "within_1px_pct": float(100 * np.mean(np.abs(err) <= 1.0)),
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="data/processed")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--min-component", type=int, default=20,
                    help="ignore blobs smaller than this (px), same default as measure_mask")
    ap.add_argument("--out", default=None, help="default: runs/measure_vs_gt/<run name>")
    args = ap.parse_args(argv)

    ckpt = Path(args.checkpoint)
    seg = CrackSegmenter(ckpt)
    import torch
    cfg = torch.load(str(ckpt), map_location="cpu", weights_only=False).get("config", {})
    d = cfg.get("data", {})
    splits = split_pairs(discover_pairs(args.data_root),
                         float(d.get("val_frac", 0.15)), float(d.get("test_frac", 0.15)))
    pairs = splits[args.split]
    print(f"checkpoint {ckpt}  threshold {seg.threshold:.2f}  model input "
          f"{seg.image_size}x{seg.image_size}  |  {len(pairs)} {args.split} images\n")

    rows = []
    for i, (img_p, mask_p) in enumerate(pairs, 1):
        gt = (cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        pred = seg.segment(str(img_p))
        ceil_ = through_grid(gt, seg.image_size)
        row = {"image": img_p.name, "h": gt.shape[0], "w": gt.shape[1]}
        for who, m in (("gt", gt), ("model", pred), ("ceiling", ceil_)):
            for k, v in stats(m, args.min_component).items():
                row[f"{who}_{k}"] = v
        rows.append(row)
        if i % 10 == 0 or i == len(pairs):
            print(f"  {i}/{len(pairs)}")

    out = Path(args.out or Path("runs/measure_vs_gt") / ckpt.parent.name)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    thin = [r for r in rows if r["gt_mean_width"] < THIN_PX]
    wide = [r for r in rows if r["gt_mean_width"] >= THIN_PX]
    gt_w = np.array([r["gt_mean_width"] for r in rows])

    lines = [
        f"# Measurement vs ground truth: {ckpt.parent.name}",
        "",
        f"{len(rows)} {args.split} images, threshold {seg.threshold:.2f}, model input "
        f"{seg.image_size}x{seg.image_size}. All values in pixels of the original image.",
        f"GT mean crack width: median {np.median(gt_w):.2f} px, range "
        f"{gt_w.min():.2f}-{gt_w.max():.2f} px; {len(thin)} images thin (< {THIN_PX:g} px), "
        f"{len(wide)} wider.",
        "",
        "Signed error = measured - GT (positive = reads too wide / too long / too many).",
        "CEILING = perfect model at this input size; error it shares with MODEL comes from the resize.",
        "",
    ]
    for title, subset in (("All images", rows), (f"Thin cracks (GT mean width < {THIN_PX:g} px)", thin),
                          (f"Wider cracks (>= {THIN_PX:g} px)", wide)):
        if not subset:
            continue
        lines += [f"## {title} (n={len(subset)})", "",
                  "| quantity | source | median signed | median abs | within 1 px | median rel % |",
                  "|---|---|---|---|---|---|"]
        for f in FIELDS:
            for s in (x for x in summarise(subset, "model") + summarise(subset, "ceiling")
                      if x["quantity"] == f):
                lines.append(f"| {f} | {s['source']} | {s['median_signed_err']:+.2f} | "
                             f"{s['median_abs_err']:.2f} | {s['within_1px_pct']:.0f}% | "
                             f"{s['median_rel_err_pct']:.0f}% |")
        lines.append("")

    text = "\n".join(lines)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"-> {out / 'summary.md'}  and  {out / 'per_image.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
