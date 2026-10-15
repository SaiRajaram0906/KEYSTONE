"""Workstream C: score a checkpoint on an EXTERNAL labelled dataset (domain gap).

The model was trained and tested on DeepCrack (road / concrete). KEYSTONE's
target is masonry, where mortar joints look like cracks. This script puts a
number on that gap, scored exactly the way src/evaluate.py scores the
DeepCrack test split (same transform, same per-image metrics, the
checkpoint's own validation-tuned threshold, no re-tuning on the new data).

    python scripts/eval_external.py --checkpoint runs/loss_focal_tversky/best.pt \
        --images <images folder> --masks <masks folder> --name masonry

It also reports what IoU alone hides on a dataset with crack-free images:
the FALSE-ALARM RATE, i.e. the share of images with no crack in the ground
truth where the model still marks crack pixels.

Steps:
  1. Pair images and masks by filename, normalise masks to 0/255 (handles
     0/1 and 0/255 label files), write them to data/<name>/images + masks.
  2. Evaluate the checkpoint on all pairs.
  3. Write runs/external_<name>/<run>/metrics.json and per_image.csv
     (worst images first: open their overlays from src.predict).

Read-only for src/ and keystone/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.prepare_data import IMG_EXTS, _find_mask
from src.data.dataset import CrackDataset
from src.data.transforms import eval_transform
from src.evaluate import collect_logits, metrics_at
from src.metrics import MetricTracker
from src.models.unet import build_model


def normalise_mask(m: np.ndarray) -> np.ndarray:
    """Any label file -> uint8 0/255. 0/1 files are scaled, 0/255 thresholded at 127."""
    if m.max() <= 1:
        return (m > 0).astype(np.uint8) * 255
    return (m > 127).astype(np.uint8) * 255


def build_pairs(images: Path, masks: Path, out: Path):
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    pairs, missing, inverted = [], 0, []
    for p in sorted(images.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        mp = _find_mask(p.stem, masks)
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp else None
        if img is None or m is None:
            missing += 1
            continue
        m = normalise_mask(m)
        if m.shape != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if (m > 0).mean() > 0.5:
            inverted.append(p.name)
        ip, mo = out / "images" / f"{p.stem}.png", out / "masks" / f"{p.stem}.png"
        cv2.imwrite(str(ip), img)
        cv2.imwrite(str(mo), m)
        pairs.append((ip, mo))
    return pairs, missing, inverted


@torch.no_grad()
def per_image(logits, targets, threshold):
    rows = []
    for i in range(logits.size(0)):
        t = MetricTracker(threshold=threshold)
        t.update(logits[i : i + 1], targets[i : i + 1])
        m = t.compute()
        pred_px = int((torch.sigmoid(logits[i]) > threshold).sum())
        gt_px = int(targets[i].sum())
        rows.append({"iou": m.iou, "dice": m.dice, "precision": m.precision,
                     "recall": m.recall, "gt_crack_px": gt_px, "pred_crack_px": pred_px})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True, help="folder of images")
    ap.add_argument("--masks", required=True, help="folder of ground-truth masks")
    ap.add_argument("--name", default="external", help="short dataset name, e.g. masonry")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args(argv)

    ckpt_path = Path(args.checkpoint)
    out_data = Path("data") / args.name
    pairs, missing, inverted = build_pairs(Path(args.images), Path(args.masks), out_data)
    if not pairs:
        raise SystemExit("no image/mask pairs matched — check --images and --masks")

    fr = [float((cv2.imread(str(m), cv2.IMREAD_GRAYSCALE) > 127).mean()) for _, m in pairs]
    n_empty = sum(f == 0 for f in fr)
    print(f"{len(pairs)} pairs written to {out_data}"
          + (f"  ({missing} images had no readable mask, skipped)" if missing else ""))
    print(f"crack pixels: {np.mean(fr):.3%} of total | images with no crack: {n_empty}")
    if inverted:
        print(f"[warn] {len(inverted)} masks are >50% crack — inverted labels? e.g. {inverted[:3]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    size = int(cfg.get("data", {}).get("image_size", 320))
    threshold = float(ckpt.get("best_threshold", 0.5))
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    loader = DataLoader(CrackDataset(pairs, eval_transform(size)),
                        batch_size=args.batch_size, shuffle=False)
    logits, targets = collect_logits(model, loader, device)

    overall = metrics_at(logits, targets, threshold)
    rows = per_image(logits, targets, threshold)
    for r, (ip, _) in zip(rows, pairs):
        r["image"] = ip.name

    has = torch.tensor([r["gt_crack_px"] > 0 for r in rows])
    crack_m = metrics_at(logits[has], targets[has], threshold) if has.any() else None
    empty_rows = [r for r in rows if r["gt_crack_px"] == 0]
    false_alarms = [r for r in empty_rows if r["pred_crack_px"] > 0]

    print(f"\nEXTERNAL '{args.name}'  checkpoint {ckpt_path}  threshold {threshold:.3f} "
          f"(tuned on DeepCrack val, not re-tuned)  input {size}x{size}")
    print(f"  all images ({len(rows)})          {overall}")
    if crack_m is not None:
        print(f"  images with a crack ({int(has.sum())})  {crack_m}")
    if empty_rows:
        print(f"  crack-free images: {len(empty_rows)}, model marked cracks on "
              f"{len(false_alarms)} ({len(false_alarms) / len(empty_rows):.0%})  <- false-alarm rate")

    rows.sort(key=lambda r: (r["gt_crack_px"] == 0, r["iou"]))
    worst = [r for r in rows if r["gt_crack_px"] > 0][:8]
    if worst:
        print("\n  worst images that DO contain a crack (open their overlays):")
        for r in worst:
            print(f"    {r['image']:<22} IoU {r['iou']:.3f}  P {r['precision']:.3f}  R {r['recall']:.3f}")
    if false_alarms:
        fa = sorted(false_alarms, key=lambda r: -r["pred_crack_px"])[:8]
        print("\n  biggest false alarms on crack-free images:")
        for r in fa:
            print(f"    {r['image']:<22} {r['pred_crack_px']} predicted crack px (model resolution)")

    out = Path("runs") / f"external_{args.name}" / ckpt_path.parent.name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "iou", "dice", "precision", "recall",
                                          "gt_crack_px", "pred_crack_px"])
        w.writeheader()
        w.writerows(rows)
    summary = {
        "dataset": args.name, "checkpoint": str(ckpt_path), "threshold": threshold,
        "n_images": len(rows), "n_crack_free": len(empty_rows),
        "all_images": overall.as_dict(),
        "images_with_crack": crack_m.as_dict() if crack_m is not None else None,
        "false_alarm_rate": (len(false_alarms) / len(empty_rows)) if empty_rows else None,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n-> {out / 'metrics.json'}  and  {out / 'per_image.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
