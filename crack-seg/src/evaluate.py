"""Evaluate a checkpoint on the held-out test split.

    python -m src.evaluate --checkpoint runs/default/best.pt

Does two things worth understanding:

1. TUNES THE DECISION THRESHOLD ON VALIDATION, NOT TEST.
   0.5 is rarely optimal for imbalanced segmentation. The sweep runs on
   the validation split, the winning threshold is applied once to test,
   and it is written back into the checkpoint so CrackSegmenter picks it
   up automatically. Tuning on test and then reporting test would be
   leakage, and it is an easy mistake to make.

2. REPORTS THE DEGENERATE BASELINE ALONGSIDE THE MODEL.
   An all-background prediction scores ~98% pixel accuracy. Printing that
   next to the model's numbers makes the comparison honest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import CrackDataset, discover_pairs, split_pairs
from src.data.transforms import eval_transform
from src.metrics import THRESHOLD_GRID, MetricTracker
from src.models.unet import build_model


@torch.no_grad()
def collect_logits(model, loader, device):
    """Run the model over a loader, returning stacked logits and targets."""
    model.eval()
    logits_all, targets_all = [], []
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        logits_all.append(model(images).float().cpu())
        targets_all.append(masks.cpu())
    if not logits_all:
        raise RuntimeError("loader produced no batches — is the split empty?")
    return torch.cat(logits_all), torch.cat(targets_all)


def metrics_at(logits, targets, threshold: float):
    tracker = MetricTracker(threshold=threshold)
    # Chunked so a large test set does not spike memory.
    for i in range(0, logits.size(0), 16):
        tracker.update(logits[i : i + 16], targets[i : i + 16])
    return tracker.compute()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default=None, help="override config data root")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--no-tune",
        action="store_true",
        help="skip the validation sweep and evaluate at 0.5",
    )
    ap.add_argument(
        "--thresholds",
        default=None,
        help="comma-separated thresholds to sweep on validation "
        "(default: 0.05-0.95 in 0.05 steps, then 0.96-0.99)",
    )
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    root = args.data_root or cfg.get("data", {}).get("root", "data/processed")
    size = int(cfg.get("data", {}).get("image_size", 320))

    pairs = discover_pairs(root)
    splits = split_pairs(
        pairs,
        val_frac=cfg.get("data", {}).get("val_frac", 0.15),
        test_frac=cfg.get("data", {}).get("test_frac", 0.15),
    )
    tf = eval_transform(size)
    mk = lambda items: DataLoader(  # noqa: E731
        CrackDataset(items, tf), batch_size=args.batch_size, shuffle=False
    )

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[model] loaded {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

    # ---- threshold selection on validation -------------------------------
    threshold = 0.5
    if not args.no_tune:
        v_logits, v_targets = collect_logits(model, mk(splits["val"]), device)
        print(f"\nTHRESHOLD SWEEP on validation (n={v_logits.size(0)})")
        print(f"{'thr':>6}{'IoU':>9}{'Dice':>9}{'prec':>9}{'recall':>9}")
        grid = (
            tuple(float(x) for x in args.thresholds.split(","))
            if args.thresholds
            else THRESHOLD_GRID
        )
        best = (-1.0, 0.5)
        for t in grid:
            m = metrics_at(v_logits, v_targets, t)
            star = ""
            if m.iou > best[0]:
                best, star = (m.iou, t), "  *"
            print(
                f"{t:>6.2f}{m.iou:>9.4f}{m.dice:>9.4f}"
                f"{m.precision:>9.4f}{m.recall:>9.4f}{star}"
            )
        threshold = best[1]
        print(f"\nselected threshold {threshold:.2f} (val IoU {best[0]:.4f})")
        if threshold in (min(grid), max(grid)):
            print(
                f"[warn] the best threshold is at the edge of the sweep "
                f"({min(grid):.2f}-{max(grid):.2f}); validation IoU may still be "
                "improving beyond it. Widen --thresholds."
            )

    # ---- single evaluation on test ---------------------------------------
    t_logits, t_targets = collect_logits(model, mk(splits["test"]), device)
    test_m = metrics_at(t_logits, t_targets, threshold)

    # Degenerate baseline: predict background everywhere.
    empty = torch.full_like(t_logits, -20.0)
    base_m = metrics_at(empty, t_targets, threshold)

    print(f"\nTEST RESULTS at threshold {threshold:.2f}  (n={t_logits.size(0)})")
    print(f"  model              {test_m}")
    print(f"  all-background     {base_m}")
    print(
        "\n  The baseline's pixel accuracy is the reason that metric is not\n"
        "  reported as a headline number."
    )

    # Persist the tuned threshold so CrackSegmenter uses it by default.
    ckpt["best_threshold"] = float(threshold)
    ckpt["test_metrics"] = test_m.as_dict()
    torch.save(ckpt, ckpt_path)

    out = ckpt_path.parent / "test_metrics.json"
    with open(out, "w") as f:
        json.dump(
            {
                "threshold": threshold,
                "model": test_m.as_dict(),
                "all_background_baseline": base_m.as_dict(),
            },
            f,
            indent=2,
        )
    print(f"\nthreshold written back to {ckpt_path}")
    print(f"metrics written to {out}")


if __name__ == "__main__":
    main()
