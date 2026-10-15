"""Training loop for crack segmentation.

    python -m src.train --config configs/default.yaml

Selects the best checkpoint on validation IoU, not on validation loss.
Loss and IoU disagree more often than people expect on imbalanced data,
and IoU is the number that goes in the report.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import (
    CrackDataset,
    compute_pos_weight,
    discover_pairs,
    split_pairs,
)
from src.data.transforms import build_transforms
from src.losses import build_loss
from src.metrics import MetricTracker
from src.models.unet import build_model


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(cfg: dict):
    d = cfg["data"]
    pairs = discover_pairs(d["root"])
    splits = split_pairs(
        pairs, val_frac=d.get("val_frac", 0.15), test_frac=d.get("test_frac", 0.15)
    )
    train_tf, eval_tf = build_transforms(cfg)

    common = dict(
        num_workers=d.get("num_workers", 2),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=d.get("num_workers", 2) > 0,
    )
    train_loader = DataLoader(
        CrackDataset(splits["train"], train_tf),
        batch_size=d.get("batch_size", 8),
        shuffle=True,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        CrackDataset(splits["val"], eval_tf),
        batch_size=d.get("batch_size", 8),
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, splits


def run_epoch(model, loader, loss_fn, device, optimiser=None, scaler=None, threshold=0.5):
    """One pass. Training if optimiser is given, evaluation otherwise."""
    training = optimiser is not None
    model.train(training)
    tracker = MetricTracker(threshold=threshold)
    total_loss, n_batches = 0.0, 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            if scaler is not None and training:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    logits = model(images)
                    loss = loss_fn(logits, masks)
                optimiser.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                logits = model(images)
                loss = loss_fn(logits, masks)
                if training:
                    optimiser.zero_grad(set_to_none=True)
                    loss.backward()
                    optimiser.step()

        tracker.update(logits.detach().float(), masks)
        total_loss += float(loss.detach())
        n_batches += 1

    return total_loss / max(n_batches, 1), tracker.compute()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None, help="override output directory")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device(
        cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = Path(args.out or cfg.get("out_dir", "runs/default"))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, splits = build_loaders(cfg)
    print(
        f"[data] train {len(splits['train'])} | val {len(splits['val'])} | "
        f"test {len(splits['test'])}"
    )

    # Estimate class imbalance from the training split only — computing it
    # over everything would leak test statistics into training.
    if cfg.get("loss", {}).get("pos_weight") == "auto":
        pw = compute_pos_weight(splits["train"])
        cfg["loss"]["pos_weight"] = pw
        print(f"[loss] auto pos_weight = {pw:.1f}")

    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg).to(device)

    t = cfg.get("train", {})
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=t.get("lr", 3e-4),
        weight_decay=t.get("weight_decay", 1e-4),
    )
    epochs = t.get("epochs", 40)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    scaler = (
        torch.amp.GradScaler(device.type)
        if (t.get("amp", True) and device.type == "cuda")
        else None
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n_params:,} trainable parameters on {device}")

    best_iou, best_epoch, patience = -1.0, -1, t.get("early_stop_patience", 12)
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_m = run_epoch(model, train_loader, loss_fn, device, optimiser, scaler)
        va_loss, va_m = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "train_iou": tr_m.iou,
                "val_iou": va_m.iou,
                "val_dice": va_m.dice,
                "val_precision": va_m.precision,
                "val_recall": va_m.recall,
                "lr": scheduler.get_last_lr()[0],
            }
        )

        marker = ""
        if va_m.iou > best_iou:
            best_iou, best_epoch, marker = va_m.iou, epoch, "  <- best"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_iou": va_m.iou,
                },
                out_dir / "best.pt",
            )

        print(
            f"epoch {epoch:3d}/{epochs}  "
            f"train {tr_loss:.4f}  val {va_loss:.4f}  "
            f"val IoU {va_m.iou:.4f}  Dice {va_m.dice:.4f}  "
            f"P {va_m.precision:.3f}  R {va_m.recall:.3f}  "
            f"({time.time() - t0:.0f}s){marker}"
        )

        if epoch - best_epoch >= patience:
            print(f"[early stop] no improvement for {patience} epochs")
            break

    torch.save({"model_state": model.state_dict(), "config": cfg}, out_dir / "last.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nbest val IoU {best_iou:.4f} at epoch {best_epoch}")
    print(f"checkpoints and history written to {out_dir}")


if __name__ == "__main__":
    main()
