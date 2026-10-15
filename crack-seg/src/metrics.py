"""Evaluation metrics for binary crack segmentation.

WHY PIXEL ACCURACY IS NOT REPORTED AS A HEADLINE NUMBER
-------------------------------------------------------
With ~2% crack pixels, predicting all-background gives ~98% accuracy. It
is computed here only so the report can show how misleading it is.

The metrics that matter:
    IoU (Jaccard)  intersection over union — the standard segmentation score
    Dice (F1)      harmonic mean of precision and recall
    Precision      of the pixels called crack, how many were crack
    Recall         of the actual crack pixels, how many were found

For crack monitoring, recall is the one to protect. A missed crack is a
missed structural signal; a false positive costs a human two seconds.

EMPTY-MASK CONVENTION
---------------------
Some images genuinely contain no crack. For those, IoU and Dice are 0/0.
This module treats a correct empty prediction as a perfect score (1.0) and
a false-positive-on-empty as 0.0, then reports how many such images there
were so the average can be interpreted. Silently averaging NaNs, or
smoothing them to ~1.0, is a common way to accidentally inflate results.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = ["binarise", "confusion_counts", "MetricTracker", "SegMetrics"]


def binarise(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Logits -> hard 0/1 predictions."""
    return (torch.sigmoid(logits) > threshold).to(torch.float32)


def confusion_counts(preds: torch.Tensor, targets: torch.Tensor):
    """Per-sample TP, FP, FN, TN over flattened spatial dimensions.

    Args:
        preds:   (N, 1, H, W) binary predictions.
        targets: (N, 1, H, W) binary ground truth.

    Returns:
        Four tensors of shape (N,).
    """
    n = preds.size(0)
    p = preds.reshape(n, -1)
    t = targets.reshape(n, -1).to(p.dtype)
    tp = (p * t).sum(dim=1)
    fp = (p * (1 - t)).sum(dim=1)
    fn = ((1 - p) * t).sum(dim=1)
    tn = ((1 - p) * (1 - t)).sum(dim=1)
    return tp, fp, fn, tn


@dataclass
class SegMetrics:
    """Aggregated results over a dataset."""

    iou: float = 0.0
    dice: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    pixel_accuracy: float = 0.0
    n_images: int = 0
    n_empty_targets: int = 0
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "iou": self.iou,
            "dice": self.dice,
            "precision": self.precision,
            "recall": self.recall,
            "pixel_accuracy": self.pixel_accuracy,
            "n_images": self.n_images,
            "n_empty_targets": self.n_empty_targets,
        }
        d.update(self.extras)
        return d

    def __str__(self) -> str:
        return (
            f"IoU {self.iou:.4f} | Dice {self.dice:.4f} | "
            f"P {self.precision:.4f} | R {self.recall:.4f} | "
            f"PixAcc {self.pixel_accuracy:.4f} "
            f"(n={self.n_images}, empty={self.n_empty_targets})"
        )


class MetricTracker:
    """Accumulates per-image metrics across batches.

    Averaging per image (macro) rather than pooling all pixels (micro) is
    the right choice here: pooling lets a handful of heavily-cracked images
    dominate the score.

    Usage:
        tracker = MetricTracker(threshold=0.5)
        for images, masks in loader:
            logits = model(images)
            tracker.update(logits, masks)
        print(tracker.compute())
    """

    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        self._iou: list[float] = []
        self._dice: list[float] = []
        self._prec: list[float] = []
        self._rec: list[float] = []
        self._acc: list[float] = []
        self._n_empty = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = binarise(logits, self.threshold)
        tp, fp, fn, tn = confusion_counts(preds, targets)

        target_empty = (tp + fn) == 0
        pred_empty = (tp + fp) == 0
        both_empty = target_empty & pred_empty

        union = tp + fp + fn
        iou = torch.where(both_empty, torch.ones_like(tp), tp / (union + self.eps))
        dice = torch.where(
            both_empty, torch.ones_like(tp), (2 * tp) / (2 * tp + fp + fn + self.eps)
        )
        # An empty prediction on an empty target is perfectly precise;
        # an empty prediction on a non-empty target has zero recall.
        precision = torch.where(
            pred_empty,
            target_empty.to(tp.dtype),
            tp / (tp + fp + self.eps),
        )
        recall = torch.where(
            target_empty,
            pred_empty.to(tp.dtype),
            tp / (tp + fn + self.eps),
        )
        accuracy = (tp + tn) / (tp + tn + fp + fn + self.eps)

        self._iou += iou.tolist()
        self._dice += dice.tolist()
        self._prec += precision.tolist()
        self._rec += recall.tolist()
        self._acc += accuracy.tolist()
        self._n_empty += int(target_empty.sum().item())

    def compute(self) -> SegMetrics:
        if not self._iou:
            return SegMetrics()
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        return SegMetrics(
            iou=mean(self._iou),
            dice=mean(self._dice),
            precision=mean(self._prec),
            recall=mean(self._rec),
            pixel_accuracy=mean(self._acc),
            n_images=len(self._iou),
            n_empty_targets=self._n_empty,
        )


# Decision thresholds swept on validation. Steps of 0.05, plus a fine tail up
# to 0.99: overlap-based losses (dice, tversky, focal_tversky) produce
# confident, slightly over-wide masks, and on DeepCrack their validation IoU
# was still rising at 0.95, the old upper limit.
THRESHOLD_GRID: tuple[float, ...] = tuple(i / 20 for i in range(1, 20)) + (
    0.96, 0.97, 0.98, 0.99,
)


@torch.no_grad()
def sweep_threshold(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thresholds: tuple[float, ...] = THRESHOLD_GRID,
) -> list[tuple[float, SegMetrics]]:
    """Evaluate at a range of thresholds.

    The default 0.5 is rarely optimal for imbalanced segmentation. Lowering
    it trades precision for recall, and the resulting curve is a good figure
    for the report.
    """
    out = []
    for t in thresholds:
        tracker = MetricTracker(threshold=t)
        tracker.update(logits, targets)
        out.append((t, tracker.compute()))
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    N, H, W = 6, 64, 64

    targets = torch.zeros(N, 1, H, W)
    targets[:4, :, 30:32, :] = 1.0          # four images with a crack
    # images 4 and 5 are deliberately empty

    perfect = torch.where(targets > 0.5, 8.0, -8.0)
    empty = torch.full((N, 1, H, W), -8.0)

    print("PERFECT PREDICTION")
    t = MetricTracker(); t.update(perfect, targets); print(" ", t.compute())

    print("\nALL-EMPTY PREDICTION  (the degenerate model)")
    t = MetricTracker(); t.update(empty, targets); print(" ", t.compute())
    print(
        "  note the pixel accuracy: high, while IoU and recall correctly collapse.\n"
        "  This is why pixel accuracy is not a headline metric."
    )

    print("\nTHRESHOLD SWEEP on a noisy prediction")
    noisy = torch.where(targets > 0.5, 1.2, -1.2) + torch.randn(N, 1, H, W) * 1.5
    for th, m in sweep_threshold(noisy, targets, (0.2, 0.35, 0.5, 0.65, 0.8)):
        print(f"  t={th:.2f}  IoU {m.iou:.4f}  P {m.precision:.4f}  R {m.recall:.4f}")
