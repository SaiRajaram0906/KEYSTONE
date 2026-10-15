"""Loss functions for crack segmentation.

THE CENTRAL PROBLEM
-------------------
Cracks occupy roughly 1-5% of the pixels in a typical image. A model
trained with plain binary cross-entropy can reach ~97% pixel accuracy by
predicting "no crack" everywhere, and the loss will look excellent while
the model is useless.

Every loss here exists to handle that imbalance, and comparing them is the
single most informative experiment in this project:

    BCE          baseline; expect it to under-segment badly
    Dice         optimises overlap directly, so empty predictions score 0
    BCE + Dice   the usual default; stable early, overlap-aware later
    Focal        down-weights easy background pixels
    Tversky      Dice with asymmetric penalties; beta > alpha favours recall
    Focal-Tversky  Tversky raised to a power, focusing on hard examples

All losses take RAW LOGITS, not probabilities. The model returns logits and
the sigmoid is applied inside each loss, which is both numerically safer
(log-sum-exp trick in BCEWithLogits) and avoids double-sigmoid bugs.

Shapes throughout: logits and targets are (N, 1, H, W); targets are float
tensors containing 0.0 or 1.0.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DiceLoss",
    "BCEDiceLoss",
    "FocalLoss",
    "TverskyLoss",
    "FocalTverskyLoss",
    "build_loss",
]


def _flatten(logits: torch.Tensor, targets: torch.Tensor):
    """Return per-sample flattened probabilities and targets."""
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} and targets {tuple(targets.shape)} "
            "must have the same shape"
        )
    probs = torch.sigmoid(logits)
    n = logits.size(0)
    return probs.reshape(n, -1), targets.reshape(n, -1).to(probs.dtype)


class DiceLoss(nn.Module):
    """1 - Dice coefficient, computed per sample then averaged.

    Dice = 2|X ∩ Y| / (|X| + |Y|). Because it is a ratio of overlap to
    total area, predicting all-background gives Dice 0 rather than a good
    score, which is exactly the behaviour BCE lacks.

    The `smooth` term keeps the gradient finite when an image genuinely
    contains no crack; without it the loss is 0/0.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs, tgts = _flatten(logits, targets)
        intersection = (probs * tgts).sum(dim=1)
        denom = probs.sum(dim=1) + tgts.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Weighted sum of BCE-with-logits and Dice.

    This is the sensible default. BCE gives dense, well-behaved gradients
    early in training when the Dice term is near-degenerate; Dice takes
    over once predictions are non-trivial.

    `pos_weight` multiplies the positive (crack) term inside BCE. Setting
    it to roughly (background pixels / crack pixels) counteracts the
    imbalance. scripts/prepare_data.py prints the ratio for your dataset.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float | None = None,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss(smooth=smooth)
        # Registered as a buffer so .to(device) moves it with the module.
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.to(logits.dtype), pos_weight=self.pos_weight
        )
        return self.bce_weight * bce + self.dice_weight * self.dice(logits, targets)


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017).

    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)

    The (1 - p_t)^gamma factor shrinks the contribution of pixels the model
    already classifies confidently. With gamma=2 an easy background pixel
    at p=0.99 contributes 10,000x less than one at p=0.5, so the loss is
    dominated by the crack boundary rather than the acres of empty wall.

    alpha additionally weights the positive class. alpha=0.75 is a
    reasonable starting point for crack data.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # p_t = p if target == 1 else 1 - p, recovered without a second sigmoid.
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        return loss.mean()


class TverskyLoss(nn.Module):
    """Tversky loss (Salehi et al., 2017): Dice with asymmetric penalties.

    Tversky = TP / (TP + alpha*FP + beta*FN)

    alpha == beta == 0.5 reduces exactly to Dice. Raising beta punishes
    false negatives harder, which trades precision for recall — usually the
    right trade for thin cracks, where missing a crack matters more than a
    few false positives that a later filter can remove.

    Suggested starting point: alpha=0.3, beta=0.7.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def _tversky_index(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs, tgts = _flatten(logits, targets)
        tp = (probs * tgts).sum(dim=1)
        fp = (probs * (1.0 - tgts)).sum(dim=1)
        fn = ((1.0 - probs) * tgts).sum(dim=1)
        return (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._tversky_index(logits, targets).mean()


class FocalTverskyLoss(TverskyLoss):
    """Focal Tversky (Abraham & Khan, 2019): (1 - Tversky)^gamma.

    gamma > 1 concentrates learning on samples with a poor Tversky index,
    i.e. the images the model is currently failing on. gamma=0.75 is the
    value reported in the paper; gamma=1 reduces to plain Tversky.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1.0,
    ):
        super().__init__(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ti = self._tversky_index(logits, targets)
        return (1.0 - ti).pow(self.gamma).mean()


def build_loss(cfg: dict) -> nn.Module:
    """Construct a loss from a config dict.

    configs/default.yaml:
        loss:
          name: bce_dice        # bce | dice | bce_dice | focal | tversky | focal_tversky
          pos_weight: null      # float, or null to disable
          bce_weight: 0.5
          dice_weight: 0.5
          alpha: 0.3
          beta: 0.7
          gamma: 2.0
    """
    c = cfg.get("loss", {})
    name = str(c.get("name", "bce_dice")).lower()

    if name == "bce":
        pw = c.get("pos_weight")
        pos_weight = torch.tensor([float(pw)]) if pw is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "dice":
        return DiceLoss(smooth=c.get("smooth", 1.0))
    if name == "bce_dice":
        return BCEDiceLoss(
            bce_weight=c.get("bce_weight", 0.5),
            dice_weight=c.get("dice_weight", 0.5),
            pos_weight=c.get("pos_weight"),
            smooth=c.get("smooth", 1.0),
        )
    if name == "focal":
        return FocalLoss(alpha=c.get("alpha", 0.75), gamma=c.get("gamma", 2.0))
    if name == "tversky":
        return TverskyLoss(
            alpha=c.get("alpha", 0.3),
            beta=c.get("beta", 0.7),
            smooth=c.get("smooth", 1.0),
        )
    if name == "focal_tversky":
        return FocalTverskyLoss(
            alpha=c.get("alpha", 0.3),
            beta=c.get("beta", 0.7),
            gamma=c.get("gamma", 0.75),
            smooth=c.get("smooth", 1.0),
        )
    raise ValueError(f"unknown loss '{name}'")


if __name__ == "__main__":
    torch.manual_seed(0)
    N, H, W = 4, 64, 64

    # A target with ~2% positive pixels, mimicking real crack sparsity.
    targets = torch.zeros(N, 1, H, W)
    targets[:, :, 30:32, :] = 1.0
    frac = targets.mean().item()

    perfect = torch.where(targets > 0.5, 10.0, -10.0)   # confident and correct
    empty = torch.full((N, 1, H, W), -10.0)             # predicts no crack at all
    random_logits = torch.randn(N, 1, H, W)

    print(f"positive pixel fraction: {frac:.3%}\n")
    header = f"{'loss':<16}{'perfect':>10}{'all-empty':>12}{'random':>10}"
    print(header)
    print("-" * len(header))

    for name in ("bce", "dice", "bce_dice", "focal", "tversky", "focal_tversky"):
        fn = build_loss({"loss": {"name": name}})
        vals = [fn(x, targets).item() for x in (perfect, empty, random_logits)]
        print(f"{name:<16}{vals[0]:>10.4f}{vals[1]:>12.4f}{vals[2]:>10.4f}")

    print(
        "\nThe all-empty column is the one that matters: BCE barely penalises a\n"
        "model that predicts nothing, while the overlap-based losses score it ~1.0."
    )
