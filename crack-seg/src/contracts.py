"""Frozen interfaces between the three workstreams.

WHY THIS FILE EXISTS
--------------------
Three people building three parts in parallel will only integrate cleanly
if the boundaries between them are written down and checked. This module
is the written-down part; tests/contracts_test.py is the checking part.

Nothing here does real work. It defines the shape of what crosses each
boundary and provides a validator for it, so any workstream can prove it
conforms without waiting for the other two to finish.

THE THREE BOUNDARIES
--------------------

    [A] DATA  ──────────▶  [B] MODEL  ──────────▶  [C] MEASURE
         build_loaders()        build_model()          CrackSegmenter
         yields batches         yields logits          yields masks
                                writes checkpoints     yields measurements

    A -> B   a batch is (images, masks)
             images : float32 (N, 3, H, W), normalised
             masks  : float32 (N, 1, H, W), values in {0.0, 1.0}

    B -> C   model(images) -> logits, float32 (N, 1, H, W), RAW LOGITS
             checkpoint    -> dict with the keys in REQUIRED_CKPT_KEYS

    C -> KEYSTONE
             segment(image)            -> uint8 (H, W), values in {0, 1}
             measure_mask(mask, scale)  -> CrackStats
             compare_captures(a, b, s)  -> CrackDelta

RULES
-----
1. Changing anything in this file requires all three people to agree,
   because it breaks the other two by definition.
2. Everything else is yours. Rewrite the U-Net, swap the augmentation,
   change the loss — as long as the boundary still validates.
3. Run `python tests/contracts_test.py` before every push. If it passes,
   your part will integrate.

THE ONE MISTAKE THIS FILE EXISTS TO PREVENT
-------------------------------------------
The model returns LOGITS, never probabilities. Losses and metrics apply
their own sigmoid. Returning probabilities trains without any error and
silently halves the learning signal, and it is very hard to find later.
validate_logits() below checks for it heuristically.
"""
from __future__ import annotations

__all__ = [
    "REQUIRED_CKPT_KEYS",
    "OPTIONAL_CKPT_KEYS",
    "ContractError",
    "validate_batch",
    "validate_logits",
    "validate_checkpoint",
    "validate_mask",
    "validate_stats",
    "validate_delta",
]

REQUIRED_CKPT_KEYS = ("model_state", "config")
OPTIONAL_CKPT_KEYS = ("epoch", "val_iou", "best_threshold", "test_metrics")


class ContractError(AssertionError):
    """Raised when something crossing a boundary has the wrong shape or type."""


# --------------------------------------------------------------------------
# A -> B : what the data pipeline hands the training loop
# --------------------------------------------------------------------------
def validate_batch(images, masks, expect_size: int | None = None) -> None:
    """Check one batch from a DataLoader.

    Owner of workstream A must make this pass.
    """
    import torch

    if not torch.is_tensor(images) or not torch.is_tensor(masks):
        raise ContractError("images and masks must both be torch tensors")

    if images.ndim != 4 or images.shape[1] != 3:
        raise ContractError(
            f"images must be (N, 3, H, W), got {tuple(images.shape)}"
        )
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ContractError(
            f"masks must be (N, 1, H, W), got {tuple(masks.shape)} "
            "— a common cause is returning (N, H, W) from the dataset"
        )
    if images.shape[0] != masks.shape[0]:
        raise ContractError("batch sizes differ between images and masks")
    if images.shape[2:] != masks.shape[2:]:
        raise ContractError(
            f"spatial dims differ: images {tuple(images.shape[2:])} "
            f"vs masks {tuple(masks.shape[2:])}"
        )
    if images.dtype != torch.float32 or masks.dtype != torch.float32:
        raise ContractError(
            f"both must be float32, got {images.dtype} and {masks.dtype}"
        )

    unique = set(masks.unique().tolist())
    if not unique.issubset({0.0, 1.0}):
        extra = sorted(unique - {0.0, 1.0})[:5]
        raise ContractError(
            f"masks must contain only 0.0 and 1.0; also found {extra} "
            "— usually bilinear interpolation applied to a mask"
        )

    if expect_size is not None and tuple(images.shape[2:]) != (expect_size, expect_size):
        raise ContractError(
            f"expected {expect_size}x{expect_size}, got {tuple(images.shape[2:])}"
        )


# --------------------------------------------------------------------------
# B -> C : what the model hands evaluation and inference
# --------------------------------------------------------------------------
def validate_logits(logits, images=None) -> None:
    """Check a model's forward output.

    Owner of workstream B must make this pass.
    """
    import torch

    if not torch.is_tensor(logits):
        raise ContractError("model output must be a torch tensor")
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ContractError(
            f"logits must be (N, 1, H, W), got {tuple(logits.shape)}"
        )
    if images is not None and logits.shape[2:] != images.shape[2:]:
        raise ContractError(
            f"output spatial size {tuple(logits.shape[2:])} must match "
            f"input {tuple(images.shape[2:])}"
        )

    with torch.no_grad():
        lo, hi = float(logits.min()), float(logits.max())
    if 0.0 <= lo and hi <= 1.0 and logits.numel() > 16:
        raise ContractError(
            f"output range [{lo:.3f}, {hi:.3f}] looks like probabilities, not "
            "logits. The model must NOT apply a sigmoid — losses and metrics "
            "apply their own. See the module docstring."
        )


def validate_checkpoint(ckpt) -> None:
    """Check a saved checkpoint dict.

    Owner of workstream B must make this pass; workstream C reads it.
    """
    if not isinstance(ckpt, dict):
        raise ContractError(f"checkpoint must be a dict, got {type(ckpt).__name__}")
    missing = [k for k in REQUIRED_CKPT_KEYS if k not in ckpt]
    if missing:
        raise ContractError(f"checkpoint missing required keys: {missing}")
    if not isinstance(ckpt["config"], dict):
        raise ContractError("checkpoint['config'] must be the config dict")
    if "best_threshold" in ckpt:
        t = ckpt["best_threshold"]
        if not isinstance(t, (int, float)) or not 0.0 < float(t) < 1.0:
            raise ContractError(f"best_threshold must be in (0, 1), got {t!r}")


# --------------------------------------------------------------------------
# C -> KEYSTONE : what the measurement layer hands the wider system
# --------------------------------------------------------------------------
def validate_mask(mask, expect_shape: tuple[int, int] | None = None) -> None:
    """Check a segmentation mask leaving CrackSegmenter.segment().

    Owner of workstream C must make this pass.
    """
    import numpy as np

    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ContractError(
            f"mask must be 2-D (H, W), got {arr.shape} — squeeze the batch "
            "and channel dimensions before returning"
        )
    if arr.dtype != np.uint8:
        raise ContractError(f"mask must be uint8, got {arr.dtype}")
    vals = set(np.unique(arr).tolist())
    if not vals.issubset({0, 1}):
        raise ContractError(
            f"mask must contain only 0 and 1, found {sorted(vals)[:5]} "
            "— did you forget to threshold, or scale to 0-255?"
        )
    if expect_shape is not None and arr.shape != expect_shape:
        raise ContractError(
            f"mask shape {arr.shape} must match the input image {expect_shape}"
        )


def validate_stats(stats) -> None:
    """Check a CrackStats object."""
    required = (
        "n_components",
        "total_length",
        "mean_width",
        "max_width",
        "p95_width",
        "area",
        "coverage_fraction",
        "units",
    )
    for f in required:
        if not hasattr(stats, f):
            raise ContractError(f"CrackStats missing field '{f}'")
    if stats.units not in {"px", "mm"}:
        raise ContractError(f"units must be 'px' or 'mm', got {stats.units!r}")
    if stats.max_width < stats.mean_width - 1e-6:
        raise ContractError("max_width must be >= mean_width")
    if not 0.0 <= stats.coverage_fraction <= 1.0:
        raise ContractError(
            f"coverage_fraction must be a fraction, got {stats.coverage_fraction}"
        )


def validate_delta(delta) -> None:
    """Check a CrackDelta object."""
    for f in ("earlier", "later", "d_max_width", "d_mean_width",
              "d_total_length", "d_area", "new_crack_area", "units"):
        if not hasattr(delta, f):
            raise ContractError(f"CrackDelta missing field '{f}'")
    validate_stats(delta.earlier)
    validate_stats(delta.later)
    if delta.new_crack_area < -1e-6:
        raise ContractError("new_crack_area cannot be negative")


if __name__ == "__main__":
    import numpy as np
    import torch

    print("contract validators — negative cases should all be caught\n")

    # A -> B
    good_i = torch.rand(2, 3, 64, 64)
    good_m = (torch.rand(2, 1, 64, 64) > 0.9).float()
    validate_batch(good_i, good_m)
    print("  valid batch                      accepted")

    for bad, why in (
        ((good_i, good_m.squeeze(1)), "mask missing channel dim"),
        ((good_i, torch.rand(2, 1, 64, 64)), "mask not binary"),
        ((good_i, good_m.double()), "wrong dtype"),
    ):
        try:
            validate_batch(*bad)
            print(f"  MISSED: {why}")
        except ContractError:
            print(f"  rejected: {why}")

    # B -> C
    validate_logits(torch.randn(2, 1, 64, 64) * 4, good_i)
    print("  valid logits                     accepted")
    try:
        validate_logits(torch.rand(2, 1, 64, 64), good_i)
        print("  MISSED: probabilities instead of logits")
    except ContractError:
        print("  rejected: probabilities instead of logits")

    validate_checkpoint({"model_state": {}, "config": {}})
    print("  valid checkpoint                 accepted")
    try:
        validate_checkpoint({"model_state": {}})
        print("  MISSED: checkpoint missing config")
    except ContractError:
        print("  rejected: checkpoint missing config")

    # C -> KEYSTONE
    validate_mask(np.zeros((32, 32), np.uint8))
    print("  valid mask                       accepted")
    for bad, why in (
        (np.zeros((1, 32, 32), np.uint8), "mask still has a batch dim"),
        (np.zeros((32, 32), np.float32), "mask not uint8"),
        ((np.ones((32, 32)) * 255).astype(np.uint8), "mask scaled to 0-255"),
    ):
        try:
            validate_mask(bad)
            print(f"  MISSED: {why}")
        except ContractError:
            print(f"  rejected: {why}")

    print("\nall contract validators behave correctly")
