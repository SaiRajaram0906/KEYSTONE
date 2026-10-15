"""Conformance tests for the three workstream boundaries.

    python tests/contracts_test.py            # everything
    python tests/contracts_test.py --only A   # just your part

Run your own section before every push. If it passes, your part will drop
into the others without an integration day spent on shape mismatches.

Each section tests one workstream against the contract, using fakes for
the other two. That means all three sections pass independently, from the
first day, before anyone has finished.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.contracts import (  # noqa: E402
    ContractError,
    validate_batch,
    validate_checkpoint,
    validate_delta,
    validate_logits,
    validate_mask,
    validate_stats,
)
from tests.fakes import (  # noqa: E402
    FakeModel,
    make_fake_dataset,
    save_fake_checkpoint,
    synthetic_crack_mask,
)

PASS, FAIL = "  PASS  ", "  FAIL  "
results: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        fn()
        results.append((name, True, ""))
        print(f"{PASS}{name}")
    except Exception as exc:  # noqa: BLE001
        results.append((name, False, str(exc)))
        print(f"{FAIL}{name}\n          {type(exc).__name__}: {exc}")


# ==========================================================================
# A — DATA
# owner: builds dataset.py and transforms.py
# must guarantee: batches match the A->B contract
# ==========================================================================
def test_workstream_a() -> None:
    print("\nWORKSTREAM A — data pipeline")
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import (
        CrackDataset,
        compute_pos_weight,
        discover_pairs,
        split_pairs,
    )
    from src.data.transforms import eval_transform, train_transform

    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_dataset(Path(tmp) / "data", n=30, size=160)
        pairs = discover_pairs(root)
        splits = split_pairs(pairs, val_frac=0.2, test_frac=0.2)

        def batch_contract():
            ds = CrackDataset(splits["train"], train_transform(128, "medium"))
            loader = DataLoader(ds, batch_size=4, shuffle=True)
            images, masks = next(iter(loader))
            validate_batch(images, masks, expect_size=128)

        def eval_batch_contract():
            ds = CrackDataset(splits["val"], eval_transform(128))
            loader = DataLoader(ds, batch_size=2)
            images, masks = next(iter(loader))
            validate_batch(images, masks, expect_size=128)

        def splits_disjoint():
            names = {k: {p[0].name for p in v} for k, v in splits.items()}
            assert not (names["train"] & names["val"]), "train/val overlap"
            assert not (names["train"] & names["test"]), "train/test overlap"
            assert not (names["val"] & names["test"]), "val/test overlap"
            assert sum(len(v) for v in splits.values()) == len(pairs), "images lost"

        def splits_deterministic():
            a = [p[0].name for p in split_pairs(pairs, 0.2, 0.2)["test"]]
            b = [p[0].name for p in split_pairs(pairs, 0.2, 0.2)["test"]]
            assert a == b, "split changes between calls"

        def pos_weight_sane():
            pw = compute_pos_weight(splits["train"])
            assert pw > 1.0, f"pos_weight {pw} — cracks should be the minority class"

        def feeds_a_model():
            ds = CrackDataset(splits["train"], train_transform(128))
            loader = DataLoader(ds, batch_size=2)
            images, masks = next(iter(loader))
            logits = FakeModel()(images)
            validate_logits(logits, images)
            assert logits.shape == masks.shape, "logits must match mask shape"

        check("batches satisfy the A->B contract (train)", batch_contract)
        check("batches satisfy the A->B contract (eval)", eval_batch_contract)
        check("splits are disjoint and lose nothing", splits_disjoint)
        check("splits are deterministic across calls", splits_deterministic)
        check("pos_weight reflects class imbalance", pos_weight_sane)
        check("a batch flows through a model unchanged in shape", feeds_a_model)


# ==========================================================================
# B — MODEL AND TRAINING
# owner: builds unet.py, losses.py, train.py
# must guarantee: logits and checkpoints match the B->C contract
# ==========================================================================
def test_workstream_b() -> None:
    print("\nWORKSTREAM B — model, losses, training")
    import torch

    from src.losses import build_loss
    from src.models.unet import UNet, build_model

    def output_contract():
        for bilinear in (True, False):
            model = UNet(base=8, depth=3, bilinear=bilinear)
            images = torch.rand(2, 3, 96, 96)
            validate_logits(model(images), images)

    def odd_sizes_survive():
        model = UNet(base=8, depth=3)
        for size in (96, 100, 130):
            images = torch.rand(1, 3, size, size)
            out = model(images)
            assert out.shape[2:] == images.shape[2:], f"size {size} not preserved"

    def losses_accept_logits():
        logits = torch.randn(2, 1, 64, 64) * 3
        targets = (torch.rand(2, 1, 64, 64) > 0.95).float()
        for name in ("bce", "dice", "bce_dice", "focal", "tversky", "focal_tversky"):
            loss = build_loss({"loss": {"name": name}})(logits, targets)
            assert torch.isfinite(loss), f"{name} produced {loss}"
            assert loss.ndim == 0, f"{name} must return a scalar"

    def losses_punish_empty_predictions():
        """The point of the whole loss section: a do-nothing model must score badly."""
        targets = torch.zeros(2, 1, 64, 64)
        targets[:, :, 30:33, :] = 1.0
        empty = torch.full((2, 1, 64, 64), -10.0)
        dice = build_loss({"loss": {"name": "dice"}})(empty, targets)
        assert dice > 0.9, (
            f"Dice on an all-background prediction was {dice:.3f}; expected ~1.0. "
            "If this is low, the imbalance handling is broken."
        )

    def gradients_flow():
        model = UNet(base=8, depth=3)
        images = torch.rand(2, 3, 64, 64)
        targets = (torch.rand(2, 1, 64, 64) > 0.95).float()
        loss = build_loss({"loss": {"name": "bce_dice"}})(model(images), targets)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, "no gradients reached the parameters"
        assert any(g.abs().sum() > 0 for g in grads), "all gradients are zero"

    def checkpoint_contract():
        with tempfile.TemporaryDirectory() as tmp:
            p = save_fake_checkpoint(Path(tmp) / "c.pt", size=128, base=8)
            validate_checkpoint(torch.load(p, map_location="cpu", weights_only=False))

    def config_builds_a_model():
        cfg = {"model": {"base": 8, "depth": 3, "bilinear": True, "dropout": 0.1}}
        validate_logits(build_model(cfg)(torch.rand(1, 3, 64, 64)))

    check("model returns logits, not probabilities", output_contract)
    check("output size matches input, including odd sizes", odd_sizes_survive)
    check("every loss accepts logits and returns a scalar", losses_accept_logits)
    check("overlap losses punish an all-empty prediction", losses_punish_empty_predictions)
    check("gradients reach the parameters", gradients_flow)
    check("checkpoints satisfy the B->C contract", checkpoint_contract)
    check("build_model works from a config dict", config_builds_a_model)


# ==========================================================================
# C — MEASUREMENT AND KEYSTONE INTERFACE
# owner: builds metrics.py, evaluate.py, keystone/interface.py
# must guarantee: masks and measurements match the C->KEYSTONE contract
# ==========================================================================
def test_workstream_c() -> None:
    print("\nWORKSTREAM C — metrics, measurement, KEYSTONE interface")
    import numpy as np
    import torch

    from keystone.interface import CrackSegmenter, compare_captures, measure_mask
    from src.metrics import MetricTracker

    def metrics_perfect_and_empty():
        targets = torch.zeros(4, 1, 64, 64)
        targets[:, :, 30:33, :] = 1.0
        perfect = torch.where(targets > 0.5, 8.0, -8.0)
        t = MetricTracker(); t.update(perfect, targets)
        assert t.compute().iou > 0.99, "perfect prediction must score IoU ~1"

        empty = torch.full((4, 1, 64, 64), -8.0)
        t = MetricTracker(); t.update(empty, targets)
        m = t.compute()
        assert m.iou < 0.01, f"all-empty scored IoU {m.iou:.3f}; must be ~0"
        assert m.pixel_accuracy > 0.9, "the misleading metric should still be high"

    def metrics_handle_empty_targets():
        targets = torch.zeros(2, 1, 32, 32)
        empty = torch.full((2, 1, 32, 32), -8.0)
        t = MetricTracker(); t.update(empty, targets)
        m = t.compute()
        assert m.iou == 1.0, "correctly predicting no crack must not score 0"
        assert m.n_empty_targets == 2, "empty targets must be counted and reported"

    def mask_contract():
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = save_fake_checkpoint(Path(tmp) / "c.pt", size=128, base=8)
            seg = CrackSegmenter(ckpt, device="cpu")
            img, _ = synthetic_crack_mask(200, 260, seed=5)
            mask = seg.segment(img)
            validate_mask(mask, expect_shape=(200, 260))

    def width_is_calibrated():
        for true_w in (1, 3, 5, 7):
            band = np.zeros((60, 200), np.uint8)
            top = 30 - true_w // 2
            band[top:top + true_w, 20:180] = 1
            est = measure_mask(band).max_width
            assert abs(est - true_w) < 1e-6, (
                f"odd width {true_w}px measured as {est}px — see the "
                "quantisation note in keystone/interface.py"
            )

    def stats_contract():
        band = np.zeros((60, 200), np.uint8)
        band[28:33, 20:180] = 1
        validate_stats(measure_mask(band))
        validate_stats(measure_mask(band, mm_per_px=0.5))

    def scale_applies_linearly():
        band = np.zeros((60, 200), np.uint8)
        band[28:33, 20:180] = 1
        px = measure_mask(band).max_width
        mm = measure_mask(band, mm_per_px=0.4).max_width
        assert abs(mm - px * 0.4) < 1e-6, "mm_per_px must scale widths linearly"

    def comparison_contract():
        a = np.zeros((120, 300), np.uint8); a[58:61, 30:200] = 1
        b = np.zeros((120, 300), np.uint8); b[57:64, 30:260] = 1
        d = compare_captures(a, b, mm_per_px=0.5)
        validate_delta(d)
        assert d.d_max_width > 0, "widening must be detected"
        assert d.d_total_length > 0, "lengthening must be detected"
        assert d.new_crack_area > 0, "new crack area must be positive"

    def no_change_reads_zero():
        """The false-positive floor: identical captures must report no change."""
        a = np.zeros((120, 300), np.uint8); a[58:61, 30:200] = 1
        d = compare_captures(a, a.copy(), mm_per_px=0.5)
        assert abs(d.d_max_width) < 1e-6, f"identical masks reported {d.d_max_width}"
        assert d.new_crack_area == 0.0, "identical masks cannot have new crack area"

    def misaligned_masks_rejected():
        a = np.zeros((100, 200), np.uint8)
        b = np.zeros((100, 210), np.uint8)
        try:
            compare_captures(a, b)
        except ValueError:
            return
        raise AssertionError("mismatched mask shapes must raise")

    check("metrics: perfect scores 1, all-empty scores 0", metrics_perfect_and_empty)
    check("metrics: empty targets handled, not NaN", metrics_handle_empty_targets)
    check("segment() satisfies the C->KEYSTONE contract", mask_contract)
    check("crack width is calibrated against ground truth", width_is_calibrated)
    check("CrackStats satisfies the contract", stats_contract)
    check("mm_per_px scales widths linearly", scale_applies_linearly)
    check("compare_captures detects widening and growth", comparison_contract)
    check("identical captures report zero change", no_change_reads_zero)
    check("misaligned masks are rejected, not silently resized", misaligned_masks_rejected)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["A", "B", "C"], help="test one workstream")
    args = ap.parse_args()

    print("=" * 62)
    print("CONTRACT CONFORMANCE")
    print("=" * 62)

    if args.only in (None, "A"):
        test_workstream_a()
    if args.only in (None, "B"):
        test_workstream_b()
    if args.only in (None, "C"):
        test_workstream_c()

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print("\n" + "=" * 62)
    print(f"{passed}/{total} checks passed")
    if passed != total:
        print("\nfailures:")
        for name, ok, err in results:
            if not ok:
                print(f"  - {name}\n      {err}")
        sys.exit(1)
    print("all boundaries conform — this will integrate")
    print("=" * 62)


if __name__ == "__main__":
    main()
