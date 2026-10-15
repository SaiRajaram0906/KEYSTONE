"""KEYSTONE integration layer.

This is the only module KEYSTONE imports. Everything under src/ is the
research project; this file is the contract between the two.

WHAT SEGMENTATION ALONE DOES NOT GIVE YOU
------------------------------------------
A U-Net outputs a binary mask: which pixels are crack. That is not a
measurement. To report "this crack widened by 3 mm" you need three further
steps, and this module provides them:

    1. skeletonise the mask          -> the crack centreline
    2. distance transform            -> at every crack pixel, the distance
                                        to the nearest non-crack pixel
    3. width at a centreline pixel   = 2 x that distance - 1

and then a scale factor (mm per pixel) to leave pixel space. Without a
surveyed scale reference in the photograph, every number below is in
pixels and is not comparable between captures. This is the same scale
requirement KEYSTONE's photogrammetry stage already satisfies.

WIDTH ACCURACY, STATED HONESTLY
-------------------------------
The distance transform returns the distance to the nearest background
pixel, which is (w+1)/2 at the centre of an odd-width band and w/2 for an
even one. No single formula recovers both exactly, so the estimate here
(2d - 1) is exact for odd widths and one pixel short for even ones:

    true width   1   2   3   4   5   6   7
    estimate     1   1   3   3   5   5   7

Absolute widths therefore carry up to 1 px of quantisation error, and
that must be reported alongside any millimetre figure. The error partly
cancels in compare_captures() when both widths share a parity, but a
delta between an even and an odd width still carries up to 1 px — so at
0.5 mm/px a reported "+2.0 mm" is really +2.0 mm +/- 0.5 mm. Set alert
thresholds above that floor.

Sub-pixel width estimation needs a different method (fitting the
grey-level profile across the crack), and is out of scope here.

COMPARISON ASSUMPTION
---------------------
compare_captures() assumes the two masks are already spatially aligned.
In KEYSTONE that alignment comes free: both captures are registered to
the same ground control points during reconstruction, and crack masks are
rasterised from the same viewpoint. Feeding it two hand-held photographs
taken from different positions will produce meaningless deltas.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "CrackStats",
    "CrackDelta",
    "measure_mask",
    "compare_captures",
    "CrackSegmenter",
]


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------
@dataclass
class CrackStats:
    """Physical description of the cracks in a single capture.

    Lengths and widths are in millimetres when mm_per_px is supplied, and
    in pixels otherwise — `units` records which.
    """

    n_components: int
    total_length: float
    mean_width: float
    max_width: float
    p95_width: float
    area: float
    coverage_fraction: float
    units: str = "px"

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        u = self.units
        return (
            f"{self.n_components} crack(s) | length {self.total_length:.1f}{u} | "
            f"width mean {self.mean_width:.2f}{u} max {self.max_width:.2f}{u} "
            f"p95 {self.p95_width:.2f}{u} | coverage {self.coverage_fraction:.3%}"
        )


def measure_mask(
    mask: np.ndarray,
    mm_per_px: float | None = None,
    min_component_px: int = 20,
) -> CrackStats:
    """Turn a binary crack mask into physical measurements.

    Args:
        mask: 2-D array; any non-zero value counts as crack.
        mm_per_px: scale factor from the capture. None leaves results in
            pixels.
        min_component_px: connected components smaller than this are
            discarded as speckle. Raise it if the model is noisy.

    Returns:
        CrackStats. All-zero mask returns zeroed stats rather than raising.
    """
    from scipy import ndimage
    from skimage.morphology import skeletonize

    binary = np.asarray(mask) > 0
    if binary.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {binary.shape}")

    scale = float(mm_per_px) if mm_per_px else 1.0
    units = "mm" if mm_per_px else "px"
    total_px = binary.size

    if not binary.any():
        return CrackStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, units)

    # Drop speckle before measuring, or a few stray pixels inflate the count.
    labelled, n_raw = ndimage.label(binary)
    if n_raw:
        sizes = ndimage.sum(binary, labelled, range(1, n_raw + 1))
        keep = {i + 1 for i, s in enumerate(sizes) if s >= min_component_px}
        binary = np.isin(labelled, list(keep)) if keep else np.zeros_like(binary)

    if not binary.any():
        return CrackStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, units)

    labelled, n_components = ndimage.label(binary)

    # Distance from each crack pixel to the nearest background pixel.
    # On the centreline this equals half the local crack width.
    dist = ndimage.distance_transform_edt(binary)
    skeleton = skeletonize(binary)

    # See the module docstring: 2d - 1 is exact for odd widths, one pixel
    # short for even ones. Floor at 1 px, since a detected crack cannot be
    # narrower than the pixel that detected it.
    widths_px = np.maximum(2.0 * dist[skeleton] - 1.0, 1.0)
    if widths_px.size == 0:
        widths_px = np.array([0.0])

    length_px = float(skeleton.sum())
    area_px = float(binary.sum())

    return CrackStats(
        n_components=int(n_components),
        total_length=length_px * scale,
        mean_width=float(widths_px.mean()) * scale,
        max_width=float(widths_px.max()) * scale,
        p95_width=float(np.percentile(widths_px, 95)) * scale,
        area=area_px * scale * scale,
        coverage_fraction=area_px / total_px,
        units=units,
    )


# --------------------------------------------------------------------------
# comparison across captures
# --------------------------------------------------------------------------
@dataclass
class CrackDelta:
    """Change between two aligned captures of the same surface."""

    earlier: CrackStats
    later: CrackStats
    d_max_width: float
    d_mean_width: float
    d_total_length: float
    d_area: float
    new_crack_area: float
    units: str = "px"

    def as_dict(self) -> dict:
        d = {
            "d_max_width": self.d_max_width,
            "d_mean_width": self.d_mean_width,
            "d_total_length": self.d_total_length,
            "d_area": self.d_area,
            "new_crack_area": self.new_crack_area,
            "units": self.units,
        }
        d["earlier"] = self.earlier.as_dict()
        d["later"] = self.later.as_dict()
        return d

    def __str__(self) -> str:
        u = self.units
        return (
            f"max width {self.d_max_width:+.2f}{u} | "
            f"mean width {self.d_mean_width:+.2f}{u} | "
            f"length {self.d_total_length:+.1f}{u} | "
            f"newly cracked area {self.new_crack_area:.1f}{u}^2"
        )


def compare_captures(
    mask_earlier: np.ndarray,
    mask_later: np.ndarray,
    mm_per_px: float | None = None,
    min_component_px: int = 20,
) -> CrackDelta:
    """Measure both captures and report the change.

    The masks must be spatially aligned and the same shape. See the module
    docstring on why that is KEYSTONE's job, not this function's.
    """
    a = np.asarray(mask_earlier) > 0
    b = np.asarray(mask_later) > 0
    if a.shape != b.shape:
        raise ValueError(
            f"masks must be aligned and the same shape: {a.shape} vs {b.shape}"
        )

    s_a = measure_mask(a, mm_per_px, min_component_px)
    s_b = measure_mask(b, mm_per_px, min_component_px)
    scale = float(mm_per_px) if mm_per_px else 1.0

    # Area that is crack in the later capture but not the earlier one.
    new_area_px = float((b & ~a).sum())

    return CrackDelta(
        earlier=s_a,
        later=s_b,
        d_max_width=s_b.max_width - s_a.max_width,
        d_mean_width=s_b.mean_width - s_a.mean_width,
        d_total_length=s_b.total_length - s_a.total_length,
        d_area=s_b.area - s_a.area,
        new_crack_area=new_area_px * scale * scale,
        units=s_a.units,
    )


# --------------------------------------------------------------------------
# the segmenter KEYSTONE calls
# --------------------------------------------------------------------------
class CrackSegmenter:
    """Load a trained checkpoint and segment images.

    Typical KEYSTONE usage:

        seg = CrackSegmenter("checkpoints/best.pt", device="cuda")
        m1 = seg.segment("capture_2026_03/north_wall.jpg")
        m2 = seg.segment("capture_2026_09/north_wall.jpg")
        delta = compare_captures(m1, m2, mm_per_px=0.8)
        if delta.d_max_width > 1.0:
            raise_alert(delta)

    The threshold is exposed deliberately: 0.5 is rarely optimal on
    imbalanced data, and evaluate.py prints the value that maximised
    validation IoU.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | None = None,
        threshold: float = 0.5,
        image_size: int = 320,
    ):
        import torch

        self.threshold = threshold
        self.image_size = image_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        ckpt = torch.load(str(checkpoint), map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})
        # Prefer the size and threshold the checkpoint was trained/tuned with.
        self.image_size = int(cfg.get("data", {}).get("image_size", image_size))
        if "best_threshold" in ckpt:
            self.threshold = float(ckpt["best_threshold"])

        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.data.transforms import eval_transform
        from src.models.unet import build_model

        self.model = build_model(cfg if cfg else {})
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()
        self.transform = eval_transform(self.image_size)

    def segment(
        self,
        image: str | Path | np.ndarray,
        return_probability: bool = False,
        original_size: bool = True,
    ) -> np.ndarray:
        """Segment one image.

        Args:
            image: path, or an RGB uint8 array.
            return_probability: return float probabilities instead of a
                binary mask.
            original_size: resize the output back to the input resolution.
                Keep True for measurement, so mm_per_px stays valid.

        Returns:
            2-D array: uint8 {0,1} mask, or float32 probabilities.
        """
        import cv2
        import torch

        if isinstance(image, (str, Path)):
            arr = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if arr is None:
                raise RuntimeError(f"failed to read {image}")
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        else:
            arr = np.asarray(image)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError("image array must be RGB with shape (H, W, 3)")

        h0, w0 = arr.shape[:2]
        tensor = self.transform(image=arr, mask=np.zeros((h0, w0), np.float32))["image"]
        tensor = tensor.unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(tensor))[0, 0].cpu().numpy()

        if original_size and (prob.shape != (h0, w0)):
            prob = cv2.resize(prob, (w0, h0), interpolation=cv2.INTER_LINEAR)

        if return_probability:
            return prob.astype(np.float32)
        return (prob > self.threshold).astype(np.uint8)

    def measure(
        self, image: str | Path | np.ndarray, mm_per_px: float | None = None
    ) -> CrackStats:
        """Segment then measure in one call."""
        return measure_mask(self.segment(image), mm_per_px=mm_per_px)


if __name__ == "__main__":
    # Width estimates against known ground truth, to show the quantisation
    # behaviour documented at the top of this file.
    print("WIDTH CALIBRATION  (true -> estimated, expect odd exact)")
    for true_w in range(1, 10):
        band = np.zeros((60, 200), np.uint8)
        top = 30 - true_w // 2
        band[top:top + true_w, 20:180] = 1
        est = measure_mask(band).max_width
        flag = "exact" if abs(est - true_w) < 1e-6 else f"{est - true_w:+.0f} px"
        print(f"  {true_w} px -> {est:.0f} px   {flag}")

    # Synthetic check: a crack that widens from 4 px to 7 px and grows longer.
    h, w = 200, 400

    earlier = np.zeros((h, w), np.uint8)
    earlier[98:102, 50:300] = 1                 # 4 px wide, 250 px long

    later = np.zeros((h, w), np.uint8)
    later[97:104, 50:360] = 1                   # 7 px wide, 310 px long

    print()

    print("EARLIER :", measure_mask(earlier))
    print("LATER   :", measure_mask(later))

    delta_px = compare_captures(earlier, later)
    print("\nDELTA (px):", delta_px)

    # With a scale of 0.5 mm per pixel the same change reads in millimetres.
    delta_mm = compare_captures(earlier, later, mm_per_px=0.5)
    print("DELTA (mm):", delta_mm)

    assert delta_px.d_max_width > 0, "widening must be detected"
    assert delta_px.d_total_length > 0, "lengthening must be detected"
    assert abs(delta_mm.d_max_width - delta_px.d_max_width * 0.5) < 1e-6, (
        "scale factor must apply linearly to widths"
    )

    # Speckle must not register as a crack.
    speckle = np.zeros((h, w), np.uint8)
    speckle[10, 10] = 1
    speckle[50, 80] = 1
    assert measure_mask(speckle).n_components == 0, "speckle should be filtered"

    print("\nall interface checks passed")
