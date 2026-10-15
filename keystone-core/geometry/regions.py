"""Step 4 of Geometry: aggregate per-point M3C2 measurements into region metrics.

This module takes the dense per-point core outputs from M3C2 and aggregates them into
standard RegionChange dataclass objects for each defined Region bounding box.
"""
from __future__ import annotations

import numpy as np

from contracts.schema import ContractError, Region, RegionChange

__all__ = ["summarise"]


def summarise(
    core_pts: np.ndarray,
    distance_mm: np.ndarray,
    lod_mm: np.ndarray,
    valid: np.ndarray,
    regions: list[Region],
) -> list[RegionChange]:
    """Summarise valid M3C2 core point measurements for each region box.

    Returns a list of RegionChange objects matching the contract schema.
    Raises ContractError if any region contains zero valid points.
    """
    pts = np.asarray(core_pts, dtype=float)
    dists = np.asarray(distance_mm, dtype=float)
    lods = np.asarray(lod_mm, dtype=float)
    val = np.asarray(valid, dtype=bool)

    if pts.shape[0] != dists.shape[0] or pts.shape[0] != lods.shape[0] or pts.shape[0] != val.shape[0]:
        raise ValueError("core_pts, distance_mm, lod_mm, and valid must have matching lengths")

    region_changes: list[RegionChange] = []

    for r in regions:
        # Bounding box filter: bounds_min <= point <= bounds_max
        min_b = np.asarray(r.bounds_min, dtype=float)
        max_b = np.asarray(r.bounds_max, dtype=float)

        in_box = np.all((pts >= min_b) & (pts <= max_b), axis=1)
        valid_in_box = in_box & val

        n_points = int(np.sum(valid_in_box))
        if n_points == 0:
            raise ContractError(f"region '{r.region_id}' has no valid core points inside its bounds")

        reg_dists = dists[valid_in_box]
        reg_lods = lods[valid_in_box]

        median_mm = float(np.median(reg_dists))
        abs_dists = np.abs(reg_dists)
        p95_abs_mm = float(np.percentile(abs_dists, 95))

        # Enforce mathematical contract: p95(|d|) >= |median(d)|
        if p95_abs_mm < abs(median_mm):
            p95_abs_mm = abs(median_mm)

        lod_region = float(np.median(reg_lods))
        if lod_region <= 0:
            raise ContractError(f"region '{r.region_id}' computed non-positive lod_mm ({lod_region})")

        # Share of points exceeding their own per-point level of detection
        significant_fraction = float(np.mean(abs_dists > reg_lods))
        significant_fraction = float(np.clip(significant_fraction, 0.0, 1.0))

        rc = RegionChange(
            region_id=r.region_id,
            n_points=n_points,
            median_mm=median_mm,
            p95_abs_mm=p95_abs_mm,
            lod_mm=lod_region,
            significant_fraction=significant_fraction,
            load_bearing=r.load_bearing,
        )
        region_changes.append(rc)

    return region_changes
