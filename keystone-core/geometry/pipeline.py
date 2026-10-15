"""Main pipeline entry point for Geometry workstream.

build_change_report(earlier, later, survey, regions) -> ChangeReport
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from contracts.inputs import CaptureInput, validate_capture_input
from contracts.schema import ChangeReport, Region, validate_change_report, validate_region
from geometry.align import apply_rigid, fit_to_survey
from geometry.m3c2 import compute_m3c2
from geometry.regions import summarise
from geometry.register import refine

__all__ = ["build_change_report"]


def build_change_report(
    earlier: CaptureInput,
    later: CaptureInput,
    survey: dict[str, list[float]],
    regions: list[Region],
) -> ChangeReport:
    """Build a complete ChangeReport comparing later capture against earlier reference capture.

    Flow:
      1. Validate input captures and region definitions.
      2. Coarse marker-based alignment to survey frame (mm). Update mm_per_unit on meta copies.
      3. Fine registration of later cloud via Trimmed Point-to-Plane ICP.
      4. Combined registration RMSE calculation from ground control marker fits.
      5. Simplified M3C2 surface displacement and LoD calculation.
      6. Summarize core point results per region box.
      7. Construct and validate ChangeReport before returning.
    """
    # Step a: Validate inputs against contract rules
    validate_capture_input(earlier, survey)
    validate_capture_input(later, survey)
    for r in regions:
        validate_region(r)

    # Step b: Marker-based coarse alignment (raw -> survey frame in mm)
    sim_earlier, rmse_earlier, _ = fit_to_survey(earlier.markers, survey)
    sim_later, rmse_later, _ = fit_to_survey(later.markers, survey)

    earlier_mm = sim_earlier.apply(earlier.points)
    later_mm = sim_later.apply(later.points)

    # Fill mm_per_unit on copies of CaptureMeta without mutating inputs
    earlier_meta = dataclasses.replace(earlier.meta, mm_per_unit=float(sim_earlier.scale))
    later_meta = dataclasses.replace(later.meta, mm_per_unit=float(sim_later.scale))

    # Step c: Fine ICP registration using stable areas only (Trimmed ICP)
    T_refine, icp_rmse = refine(earlier_mm, later_mm)
    later_mm_refined = apply_rigid(later_mm, T_refine)

    # Step d: Combined registration error in placing both clouds in the shared survey frame
    registration_rmse_mm = float(math.sqrt(rmse_earlier**2 + rmse_later**2))

    # Step e: Compute M3C2 surface displacement and summarise by region
    core_pts, distance_mm, lod_mm, valid = compute_m3c2(
        earlier_mm,
        later_mm_refined,
        registration_error_mm=registration_rmse_mm,
    )

    region_changes = summarise(core_pts, distance_mm, lod_mm, valid, regions)

    # Step f: Build ChangeReport
    report = ChangeReport(
        monument_id=earlier_meta.monument_id,
        earlier=earlier_meta,
        later=later_meta,
        registration_rmse_mm=registration_rmse_mm,
        regions=region_changes,
        crack_changes=[],
    )

    # Step g: Validate ChangeReport gate before returning
    validate_change_report(report)

    return report
