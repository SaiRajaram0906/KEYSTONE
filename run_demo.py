"""End-to-end demo: every workstream, one chain, synthetic data.

    python run_demo.py

    1. Geometry (G)   two raw point clouds of one wall -> ChangeReport in mm
    2. crack-seg (C)  two aligned crack masks          -> CrackDelta in mm
    3. Adapter        CrackDelta -> CrackChange, attached to the report
    4. Decision (D)   the report crosses the JSON boundary -> Alerts
    5. Decision (D)   a 10-capture year with a hidden structural drift ->
                      does the system separate it from seasonal movement?

Everything here is synthetic ground truth from contracts/fakes.py. No real
photographs, no trained model: this shows the pieces connect, not that they
work on a real monument. See README.md, section "What is and is not proven".
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "keystone-core"))   # contracts, geometry, decision, preservation
sys.path.insert(0, str(ROOT / "crack-seg"))       # keystone.interface (crack measurement)

import cv2                                         # noqa: E402
import numpy as np                                 # noqa: E402

from contracts.crack_adapter import from_crack_delta                  # noqa: E402
from contracts.fakes import make_geometry_case, make_report_series    # noqa: E402
from contracts.schema import ChangeReport, validate_change_report     # noqa: E402
from decision.pipeline import evaluate                                # noqa: E402
from geometry.pipeline import build_change_report                     # noqa: E402
from keystone.interface import compare_captures                       # noqa: E402


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def crack_mask(width_px: int, shape=(400, 600)) -> np.ndarray:
    """A single diagonal crack of a known width, as a 0/1 mask."""
    m = np.zeros(shape, np.uint8)
    cv2.line(m, (40, 60), (560, 340), 1, thickness=width_px)
    return m


def main() -> int:
    # ---- 1. Geometry -------------------------------------------------------
    banner("1. GEOMETRY: two raw captures -> measured movement per region")
    case = make_geometry_case(seed=0, displaced={"east_upper": 12.0})
    report = build_change_report(case.earlier, case.later, case.survey, case.regions)
    print(f"registration error: {report.registration_rmse_mm:.2f} mm")
    print(f"{'region':<16}{'median mm':>11}{'LoD mm':>9}{'moved >LoD':>12}")
    for r in report.regions:
        print(f"{r.region_id:<16}{r.median_mm:>+11.2f}{r.lod_mm:>9.2f}{r.significant_fraction:>12.0%}")
    print("truth: east_upper was displaced outward by 12.0 mm; every other region by 0.")

    # ---- 2. crack-seg measurement -----------------------------------------
    banner("2. CRACK-SEG: two aligned masks -> crack widening in mm")
    mm_per_px = 0.5
    delta = compare_captures(crack_mask(3), crack_mask(7), mm_per_px=mm_per_px)
    print(f"crack width  {delta.earlier.mean_width:.1f} -> {delta.later.mean_width:.1f} mm (mean)")
    print(f"d_max_width  {delta.d_max_width:+.2f} mm    d_mean_width {delta.d_mean_width:+.2f} mm")
    print(f"quantisation floor: +/- {mm_per_px:.1f} mm (1 px at {mm_per_px} mm/px)")
    print("truth: crack was drawn 3 px -> 7 px wide = +2.0 mm at 0.5 mm/px.")

    # ---- 3. adapter --------------------------------------------------------
    banner("3. ADAPTER: CrackDelta -> CrackChange, attached to the report")
    change = from_crack_delta(delta.as_dict(), "east_upper", mm_per_px)
    report.crack_changes = [change]
    validate_change_report(report)
    print(change)

    # ---- 4. Decision on one report, across JSON ---------------------------
    banner("4. DECISION: the report crosses the JSON boundary -> alerts")
    wire = ChangeReport.from_json(report.to_json())
    for a in evaluate([wire]):
        print(f"{a.region_id:<16}{a.severity:<10}-> {a.recipient}")
        print(f"{'':<16}{a.reason}")

    # ---- 5. Decision on a year of data -------------------------------------
    banner("5. DECISION: a year of captures with a hidden structural drift")
    series, truth = make_report_series(n_captures=10, drift_region="east_upper")
    alerts = evaluate(series)
    print(f"{len(series)} reports; truth: east_upper drifts outward from capture 5 and never recovers.")
    for a in alerts:
        print(f"{a.region_id:<16}{a.severity:<10}-> {a.recipient}")
    print("\nOther regions move with temperature only; the point is that they are NOT escalated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
