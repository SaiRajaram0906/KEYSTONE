"""Synthetic scenarios with known ground truth, for both workstreams.

Neither team waits for the other, and neither needs a real monument to
start. More importantly, every scenario carries its TRUTH — the real
displacement, the real scale, which region is genuinely failing — so the
acceptance tests can check answers, not just shapes.

    make_geometry_case()   two raw captures of a masonry wall, arbitrary
                           scale and pose each, one region pushed outward
                           by a known amount.                    -> Geometry

    make_report_series()   a year of ChangeReports for one monument, with
                           temperature-driven seasonal movement everywhere
                           and, optionally, a genuine structural drift in
                           one region from a known capture.      -> Decision

SYNTHETIC PARAMETERS ARE NOT CLAIMS
-----------------------------------
The seasonal amplitude used here (roughly 0.6 mm per degree C, larger at
height) is chosen to make the separation problem non-trivial, not measured
from any monument. Do not quote these numbers as properties of real stone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from contracts.inputs import CaptureInput
from contracts.schema import (
    CaptureMeta,
    ChangeReport,
    CrackChange,
    Region,
    RegionChange,
)

__all__ = [
    "WALL_REGIONS",
    "GeometryCase",
    "make_geometry_case",
    "SeriesTruth",
    "make_report_series",
]

# A 6 m x 3 m wall face in the survey frame (mm). x = along the wall,
# z = up, y = outward normal. The two centre regions form a load-bearing pier.
WALL_W, WALL_H = 6000.0, 3000.0
WALL_REGIONS = [
    Region(f"{col}_{row}",
           [x0, -300.0, z0], [x0 + 2000.0, 300.0, z0 + 1500.0],
           load_bearing=(col == "centre"))
    for col, x0 in (("west", 0.0), ("centre", 2000.0), ("east", 4000.0))
    for row, z0 in (("lower", 0.0), ("upper", 1500.0))
]

SURVEY = {
    "GCP-1": [50.0, 5.0, 50.0],
    "GCP-2": [5950.0, 5.0, 50.0],
    "GCP-3": [50.0, 5.0, 2950.0],
    "GCP-4": [5950.0, 5.0, 2950.0],
    "GCP-5": [3000.0, 5.0, 60.0],
    "GCP-6": [3000.0, 5.0, 2940.0],
}


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def _masonry_relief(x: np.ndarray, z: np.ndarray, seed: int = 7) -> np.ndarray:
    """Surface relief of coursed ashlar: stones proud by a few mm, joints recessed."""
    bw, bh, joint = 600.0, 300.0, 15.0
    row = np.floor(z / bh)
    offset = (row % 2) * bw / 2
    col = np.floor((x + offset) / bw)
    rng = np.random.default_rng(seed)
    stone_offsets = rng.normal(0.0, 2.5, 5000)
    idx = ((row * 131 + col * 37).astype(int)) % 5000
    relief = stone_offsets[idx]
    in_joint = (
        ((z % bh) < joint) | ((z % bh) > bh - joint)
        | (((x + offset) % bw) < joint) | (((x + offset) % bw) > bw - joint)
    )
    return np.where(in_joint, relief - 10.0, relief)


def _random_rotation(rng) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a - b*b + c*c - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a - b*b - c*c + d*d],
    ])


def _to_raw(p_mm: np.ndarray, mm_per_unit: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """survey mm -> raw capture frame: rotate, scale into raw units, translate."""
    return (p_mm @ R.T) / mm_per_unit + t


def _sample_wall(n: int, rng, displaced: dict[str, float] | None) -> np.ndarray:
    x = rng.uniform(0, WALL_W, n)
    z = rng.uniform(0, WALL_H, n)
    y = _masonry_relief(x, z)
    if displaced:
        for region_id, mm in displaced.items():
            r = next(r for r in WALL_REGIONS if r.region_id == region_id)
            inside = ((x >= r.bounds_min[0]) & (x < r.bounds_max[0])
                      & (z >= r.bounds_min[2]) & (z < r.bounds_max[2]))
            y = y + np.where(inside, mm, 0.0)
    y = y + rng.normal(0.0, 1.5, n)            # photogrammetric noise
    return np.column_stack([x, y, z])


@dataclass
class GeometryCase:
    earlier: CaptureInput
    later: CaptureInput
    survey: dict[str, list[float]]
    regions: list[Region]
    truth_displacement_mm: dict[str, float]
    truth_mm_per_unit: dict[str, float]


def make_geometry_case(
    seed: int = 0,
    displaced: dict[str, float] | None = None,
    n_points: int = 60000,
    marker_noise_mm: float = 1.0,
) -> GeometryCase:
    """Two captures of the same wall; `displaced` regions moved outward by mm.

    Each capture gets its own random scale, rotation and translation — exactly
    what an uncontrolled photogrammetry run produces — and its own independent
    point sampling, so there are no point-to-point correspondences to cheat with.
    """
    displaced = {"east_upper": 12.0} if displaced is None else displaced
    rng = np.random.default_rng(seed)
    survey_arr = {k: np.array(v) for k, v in SURVEY.items()}

    captures, scales = [], {}
    for which, disp in (("earlier", None), ("later", displaced)):
        mm_per_unit = float(rng.uniform(4.0, 25.0))
        R = _random_rotation(rng)
        t = rng.uniform(-500, 500, 3)
        pts_mm = _sample_wall(n_points, rng, disp)
        markers = {
            k: _to_raw(v + rng.normal(0, marker_noise_mm, 3), mm_per_unit, R, t)
            for k, v in survey_arr.items()
        }
        cid = "C000" if which == "earlier" else "C001"
        meta = CaptureMeta(cid, "M-01", "2026-01-10" if which == "earlier" else "2026-06-10",
                           24.0 if which == "earlier" else 33.0,
                           0.0 if which == "earlier" else 8.0, None)
        captures.append(CaptureInput(meta, _to_raw(pts_mm, mm_per_unit, R, t), markers))
        scales[cid] = mm_per_unit

    truth = {r.region_id: float(displaced.get(r.region_id, 0.0)) for r in WALL_REGIONS}
    return GeometryCase(captures[0], captures[1], dict(SURVEY), list(WALL_REGIONS), truth, scales)


# --------------------------------------------------------------------------
# decision
# --------------------------------------------------------------------------
@dataclass
class SeriesTruth:
    drift_region: str | None
    drift_onset_index: int | None     # index into the returned report list
    seasonal_coeff_mm_per_c: dict[str, float]


def _temperature(d: date) -> float:
    """Chennai-like annual cycle: coolest in January, hottest around late May."""
    doy = d.timetuple().tm_yday
    return 29.0 + 5.0 * math.sin(2 * math.pi * (doy - 60) / 365.0)


def _rainfall_7d(d: date, rng) -> float:
    """North-east monsoon peak in November, lighter south-west rains mid-year."""
    doy = d.timetuple().tm_yday
    ne = 70.0 * math.exp(-((doy - 315) / 25.0) ** 2)
    sw = 20.0 * math.exp(-((doy - 210) / 40.0) ** 2)
    return max(0.0, ne + sw + rng.normal(0, 4.0))


def make_report_series(
    n_captures: int = 10,
    start: str = "2026-01-15",
    interval_days: int = 45,
    seed: int = 0,
    drift_region: str | None = "east_upper",
    drift_onset_capture: int = 5,
    drift_mm_per_capture: float = 3.5,
    monument_id: str = "M-01",
) -> tuple[list[ChangeReport], SeriesTruth]:
    """A chronological list of ChangeReports, each vs the reference capture.

    Capture 0 is the reference, so n_captures captures produce n_captures - 1
    reports. Every region moves with temperature (more at height). If
    drift_region is set, that region additionally drifts outward by
    drift_mm_per_capture per capture from drift_onset_capture onward, and
    that movement never comes back — which is what "structural" means.
    """
    rng = np.random.default_rng(seed)
    d0 = date.fromisoformat(start)
    dates = [d0 + timedelta(days=interval_days * i) for i in range(n_captures)]
    temps = [_temperature(d) + rng.normal(0, 0.8) for d in dates]
    rains = [_rainfall_7d(d, rng) for d in dates]

    coeff = {}
    for r in WALL_REGIONS:
        height_factor = 0.5 if r.region_id.endswith("lower") else 1.0
        coeff[r.region_id] = 0.6 * height_factor * rng.uniform(0.85, 1.15)
    lod = {r.region_id: float(rng.uniform(3.0, 4.5)) for r in WALL_REGIONS}

    metas = [
        CaptureMeta(f"C{i:03d}", monument_id, dates[i].isoformat(),
                    round(temps[i], 2), round(rains[i], 1), round(float(rng.uniform(4, 25)), 4))
        for i in range(n_captures)
    ]

    reports, drift_accum = [], 0.0
    for i in range(1, n_captures):
        if drift_region and i >= drift_onset_capture:
            drift_accum += drift_mm_per_capture
        regions, cracks = [], []
        for r in WALL_REGIONS:
            rid = r.region_id
            seasonal = coeff[rid] * (temps[i] - temps[0])
            moisture = 0.03 * (rains[i] - rains[0])
            drift = drift_accum if rid == drift_region else 0.0
            median = seasonal + moisture + drift + rng.normal(0, lod[rid] / 3.0)
            spread = abs(rng.normal(0, lod[rid] / 2.0))
            p95 = abs(median) + 1.6 * lod[rid] / 2.0 + spread
            sig = 1.0 / (1.0 + math.exp(-(abs(median) - lod[rid]) / (0.3 * lod[rid])))
            regions.append(RegionChange(
                rid, int(rng.integers(4000, 9000)), round(median, 3), round(p95, 3),
                round(lod[rid], 3), round(sig, 4), r.load_bearing))

            if rid == drift_region and drift_accum > 0:
                cracks.append(CrackChange(rid, round(0.15 * drift_accum + rng.normal(0, 0.1), 3),
                                          round(40.0 * drift_accum, 1), 0.5))
            elif rng.random() < 0.3:
                cracks.append(CrackChange(rid, round(rng.normal(0, 0.2), 3), 0.0, 0.5))

        reports.append(ChangeReport(
            monument_id=monument_id, earlier=metas[0], later=metas[i],
            registration_rmse_mm=round(float(rng.uniform(0.8, 2.0)), 3),
            regions=regions, crack_changes=cracks))

    onset_report_index = (drift_onset_capture - 1) if drift_region else None
    return reports, SeriesTruth(drift_region, onset_report_index, coeff)


if __name__ == "__main__":
    from contracts.schema import validate_change_report

    case = make_geometry_case(seed=1)
    print("GEOMETRY CASE")
    print(f"  earlier: {case.earlier.points.shape[0]} pts, "
          f"true mm/unit {case.truth_mm_per_unit['C000']:.3f}")
    print(f"  later  : {case.later.points.shape[0]} pts, "
          f"true mm/unit {case.truth_mm_per_unit['C001']:.3f}")
    print(f"  markers: {sorted(case.earlier.markers)}")
    print(f"  truth  : {case.truth_displacement_mm}")

    reports, truth = make_report_series(seed=1)
    for rep in reports:
        validate_change_report(rep)
    print("\nDECISION SERIES")
    print(f"  {len(reports)} reports, all valid; drift in {truth.drift_region} "
          f"from report index {truth.drift_onset_index}")
    print(f"  {'report':<8}{'date':<12}{'temp':>6}" + "".join(f"{r.region_id:>14}" for r in WALL_REGIONS))
    for k, rep in enumerate(reports):
        row = "".join(f"{rc.median_mm:>14.1f}" for rc in rep.regions)
        print(f"  {k:<8}{rep.later.captured_on:<12}{rep.later.temperature_c:>6.1f}{row}")
