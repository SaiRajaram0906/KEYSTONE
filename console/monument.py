"""A simulated stone temple, watched for a year (no Streamlit in here).

EVERYTHING HERE IS SIMULATED. The temple, its cracks and its movement are made up
so the truth is known and can be compared with what KEYSTONE reports. Nothing is
measured from a real monument. The decisions on the way are made by the project's
real Decision code (decision.pipeline.analyse), fed with change reports built from
this simulation.

Frame (mm): x = east-west, y = depth (the front of the temple faces -y), z = up.

The story planted in the year (month 0 is the reference capture):
  * the tower leans east, slowly at first, from month 2
  * the front-right column (col_b) settles into soft ground from month 4, which
    sags the beam above it (lintel) and opens a crack there and one in the plinth
  * a hairline crack on the sanctum wall never changes (a control: no alarm)
  * everything breathes a little with the temperature, upper parts more
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "keystone-core",):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from contracts.fakes import _rainfall_7d, _temperature  # noqa: E402
from contracts.schema import CaptureMeta, ChangeReport, CrackChange, RegionChange  # noqa: E402
from decision.pipeline import analyse  # noqa: E402

N_MONTHS = 12
START = date(2026, 1, 15)
MONUMENT_ID = "M-02"

LOAD_BEARING = {"col_a_lower", "col_b_lower", "col_c_lower", "col_d_lower",
                "cella_front_upper", "cella_back_upper", "cella_east_upper", "cella_west_upper"}


# ------------------------------------------------------------------ geometry
def _box_face_points(rng, lo, hi, face, dens):
    """Random points on one face of a box. dens = points per square metre."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    axis, side = face[0], face[1]
    ext = {"x": (x1 - x0, y1 - y0, z1 - z0)}
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    if axis == "x":
        area, a = dy * dz, (dy, dz)
    elif axis == "y":
        area, a = dx * dz, (dx, dz)
    else:
        area, a = dx * dy, (dx, dy)
    n = max(8, int(area / 1e6 * dens))
    u, v = rng.uniform(0, a[0], n), rng.uniform(0, a[1], n)
    if axis == "x":
        pts = np.column_stack([np.full(n, x1 if side == "+" else x0), y0 + u, z0 + v])
    elif axis == "y":
        pts = np.column_stack([x0 + u, np.full(n, y1 if side == "+" else y0), z0 + v])
    else:
        pts = np.column_stack([x0 + u, y0 + v, np.full(n, z1 if side == "+" else z0)])
    return pts


@dataclass
class Temple:
    points: np.ndarray                 # (n, 3) mm
    region: np.ndarray                 # (n,) region id per point
    region_ids: list[str]
    cracks: list[dict]                 # see _cracks()
    coef: dict[str, float]             # seasonal mm per degree C
    lod: dict[str, float]
    temps: list[float]
    rains: list[float]
    dates: list[str]
    tower_base_z: float = 5200.0
    tower_height: float = 4500.0


def build_temple(seed: int = 0, density: float = 85.0) -> Temple:
    rng = np.random.default_rng(seed)
    pts, reg = [], []

    def add(p, name):
        pts.append(p)
        reg.extend([name] * len(p))

    # plinth (top and four sides)
    lo, hi = (-5000, -4000, 0), (5000, 4000, 1000)
    for f in ("z+", "x+", "x-", "y+", "y-"):
        add(_box_face_points(rng, lo, hi, f, density), "plinth_lower")

    # four columns
    cols = {"col_a_lower": (-2800, -3000), "col_b_lower": (2800, -3000),
            "col_c_lower": (-2800, -1000), "col_d_lower": (2800, -1000)}
    r, z0, z1 = 350.0, 1000.0, 4000.0
    for name, (cx, cy) in cols.items():
        n = int(2 * np.pi * r * (z1 - z0) / 1e6 * density * 1.6)
        th, z = rng.uniform(0, 2 * np.pi, n), rng.uniform(z0, z1, n)
        add(np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th), z]), name)

    # porch slab: its front face is the lintel (beam) region
    lo, hi = (-3500, -3500, 4000), (3500, -500, 4600)
    add(_box_face_points(rng, lo, hi, "y-", density * 1.5), "lintel_upper")
    for f in ("z+", "x+", "x-", "y+", "z-"):
        add(_box_face_points(rng, lo, hi, f, density), "porch_roof_upper")

    # sanctum (cella) walls, four faces
    lo, hi = (-1800, 0, 1000), (1800, 3500, 5200)
    for f, name in (("y-", "cella_front_upper"), ("y+", "cella_back_upper"),
                    ("x+", "cella_east_upper"), ("x-", "cella_west_upper")):
        add(_box_face_points(rng, lo, hi, f, density), name)

    # tower: five tiers shrinking upward, centred on the sanctum
    cx, cy, zb = 0.0, 1750.0, 5200.0
    halfs = [1700, 1400, 1100, 800, 500]
    for k, h in enumerate(halfs):
        lo, hi = (cx - h, cy - h, zb + 900 * k), (cx + h, cy + h, zb + 900 * (k + 1))
        for f in ("x+", "x-", "y+", "y-"):
            add(_box_face_points(rng, lo, hi, f, density), "tower_upper")
        top = _box_face_points(rng, lo, hi, "z+", density)
        if k + 1 < len(halfs):
            nh = halfs[k + 1]
            keep = ~((np.abs(top[:, 0] - cx) < nh) & (np.abs(top[:, 1] - cy) < nh))
            top = top[keep]
        add(top, "tower_upper")

    points = np.vstack(pts)
    region = np.array(reg)
    ids = sorted(set(reg), key=lambda s: (s.rsplit("_", 1)[-1], s))

    # weather and per-region constants (same recipe as the wall simulator)
    dates = [START + timedelta(days=int(round(30.4 * i))) for i in range(N_MONTHS + 1)]
    temps = [_temperature(d) + rng.normal(0, 0.8) for d in dates]
    rains = [_rainfall_7d(d, rng) for d in dates]
    coef = {rid: 0.6 * (0.5 if rid.endswith("lower") else 1.0) * rng.uniform(0.85, 1.15) for rid in ids}
    lod = {rid: float(rng.uniform(3.0, 4.5)) for rid in ids}

    return Temple(points, region, ids, _cracks(), coef, lod, temps, rains,
                  [d.isoformat() for d in dates])


# -------------------------------------------------------------------- cracks
def _cracks() -> list[dict]:
    """Each crack: where it sits, how it grows. width_mm(t) and length fraction(t)."""
    return [
        dict(id="beam crack", region="lintel_upper", onset=4.5, gain=0.45, w0=0.3,
             # vertical crack on the front of the beam above the settling column
             p0=np.array([2750.0, -3500.0, 4000.0]), p1=np.array([2850.0, -3500.0, 4600.0]),
             lmin=0.30, lmax=1.0, structural=True),
        dict(id="plinth crack", region="plinth_lower", onset=5.0, gain=0.35, w0=0.3,
             # runs along the plinth top away from the settling column
             p0=np.array([2800.0, -3000.0, 1000.0]), p1=np.array([1500.0, -1100.0, 1000.0]),
             lmin=0.2, lmax=1.0, structural=True),
        dict(id="tower joint crack", region="tower_upper", onset=3.0, gain=0.50, w0=0.3,
             # horizontal crack at the joint between tier 1 and 2, west face (the side that opens as it leans east)
             p0=np.array([-1700.0, 50.0, 6100.0]), p1=np.array([-1700.0, 3450.0, 6100.0]),
             lmin=0.25, lmax=1.0, structural=True),
        dict(id="hairline (control)", region="cella_front_upper", onset=99.0, gain=0.0, w0=0.3,
             p0=np.array([-900.0, 0.0, 2500.0]), p1=np.array([-400.0, 0.0, 3500.0]),
             lmin=1.0, lmax=1.0, structural=False),
    ]


def crack_width(c: dict, t: float) -> float:
    return c["w0"] + c["gain"] * max(0.0, t - c["onset"]) ** 1.2


def crack_polyline(c: dict, t: float, n: int = 14) -> np.ndarray:
    """Crack path at month t: starts at p0 and creeps toward p1 as it lengthens."""
    span = np.clip((t - c["onset"] + 1) / (N_MONTHS - c["onset"] + 1), 0, 1) if c["structural"] else 1.0
    frac = c["lmin"] + (c["lmax"] - c["lmin"]) * span
    s = np.linspace(0, frac, n)[:, None]
    wob = np.sin(np.linspace(0, 3.1, n))[:, None] * np.array([40.0, 0.0, 30.0])
    return c["p0"] + (c["p1"] - c["p0"]) * s + wob * (s > 0)


# -------------------------------------------------------------- the movement
def _structural(points: np.ndarray, region: np.ndarray, t: float) -> np.ndarray:
    """Truth: structural displacement vectors (mm) at month t, one per point."""
    d = np.zeros_like(points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # tower leans east
    L = 3.4 * max(0.0, t - 2.0) ** 1.15
    m = region == "tower_upper"
    f = np.clip((z[m] - 5200.0) / 4500.0, 0, 1) ** 1.5
    d[m, 0] += L * f

    # col_b settles; the ground around it, the porch beam and roof sag with it
    S = 1.6 * max(0.0, t - 4.0) ** 1.1
    bx, by = 2800.0, -3000.0
    m = region == "col_b_lower"
    d[m, 2] -= S
    m = region == "plinth_lower"
    near = np.exp(-(((x[m] - bx) ** 2 + (y[m] - by) ** 2) / (2 * 1800.0 ** 2)))
    d[m, 2] -= S * 0.8 * near
    m = region == "lintel_upper"
    d[m, 2] -= S * 0.55 * np.exp(-((x[m] - bx) / 2200.0) ** 2)
    m = region == "porch_roof_upper"
    d[m, 2] -= S * 0.35 * np.exp(-((x[m] - bx) / 2600.0) ** 2)
    return d


def _seasonal(points: np.ndarray, region: np.ndarray, coef: dict, dT: float) -> np.ndarray:
    """Breathing with temperature: outward from the temple's axis, per region."""
    out = points[:, :2] - np.array([0.0, 1750.0])
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    h = np.where(nrm > 1, out / np.maximum(nrm, 1), 0.0)
    c = np.array([coef[r] for r in region])
    d = np.zeros_like(points)
    d[:, :2] = h * (c * dT)[:, None]
    return d


@dataclass
class MonthState:
    month: int
    disp: np.ndarray                   # (n,3) truth displacement vs month 0, mm
    mag: np.ndarray                    # (n,) signed magnitude along the dominant direction, mm
    measured: np.ndarray               # (n,) what a noisy capture would show, mm
    significant: np.ndarray            # (n,) bool, |measured| > per-point detection limit
    cracks: list[dict] = field(default_factory=list)


def month_state(tp: Temple, month: int, seed: int = 0) -> MonthState:
    rng = np.random.default_rng(1000 + seed * 100 + month)
    dT = tp.temps[month] - tp.temps[0]
    d = _structural(tp.points, tp.region, month) + _seasonal(tp.points, tp.region, tp.coef, dT)
    # signed size: sign of the largest component, so settling reads negative, leaning positive
    dom = np.argmax(np.abs(d), axis=1)
    sign = np.sign(d[np.arange(len(d)), dom])
    sign[sign == 0] = 1
    mag = np.linalg.norm(d, axis=1) * sign
    noise = rng.normal(0, 1.5, len(mag))
    measured = mag + noise
    lod_pt = np.array([tp.lod[r] for r in tp.region])
    sig = np.abs(measured) > lod_pt
    cr = []
    for c in tp.cracks:
        poly = crack_polyline(c, month)
        reg = np.full(len(poly), c["region"])
        dc = _structural(poly, reg, month) + _seasonal(poly, reg, tp.coef, dT)
        cr.append(dict(id=c["id"], region=c["region"], poly=poly, disp=dc,
                       width=crack_width(c, month), width0=crack_width(c, 0)))
    return MonthState(month, d, mag, measured, sig, cr)


# ------------------------------------------------------------ through KEYSTONE
@dataclass
class YearResult:
    reports: list
    analyses: list                     # analyses[k] = analyse(reports[:k+1]) for the k-th report (month k+1)
    severity: dict                     # region -> list of severities for months 1..12
    first_alert: dict                  # region -> first month at ALERT or CRITICAL (or None)
    naive_first: dict                  # region -> first month a fixed 2 x lod threshold is crossed
    series: dict                       # region -> list of measured medians (months 1..12)
    crack_series: dict                 # crack id -> list of widths (months 0..12)


def run_year(tp: Temple, seed: int = 0) -> YearResult:
    reports = []
    rng = np.random.default_rng(seed + 7)
    metas = [CaptureMeta(f"C{i:03d}", MONUMENT_ID, tp.dates[i], round(tp.temps[i], 2),
                         round(tp.rains[i], 1), round(float(rng.uniform(4, 25)), 4))
             for i in range(N_MONTHS + 1)]
    series = {r: [] for r in tp.region_ids}
    for m in range(1, N_MONTHS + 1):
        st = month_state(tp, m, seed)
        regions = []
        for rid in tp.region_ids:
            sel = tp.region == rid
            truth_med = float(np.median(st.mag[sel]))
            med = truth_med + rng.normal(0, tp.lod[rid] / 3.0)
            p95 = float(np.percentile(np.abs(st.measured[sel]), 95))
            sig = float(np.mean(st.significant[sel]))
            series[rid].append(med)
            regions.append(RegionChange(rid, int(sel.sum()), round(med, 3), round(p95, 3),
                                        round(tp.lod[rid], 3), round(sig, 4), rid in LOAD_BEARING))
        cracks = []
        by_region: dict[str, dict] = {}
        for c in st.cracks:
            if not by_region.get(c["region"]) or c["width"] > by_region[c["region"]]["width"]:
                by_region[c["region"]] = c
        for rid, c in by_region.items():
            dw = c["width"] - c["width0"] + rng.normal(0, 0.1)
            length = float(np.linalg.norm(np.diff(c["poly"], axis=0), axis=1).sum())
            cracks.append(CrackChange(rid, round(dw, 3), round(max(0.0, dw) * length, 1), 0.5))
        reports.append(ChangeReport(MONUMENT_ID, metas[0], metas[m], round(float(rng.uniform(0.8, 2.0)), 3),
                                    regions, cracks))

    analyses = [analyse(reports[:k + 1]) for k in range(len(reports))]
    severity = {r: [] for r in tp.region_ids}
    for an in analyses:
        for ra in an:
            severity[ra.region_id].append(ra.alert.severity)
    first_alert, naive_first = {}, {}
    for rid in tp.region_ids:
        first_alert[rid] = next((i + 1 for i, s in enumerate(severity[rid]) if s in ("ALERT", "CRITICAL")), None)
        naive_first[rid] = next((i + 1 for i, v in enumerate(series[rid]) if abs(v) > 2 * tp.lod[rid]), None)
    crack_series = {c["id"]: [crack_width(c, m) for m in range(N_MONTHS + 1)] for c in tp.cracks}
    return YearResult(reports, analyses, severity, first_alert, naive_first, series, crack_series)


if __name__ == "__main__":  # quick check from a terminal
    tp = build_temple()
    print(len(tp.points), "points,", len(tp.region_ids), "regions")
    res = run_year(tp)
    for rid in tp.region_ids:
        print(f"{rid:20s} first ALERT month {res.first_alert[rid]}  naive {res.naive_first[rid]}  "
              f"final {res.severity[rid][-1]}")
