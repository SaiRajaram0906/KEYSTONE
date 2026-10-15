"""Console engine: runs the real KEYSTONE code and returns plain results for the UI.

No Streamlit in here, so every function can be tested from a terminal.
Everything uses the project's own modules (preservation, geometry, decision,
contracts, crack-seg measurement). The data is the project's synthetic fixtures
with known ground truth, unless the caller passes real photos to `reconstruct_photos`.
"""
from __future__ import annotations

import hashlib
import html
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("crack-seg", "keystone-core"):          # keystone-core ends up first
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2  # noqa: E402

from contracts.crack_adapter import from_crack_delta                                # noqa: E402
from contracts.fakes import WALL_REGIONS, make_geometry_case, make_report_series    # noqa: E402
from contracts.fakes_capture import (SCENARIOS, coverage_grid, make_monument,       # noqa: E402
                                     make_session, make_wall_cloud)
from contracts.schema import ChangeReport, validate_change_report                   # noqa: E402
from decision.pipeline import analyse, evaluate                                     # noqa: E402
from decision.routing import ROUTES                                                 # noqa: E402
from geometry.align import apply_rigid, fit_to_survey                               # noqa: E402
from geometry.m3c2 import compute_m3c2                                              # noqa: E402
from geometry.pipeline import build_change_report                                   # noqa: E402
from geometry.register import refine                                                # noqa: E402
from keystone.interface import compare_captures                                     # noqa: E402
from preservation.archive import Archive                                            # noqa: E402
from preservation.coverage import coverage_grid as gate_grid                       # noqa: E402
from preservation.export import extract_section                                     # noqa: E402
from preservation.gate import check_capture                                         # noqa: E402

REGION_IDS = [r.region_id for r in WALL_REGIONS]
REGION_BY_ID = {r.region_id: r for r in WALL_REGIONS}
CAPTURE_SCENARIOS = list(SCENARIOS)


# --------------------------------------------------------------------------- P
@dataclass
class CaptureCheck:
    scenario: str
    qc: object
    grid: np.ndarray
    truth: object


def capture_check(scenario: str, seed: int = 0) -> CaptureCheck:
    monument = make_monument()
    session, truth = make_session(scenario, seed=seed, monument=monument)
    qc = check_capture(session, monument)
    return CaptureCheck(scenario, qc, gate_grid(session, monument), truth)


# --------------------------------------------------------------------------- G
@dataclass
class GeometryRun:
    report: ChangeReport
    core: np.ndarray            # (N,3) core points, survey mm
    dist: np.ndarray            # signed displacement mm
    lod: np.ndarray             # per-point level of detection mm
    valid: np.ndarray
    truth_mm: dict
    scale_found: dict           # capture id -> recovered mm per unit
    scale_true: dict
    registration_rmse_mm: float
    planted_region: str
    planted_mm: float


def run_geometry(seed: int, region: str, mm: float, n_points: int = 60000) -> GeometryRun:
    displaced = {region: float(mm)} if mm else {}
    case = make_geometry_case(seed=seed, displaced=displaced, n_points=n_points)

    # the same steps as geometry.pipeline.build_change_report, kept so we can draw the points
    sim_e, rmse_e, _ = fit_to_survey(case.earlier.markers, case.survey)
    sim_l, rmse_l, _ = fit_to_survey(case.later.markers, case.survey)
    earlier_mm = sim_e.apply(case.earlier.points)
    later_mm = sim_l.apply(case.later.points)
    T, _ = refine(earlier_mm, later_mm)
    later_ref = apply_rigid(later_mm, T)
    reg = float(math.sqrt(rmse_e ** 2 + rmse_l ** 2))
    core, dist, lod, valid = compute_m3c2(earlier_mm, later_ref, registration_error_mm=reg)

    report = build_change_report(case.earlier, case.later, case.survey, case.regions)
    return GeometryRun(
        report=report, core=core, dist=dist, lod=lod, valid=valid.astype(bool),
        truth_mm=case.truth_displacement_mm,
        scale_found={case.earlier.meta.capture_id: float(sim_e.scale),
                     case.later.meta.capture_id: float(sim_l.scale)},
        scale_true=case.truth_mm_per_unit, registration_rmse_mm=reg,
        planted_region=region, planted_mm=float(mm))


# ------------------------------------------------------------------ crack-seg
@dataclass
class CrackRun:
    delta: object
    change: object
    mask_a: np.ndarray
    mask_b: np.ndarray
    mm_per_px: float


def example_masks(w_before: int = 3, w_after: int = 7, shape=(400, 600)):
    def one(w):
        m = np.zeros(shape, np.uint8)
        cv2.line(m, (40, 60), (shape[1] - 40, shape[0] - 60), 1, thickness=w)
        return m
    return one(w_before), one(w_after)


def crack_run(mask_a: np.ndarray, mask_b: np.ndarray, region: str, mm_per_px: float) -> CrackRun:
    delta = compare_captures(mask_a, mask_b, mm_per_px=mm_per_px)
    change = from_crack_delta(delta.as_dict(), region, mm_per_px)
    return CrackRun(delta, change, mask_a, mask_b, mm_per_px)


def chain_alerts(geo: GeometryRun, crack: CrackRun | None):
    """Geometry report (+ crack change) crosses JSON, then Decision."""
    report = ChangeReport.from_json(geo.report.to_json())
    if crack is not None:
        report.crack_changes = [crack.change]
    validate_change_report(report)
    report = ChangeReport.from_json(report.to_json())
    return report, evaluate([report])


# --------------------------------------------------------------------------- D
@dataclass
class YearRun:
    reports: list
    analyses: list
    truth: object


def year_run(seed: int, region: str, rate: float, n_captures: int = 10) -> YearRun:
    reports, truth = make_report_series(n_captures=n_captures, seed=seed,
                                        drift_region=region or None,
                                        drift_mm_per_capture=rate)
    return YearRun(reports, analyse(reports), truth)


# ------------------------------------------------------------- act and record
def order_number(alert) -> str:
    h = hashlib.sha1(f"{alert.monument_id}{alert.region_id}{alert.latest_capture_id}".encode()).hexdigest()[:6].upper()
    return f"KS-{alert.decided_on.replace('-', '')}-{h}"


def inspection_order_html(alert, analysis=None, region_change=None) -> str:
    route = ROUTES[alert.severity]
    esc = html.escape
    rows = [("Order number", order_number(alert)), ("Date", alert.decided_on),
            ("Monument", alert.monument_id), ("Region", alert.region_id),
            ("Severity", alert.severity), ("Sent to", alert.recipient),
            ("Reference capture", alert.reference_capture_id), ("Latest capture", alert.latest_capture_id)]
    if analysis is not None:
        rows.append(("Load-bearing", "yes" if analysis.load_bearing else "no"))
    if region_change is not None:
        rows.append(("Measured movement", f"{region_change.median_mm:+.1f} mm (level of detection {region_change.lod_mm:.1f} mm)"))
    table = "".join(f"<tr><th>{esc(k)}</th><td>{esc(str(v))}</td></tr>" for k, v in rows)
    colour = {"INFORMATIONAL": "#2f7f4f", "WATCH": "#b7791f", "ALERT": "#c4561d", "CRITICAL": "#b3261e"}[alert.severity]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Inspection order {esc(order_number(alert))}</title>
<style>
body{{font:15px/1.5 Georgia,serif;color:#1e2b33;max-width:720px;margin:2rem auto;padding:0 1rem}}
h1{{font-size:1.5rem;margin:0}} .sev{{display:inline-block;color:#fff;background:{colour};padding:.15rem .6rem;border-radius:3px;font:600 .9rem sans-serif}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}} th,td{{text-align:left;padding:.35rem .6rem;border-bottom:1px solid #ccd}} th{{width:11rem;color:#4a5a63;font-weight:600}}
.box{{border-left:4px solid {colour};background:#f2f4f0;padding:.7rem 1rem;margin:1rem 0}}
.sign{{margin-top:2.5rem}} .sign div{{border-bottom:1px solid #1e2b33;height:2rem;margin:.4rem 0 .1rem}} small{{color:#4a5a63}}
</style></head><body>
<h1>Inspection order</h1><p><span class="sev">{esc(alert.severity)}</span></p>
<table>{table}</table>
<div class="box"><b>Why this was raised</b><br>{esc(alert.reason)}</div>
<div class="box"><b>Recommended action</b><br>{esc(route.action)}</div>
<div class="sign"><b>Validated by conservator</b><div></div><small>Name, signature, date</small></div>
<p><small>Generated by KEYSTONE from measurements, not by an inspector. The system recommends; the custodian decides.
Every figure above traces to the two captures named in this order.</small></p>
</body></html>"""


@dataclass
class RecordRun:
    stats: object
    verified: bool
    entries: list
    sections: dict = field(default_factory=dict)


def record_run(geo: GeometryRun) -> RecordRun:
    """Archive the reference and latest clouds, verify integrity, extract measured sections."""
    tmp = Path(tempfile.mkdtemp(prefix="keystone_archive_"))
    arch = Archive(tmp)
    mon = make_monument()
    rep = geo.report
    ref = arch.store(np.asarray(geo.core, np.float32).tobytes(), monument_id=rep.monument_id, kind="pointcloud",
                     created_on=rep.earlier.captured_on, capture_id=rep.earlier.capture_id, is_reference=True)
    change_map = np.column_stack([geo.core, geo.dist, geo.lod]).astype(np.float32)
    later = arch.store(change_map.tobytes(), monument_id=rep.monument_id, kind="pointcloud",
                       created_on=rep.later.captured_on, capture_id=rep.later.capture_id)
    verified = arch.verify(ref.artefact_id) and arch.verify(later.artefact_id)
    cloud = make_wall_cloud(buttress=True)
    sections = {
        "plan": extract_section(cloud, mon, rep.later.capture_id, "plan", 1500.0, 100.0),
        "elevation": extract_section(cloud, mon, rep.later.capture_id, "elevation", 0.0, 80.0),
        "section": extract_section(cloud, mon, rep.later.capture_id, "section", 2200.0, 100.0),
    }
    return RecordRun(arch.stats(rep.monument_id), verified, arch.entries(rep.monument_id), sections)


def section_svg(section, width_px: int = 900) -> str:
    """A measured drawing as SVG, in millimetres, with a scale bar."""
    x0, y0, x1, y1 = section.extent_mm
    w, h = x1 - x0, y1 - y0
    s = (width_px - 40) / max(w, 1.0)
    H = int(h * s) + 70
    pts = " ".join(f"{20 + (x - x0) * s:.1f},{H - 40 - (y - y0) * s:.1f}" for x, y in section.polylines[0][::3])
    bar_mm = 1000
    bar = bar_mm * s
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" height="{H}" viewBox="0 0 {width_px} {H}">'
            f'<rect width="100%" height="100%" fill="#fff"/>'
            f'<polyline points="{pts}" fill="none" stroke="#1e2b33" stroke-width="1"/>'
            f'<line x1="20" y1="{H - 18}" x2="{20 + bar:.1f}" y2="{H - 18}" stroke="#1e2b33" stroke-width="2"/>'
            f'<text x="20" y="{H - 4}" font-family="sans-serif" font-size="12">1 m  ({section.kind} at {section.at_mm:.0f} mm, '
            f'{section.n_points} measured points)</text></svg>')


# ------------------------------------------------------------ your own photos
def reconstruct_photos(image_dir: str, out_dir: str, tries: int = 3, max_image_size: int = 2000) -> tuple[np.ndarray | None, str]:
    """Photos to a sparse point cloud via tools/sparse_reconstruct.py. Returns (points, log)."""
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "sparse_reconstruct.py"),
        image_dir,
        out_dir,
        "--tries", str(tries),
        "--max-image-size", str(max_image_size),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = (proc.stdout + proc.stderr).strip()
    ply = Path(out_dir) / "sparse_points.ply"
    if proc.returncode != 0 or not ply.exists():
        return None, log
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(ply))
    return np.asarray(pcd.points), log
