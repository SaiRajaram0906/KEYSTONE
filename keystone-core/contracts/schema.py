"""KEYSTONE shared contract — the only thing both workstreams must agree on.

    WORKSTREAM G (Geometry)                  WORKSTREAM D (Decision)
    two point clouds  ──▶  ChangeReport  ──▶  alerts, archive, dashboard
                               ▲
          crack-seg team ──────┘  (CrackChange, via crack_adapter.py)

RULES
-----
1. Changing this file needs both workstreams to agree. It breaks the other
   side by definition. Everything outside contracts/ is yours to rewrite.
2. Units are millimetres everywhere in this file. No pixels, no metres,
   no "cloud units". Conversion happens inside Geometry, before anything
   crosses the boundary.
3. `earlier` in a ChangeReport is always the monument's REFERENCE capture
   (its first capture), never "the previous one". So region displacement
   is cumulative since the reference. Decision relies on this to see
   seasonal movement as an oscillation and structural movement as a trend.
4. Serialise with to_json / from_json. Both sides test against JSON, so
   the boundary works across processes, machines and languages.

Depends on the standard library only, so it installs anywhere.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date

__all__ = [
    "SCHEMA_VERSION",
    "SEVERITIES",
    "ContractError",
    "CaptureMeta",
    "Region",
    "RegionChange",
    "CrackChange",
    "ChangeReport",
    "Alert",
    "validate_capture_meta",
    "validate_region",
    "validate_change_report",
    "validate_alert",
]

SCHEMA_VERSION = "1.0"

# Ordered, least to most severe. Decision must use exactly these strings.
SEVERITIES = ("INFORMATIONAL", "WATCH", "ALERT", "CRITICAL")


class ContractError(ValueError):
    """Something crossing the boundary has the wrong shape, type or value."""


def _parse_date(s: str, what: str) -> date:
    try:
        return date.fromisoformat(s)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{what} must be an ISO date YYYY-MM-DD, got {s!r}") from exc


# --------------------------------------------------------------------------
# inputs and shared vocabulary
# --------------------------------------------------------------------------
@dataclass
class CaptureMeta:
    """One visit to one monument.

    mm_per_unit is how many millimetres one unit of the RAW capture spans.
    It is unknown when a capture arrives (None) and Geometry fills it in
    after fitting the surveyed markers. A ChangeReport must never carry None.

    temperature_c and rainfall_mm_7d are the environmental metadata Decision
    uses to explain seasonal movement. Rainfall is the total over the seven
    days before capture.
    """

    capture_id: str
    monument_id: str
    captured_on: str
    temperature_c: float | None = None
    rainfall_mm_7d: float | None = None
    mm_per_unit: float | None = None


@dataclass
class Region:
    """A named part of the monument, as an axis-aligned box in the survey frame (mm).

    The survey frame is fixed by the surveyed marker coordinates, so the same
    box means the same stones in every capture. load_bearing marks walls,
    columns and arches carrying structural load; Decision escalates faster
    there.
    """

    region_id: str
    bounds_min: list[float]
    bounds_max: list[float]
    load_bearing: bool = False


# --------------------------------------------------------------------------
# G -> D
# --------------------------------------------------------------------------
@dataclass
class RegionChange:
    """Cumulative surface movement of one region since the reference capture.

    median_mm            signed median displacement along the surface normal.
                         Positive = surface moved outward.
    p95_abs_mm           95th percentile of absolute per-point displacement.
                         Catches localised movement a median hides.
    lod_mm               level of detection at 95% confidence, PER POINT: the
                         median over this region of each core point's
                         M3C2-style LoD (local roughness of both clouds plus
                         registration error). A single point moving less than
                         this cannot be told apart from noise.
    significant_fraction share of points whose |displacement| exceeds their
                         own level of detection.
    """

    region_id: str
    n_points: int
    median_mm: float
    p95_abs_mm: float
    lod_mm: float
    significant_fraction: float
    load_bearing: bool = False


@dataclass
class CrackChange:
    """Crack widening in one region, from the crack-seg team's output.

    Built by contracts.crack_adapter.from_crack_delta; neither workstream
    constructs these by hand. width_uncertainty_mm carries the crack model's
    +/- 1 px quantisation error so Decision can ignore changes inside it.
    """

    region_id: str
    d_max_width_mm: float
    new_crack_area_mm2: float
    width_uncertainty_mm: float


@dataclass
class ChangeReport:
    """Geometry's complete output for one capture compared to the reference."""

    monument_id: str
    earlier: CaptureMeta
    later: CaptureMeta
    registration_rmse_mm: float
    regions: list[RegionChange]
    crack_changes: list[CrackChange] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "ChangeReport":
        d = json.loads(text)
        return cls(
            monument_id=d["monument_id"],
            earlier=CaptureMeta(**d["earlier"]),
            later=CaptureMeta(**d["later"]),
            registration_rmse_mm=d["registration_rmse_mm"],
            regions=[RegionChange(**r) for r in d["regions"]],
            crack_changes=[CrackChange(**c) for c in d.get("crack_changes", [])],
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )


# --------------------------------------------------------------------------
# D -> the world
# --------------------------------------------------------------------------
@dataclass
class Alert:
    """One decision, for one region, after one capture.

    Every alert names the two capture ids it was derived from, so any
    decision can be traced back to the measurements that caused it.
    """

    monument_id: str
    region_id: str
    severity: str
    recipient: str
    reason: str
    reference_capture_id: str
    latest_capture_id: str
    decided_on: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Alert":
        return cls(**json.loads(text))


# --------------------------------------------------------------------------
# validators
# --------------------------------------------------------------------------
def validate_capture_meta(m: CaptureMeta, require_scale: bool = False) -> None:
    if not isinstance(m, CaptureMeta):
        raise ContractError(f"expected CaptureMeta, got {type(m).__name__}")
    if not m.capture_id or not m.monument_id:
        raise ContractError("capture_id and monument_id are required")
    _parse_date(m.captured_on, "captured_on")
    if require_scale:
        if m.mm_per_unit is None:
            raise ContractError(
                f"capture {m.capture_id} has no mm_per_unit — Geometry must fill "
                "it in from the surveyed markers before building a ChangeReport"
            )
        if not m.mm_per_unit > 0:
            raise ContractError(f"mm_per_unit must be positive, got {m.mm_per_unit}")


def validate_region(r: Region) -> None:
    if not r.region_id:
        raise ContractError("region_id is required")
    if len(r.bounds_min) != 3 or len(r.bounds_max) != 3:
        raise ContractError(f"region {r.region_id}: bounds must be [x, y, z]")
    if any(lo >= hi for lo, hi in zip(r.bounds_min, r.bounds_max)):
        raise ContractError(f"region {r.region_id}: bounds_min must be < bounds_max on every axis")


def validate_change_report(rep: ChangeReport) -> None:
    """The G -> D gate. Geometry must make this pass; Decision may assume it."""
    if not isinstance(rep, ChangeReport):
        raise ContractError(f"expected ChangeReport, got {type(rep).__name__}")
    if rep.schema_version != SCHEMA_VERSION:
        raise ContractError(f"schema_version {rep.schema_version} != {SCHEMA_VERSION}")

    validate_capture_meta(rep.earlier, require_scale=True)
    validate_capture_meta(rep.later, require_scale=True)
    if not (rep.monument_id == rep.earlier.monument_id == rep.later.monument_id):
        raise ContractError("monument_id must match across the report and both captures")
    if _parse_date(rep.later.captured_on, "later") <= _parse_date(rep.earlier.captured_on, "earlier"):
        raise ContractError("later.captured_on must be after earlier.captured_on")
    if rep.earlier.capture_id == rep.later.capture_id:
        raise ContractError("a capture cannot be compared with itself")
    if not rep.registration_rmse_mm >= 0:
        raise ContractError("registration_rmse_mm must be >= 0")

    if not rep.regions:
        raise ContractError("a ChangeReport must contain at least one region")
    seen = set()
    for r in rep.regions:
        if r.region_id in seen:
            raise ContractError(f"duplicate region_id {r.region_id}")
        seen.add(r.region_id)
        if r.n_points <= 0:
            raise ContractError(f"region {r.region_id}: n_points must be > 0")
        if not r.lod_mm > 0:
            raise ContractError(f"region {r.region_id}: lod_mm must be > 0")
        if not 0.0 <= r.significant_fraction <= 1.0:
            raise ContractError(f"region {r.region_id}: significant_fraction must be in [0, 1]")
        # |median| <= median(|d|) <= p95(|d|): a violation means the stats are inconsistent.
        if r.p95_abs_mm + 1e-9 < abs(r.median_mm):
            raise ContractError(
                f"region {r.region_id}: p95_abs_mm ({r.p95_abs_mm}) cannot be below "
                f"|median_mm| ({abs(r.median_mm)})"
            )

    for c in rep.crack_changes:
        if c.region_id not in seen:
            raise ContractError(f"crack change refers to unknown region {c.region_id}")
        if c.width_uncertainty_mm < 0 or c.new_crack_area_mm2 < 0:
            raise ContractError(f"crack change in {c.region_id} has a negative uncertainty or area")


def validate_alert(a: Alert) -> None:
    """The D -> world gate."""
    if not isinstance(a, Alert):
        raise ContractError(f"expected Alert, got {type(a).__name__}")
    if a.severity not in SEVERITIES:
        raise ContractError(f"severity must be one of {SEVERITIES}, got {a.severity!r}")
    for f in ("monument_id", "region_id", "recipient", "reason",
              "reference_capture_id", "latest_capture_id"):
        if not getattr(a, f):
            raise ContractError(f"alert field '{f}' must not be empty")
    _parse_date(a.decided_on, "decided_on")


if __name__ == "__main__":
    ref = CaptureMeta("C000", "M-01", "2026-01-10", 24.0, 0.0, 0.8)
    new = CaptureMeta("C004", "M-01", "2026-05-10", 34.0, 12.0, 0.8)
    rep = ChangeReport(
        monument_id="M-01", earlier=ref, later=new, registration_rmse_mm=1.4,
        regions=[RegionChange("north_upper", 5200, 6.1, 9.8, 3.2, 0.61, True)],
    )
    validate_change_report(rep)
    again = ChangeReport.from_json(rep.to_json())
    validate_change_report(again)
    assert again == rep, "JSON round-trip must be lossless"
    print("valid report, JSON round-trip lossless")

    bad_cases = {
        "missing scale": lambda: validate_change_report(ChangeReport(
            "M-01", CaptureMeta("C0", "M-01", "2026-01-10"), new, 1.0, rep.regions)),
        "later not after earlier": lambda: validate_change_report(ChangeReport(
            "M-01", new, ref, 1.0, rep.regions)),
        "inconsistent stats": lambda: validate_change_report(ChangeReport(
            "M-01", ref, new, 1.0,
            [RegionChange("r", 10, 8.0, 2.0, 1.0, 0.5)])),
        "unknown severity": lambda: validate_alert(Alert(
            "M-01", "r", "URGENT", "office", "x", "C0", "C4", "2026-05-10")),
    }
    for name, fn in bad_cases.items():
        try:
            fn()
            print(f"MISSED: {name}")
        except ContractError:
            print(f"rejected: {name}")
