"""KEYSTONE capture and preservation contract.

This file is ADDITIVE. It does not change schema.py or inputs.py, so the
Geometry and Decision workstreams are unaffected by anything here.

WHERE THIS SITS
---------------
    [P] field capture ──▶ photogrammetry ──▶ [G] geometry ──▶ [D] decision
         registry              (off the             │              │
         quality gate           shelf)              │              │
              ▲                                     ▼              ▼
              └──────────────── [P] archive ◀───────┴──────────────┘

Workstream P owns both ends:

  FRONT DOOR   A monument must be described before it can be measured
               (where its survey markers are, which regions matter), and
               a photo set must be checked BEFORE the conservator leaves
               the site. A capture that fails quality checks back at the
               office means another trip to a remote monument, which in
               practice means it never gets re-measured at all.

  BACK DOOR    Every capture, report and alert is kept, because the
               archive is half of what KEYSTONE promises: if the monument
               is lost, the record survives it. That archive only ever
               grows, so it needs a retention policy, and the record is
               only worth keeping if something can be got back out of it
               — plans, elevations and sections.

OUT OF SCOPE FOR P: turning photos into a point cloud. That is
photogrammetry, done by an off-the-shelf tool (COLMAP, or a phone app).
P decides whether a photo set is GOOD ENOUGH to reconstruct; it does not
reconstruct.

UNITS
-----
Millimetres for anything spatial, matching schema.py. Photo footprints,
region boxes and marker positions are all in the monument's survey frame:
x along the face, y outward, z up.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date

__all__ = [
    "PRESERVATION_SCHEMA_VERSION",
    "QC_VERDICTS",
    "PreservationError",
    "MarkerSpec",
    "MonumentRecord",
    "PhotoMeta",
    "CaptureSession",
    "CheckResult",
    "CaptureQC",
    "ArchiveEntry",
    "ArchiveStats",
    "Section",
    "validate_monument_record",
    "validate_capture_session",
    "validate_capture_qc",
    "validate_archive_entry",
    "validate_section",
]

PRESERVATION_SCHEMA_VERSION = "1.0"

# Ordered, best to worst. The gate must use exactly these strings.
QC_VERDICTS = ("PASS", "PASS_WITH_GAPS", "RESHOOT")


class PreservationError(ValueError):
    """Something crossing a P boundary has the wrong shape, type or value."""


def _parse_date(s: str, what: str) -> date:
    try:
        return date.fromisoformat(s)
    except (TypeError, ValueError) as exc:
        raise PreservationError(
            f"{what} must be an ISO date YYYY-MM-DD, got {s!r}"
        ) from exc


# ==========================================================================
# the registry — what a monument IS, before anything is measured
# ==========================================================================
@dataclass
class MarkerSpec:
    """One surveyed ground-control marker.

    position_mm is where the marker really is in the monument's survey
    frame. Geometry fits a capture's observed marker positions onto these
    to recover scale and pose, so an error here becomes an error in every
    measurement of that monument, forever.

    accuracy_mm is how well the position itself is known. A marker
    established with a total station is sub-millimetre; one taped off a
    doorway is centimetres, and the whole monument inherits that.

    target_radius_mm is the size of the printed target on the wall. Targets
    in a set are deliberately different sizes, because a symmetric
    arrangement of identical markers cannot be told apart from geometry
    alone — the constellation fits equally well rotated 180 degrees, and
    every measurement then comes out mirrored. Size, or a printed code,
    is what breaks that tie.
    """

    marker_id: str
    position_mm: list[float]
    accuracy_mm: float = 2.0
    description: str = ""
    target_radius_mm: float = 40.0


@dataclass
class MonumentRecord:
    """Everything Geometry needs before it can measure a monument.

    reference_capture_id is the capture every later one is compared
    against. It is set once, when the first capture passes quality checks,
    and changing it afterwards invalidates the entire measurement history
    — so P must treat it as write-once.
    """

    monument_id: str
    name: str
    markers: list[MarkerSpec]
    regions: list[dict]                       # Region from schema.py, as dicts
    face_size_mm: list[float] = field(default_factory=lambda: [6000.0, 3000.0])
    reference_capture_id: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    schema_version: str = PRESERVATION_SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "MonumentRecord":
        d = json.loads(text)
        return cls(
            monument_id=d["monument_id"],
            name=d["name"],
            markers=[MarkerSpec(**m) for m in d["markers"]],
            regions=list(d["regions"]),
            face_size_mm=list(d.get("face_size_mm", [6000.0, 3000.0])),
            reference_capture_id=d.get("reference_capture_id"),
            latitude=d.get("latitude"),
            longitude=d.get("longitude"),
            schema_version=d.get("schema_version", PRESERVATION_SCHEMA_VERSION),
        )

    def survey(self) -> dict[str, list[float]]:
        """The survey dict Geometry's build_change_report() expects."""
        return {m.marker_id: list(m.position_mm) for m in self.markers}


# ==========================================================================
# the field — what comes back from a visit
# ==========================================================================
@dataclass
class PhotoMeta:
    """One photograph, described by what a quality gate needs to judge it.

    The person building P computes these FROM the image files. They are
    not metadata the camera supplies:

        sharpness       variance of the Laplacian. High is sharp. The
                        absolute value depends on the subject, so it is
                        compared within a session, not against a constant.
        exposure_clip   fraction of pixels at 0 or 255. Blown highlights
                        and crushed shadows carry no texture, and
                        photogrammetry needs texture.
        footprint_mm    the rectangle of the monument face this photo
                        covers, [x_min, z_min, x_max, z_max]. In a real
                        build this comes from the capture app's pose
                        estimate or from a coarse match; in the fakes it
                        is given.
        markers_visible which surveyed markers appear in this frame.
    """

    photo_id: str
    width: int
    height: int
    sharpness: float
    exposure_clip: float
    footprint_mm: list[float]
    markers_visible: list[str] = field(default_factory=list)
    iso: int | None = None
    path: str | None = None


@dataclass
class CaptureSession:
    """One visit: every photo taken, plus the conditions it was taken in.

    temperature_c and rainfall_mm_7d ride along because Decision needs
    them much later to explain seasonal movement. P records them at the
    site; nobody can reconstruct them afterwards.
    """

    capture_id: str
    monument_id: str
    captured_on: str
    photos: list[PhotoMeta]
    device: str = "smartphone"
    temperature_c: float | None = None
    rainfall_mm_7d: float | None = None
    operator: str = ""
    schema_version: str = PRESERVATION_SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "CaptureSession":
        d = json.loads(text)
        return cls(
            capture_id=d["capture_id"],
            monument_id=d["monument_id"],
            captured_on=d["captured_on"],
            photos=[PhotoMeta(**p) for p in d["photos"]],
            device=d.get("device", "smartphone"),
            temperature_c=d.get("temperature_c"),
            rainfall_mm_7d=d.get("rainfall_mm_7d"),
            operator=d.get("operator", ""),
            schema_version=d.get("schema_version", PRESERVATION_SCHEMA_VERSION),
        )


# ==========================================================================
# the gate — is this capture good enough to reconstruct?
# ==========================================================================
@dataclass
class CheckResult:
    """One named check, with enough detail to act on."""

    name: str
    passed: bool
    detail: str
    value: float | None = None
    threshold: float | None = None


@dataclass
class CaptureQC:
    """The gate's verdict on one capture session.

    verdict:
        PASS             reconstruct it
        PASS_WITH_GAPS   reconstruct it, but some regions will be missing
                         or weak; those region_ids are listed so Geometry
                         does not report confident numbers for them
        RESHOOT          do not reconstruct; the listed photos or areas
                         must be shot again before the operator leaves

    reshoot_guidance must be specific enough to act on at the site. "Poor
    quality" sends someone home none the wiser; "re-shoot the lower east
    face, 8 photos, markers GCP-1 and GCP-5 must be in frame" does not.
    """

    capture_id: str
    monument_id: str
    verdict: str
    checks: list[CheckResult]
    coverage_fraction: float
    weak_region_ids: list[str] = field(default_factory=list)
    reshoot_guidance: list[str] = field(default_factory=list)
    schema_version: str = PRESERVATION_SCHEMA_VERSION

    @property
    def usable(self) -> bool:
        return self.verdict in ("PASS", "PASS_WITH_GAPS")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "CaptureQC":
        d = json.loads(text)
        return cls(
            capture_id=d["capture_id"],
            monument_id=d["monument_id"],
            verdict=d["verdict"],
            checks=[CheckResult(**c) for c in d["checks"]],
            coverage_fraction=d["coverage_fraction"],
            weak_region_ids=list(d.get("weak_region_ids", [])),
            reshoot_guidance=list(d.get("reshoot_guidance", [])),
            schema_version=d.get("schema_version", PRESERVATION_SCHEMA_VERSION),
        )


# ==========================================================================
# the archive
# ==========================================================================
@dataclass
class ArchiveEntry:
    """One stored artefact.

    content_hash is what makes the archive trustworthy: it proves the
    bytes read back are the bytes written. An archive that cannot prove
    that is a folder, not a record.

    density is how much of the original point cloud is retained. 1.0 is
    everything; a decimated older capture might be 0.1. The reference
    capture is never decimated, because every future comparison is made
    against it.
    """

    artefact_id: str
    monument_id: str
    kind: str                                 # capture | pointcloud | report | alert | export
    created_on: str
    size_bytes: int
    content_hash: str
    density: float = 1.0
    is_reference: bool = False
    capture_id: str | None = None


@dataclass
class ArchiveStats:
    """What the archive currently costs, and what it would cost after tiering."""

    monument_id: str
    n_entries: int
    total_bytes: int
    reference_bytes: int
    reclaimable_bytes: int
    oldest: str | None = None
    newest: str | None = None


# ==========================================================================
# export — getting something back out
# ==========================================================================
@dataclass
class Section:
    """A 2D slice through a point cloud: a plan, elevation or section.

    This is what backs the claim that the archive "guides restoration".
    A point cloud is not a drawing; a slice through it, at a stated
    thickness and position, with real dimensions, is the start of one.

    polylines are ordered [u, v] points in millimetres, in the plane of
    the slice. extent_mm is [u_min, v_min, u_max, v_max].
    """

    monument_id: str
    capture_id: str
    kind: str                                 # plan | elevation | section
    at_mm: float
    thickness_mm: float
    polylines: list[list[list[float]]]
    extent_mm: list[float]
    n_points: int
    schema_version: str = PRESERVATION_SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# ==========================================================================
# validators
# ==========================================================================
def validate_monument_record(m: MonumentRecord) -> None:
    """The registry gate. Geometry may assume all of this."""
    if not isinstance(m, MonumentRecord):
        raise PreservationError(f"expected MonumentRecord, got {type(m).__name__}")
    if not m.monument_id or not m.name:
        raise PreservationError("monument_id and name are required")

    if len(m.markers) < 3:
        raise PreservationError(
            f"{m.monument_id}: {len(m.markers)} marker(s). At least 3 are needed "
            "to fix scale and pose, and they must not be collinear."
        )
    seen = set()
    for mk in m.markers:
        if mk.marker_id in seen:
            raise PreservationError(f"duplicate marker_id {mk.marker_id}")
        seen.add(mk.marker_id)
        if len(mk.position_mm) != 3:
            raise PreservationError(f"marker {mk.marker_id}: position must be [x, y, z]")
        if not mk.accuracy_mm > 0:
            raise PreservationError(f"marker {mk.marker_id}: accuracy_mm must be positive")

    # Three markers on one line fix scale but leave rotation about that line
    # free, so the fit looks fine and the cloud is free to pivot.
    if _markers_collinear([mk.position_mm for mk in m.markers]):
        raise PreservationError(
            f"{m.monument_id}: all markers are collinear, which leaves rotation "
            "about that line undetermined. Spread them across the face."
        )

    if not m.regions:
        raise PreservationError(f"{m.monument_id}: at least one region is required")
    region_ids = set()
    for r in m.regions:
        rid = r.get("region_id")
        if not rid:
            raise PreservationError("every region needs a region_id")
        if rid in region_ids:
            raise PreservationError(f"duplicate region_id {rid}")
        region_ids.add(rid)
        lo, hi = r.get("bounds_min"), r.get("bounds_max")
        if not lo or not hi or len(lo) != 3 or len(hi) != 3:
            raise PreservationError(f"region {rid}: bounds must be [x, y, z]")
        if any(a >= b for a, b in zip(lo, hi)):
            raise PreservationError(f"region {rid}: bounds_min must be < bounds_max")

    if len(m.face_size_mm) != 2 or any(v <= 0 for v in m.face_size_mm):
        raise PreservationError("face_size_mm must be [width, height], both positive")


def _markers_collinear(positions: list[list[float]], tol: float = 1e-6) -> bool:
    """True if every marker lies on one straight line."""
    if len(positions) < 3:
        return True
    ax, ay, az = positions[0]
    best = None
    for p in positions[1:]:
        v = (p[0] - ax, p[1] - ay, p[2] - az)
        n = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
        if n > tol:
            best = (v[0] / n, v[1] / n, v[2] / n)
            break
    if best is None:
        return True
    for p in positions[1:]:
        v = (p[0] - ax, p[1] - ay, p[2] - az)
        cross = (
            v[1] * best[2] - v[2] * best[1],
            v[2] * best[0] - v[0] * best[2],
            v[0] * best[1] - v[1] * best[0],
        )
        if (cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2) ** 0.5 > 1.0:  # 1 mm
            return False
    return True


def validate_capture_session(s: CaptureSession) -> None:
    if not isinstance(s, CaptureSession):
        raise PreservationError(f"expected CaptureSession, got {type(s).__name__}")
    if not s.capture_id or not s.monument_id:
        raise PreservationError("capture_id and monument_id are required")
    _parse_date(s.captured_on, "captured_on")
    if not s.photos:
        raise PreservationError(f"capture {s.capture_id} contains no photos")

    seen = set()
    for p in s.photos:
        if p.photo_id in seen:
            raise PreservationError(f"duplicate photo_id {p.photo_id}")
        seen.add(p.photo_id)
        if p.width <= 0 or p.height <= 0:
            raise PreservationError(f"photo {p.photo_id}: bad dimensions")
        if len(p.footprint_mm) != 4:
            raise PreservationError(
                f"photo {p.photo_id}: footprint must be [x_min, z_min, x_max, z_max]"
            )
        x0, z0, x1, z1 = p.footprint_mm
        if x1 <= x0 or z1 <= z0:
            raise PreservationError(f"photo {p.photo_id}: footprint has no area")
        if not 0.0 <= p.exposure_clip <= 1.0:
            raise PreservationError(f"photo {p.photo_id}: exposure_clip must be a fraction")


def validate_capture_qc(q: CaptureQC) -> None:
    """The P -> G gate. Geometry reconstructs only what passes this."""
    if not isinstance(q, CaptureQC):
        raise PreservationError(f"expected CaptureQC, got {type(q).__name__}")
    if q.verdict not in QC_VERDICTS:
        raise PreservationError(f"verdict must be one of {QC_VERDICTS}, got {q.verdict!r}")
    if not q.checks:
        raise PreservationError("a QC report must contain at least one check")
    if not 0.0 <= q.coverage_fraction <= 1.0:
        raise PreservationError(
            f"coverage_fraction must be in [0, 1], got {q.coverage_fraction}"
        )
    if q.verdict == "RESHOOT" and not q.reshoot_guidance:
        raise PreservationError(
            f"capture {q.capture_id}: RESHOOT with no guidance. The operator is still "
            "at the site and needs to be told what to shoot again."
        )
    if q.verdict == "PASS" and q.weak_region_ids:
        raise PreservationError(
            "PASS cannot list weak regions — use PASS_WITH_GAPS so Geometry knows "
            "not to report confident numbers for them"
        )
    if q.verdict == "PASS_WITH_GAPS" and not q.weak_region_ids:
        raise PreservationError("PASS_WITH_GAPS must name the weak regions")
    for c in q.checks:
        if not c.name or not c.detail:
            raise PreservationError("every check needs a name and a detail")


def validate_archive_entry(e: ArchiveEntry) -> None:
    if not isinstance(e, ArchiveEntry):
        raise PreservationError(f"expected ArchiveEntry, got {type(e).__name__}")
    if e.kind not in ("capture", "pointcloud", "report", "alert", "export"):
        raise PreservationError(f"unknown archive kind {e.kind!r}")
    if not e.content_hash or len(e.content_hash) < 16:
        raise PreservationError(f"{e.artefact_id}: content_hash looks too short to be a digest")
    if e.size_bytes < 0:
        raise PreservationError(f"{e.artefact_id}: negative size")
    if not 0.0 < e.density <= 1.0:
        raise PreservationError(f"{e.artefact_id}: density must be in (0, 1]")
    if e.is_reference and e.density < 1.0:
        raise PreservationError(
            f"{e.artefact_id}: the reference capture must never be decimated — "
            "every future comparison is made against it"
        )
    _parse_date(e.created_on, "created_on")


def validate_section(s: Section) -> None:
    if not isinstance(s, Section):
        raise PreservationError(f"expected Section, got {type(s).__name__}")
    if s.kind not in ("plan", "elevation", "section"):
        raise PreservationError(f"unknown section kind {s.kind!r}")
    if not s.thickness_mm > 0:
        raise PreservationError("thickness_mm must be positive")
    if len(s.extent_mm) != 4:
        raise PreservationError("extent must be [u_min, v_min, u_max, v_max]")
    u0, v0, u1, v1 = s.extent_mm
    if u1 <= u0 or v1 <= v0:
        raise PreservationError("extent has no area")
    for line in s.polylines:
        for pt in line:
            if len(pt) != 2:
                raise PreservationError("polyline points must be [u, v] in the slice plane")


if __name__ == "__main__":
    markers = [
        MarkerSpec("GCP-1", [50.0, 5.0, 50.0], 1.0),
        MarkerSpec("GCP-2", [5950.0, 5.0, 50.0], 1.0),
        MarkerSpec("GCP-3", [50.0, 5.0, 2950.0], 1.0),
    ]
    rec = MonumentRecord(
        monument_id="M-01", name="Test wall", markers=markers,
        regions=[{"region_id": "west_lower", "bounds_min": [0, -300, 0],
                  "bounds_max": [2000, 300, 1500], "load_bearing": False}],
    )
    validate_monument_record(rec)
    assert MonumentRecord.from_json(rec.to_json()) == rec, "JSON round-trip must be lossless"
    print("valid monument record, JSON round-trip lossless")
    print(f"survey dict for Geometry: {sorted(rec.survey())}")

    bad = {
        "two markers": lambda: validate_monument_record(
            MonumentRecord("M", "n", markers[:2], rec.regions)),
        "collinear markers": lambda: validate_monument_record(
            MonumentRecord("M", "n", [
                MarkerSpec("A", [0.0, 0.0, 0.0]),
                MarkerSpec("B", [1000.0, 0.0, 0.0]),
                MarkerSpec("C", [2000.0, 0.0, 0.0]),
            ], rec.regions)),
        "RESHOOT without guidance": lambda: validate_capture_qc(
            CaptureQC("C1", "M-01", "RESHOOT",
                      [CheckResult("x", False, "d")], 0.4)),
        "PASS that admits gaps": lambda: validate_capture_qc(
            CaptureQC("C1", "M-01", "PASS",
                      [CheckResult("x", True, "d")], 0.9, ["west_lower"])),
        "decimated reference": lambda: validate_archive_entry(
            ArchiveEntry("a", "M-01", "pointcloud", "2026-01-10", 10,
                         "0" * 32, density=0.1, is_reference=True)),
    }
    for name, fn in bad.items():
        try:
            fn()
            print(f"  MISSED: {name}")
        except PreservationError:
            print(f"  rejected: {name}")
