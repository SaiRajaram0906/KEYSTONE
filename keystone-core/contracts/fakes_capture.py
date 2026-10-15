"""Synthetic capture sessions with known defects, for workstream P.

Nobody has a labelled corpus of good and bad heritage photo sets, so the
quality gate is developed and SCORED against sessions whose faults are
known exactly.

Each scenario names one realistic way a field capture goes wrong:

    good            what a careful capture looks like
    too_few         shot from too far back: only 6 frames, each covering a
                    huge span of wall. Coverage and overlap are both fine,
                    but every pixel spans several millimetres of stone, so
                    a 3 mm crack is smaller than a pixel. Caught by ground
                    sampling distance, not by counting photos.
    blurry          handheld in low light; a third of the frames shaken
    dark_gap        the lower east face never photographed
    markers_missing fewer than 3 surveyed markers ever in frame, so the
                    cloud can never be scaled or placed
    overexposed     shot into afternoon sun; highlights blown, no texture
    thin_overlap    THE TRAP. Every part of the wall appears in some photo,
                    so a naive "is it covered?" check passes. But almost no
                    point appears in three or more photos, and
                    photogrammetry needs multiple views per point to
                    triangulate. A gate that checks coverage but not
                    redundancy will wave this through, and reconstruction
                    fails back at the office.

Note that too_few and thin_overlap fail in opposite ways: one is at the
wrong distance with good overlap, the other at the right distance with
too few views per point. A gate needs both checks; neither alone is enough.

PHOTO METADATA vs REAL IMAGES
-----------------------------
The contract is on PhotoMeta, so the acceptance tests need no image files
and run in under a second. But sharpness and exposure_clip are numbers a
real build must COMPUTE from photographs, so write_sample_images() emits
real JPEGs — sharp, shaken and blown-out — to develop that extractor
against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from contracts.preservation import (
    CaptureSession,
    MarkerSpec,
    MonumentRecord,
    PhotoMeta,
)

__all__ = [
    "SCENARIOS",
    "WALL_W",
    "WALL_H",
    "SessionTruth",
    "make_monument",
    "make_session",
    "make_wall_cloud",
    "make_capture_cloud",
    "CloudTruth",
    "coverage_grid",
    "write_sample_images",
]

WALL_W, WALL_H = 6000.0, 3000.0          # matches contracts/fakes.py

SCENARIOS = (
    "good",
    "too_few",
    "blurry",
    "dark_gap",
    "markers_missing",
    "overexposed",
    "thin_overlap",
)

# Which scenarios must never be reconstructed, and which are salvageable.
_MUST_RESHOOT = {"too_few", "blurry", "markers_missing", "overexposed", "thin_overlap"}
_SALVAGEABLE = {"dark_gap"}


@dataclass
class SessionTruth:
    """What is really wrong with a session — for scoring, never for the gate."""

    scenario: str
    should_reshoot: bool
    should_flag_gaps: bool
    bad_photo_ids: list[str]
    uncovered_region_ids: list[str]


def make_monument(monument_id: str = "M-01") -> MonumentRecord:
    """The same 6 m x 3 m wall the Geometry and Decision fakes use."""
    markers = [
        # Sizes differ on purpose: this arrangement is symmetric under a
        # 180-degree rotation, so identical targets could not be told apart.
        MarkerSpec("GCP-1", [50.0, 5.0, 50.0], 1.0, "west end, low", 30.0),
        MarkerSpec("GCP-2", [5950.0, 5.0, 50.0], 1.0, "east end, low", 36.0),
        MarkerSpec("GCP-3", [50.0, 5.0, 2950.0], 1.0, "west end, high", 42.0),
        MarkerSpec("GCP-4", [5950.0, 5.0, 2950.0], 1.0, "east end, high", 48.0),
        MarkerSpec("GCP-5", [3000.0, 5.0, 60.0], 1.5, "centre, low", 54.0),
        MarkerSpec("GCP-6", [3000.0, 5.0, 2940.0], 1.5, "centre, high", 60.0),
    ]
    regions = [
        {
            "region_id": f"{col}_{row}",
            "bounds_min": [x0, -300.0, z0],
            "bounds_max": [x0 + 2000.0, 300.0, z0 + 1500.0],
            "load_bearing": col == "centre",
        }
        for col, x0 in (("west", 0.0), ("centre", 2000.0), ("east", 4000.0))
        for row, z0 in (("lower", 0.0), ("upper", 1500.0))
    ]
    return MonumentRecord(
        monument_id=monument_id,
        name="Synthetic masonry wall",
        markers=markers,
        regions=regions,
        face_size_mm=[WALL_W, WALL_H],
    )


def _markers_in(footprint: list[float], monument: MonumentRecord) -> list[str]:
    x0, z0, x1, z1 = footprint
    return [
        m.marker_id
        for m in monument.markers
        if x0 <= m.position_mm[0] <= x1 and z0 <= m.position_mm[2] <= z1
    ]


def _tile_footprints(cols: int, rows: int, overlap: float) -> list[list[float]]:
    """Grid of photo footprints covering the wall with the given overlap.

    overlap is the fraction of a frame shared with its neighbour. 0.75 is
    a careful capture; 0.05 covers the wall while overlapping almost
    nothing, which is the thin_overlap trap.
    """
    step_x = WALL_W / cols
    step_z = WALL_H / rows
    fw = step_x / max(1e-6, (1.0 - overlap))
    fh = step_z / max(1e-6, (1.0 - overlap))
    out = []
    for r in range(rows):
        for c in range(cols):
            cx = step_x * (c + 0.5)
            cz = step_z * (r + 0.5)
            out.append([
                max(0.0, cx - fw / 2), max(0.0, cz - fh / 2),
                min(WALL_W, cx + fw / 2), min(WALL_H, cz + fh / 2),
            ])
    return out


def make_session(
    scenario: str = "good",
    seed: int = 0,
    monument: MonumentRecord | None = None,
    capture_id: str = "C001",
    captured_on: str = "2026-03-15",
) -> tuple[CaptureSession, SessionTruth]:
    """Build one synthetic capture session and the truth about it."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    monument = monument or make_monument()
    rng = np.random.default_rng(seed)

    cols, rows, overlap = 8, 5, 0.72
    if scenario == "too_few":
        cols, rows = 2, 2                    # far back: few, very wide frames
    elif scenario == "thin_overlap":
        overlap = 0.04

    footprints = _tile_footprints(cols, rows, overlap)

    # dark_gap: the operator never pointed the camera at the lower east
    # quarter, so drop every frame that sees any part of it. Dropping only
    # frames centred there is not enough — neighbours spill in and fill it.
    uncovered: list[str] = []
    if scenario == "dark_gap":
        gx0, gz0, gx1, gz1 = 4000.0, 0.0, WALL_W, 1500.0
        footprints = [
            f for f in footprints
            if not (f[0] < gx1 and f[2] > gx0 and f[1] < gz1 and f[3] > gz0)
        ]
        uncovered = ["east_lower"]

    bad_ids: list[str] = []
    photos: list[PhotoMeta] = []
    for i, fp in enumerate(footprints):
        pid = f"{capture_id}-P{i:03d}"
        sharpness = float(rng.normal(180.0, 25.0))
        clip = float(abs(rng.normal(0.004, 0.002)))
        visible = _markers_in(fp, monument)

        if scenario == "blurry" and i % 3 == 0:
            sharpness = float(rng.uniform(8.0, 28.0))       # shaken
            bad_ids.append(pid)
        if scenario == "overexposed" and i % 2 == 0:
            clip = float(rng.uniform(0.22, 0.40))           # highlights gone
            bad_ids.append(pid)
        if scenario == "markers_missing":
            # Framed tight on the stonework, so the markers out at the ends
            # never made it into a single frame. Only the two centre markers
            # are ever seen, and two cannot fix rotation.
            visible = [m for m in visible if m in ("GCP-5", "GCP-6")]

        photos.append(PhotoMeta(
            photo_id=pid,
            width=4032, height=3024,
            sharpness=round(max(1.0, sharpness), 2),
            exposure_clip=round(min(1.0, clip), 4),
            footprint_mm=[round(v, 1) for v in fp],
            markers_visible=visible,
            iso=int(rng.choice([100, 200, 400, 800])),
        ))

    session = CaptureSession(
        capture_id=capture_id,
        monument_id=monument.monument_id,
        captured_on=captured_on,
        photos=photos,
        device="smartphone",
        temperature_c=round(float(rng.uniform(24, 34)), 1),
        rainfall_mm_7d=round(float(abs(rng.normal(5, 8))), 1),
        operator="field-officer",
    )
    truth = SessionTruth(
        scenario=scenario,
        should_reshoot=scenario in _MUST_RESHOOT,
        should_flag_gaps=scenario in _SALVAGEABLE,
        bad_photo_ids=sorted(set(bad_ids)),
        uncovered_region_ids=uncovered,
    )
    return session, truth


def coverage_grid(session: CaptureSession, cell_mm: float = 100.0) -> np.ndarray:
    """How many photos see each cell of the wall.

    A reference implementation of the idea the gate is built on, provided
    so nobody has to reinvent it: coverage is cells seen at least once,
    redundancy is cells seen at least three times. Photogrammetry
    triangulates a point from multiple views, so redundancy is what
    actually predicts whether reconstruction succeeds.
    """
    nx = max(1, int(WALL_W / cell_mm))
    nz = max(1, int(WALL_H / cell_mm))
    grid = np.zeros((nz, nx), dtype=np.int32)
    for p in session.photos:
        x0, z0, x1, z1 = p.footprint_mm
        c0, c1 = int(x0 / cell_mm), int(math.ceil(x1 / cell_mm))
        r0, r1 = int(z0 / cell_mm), int(math.ceil(z1 / cell_mm))
        grid[max(0, r0):min(nz, r1), max(0, c0):min(nx, c1)] += 1
    return grid


def make_wall_cloud(n_points: int = 40000, seed: int = 0,
                    buttress: bool = True) -> np.ndarray:
    """A masonry wall as an (N, 3) point cloud, already in survey millimetres.

    x runs along the face (0 to WALL_W), z is height (0 to WALL_H), y is the
    outward normal and stays near zero apart from surface relief. With
    buttress=True a 400 mm deep pier stands proud of the face between
    x = 2000 and x = 2400 — something a plan view should show and an
    elevation should not, which is how an export can be checked rather
    than merely eyeballed.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, WALL_W, n_points)
    z = rng.uniform(0.0, WALL_H, n_points)

    # coursed ashlar: stones proud by a few mm, joints recessed
    course_h, stone_w, joint = 300.0, 600.0, 15.0
    row = np.floor(z / course_h)
    offset = (row % 2) * stone_w / 2
    col = np.floor((x + offset) / stone_w)
    stone = rng.normal(0.0, 2.5, 5000)[((row * 131 + col * 37).astype(int)) % 5000]
    in_joint = (
        ((z % course_h) < joint) | ((z % course_h) > course_h - joint)
        | (((x + offset) % stone_w) < joint) | (((x + offset) % stone_w) > stone_w - joint)
    )
    y = np.where(in_joint, stone - 10.0, stone) + rng.normal(0.0, 1.0, n_points)

    if buttress:
        y = np.where((x >= 2000.0) & (x <= 2400.0), y + 400.0, y)

    return np.column_stack([x, y, z])


@dataclass
class CloudTruth:
    """Where the markers really are in a raw reconstructed cloud."""

    marker_positions_raw: dict[str, np.ndarray]
    mm_per_unit: float


def make_capture_cloud(
    monument: MonumentRecord | None = None,
    seed: int = 0,
    n_points: int = 45000,
    disc_points: int = 400,
    disc_radius_mm: float = 40.0,
    noise_mm: float = 1.0,
) -> tuple[np.ndarray, "CloudTruth"]:
    """A reconstructed cloud as photogrammetry actually delivers one.

    Two things make this different from make_wall_cloud():

      1. ARBITRARY UNITS AND POSE. Photogrammetry has no idea how big the
         wall is or which way is up, so the cloud comes out at a random
         scale, rotation and translation. Recovering that is Geometry's
         job — but Geometry cannot start until somebody says where the
         markers are IN THIS CLOUD.

      2. THE MARKERS ARE IN IT. Each surveyed marker is a printed target
         stuck on the wall, so it reconstructs as a small dense disc
         standing slightly proud of the face. It is much denser than the
         surrounding masonry, which is what makes it findable.

    Returns the raw (N, 3) cloud and the truth: each marker's real position
    in that raw frame, and the scale it was built at. The truth exists for
    scoring only — a real build has neither.
    """
    monument = monument or make_monument()
    rng = np.random.default_rng(seed)

    wall = make_wall_cloud(n_points=n_points, seed=seed, buttress=False)

    # Printed targets: dense discs on the face at each surveyed position.
    centres_mm: dict[str, np.ndarray] = {}
    discs = []
    for mk in monument.markers:
        cx, _, cz = mk.position_mm
        radius = mk.target_radius_mm or disc_radius_mm
        # Point count scales with area, so every target has the same
        # printed density and a bigger target is not simply a denser one.
        n_disc = max(120, int(disc_points * (radius / 40.0) ** 2))
        theta = rng.uniform(0, 2 * math.pi, n_disc)
        r = radius * np.sqrt(rng.uniform(0, 1, n_disc))
        dx = cx + r * np.cos(theta)
        dz = cz + r * np.sin(theta)
        dy = np.full(n_disc, 8.0) + rng.normal(0, 0.4, n_disc)
        discs.append(np.column_stack([dx, dy, dz]))
        centres_mm[mk.marker_id] = np.array([cx, 8.0, cz])

    pts_mm = np.vstack([wall] + discs)
    pts_mm = pts_mm + rng.normal(0, noise_mm, pts_mm.shape)

    # Random similarity transform into the raw reconstruction frame.
    mm_per_unit = float(rng.uniform(4.0, 25.0))
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    R = np.array([
        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d],
    ])
    t = rng.uniform(-500, 500, 3)

    def to_raw(p: np.ndarray) -> np.ndarray:
        return (p @ R.T) / mm_per_unit + t

    raw = to_raw(pts_mm)
    truth = CloudTruth(
        marker_positions_raw={k: to_raw(v.reshape(1, 3))[0] for k, v in centres_mm.items()},
        mm_per_unit=mm_per_unit,
    )
    return raw, truth


def write_sample_images(out_dir: str, n: int = 9, size: int = 480) -> str:
    """Write real JPEGs — sharp, shaken and blown-out — for the extractor.

    The acceptance tests never touch these. They exist so sharpness() and
    exposure_clip() can be written against real pixels instead of guessed.
    Filenames state the truth: sharp_*, blurry_*, blown_*.
    """
    try:
        import cv2
    except ImportError as exc:                              # pragma: no cover
        raise ImportError(
            "write_sample_images needs opencv: pip install opencv-python-headless"
        ) from exc
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    for i in range(n):
        # coursed masonry: stones proud, joints recessed and darker
        img = rng.integers(95, 165, (size, size, 3), dtype=np.uint8)
        for z in range(0, size, 60):
            img[z:z + 6, :] = rng.integers(40, 70)
        offset = 0
        for z in range(0, size, 60):
            offset = 0 if offset else 55
            for x in range(offset, size, 110):
                img[z:z + 60, x:x + 5] = rng.integers(40, 70)

        kind = ("sharp", "blurry", "blown")[i % 3]
        if kind == "blurry":
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=3.5)
        elif kind == "blown":
            # Shot into the sun: gained up and lifted, so the stone faces
            # saturate and only the dark joints keep any texture. Aiming for
            # roughly a quarter of pixels at 255, matching the exposure_clip
            # values the metadata scenarios use.
            img = np.clip(img.astype(np.float32) * 1.9 + 60.0, 0, 255).astype(np.uint8)

        cv2.imwrite(str(out / f"{kind}_{i:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return str(out)


if __name__ == "__main__":
    from contracts.preservation import validate_capture_session, validate_monument_record

    monument = make_monument()
    validate_monument_record(monument)
    print(f"monument {monument.monument_id}: {len(monument.markers)} markers, "
          f"{len(monument.regions)} regions, face {monument.face_size_mm}\n")

    header = (f"{'scenario':<18}{'photos':>7}{'cover':>7}{'redund':>8}"
              f"{'mm/px':>7}{'soft':>6}{'blown':>7}{'markers':>9}")
    print(header)
    print("-" * len(header))
    for name in SCENARIOS:
        s, truth = make_session(name, seed=1, monument=monument)
        validate_capture_session(s)
        g = coverage_grid(s)
        cover = float((g >= 1).mean())
        redund = float((g >= 3).mean())
        gsd = float(np.median([
            (p.footprint_mm[2] - p.footprint_mm[0]) / p.width for p in s.photos
        ]))
        soft = sum(1 for p in s.photos if p.sharpness < 50)
        blown = sum(1 for p in s.photos if p.exposure_clip > 0.1)
        seen = len({m for p in s.photos for m in p.markers_visible})
        print(f"{name:<18}{len(s.photos):>7}{cover:>7.2f}{redund:>8.2f}"
              f"{gsd:>7.2f}{soft:>6}{blown:>7}{seen:>9}")

    print("\n  thin_overlap : coverage 1.00 but redundancy near 0 — a gate that")
    print("                 checks only coverage waves it through, and")
    print("                 reconstruction then fails back at the office.")
    print("  too_few      : coverage and redundancy both fine; it fails on mm/px.")
    print("                 At that scale a 3 mm crack is smaller than a pixel.")
