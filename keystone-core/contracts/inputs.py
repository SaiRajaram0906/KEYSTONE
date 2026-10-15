"""What Geometry receives: one raw capture of one monument.

Raw point clouds from photogrammetry arrive in arbitrary units, in an
arbitrary position and orientation. The only thing tying a capture to the
real world is the set of surveyed ground-control markers visible in it.

    points      (N, 3) float  — the dense cloud, raw units, raw frame
    markers     {marker_id: (3,)} — where each marker appears IN THIS CLOUD
    survey      {marker_id: [x, y, z]} — where each marker really is, in mm,
                in the fixed survey frame (shared by every capture)

Fitting markers -> survey gives scale, rotation and translation in one
step. After that, every capture of the monument sits in the same frame,
in millimetres, and the Region boxes mean the same stones every time.

Real captures (a phone-app export, a COLMAP reconstruction) are loaded
into exactly this structure — see load_ply below.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from contracts.schema import CaptureMeta, ContractError, validate_capture_meta

__all__ = ["CaptureInput", "validate_capture_input", "load_ply"]


@dataclass
class CaptureInput:
    meta: CaptureMeta
    points: np.ndarray
    markers: dict[str, np.ndarray] = field(default_factory=dict)


def validate_capture_input(c: CaptureInput, survey: dict[str, list[float]]) -> None:
    validate_capture_meta(c.meta)
    pts = np.asarray(c.points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ContractError(f"points must be (N, 3), got {pts.shape}")
    if pts.shape[0] < 1000:
        raise ContractError(f"only {pts.shape[0]} points — too sparse to compare")
    if not np.isfinite(pts).all():
        raise ContractError("points contain NaN or inf")
    common = set(c.markers) & set(survey)
    if len(common) < 3:
        raise ContractError(
            f"capture {c.meta.capture_id} shares only {len(common)} marker(s) with the "
            "survey; at least 3 non-collinear markers are needed to fix scale and pose"
        )


def load_ply(path: str) -> np.ndarray:
    """Read an ASCII or binary PLY into an (N, 3) array.

    Uses Open3D when available (Geometry's own dependency). Kept here so both
    workstreams read real captures the same way.
    """
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover
        raise ImportError("load_ply needs open3d: pip install open3d") from exc
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise ContractError(f"no points read from {path}")
    return pts
