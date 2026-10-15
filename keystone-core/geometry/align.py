"""Step 1 of Geometry: scale + coarse alignment from ground-control markers.

Each raw capture lives in its own arbitrary units, position and orientation.
The surveyed markers are the only link to the real world, so we fit the
similarity transform (scale s, rotation R, translation t) that best maps
"where the markers appear in the cloud" onto "where the markers really are":

        survey_mm  ~=  s * R @ raw + t

`s` is exactly mm_per_unit, because raw units go in and millimetres come out.
After this, every capture sits in the one survey frame, in mm, and the
Region boxes mean the same stones every time.

The fit is Umeyama (1991): a closed-form least-squares solution, no iteration,
no initial guess.  This file also holds the small helpers the later steps
share (apply a transform, apply a rigid 4x4 from ICP).
"""
from __future__ import annotations

from dataclasses import dataclass

# pyrefly: ignore [missing-import]
import numpy as np

__all__ = [
    "Similarity",
    "AlignmentError",
    "umeyama",
    "fit_to_survey",
    "apply_rigid",
]


class AlignmentError(ValueError):
    """The markers cannot fix scale and pose (too few, collinear, or mismatched)."""


@dataclass
class Similarity:
    """dst = scale * R @ src + t   (row-vector points: dst = scale * src @ R.T + t)."""

    scale: float
    R: np.ndarray          # (3, 3) proper rotation, det = +1
    t: np.ndarray          # (3,)

    def apply(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        return self.scale * (pts @ self.R.T) + self.t

    def inverse(self) -> "Similarity":
        s_inv = 1.0 / self.scale
        R_inv = self.R.T
        return Similarity(s_inv, R_inv, -s_inv * (R_inv @ self.t))

    def as_matrix(self) -> np.ndarray:
        """4x4 homogeneous matrix with the scale folded into the rotation block."""
        M = np.eye(4)
        M[:3, :3] = self.scale * self.R
        M[:3, 3] = self.t
        return M


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Similarity:
    """Least-squares similarity transform mapping src -> dst (Umeyama 1991).

    src, dst: (n, 3) corresponding points, n >= 3, not all on one line.
    Returns the Similarity minimising  sum_i || dst_i - (s R src_i + t) ||^2.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise AlignmentError(f"src and dst must both be (n, 3); got {src.shape} and {dst.shape}")
    n = src.shape[0]
    if n < 3:
        raise AlignmentError(f"need at least 3 point pairs, got {n}")

    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst

    # Collinear points leave rotation about their shared line undetermined.
    sv = np.linalg.svd(src_c, compute_uv=False)
    if sv[0] <= 0 or sv[1] < 1e-6 * sv[0]:
        raise AlignmentError("points are (nearly) collinear; rotation is not determined")

    # Cross-covariance between the two centred sets, then its SVD.
    cov = dst_c.T @ src_c / n
    U, D, Vt = np.linalg.svd(cov)

    # If the best orthogonal matrix is a reflection (det = -1), flip the weakest axis
    # so we return a proper rotation instead of a mirror image.
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    var_src = (src_c ** 2).sum() / n
    scale = float((D * np.diag(S)).sum() / var_src) if with_scale else 1.0
    t = mu_dst - scale * (R @ mu_src)
    return Similarity(scale, R, t)


def fit_to_survey(
    markers: dict[str, np.ndarray],
    survey: dict[str, list[float]],
) -> tuple[Similarity, float, dict[str, float]]:
    """Fit one capture's markers (raw units) to the surveyed positions (mm).

    Returns (similarity, rmse_mm, residuals_mm):
      similarity   raw -> survey-mm transform; .scale is mm_per_unit
      rmse_mm      root-mean-square marker residual after the fit
      residuals_mm per-marker distance between fitted and surveyed position,
                   so a single badly placed marker can be spotted
    Only marker ids present in BOTH dicts are used, in sorted order.
    """
    ids = sorted(set(markers) & set(survey))
    if len(ids) < 3:
        raise AlignmentError(f"only {len(ids)} marker(s) in common with the survey; need >= 3")

    raw = np.array([np.asarray(markers[k], dtype=float) for k in ids])
    mm = np.array([np.asarray(survey[k], dtype=float) for k in ids])

    sim = umeyama(raw, mm, with_scale=True)
    if not sim.scale > 0:
        raise AlignmentError(f"fitted scale is not positive ({sim.scale})")

    dist = np.linalg.norm(sim.apply(raw) - mm, axis=1)
    rmse = float(np.sqrt(np.mean(dist ** 2)))
    return sim, rmse, {k: float(d) for k, d in zip(ids, dist)}


def apply_rigid(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform (e.g. from ICP) to (N, 3) points."""
    T = np.asarray(T, dtype=float)
    return np.asarray(points, dtype=float) @ T[:3, :3].T + T[:3, 3]