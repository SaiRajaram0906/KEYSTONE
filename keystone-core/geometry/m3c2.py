"""Step 3 of Geometry: Multiscale Model-to-Model Cloud Comparison (M3C2).

WHY PLAIN NEAREST-NEIGHBOUR DISTANCE FAILS ON MASONRY:
Masonry monument surfaces are non-smooth ashlar wall faces with recessed mortar joints
(~15 mm deep) and varying stone face relief. If plain nearest-neighbour distance (e.g. C2C)
is used to compare two point clouds, random photogrammetric sampling points from the later
cloud often fall into a recessed joint or onto an adjacent stone face. As a result, surface
roughness and joint topography register false-positive structural movement.

HOW M3C2 SOLVES THIS:
M3C2 (Lague et al., 2013) measures true surface displacement by:
1. Fitting local orientation normals at core points over a diameter larger than individual
   mortar joints (normal_radius = 100 mm), effectively averaging out joint roughness.
2. Defining a projection cylinder (radius 50 mm, half-length 100 mm) along the normal.
3. Projecting all points from both clouds inside the cylinder onto the normal vector,
   computing the mean position for each cloud: distance = mean(later) - mean(earlier).
4. Estimating Level of Detection (LoD at 95% confidence) using local standard deviations
   s1, s2 (local roughness of each cloud) and the registration error:
   LoD = 1.96 * sqrt(s1^2 / n1 + s2^2 / n2) + registration_error_mm.
   Thus, high local surface roughness naturally expands the uncertainty error bar (LoD),
   preventing noise from being misidentified as wall displacement.
"""
from __future__ import annotations

import numpy as np
import scipy.spatial

__all__ = ["compute_m3c2"]


def compute_m3c2(
    earlier_pts: np.ndarray,
    later_pts: np.ndarray,
    registration_error_mm: float = 0.0,
    n_core_pts: int = 20000,
    normal_radius: float = 100.0,
    cylinder_radius: float = 50.0,
    cylinder_half_length: float = 100.0,
    min_points_per_cylinder: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute simplified M3C2 distance and LoD between reference (earlier) and moving (later) clouds.

    Returns:
      core_pts:    (N_core, 3) sampled reference core points
      distance_mm: (N_core,) signed displacement along outward normal (+ = outward movement)
      lod_mm:      (N_core,) 95% confidence level of detection in mm
      valid:       (N_core,) boolean mask indicating valid M3C2 measurement points
    """
    earlier = np.asarray(earlier_pts, dtype=float)
    later = np.asarray(later_pts, dtype=float)

    if earlier.ndim != 2 or earlier.shape[1] != 3 or later.ndim != 2 or later.shape[1] != 3:
        raise ValueError("earlier_pts and later_pts must both be (N, 3) arrays")

    # 1. Randomly sample up to n_core_pts core points from the reference cloud (fixed seed)
    n_pts = earlier.shape[0]
    rng = np.random.default_rng(seed)
    if n_pts <= n_core_pts:
        core_indices = np.arange(n_pts)
    else:
        core_indices = rng.choice(n_pts, size=n_core_pts, replace=False)

    core_pts = earlier[core_indices]
    num_core = len(core_pts)

    # Build cKDTrees for spatial querying
    earlier_tree = scipy.spatial.cKDTree(earlier)
    later_tree = scipy.spatial.cKDTree(later)

    # 2. Compute local surface normals via PCA for each core point
    # Query neighbours in reference cloud within normal_radius = 100 mm
    neighbour_indices = earlier_tree.query_ball_point(core_pts, r=normal_radius)

    normals = np.zeros((num_core, 3), dtype=float)
    for i in range(num_core):
        idxs = neighbour_indices[i]
        if len(idxs) < 3:
            # Fallback normal pointing outward (+y in survey frame)
            normals[i] = [0.0, 1.0, 0.0]
            continue

        pts_local = earlier[idxs]
        # PCA plane fit: covariance matrix of centered neighbour points
        centered = pts_local - pts_local.mean(axis=0)
        cov = centered.T @ centered / len(idxs)
        # Eigenvalues and eigenvectors (eigh returns in ascending order)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]  # eigenvector with smallest eigenvalue

        # Orient normal so its y component is > 0 (outward in survey frame)
        if normal[1] < 0:
            normal = -normal
        elif normal[1] == 0:
            normal = np.array([0.0, 1.0, 0.0])

        norm_len = np.linalg.norm(normal)
        normals[i] = normal / (norm_len if norm_len > 0 else 1.0)

    # 3 & 4. Gather cylinder points and compute M3C2 distance & LoD per core point
    distance_mm = np.full(num_core, np.nan, dtype=float)
    lod_mm = np.full(num_core, np.nan, dtype=float)
    valid = np.zeros(num_core, dtype=bool)

    # Bounding search sphere radius for cylinder query
    search_r = np.sqrt(cylinder_radius**2 + cylinder_half_length**2)
    cyl_r_sq = cylinder_radius**2

    # Query candidate points within bounding sphere
    earlier_candidates = earlier_tree.query_ball_point(core_pts, r=search_r)
    later_candidates = later_tree.query_ball_point(core_pts, r=search_r)

    for i in range(num_core):
        cp = core_pts[i]
        n_vec = normals[i]

        # Process earlier cloud candidates inside cylinder
        earlier_cand_pts = earlier[earlier_candidates[i]]
        if len(earlier_cand_pts) < min_points_per_cylinder:
            continue

        rel_1 = earlier_cand_pts - cp
        d_ax_1 = rel_1 @ n_vec  # axial projection onto normal
        d_rad_sq_1 = np.sum(rel_1**2, axis=1) - d_ax_1**2  # squared radial distance

        in_cyl_1 = (np.abs(d_ax_1) <= cylinder_half_length) & (d_rad_sq_1 <= cyl_r_sq)
        proj_1 = d_ax_1[in_cyl_1]
        n1 = len(proj_1)
        if n1 < min_points_per_cylinder:
            continue

        # Process later cloud candidates inside cylinder
        later_cand_pts = later[later_candidates[i]]
        if len(later_cand_pts) < min_points_per_cylinder:
            continue

        rel_2 = later_cand_pts - cp
        d_ax_2 = rel_2 @ n_vec
        d_rad_sq_2 = np.sum(rel_2**2, axis=1) - d_ax_2**2

        in_cyl_2 = (np.abs(d_ax_2) <= cylinder_half_length) & (d_rad_sq_2 <= cyl_r_sq)
        proj_2 = d_ax_2[in_cyl_2]
        n2 = len(proj_2)
        if n2 < min_points_per_cylinder:
            continue

        # Compute M3C2 displacement: positive = outward movement
        mean_1 = float(np.mean(proj_1))
        mean_2 = float(np.mean(proj_2))
        dist = mean_2 - mean_1

        # Local standard deviations (roughness) along normal
        s1 = float(np.std(proj_1, ddof=1)) if n1 > 1 else 0.0
        s2 = float(np.std(proj_2, ddof=1)) if n2 > 1 else 0.0

        # Level of detection at 95% confidence
        lod = 1.96 * np.sqrt((s1**2 / n1) + (s2**2 / n2)) + registration_error_mm

        valid[i] = True
        distance_mm[i] = dist
        lod_mm[i] = lod

    return core_pts, distance_mm, lod_mm, valid
