"""Step 2 of Geometry: fine registration via Trimmed Point-to-Plane ICP.

Ground-control markers provide an initial coarse alignment in survey coordinates.
However, small residual alignment errors (e.g., marker placement noise) can remain.
To refine registration, we use Point-to-Plane ICP on voxel-downsampled point clouds.

Crucially, to prevent real structural movement (e.g. a bulging wall region) from being
aligned away during ICP, we implement TRIMMED ICP (Chetverikov et al., 2002).
In each ICP pass, per-point residuals to the reference surface are evaluated, and only
the best 70% of points (smallest residuals) are retained to update the transformation.
This automatically ignores displaced or deformed regions without relying on prior region labels.
"""
from __future__ import annotations

import numpy as np
import scipy.spatial
import open3d as o3d

from geometry.align import apply_rigid

__all__ = ["refine"]


def _compute_trimmed_rmse(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    target_normals: np.ndarray,
    target_tree: scipy.spatial.cKDTree,
    T: np.ndarray,
    max_correspondence_distance: float = 40.0,
    trim_fraction: float = 0.70,
) -> tuple[float, np.ndarray]:
    """Compute point-to-plane trimmed RMSE and return the indices of the best trimmed points.

    WHY point-to-plane residual: Point-to-plane distance |(p - q) . n| measures true surface-to-surface
    offset, ignoring tangential sliding along planar masonry walls.
    """
    # Transform source points by current candidate rigid transformation T
    pts_trans = apply_rigid(source_pts, T)

    # Find nearest target point for each transformed source point using cKDTree
    dists, idxs = target_tree.query(pts_trans, k=1)

    # Filter correspondences within max search distance
    valid_mask = dists <= max_correspondence_distance
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 3:
        # Fallback if too few correspondences match
        return float("inf"), np.array([], dtype=int)

    # Point-to-plane residual computation: |(p - q) . n|
    diffs = pts_trans[valid_indices] - target_pts[idxs[valid_indices]]
    normals = target_normals[idxs[valid_indices]]
    pt2plane_res = np.abs(np.einsum("ij,ij->i", diffs, normals))

    # Sort residuals and select top trim_fraction (e.g., 70%) best points
    n_keep = max(3, int(np.ceil(trim_fraction * len(pt2plane_res))))
    sorted_order = np.argsort(pt2plane_res)
    best_in_valid = sorted_order[:n_keep]

    trimmed_residuals = pt2plane_res[best_in_valid]
    trimmed_rmse = float(np.sqrt(np.mean(trimmed_residuals**2)))
    trimmed_source_indices = valid_indices[best_in_valid]

    return trimmed_rmse, trimmed_source_indices


def refine(
    reference_pts: np.ndarray,
    moving_pts: np.ndarray,
    voxel_size: float = 20.0,
    max_correspondence_distance: float = 40.0,
    trim_fraction: float = 0.70,
    passes: int = 3,
) -> tuple[np.ndarray, float]:
    """Refine moving_pts alignment to reference_pts using Trimmed Point-to-Plane ICP.

    Both reference_pts and moving_pts must already be in the survey-mm frame.
    Returns (T_rigid_4x4, icp_rmse_mm). If refined fit has worse trimmed RMSE than identity,
    returns identity transformation matrix.
    """
    ref_pts = np.asarray(reference_pts, dtype=float)
    mov_pts = np.asarray(moving_pts, dtype=float)

    if ref_pts.ndim != 2 or ref_pts.shape[1] != 3 or mov_pts.ndim != 2 or mov_pts.shape[1] != 3:
        raise ValueError("reference_pts and moving_pts must both be (N, 3) arrays")

    # Step A: Convert to Open3D point clouds and voxel downsample to ~20 mm
    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(ref_pts)
    ref_down = ref_pcd.voxel_down_sample(voxel_size=voxel_size)

    mov_pcd = o3d.geometry.PointCloud()
    mov_pcd.points = o3d.utility.Vector3dVector(mov_pts)
    mov_down = mov_pcd.voxel_down_sample(voxel_size=voxel_size)

    # Extract downsampled numpy arrays
    target_down_pts = np.asarray(ref_down.points, dtype=float)
    source_down_pts = np.asarray(mov_down.points, dtype=float)

    if len(target_down_pts) < 3 or len(source_down_pts) < 3:
        # Not enough points after downsampling to fit ICP
        return np.eye(4), 0.0

    # Step B: Compute surface normals on reference cloud for Point-to-Plane ICP
    ref_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100.0, max_nn=30)
    )
    target_normals = np.asarray(ref_down.normals, dtype=float)
    # Ensure unit normal vectors
    norm_lengths = np.linalg.norm(target_normals, axis=1, keepdims=True)
    norm_lengths[norm_lengths == 0] = 1.0
    target_normals = target_normals / norm_lengths

    # Build cKDTree on target downsampled points for fast correspondence searching
    target_tree = scipy.spatial.cKDTree(target_down_pts)

    # Initial candidate transformation is identity T_0 = I_4
    T_init = np.eye(4)
    initial_rmse, _ = _compute_trimmed_rmse(
        source_down_pts,
        target_down_pts,
        target_normals,
        target_tree,
        T_init,
        max_correspondence_distance=max_correspondence_distance,
        trim_fraction=trim_fraction,
    )

    T_current = T_init.copy()

    # Step C: Trimmed ICP passes
    for pass_idx in range(passes):
        # 1. Compute current trimmed RMSE and select best 70% stable points
        current_rmse, best_source_indices = _compute_trimmed_rmse(
            source_down_pts,
            target_down_pts,
            target_normals,
            target_tree,
            T_current,
            max_correspondence_distance=max_correspondence_distance,
            trim_fraction=trim_fraction,
        )

        if len(best_source_indices) < 3:
            break

        # 2. Extract trimmed source subset for Point-to-Plane ICP pass
        trimmed_source_pts = source_down_pts[best_source_indices]
        sub_pcd = o3d.geometry.PointCloud()
        sub_pcd.points = o3d.utility.Vector3dVector(trimmed_source_pts)

        # 3. Perform point-to-plane ICP on trimmed subset
        icp_result = o3d.pipelines.registration.registration_icp(
            sub_pcd,
            ref_down,
            max_correspondence_distance=max_correspondence_distance,
            init=T_current,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
        )

        T_current = np.asarray(icp_result.transformation, dtype=float)

    # Final evaluation of trimmed RMSE after all passes
    final_rmse, _ = _compute_trimmed_rmse(
        source_down_pts,
        target_down_pts,
        target_normals,
        target_tree,
        T_current,
        max_correspondence_distance=max_correspondence_distance,
        trim_fraction=trim_fraction,
    )

    # Step D: Safety check - if refined fit has worse trimmed RMSE than starting fit, return identity
    if final_rmse >= initial_rmse or np.isnan(final_rmse):
        return np.eye(4), float(initial_rmse if not np.isnan(initial_rmse) else 0.0)

    return T_current, float(final_rmse)
