"""Locate and identify surveyed targets in a raw, arbitrary-scale cloud.

LOCATE   Every point gets a scale-free density score: the distance to its
         k-th nearest neighbour, expressed relative to the cloud's median.
         Target discs are several times denser than masonry, so points whose
         relative k-NN distance is small are target candidates.  They are
         grouped into connected clusters (no absolute radius in mm anywhere).

IDENTIFY Position alone cannot separate GCP-1 from GCP-4 when the layout is
         symmetric under a 180-degree rotation.  Target *size* order is
         scale-free, so cluster sizes are rank-matched to
         MarkerSpec.target_radius_mm.  If the survey carries no target sizes
         and the layout is ambiguous, this raises instead of silently
         returning a mirrored answer.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

__all__ = ["locate_markers"]

K_NEIGHBOURS = 10          # k for the k-th nearest-neighbour density score
DENSITY_RATIO = 0.55       # candidate if kth-dist < ratio * cloud median
MIN_CLUSTER_POINTS = 12
MAX_EXHAUSTIVE = 7         # 7! = 5040 permutations is still quick


def _validate_points(points) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 50:
        raise ValueError("points must be an (N, 3) array with at least 50 points")
    if not np.isfinite(pts).all():
        raise ValueError("points contain NaN or inf")
    return pts


def similarity_fit(source, target):
    """Least-squares scale, proper rotation (no reflection), translation.

    Row-vector convention:  fitted = scale * (source @ R) + t
    """
    a = np.asarray(source, float)
    b = np.asarray(target, float)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    xa, xb = a - ca, b - cb
    denom = float(np.sum(xa * xa))
    if denom <= 1e-15:
        raise ValueError("degenerate marker constellation")
    u, _, vt = np.linalg.svd(xa.T @ xb)
    d = np.sign(np.linalg.det(u @ vt)) or 1.0
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    scale = float(np.sum((xa @ r) * xb) / denom)
    t = cb - scale * (ca @ r)
    fitted = scale * (a @ r) + t
    return scale, r, t, fitted


def _rmse(fitted, expected) -> float:
    return float(np.sqrt(np.mean(np.sum((fitted - expected) ** 2, axis=1))))


def _find_clusters(points: np.ndarray):
    """Return [(n_points, size, centre)] for dense clusters, largest first."""
    n = len(points)
    k = min(K_NEIGHBOURS, n - 1)
    tree = cKDTree(points)
    dist, _ = tree.query(points, k=k + 1)
    kth = dist[:, -1]
    med = float(np.median(kth))
    if med <= 0:
        raise ValueError("cloud has no spatial extent")

    cand = np.flatnonzero(kth < DENSITY_RATIO * med)
    if len(cand) < MIN_CLUSTER_POINTS:
        return []
    cpts = points[cand]
    # Link radius comes from the candidates' own spacing, so it is in the
    # cloud's units whatever they are.
    eps = 2.5 * float(np.median(kth[cand]))
    pairs = cKDTree(cpts).query_pairs(eps, output_type="ndarray")
    if len(pairs) == 0:
        return []
    m = len(cand)
    graph = coo_matrix((np.ones(len(pairs), np.uint8), (pairs[:, 0], pairs[:, 1])),
                       shape=(m, m))
    n_comp, labels = connected_components(graph, directed=False)

    clusters = []
    for lab in range(n_comp):
        idx = np.flatnonzero(labels == lab)
        if len(idx) < MIN_CLUSTER_POINTS:
            continue
        p = cpts[idx]
        centre = np.median(p, axis=0)
        # Size proxy: RMS distance from the centre.  For a uniform disc
        # RMS = R / sqrt(2), so it is monotone in target radius.
        size = float(np.sqrt(np.mean(np.sum((p - centre) ** 2, axis=1))))
        clusters.append((len(idx), size, p.mean(axis=0)))
    clusters.sort(key=lambda c: c[0], reverse=True)
    return clusters


def _rank(values) -> np.ndarray:
    return np.argsort(np.argsort(np.asarray(values, float))).astype(float)


def _spearman(a, b) -> float:
    ra, rb = _rank(a), _rank(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def locate_markers(points, monument):
    """Return {marker_id: position in the RAW cloud's own coordinates}."""
    pts = _validate_points(points)
    markers = list(monument.markers)
    m = len(markers)
    if m < 3:
        raise ValueError("at least three surveyed markers are required")

    clusters = _find_clusters(pts)
    if len(clusters) < m:
        raise ValueError(f"found only {len(clusters)} dense target cluster(s); need {m}")

    chosen = clusters[:m]                      # the m largest dense clusters
    centres = np.asarray([c[2] for c in chosen], float)
    sizes = np.asarray([c[1] for c in chosen], float)
    expected = np.asarray([mk.position_mm for mk in markers], float)
    radii = [getattr(mk, "target_radius_mm", None) for mk in markers]
    have_radii = all(r is not None for r in radii)

    tol = max(25.0, 0.02 * float(np.ptp(expected, axis=0).mean()))

    if m > MAX_EXHAUSTIVE:
        if not have_radii:
            raise ValueError("too many markers to identify by position alone; "
                             "survey needs target sizes")
        order_det = np.argsort(sizes)
        order_exp = np.argsort(radii)
        assign = np.empty(m, int)
        assign[order_exp] = order_det          # smallest<->smallest, ...
        _, _, _, fitted = similarity_fit(centres[assign], expected)
        if _rmse(fitted, expected) > tol:
            raise ValueError("marker constellation does not match the survey")
        return {markers[i].marker_id: centres[assign[i]].copy() for i in range(m)}

    fits = []                                  # (rmse, perm)
    for perm in permutations(range(m)):
        try:
            scale, _, _, fitted = similarity_fit(centres[list(perm)], expected)
        except ValueError:
            continue
        if scale <= 0:
            continue
        fits.append((_rmse(fitted, expected), perm))
    if not fits:
        raise ValueError("could not fit the marker constellation")
    fits.sort(key=lambda f: f[0])
    plausible = [f for f in fits if f[0] <= tol]
    if not plausible:
        raise ValueError(f"marker constellation fit failed; best RMSE {fits[0][0]:.1f} mm")

    if have_radii:
        # Among geometrically plausible mappings, take the one whose target
        # sizes are ordered like the surveyed sizes.  Geometry breaks ties.
        def score(f):
            rmse, perm = f
            return (-_spearman(sizes[list(perm)], radii), rmse)
        best_rmse, best = min(plausible, key=score)
    else:
        best_rmse, best = plausible[0]
        # A second, different mapping that fits almost as well means position
        # alone cannot say which target is which (e.g. 180-degree symmetry).
        if len(plausible) > 1 and plausible[1][0] < max(3.0 * best_rmse, 0.5 * tol):
            raise ValueError(
                "marker layout is ambiguous by position alone (symmetric); "
                "add MarkerSpec.target_radius_mm so target sizes can identify them")

    return {markers[i].marker_id: centres[best[i]].copy() for i in range(m)}
