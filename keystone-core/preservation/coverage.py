from __future__ import annotations
import math
import numpy as np

def _face_size(monument):
    return float(monument.face_size_mm[0]), float(monument.face_size_mm[1])

def coverage_grid(session, monument=None, cell_mm: float = 100.0) -> np.ndarray:
    if cell_mm <= 0: raise ValueError("cell_mm must be positive")
    width, height = _face_size(monument) if monument is not None else (6000.0, 3000.0)
    nx, nz = max(1, int(math.ceil(width / cell_mm))), max(1, int(math.ceil(height / cell_mm)))
    grid = np.zeros((nz, nx), dtype=np.int32)
    for p in session.photos:
        x0, z0, x1, z1 = map(float, p.footprint_mm)
        c0, c1 = max(0, int(math.floor(x0/cell_mm))), min(nx, int(math.ceil(x1/cell_mm)))
        r0, r1 = max(0, int(math.floor(z0/cell_mm))), min(nz, int(math.ceil(z1/cell_mm)))
        if c0 < c1 and r0 < r1: grid[r0:r1, c0:c1] += 1
    return grid

def coverage_fraction(session, monument, cell_mm=100.0):
    return float(np.mean(coverage_grid(session, monument, cell_mm) >= 1))

def redundancy_fraction(session, monument, min_views=3, cell_mm=100.0):
    if min_views < 1: raise ValueError("min_views must be >= 1")
    return float(np.mean(coverage_grid(session, monument, cell_mm) >= min_views))

def ground_sampling_distance(photo):
    x0, _, x1, _ = map(float, photo.footprint_mm)
    if photo.width <= 0 or x1 <= x0: raise ValueError(f"photo {photo.photo_id}: invalid footprint or width")
    return (x1-x0)/float(photo.width)

def session_gsd(session): return [ground_sampling_distance(p) for p in session.photos]
def median_gsd(session):
    vals = session_gsd(session)
    return float(np.median(vals)) if vals else math.inf

def region_coverage(session, monument, cell_mm=100.0):
    grid = coverage_grid(session, monument, cell_mm)
    width, height = _face_size(monument)
    out = {}
    for region in monument.regions:
        rid = str(region["region_id"]); lo, hi = region["bounds_min"], region["bounds_max"]
        x0, z0, x1, z1 = max(0., float(lo[0])), max(0., float(lo[2])), min(width, float(hi[0])), min(height, float(hi[2]))
        c0, c1 = max(0, int(math.floor(x0/cell_mm))), min(grid.shape[1], int(math.ceil(x1/cell_mm)))
        r0, r1 = max(0, int(math.floor(z0/cell_mm))), min(grid.shape[0], int(math.ceil(z1/cell_mm)))
        out[rid] = 0.0 if c0 >= c1 or r0 >= r1 else float(np.mean(grid[r0:r1,c0:c1] >= 1))
    return out

def weak_regions(session, monument, threshold=0.80):
    return [rid for rid, frac in region_coverage(session, monument).items() if frac < threshold]
