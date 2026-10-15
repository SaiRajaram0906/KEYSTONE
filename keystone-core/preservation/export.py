"""Measured plan/elevation/section extraction from a survey-frame cloud."""
from __future__ import annotations
import numpy as np
from contracts.preservation import Section, validate_monument_record, validate_section

__all__=["extract_section"]

def _polyline(projected):
    if len(projected)==0: return []
    # A deterministic order is useful to downstream drawing code.  The output
    # remains a measured point trace rather than pretending to be a CAD wall.
    order=np.lexsort((projected[:,1],projected[:,0]))
    return projected[order].tolist()

def extract_section(points, monument, capture_id, kind, at_mm, thickness_mm):
    validate_monument_record(monument)
    if kind not in ("plan","elevation","section"): raise ValueError("kind must be plan, elevation or section")
    if thickness_mm<=0: raise ValueError("thickness_mm must be positive")
    pts=np.asarray(points,dtype=float)
    if pts.ndim!=2 or pts.shape[1]!=3 or len(pts)==0: raise ValueError("points must be a non-empty (N,3) array")
    half=float(thickness_mm)/2.0
    if kind=="elevation":
        mask=np.abs(pts[:,1]-float(at_mm))<=half; selected=pts[mask]; projected=selected[:,[0,2]]
    elif kind=="plan":
        mask=np.abs(pts[:,2]-float(at_mm))<=half; selected=pts[mask]; projected=selected[:,[0,1]]
    else:
        # A vertical section at x=at_mm, showing outward depth vs height.
        mask=np.abs(pts[:,0]-float(at_mm))<=half; selected=pts[mask]; projected=selected[:,[1,2]]
    if len(selected)==0: raise ValueError(f"no points intersect the {kind} slice at {at_mm} mm")
    extent=[float(projected[:,0].min()),float(projected[:,1].min()),float(projected[:,0].max()),float(projected[:,1].max())]
    # Avoid a zero-area extent for an unusually thin synthetic slice.
    if extent[2]<=extent[0]: extent[2]=extent[0]+1e-6
    if extent[3]<=extent[1]: extent[3]=extent[1]+1e-6
    sec=Section(monument_id=monument.monument_id,capture_id=capture_id,kind=kind,at_mm=float(at_mm),thickness_mm=float(thickness_mm),polylines=[_polyline(projected)],extent_mm=extent,n_points=int(len(selected)))
    validate_section(sec); return sec
