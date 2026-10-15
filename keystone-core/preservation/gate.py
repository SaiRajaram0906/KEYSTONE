from __future__ import annotations
import numpy as np
from contracts.preservation import CaptureQC, CheckResult, validate_capture_session, validate_monument_record
from .coverage import coverage_fraction, ground_sampling_distance, redundancy_fraction, region_coverage
MAX_GSD_MM_PER_PIXEL=1.0
MIN_REDUNDANCY_FRACTION=0.20
MIN_COVERAGE_FRACTION=0.80
MIN_REGION_COVERAGE=0.80
MIN_MARKERS=3
MAX_EXPOSURE_CLIP=0.10
SHARPNESS_RATIO=0.40

def _check(name, passed, detail, value=None, threshold=None):
    return CheckResult(name=name, passed=bool(passed), detail=detail, value=None if value is None else float(value), threshold=None if threshold is None else float(threshold))

def evaluate_capture(session, monument):
    validate_capture_session(session); validate_monument_record(monument)
    if session.monument_id != monument.monument_id: raise ValueError("capture monument_id does not match the monument record")
    checks=[]
    gsd_values=[ground_sampling_distance(p) for p in session.photos]; gsd=float(np.median(gsd_values))
    gsd_bad=[p for p,v in zip(session.photos,gsd_values) if v>MAX_GSD_MM_PER_PIXEL]
    checks.append(_check("ground_sampling_distance", not gsd_bad, f"median ground sampling distance is {gsd:.2f} mm/pixel", gsd, MAX_GSD_MM_PER_PIXEL))
    redundancy=redundancy_fraction(session,monument)
    checks.append(_check("redundancy", redundancy>=MIN_REDUNDANCY_FRACTION, f"{redundancy:.2%} of the face is seen in at least three photos", redundancy, MIN_REDUNDANCY_FRACTION))
    sharp_values=np.asarray([float(p.sharpness) for p in session.photos]); median=float(np.median(sharp_values)); sharp_threshold=median*SHARPNESS_RATIO
    blurry=[p for p in session.photos if p.sharpness<sharp_threshold]
    checks.append(_check("sharpness", not blurry, f"session median sharpness {median:.2f}; {len(blurry)} of {len(session.photos)} frames below {SHARPNESS_RATIO:.2f}× median", len(blurry)/len(session.photos), 0.25))
    blown=[p for p in session.photos if p.exposure_clip>MAX_EXPOSURE_CLIP]
    checks.append(_check("exposure_clipping", not blown, f"{len(blown)} of {len(session.photos)} frames exceed {MAX_EXPOSURE_CLIP:.0%} clipped pixels", len(blown)/len(session.photos), 0.0))
    seen=sorted({mid for p in session.photos for mid in p.markers_visible if mid}); missing=[m.marker_id for m in monument.markers if m.marker_id not in seen]
    checks.append(_check("surveyed_markers", len(seen)>=MIN_MARKERS, f"{len(seen)} distinct surveyed markers visible; need at least {MIN_MARKERS}"+(f"; missing {', '.join(missing)}" if missing else ""), len(seen), MIN_MARKERS))
    coverage=coverage_fraction(session,monument)
    checks.append(_check("coverage", coverage>=MIN_COVERAGE_FRACTION, f"{coverage:.2%} of the monument face is photographed", coverage, MIN_COVERAGE_FRACTION))
    regions=region_coverage(session,monument); weak=sorted(r for r,v in regions.items() if v<MIN_REGION_COVERAGE)
    hard_fail=[c.name for c in checks if not c.passed]
    verdict="RESHOOT" if hard_fail else ("PASS_WITH_GAPS" if weak else "PASS")
    guidance=[]
    if gsd_bad: guidance.append(f"Move closer and re-shoot {len(gsd_bad)} frame(s) with finer than {MAX_GSD_MM_PER_PIXEL:.1f} mm/pixel detail: {', '.join(p.photo_id for p in gsd_bad[:12])}.")
    if not checks[1].passed: guidance.append(f"Re-shoot with more overlapping viewpoints; redundancy is {redundancy:.0%}, below {MIN_REDUNDANCY_FRACTION:.0%}.")
    if blurry: guidance.append(f"Re-shoot {len(blurry)} blurred frame(s) with the camera steady and in focus: {', '.join(p.photo_id for p in blurry[:12])}.")
    if blown: guidance.append(f"Re-shoot {len(blown)} overexposed frame(s) with texture visible in the highlights: {', '.join(p.photo_id for p in blown[:12])}.")
    if len(seen)<MIN_MARKERS: guidance.append(f"Re-shoot wider framing so at least {MIN_MARKERS} surveyed markers are visible; currently visible: {', '.join(seen) or 'none'}.")
    if not checks[5].passed: guidance.append(f"Re-shoot uncovered parts of the monument; face coverage is only {coverage:.0%}.")
    if verdict=="PASS_WITH_GAPS":
        for rid in weak: guidance.append(f"Gap: re-shoot region {rid} before leaving site if possible; Geometry must treat it as weak coverage.")
    return checks, coverage, weak, guidance, verdict

def check_capture(session,monument):
    checks,coverage,weak,guidance,verdict=evaluate_capture(session,monument)
    return CaptureQC(capture_id=session.capture_id, monument_id=session.monument_id, verdict=verdict, checks=checks, coverage_fraction=float(coverage), weak_region_ids=[] if verdict=="PASS" else weak, reshoot_guidance=guidance if verdict!="PASS" else [])
