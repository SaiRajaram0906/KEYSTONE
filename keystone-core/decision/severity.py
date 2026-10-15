"""Severity - turn the evidence for one region into one of four bands.

    INFORMATIONAL  movement is explained by the season. Recorded, nobody told.
    WATCH          something is off, but it could still be one noisy capture.
                   Look again sooner.
    ALERT          movement beyond the seasonal band on two captures running,
                   same direction, not coming back. Someone should inspect.
    CRITICAL       as ALERT, and large, or in a load-bearing element, or
                   backed by crack widening. An engineer should assess, and
                   restricting access is recommended.

THE RULES, IN ORDER
-------------------
1. Evidence for ALERT is PERSISTENCE, never size. A single capture - however
   far out - cannot raise ALERT, because one capture cannot tell a failing
   wall from a hot day or a bad registration.
2. Fewer than three captures (reference + two reports) caps the result at
   WATCH. There is no persistence to measure yet.
3. The residual must exceed the region's own detection limit (lod_mm) before
   it counts as ALERT evidence: statistically surprising is not enough, it
   must also be physically measurable.
4. Load-bearing regions escalate faster: the run of captures may start from a
   slightly lower bar, and CRITICAL is reached at a smaller residual.
5. Crack widening from the crack-seg team is independent evidence. Widening
   beyond CRACK_SIGNIFICANT x its uncertainty raises at least WATCH on its
   own; seen on two captures running it raises ALERT; alongside persistent
   displacement it raises CRITICAL.

This module never looks at weather or at raw displacement. Its only inputs
are the ResidualAnalysis, the crack evidence, and the capture count - so
seasonal movement cannot reach ALERT by any path through these rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from contracts.schema import SEVERITIES, ChangeReport
from decision.residuals import Z_STRONG, ResidualAnalysis, run_length

__all__ = [
    "MIN_CAPTURES_FOR_ALERT",
    "CRACK_SIGNIFICANT",
    "CrackEvidence",
    "SeverityDecision",
    "crack_evidence",
    "classify",
    "rank",
]

MIN_CAPTURES_FOR_ALERT = 3       # reference capture + two reports
Z_WATCH_SINGLE = 2.5             # one capture this far out is worth a WATCH
Z_STRONG_LOAD_BEARING = 2.5      # load-bearing: persistence counted from a lower bar
CRACK_SIGNIFICANT = 3.0          # widening must exceed this many x its uncertainty
CRITICAL_LOD_MULTIPLE = 4.0      # residual this many x lod -> CRITICAL
CRITICAL_LOD_MULTIPLE_LOAD_BEARING = 2.0


def rank(severity: str) -> int:
    return SEVERITIES.index(severity)


@dataclass
class CrackEvidence:
    """Crack widening in one region across the history."""

    latest_width_mm: float = 0.0
    latest_uncertainty_mm: float = 0.0
    latest_new_area_mm2: float = 0.0
    significant: bool = False        # latest widening beyond CRACK_SIGNIFICANT x uncertainty
    consecutive: int = 0             # latest captures in a row with significant widening


@dataclass
class SeverityDecision:
    severity: str
    findings: list[str] = field(default_factory=list)     # plain-language evidence
    capped_by_history: bool = False                        # would be higher with more captures


def crack_evidence(history: list[ChangeReport], region_id: str) -> CrackEvidence:
    """Collect crack widening for one region; widening inside its uncertainty is ignored."""
    ev = CrackEvidence()
    flags: list[bool] = []
    for k, rep in enumerate(history):
        widening = [c for c in rep.crack_changes if c.region_id == region_id]
        if not widening:
            flags.append(False)
            continue
        worst = max(widening, key=lambda c: c.d_max_width_mm)
        significant = (worst.d_max_width_mm > 0
                       and worst.d_max_width_mm >= CRACK_SIGNIFICANT * worst.width_uncertainty_mm)
        flags.append(significant)
        if k == len(history) - 1:
            ev.latest_width_mm = float(worst.d_max_width_mm)
            ev.latest_uncertainty_mm = float(worst.width_uncertainty_mm)
            ev.latest_new_area_mm2 = float(worst.new_crack_area_mm2)
            ev.significant = significant
    for flag in reversed(flags):
        if not flag:
            break
        ev.consecutive += 1
    return ev


def classify(analysis: ResidualAnalysis, n_captures: int, load_bearing: bool,
             crack: CrackEvidence | None = None) -> SeverityDecision:
    """Apply the rules above. n_captures counts the reference capture too."""
    crack = crack or CrackEvidence()
    findings: list[str] = []
    a = analysis
    size = abs(a.latest_residual_mm)
    lod = a.latest_lod_mm
    direction = a.direction if a.direction != "none" else ("outward" if a.latest_residual_mm >= 0 else "inward")

    # ---- displacement evidence
    z_bar = Z_STRONG_LOAD_BEARING if load_bearing else Z_STRONG
    run = a.run_strong if not load_bearing else run_length(a.z, z_bar)
    measurable = lod > 0 and size >= lod
    displacement_persistent = run >= 2 and a.retained and measurable

    # ---- crack evidence
    crack_persistent = crack.consecutive >= 2

    level = "INFORMATIONAL"
    if displacement_persistent:
        level = "ALERT"
        findings.append(
            f"{size:.1f} mm further {direction} than the seasonal model expects "
            f"(expected {a.expected[-1]:+.1f} mm, measured {a.observed[-1]:+.1f} mm, "
            f"detection limit {lod:.1f} mm); outside the seasonal band on the last {run} "
            f"captures in the same direction and not returning "
            f"(trend {a.trend_mm_per_capture:+.1f} mm per capture)")
        critical_multiple = CRITICAL_LOD_MULTIPLE_LOAD_BEARING if load_bearing else CRITICAL_LOD_MULTIPLE
        if size >= critical_multiple * lod:
            level = "CRITICAL"
            findings.append(
                f"unexplained movement is {size / lod:.1f}x the detection limit"
                + (" in a load-bearing element" if load_bearing else ""))
        if crack.significant:
            level = "CRITICAL"
    elif abs(a.latest_z) >= Z_WATCH_SINGLE or a.run_mild >= 2:
        level = "WATCH"
        if a.run_mild >= 2:
            findings.append(
                f"{size:.1f} mm {direction} of the seasonal expectation, outside the band on the "
                f"last {a.run_mild} captures but not yet conclusive")
        else:
            findings.append(
                f"{size:.1f} mm {direction} of the seasonal expectation on the latest capture only "
                f"({abs(a.latest_z):.1f} sigma); one capture cannot separate movement from noise")

    if crack.significant:
        ratio = crack.latest_width_mm / crack.latest_uncertainty_mm if crack.latest_uncertainty_mm else float("inf")
        findings.append(
            f"crack widened {crack.latest_width_mm:.1f} mm, {ratio:.0f}x its "
            f"{crack.latest_uncertainty_mm:.1f} mm measurement uncertainty"
            + (f", with {crack.latest_new_area_mm2:.0f} mm2 of new crack area" if crack.latest_new_area_mm2 else "")
            + (f"; widening seen on the last {crack.consecutive} captures" if crack_persistent else ""))
        floor = "ALERT" if crack_persistent else "WATCH"
        if rank(level) < rank(floor):
            level = floor

    # ---- history cap: no ALERT until there is persistence to measure
    capped = False
    if n_captures < MIN_CAPTURES_FOR_ALERT and rank(level) > rank("WATCH"):
        level, capped = "WATCH", True
    if n_captures < MIN_CAPTURES_FOR_ALERT and rank(level) == rank("WATCH"):
        findings.append(
            f"only {n_captures} captures exist, so persistence cannot be judged yet "
            f"({MIN_CAPTURES_FOR_ALERT} are required before an ALERT)")

    if not findings:
        findings.append(
            f"movement of {a.observed[-1]:+.1f} mm is consistent with the seasonal expectation of "
            f"{a.expected[-1]:+.1f} mm (+/- {2 * a.sigma[-1]:.1f} mm)" if a.observed else
            "no measurements for this region")
    return SeverityDecision(severity=level, findings=findings, capped_by_history=capped)


if __name__ == "__main__":
    from contracts.fakes import make_report_series
    from decision.residuals import analyse_residuals
    from decision.seasonal import all_region_ids, fit_expected

    reports, _ = make_report_series(seed=1)
    load_bearing = {rc.region_id: rc.load_bearing for rc in reports[-1].regions}
    print("SEVERITY after each capture (seed 1, drift in east_upper from report 4)\n")
    ids = all_region_ids(reports)
    print(f"  {'report':<8}" + "".join(f"{r:>15}" for r in ids))
    for n in range(1, len(reports) + 1):
        prefix = reports[:n]
        row = []
        for rid in ids:
            a = analyse_residuals(prefix, fit_expected(prefix, rid))
            row.append(classify(a, n + 1, load_bearing[rid], crack_evidence(prefix, rid)).severity)
        print(f"  {n - 1:<8}" + "".join(f"{s:>15}" for s in row))
    print("\n(no peer screening here - pipeline.py adds that, which keeps the drifting\n"
          " region from distorting its neighbours' expectations)")
