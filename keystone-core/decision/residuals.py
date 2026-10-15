"""Residual analysis - what is left once the weather is accounted for.

    residual = observed movement - expected seasonal movement
    z        = residual / band sigma        (how surprising it is)

A single large residual proves little: one noisy capture produces one. What
identifies STRUCTURAL movement is its behaviour over time:

    it persists   several consecutive captures are off in the same direction
    it is kept    the latest residual has not fallen back towards zero
    it grows      the trend across recent captures points away from zero

Seasonal movement fails all three: once the model has removed the weather it
is centred on zero, flips sign, and comes back. This module only measures
these things; deciding what they mean is severity.py's job.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from contracts.schema import ChangeReport
from decision.seasonal import SeasonalFit, region_observations

__all__ = [
    "Z_MILD",
    "Z_STRONG",
    "RETAINED_FRACTION",
    "ResidualAnalysis",
    "run_length",
    "theil_sen_slope",
    "analyse_residuals",
]

Z_MILD = 2.0            # worth noticing
Z_STRONG = 3.0          # very unlikely to be noise
RETAINED_FRACTION = 0.7 # latest residual must keep this share of the run's peak


@dataclass
class ResidualAnalysis:
    """The residual story of one region. Lists are aligned, oldest first."""

    region_id: str
    report_index: list[int] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    observed: list[float] = field(default_factory=list)
    expected: list[float] = field(default_factory=list)
    sigma: list[float] = field(default_factory=list)
    residual: list[float] = field(default_factory=list)
    z: list[float] = field(default_factory=list)

    latest_residual_mm: float = 0.0
    latest_z: float = 0.0
    latest_lod_mm: float = 0.0
    direction: str = "none"          # outward | inward | none
    run_mild: int = 0                # latest consecutive captures with |z| >= Z_MILD, same sign
    run_strong: int = 0              # ... with |z| >= Z_STRONG, same sign
    retained: bool = False           # has the residual stayed out rather than come back?
    trend_mm_per_capture: float = 0.0
    trend_mm_per_year: float = 0.0

    @property
    def persistent(self) -> bool:
        """Off by a lot, twice running, same direction, and not coming back."""
        return self.run_strong >= 2 and self.retained


def run_length(z: list[float], threshold: float) -> int:
    """How many of the MOST RECENT captures in a row have |z| >= threshold
    with the same sign as the latest one. 0 if the latest does not qualify."""
    if not z or abs(z[-1]) < threshold:
        return 0
    sign = 1.0 if z[-1] > 0 else -1.0
    n = 0
    for value in reversed(z):
        if value * sign >= threshold:
            n += 1
        else:
            break
    return n


def theil_sen_slope(x: list[float], y: list[float]) -> float:
    """Median of all pairwise slopes - a trend estimate one bad capture cannot bend."""
    slopes = [(y[j] - y[i]) / (x[j] - x[i])
              for i in range(len(x)) for j in range(i + 1, len(x)) if x[j] != x[i]]
    return float(np.median(slopes)) if slopes else 0.0


def analyse_residuals(history: list[ChangeReport], fit: SeasonalFit,
                      trend_window: int = 4) -> ResidualAnalysis:
    """Build the residual series for fit.region_id and measure its persistence."""
    out = ResidualAnalysis(region_id=fit.region_id)
    own = region_observations(history, fit.region_id)

    for k in sorted(own):
        if k not in fit.expected:
            continue
        observed, lod = own[k]
        sigma = fit.sigma[k]
        residual = observed - fit.expected[k]
        out.report_index.append(k)
        out.dates.append(history[k].later.captured_on)
        out.observed.append(observed)
        out.expected.append(fit.expected[k])
        out.sigma.append(sigma)
        out.residual.append(residual)
        out.z.append(residual / sigma if sigma > 0 else 0.0)
        out.latest_lod_mm = lod

    if not out.z:
        return out

    out.latest_residual_mm = out.residual[-1]
    out.latest_z = out.z[-1]
    out.run_mild = run_length(out.z, Z_MILD)
    out.run_strong = run_length(out.z, Z_STRONG)
    if abs(out.latest_z) >= Z_MILD:
        out.direction = "outward" if out.latest_z > 0 else "inward"

    # "kept": compare the latest residual with the largest one in the current run
    run = max(out.run_mild, 1)
    peak = max(abs(r) for r in out.residual[-run:])
    out.retained = abs(out.latest_residual_mm) >= RETAINED_FRACTION * peak

    # trend over the most recent captures, in mm per capture and mm per year
    recent = slice(-trend_window, None)
    idx = [float(i) for i in out.report_index[recent]]
    res = out.residual[recent]
    out.trend_mm_per_capture = theil_sen_slope(idx, res)
    days = [float(date.fromisoformat(d).toordinal()) for d in out.dates[recent]]
    out.trend_mm_per_year = theil_sen_slope(days, res) * 365.0
    return out


if __name__ == "__main__":
    from contracts.fakes import make_report_series
    from decision.seasonal import fit_expected

    reports, _ = make_report_series(seed=1)
    print("RESIDUALS (seed 1, drift in east_upper from report 4)\n")
    for rid in ("centre_upper", "east_upper"):
        a = analyse_residuals(reports, fit_expected(reports, rid, exclude={"east_upper"}))
        print(f"  {rid}")
        print(f"    {'date':<12}{'observed':>9}{'expected':>9}{'residual':>9}{'z':>7}")
        for d, o, e, r, z in zip(a.dates, a.observed, a.expected, a.residual, a.z):
            print(f"    {d:<12}{o:>9.1f}{e:>9.1f}{r:>9.1f}{z:>7.1f}")
        print(f"    run of strong residuals: {a.run_strong}   retained: {a.retained}   "
              f"trend: {a.trend_mm_per_capture:+.1f} mm/capture   persistent: {a.persistent}\n")
