"""Seasonal model - how far SHOULD this region have moved, given the weather?

THE IDEA
--------
Stone expands with heat and swells with moisture, then comes back. A region
that is 9 mm out in late May is doing what every wall does. So before judging
a region we ask what ordinary weather-driven movement looks like today, and
only the part that is NOT explained by weather is handed on to residuals.py.

The model is deliberately small:

    expected_mm = a * (temperature - temperature_at_reference)
                + b * (rainfall_7d  - rainfall_7d_at_reference)

There is no intercept: every report is measured against the reference
capture, so at reference weather the expected movement is zero by definition.

TWO DESIGN DECISIONS THAT MATTER
--------------------------------
1. POOLING BY HEIGHT. One region gives one number per capture - far too few
   to fit two coefficients in the first year. Regions at the same height
   respond alike (the top of a wall moves more than its base), so we pool
   every region at the same height into one fit. Height is read from the
   region id suffix ("east_upper" -> "upper").

2. LEAVE THE JUDGED REGION OUT. The region being judged never contributes to
   its own expectation. If it did, a region that is slowly failing would drag
   the fit towards itself, its drift would be absorbed into the "seasonal"
   coefficients, and the residual that should expose it would shrink. Its
   neighbours are the only honest witness to what the weather did.

   The same logic runs one step further: pipeline.py first screens for a
   region that is already wildly off, and passes it in `exclude` so that it
   cannot distort the expectation for its healthy neighbours either.

Because the pooled fit includes the peers' measurements from the LATEST
capture, the model never has to extrapolate to weather it has not seen: the
first monsoon week is explained by what the neighbours did that same week.

THE BAND
--------
Each expectation carries a one-sigma uncertainty made of three parts:

    measurement   lod_mm * LOD_TO_SIGMA for the judged region
    fit           the regression's own prediction variance
    individuality REL_COEFF_SPREAD * |expected| - neighbours at the same
                  height do not respond identically, and that mismatch grows
                  with the size of the seasonal swing

The seasonal coefficients recovered here describe the synthetic data in
contracts/fakes.py. A real monument needs roughly a year of captures before
its own coefficients are calibrated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from contracts.schema import ChangeReport

__all__ = [
    "LOD_TO_SIGMA",
    "REL_COEFF_SPREAD",
    "SeasonalFit",
    "all_region_ids",
    "height_group",
    "choose_peers",
    "weather_deltas",
    "region_observations",
    "fit_expected",
]

# lod_mm is a PER-POINT detection limit at 95% confidence, i.e. about two
# sigma for a single point. A region's median over thousands of points is
# steadier than any one point, but registration error is shared by all of
# them and does not average away. lod / 3 is the working value; the peers'
# own scatter overrides it upward whenever the data say the noise is larger.
LOD_TO_SIGMA = 1.0 / 3.0

# Allowed difference in seasonal response between a region and its peers,
# as a fraction of the expected movement.
REL_COEFF_SPREAD = {"same-height": 0.20, "all-regions": 0.50, "self": 0.20, "none": 0.0}

# Residual degrees of freedom needed before the peers' scatter is trusted
# as a noise estimate.
MIN_DOF_FOR_EMPIRICAL_NOISE = 6


@dataclass
class SeasonalFit:
    """What one region was expected to do, report by report.

    expected and sigma are keyed by the report's index in `history`, and only
    contain the reports in which the region actually appears.
    """

    region_id: str
    pool: str                                  # same-height | all-regions | self | none
    peers: list[str] = field(default_factory=list)
    coef_temp_mm_per_c: float | None = None
    coef_rain_mm_per_mm: float | None = None
    n_observations: int = 0
    noise_sigma_mm: float = 0.0
    expected: dict[int, float] = field(default_factory=dict)
    sigma: dict[int, float] = field(default_factory=dict)

    def describe(self) -> str:
        if self.pool == "none":
            return "no seasonal model (no usable observations)"
        source = {
            "same-height": f"{len(self.peers)} peer region(s) at the same height",
            "all-regions": f"{len(self.peers)} other region(s) at mixed heights",
            "self": "this region's own history (no peers available)",
        }[self.pool]
        if self.coef_temp_mm_per_c is None:
            return f"peer average from {source}"
        return (f"{self.coef_temp_mm_per_c:+.2f} mm/C and "
                f"{self.coef_rain_mm_per_mm:+.3f} mm per mm of 7-day rain, from {source}")


# --------------------------------------------------------------------------
# reading the history
# --------------------------------------------------------------------------
def all_region_ids(history: list[ChangeReport]) -> list[str]:
    """Every region id seen anywhere in the history, in first-seen order."""
    seen: dict[str, None] = {}
    for rep in history:
        for rc in rep.regions:
            seen.setdefault(rc.region_id, None)
    return list(seen)


def height_group(region_id: str) -> str:
    """'east_upper' -> 'upper'. Regions without a suffix share the group ''."""
    if "_" not in region_id:
        return ""
    return region_id.rsplit("_", 1)[-1].lower()


def choose_peers(region_id: str, region_ids: list[str],
                 exclude: frozenset[str] | set[str] = frozenset()) -> tuple[list[str], str]:
    """Pick the regions allowed to define 'normal' for region_id.

    Preference order: trusted regions at the same height, then any trusted
    region, then (only if nothing else exists) the region's own history.
    The judged region itself is never a peer.
    """
    others = [r for r in region_ids if r != region_id and r not in exclude]
    same = [r for r in others if height_group(r) == height_group(region_id)]
    if same:
        return same, "same-height"
    if others:
        return others, "all-regions"
    return [], "self"


def weather_deltas(history: list[ChangeReport]) -> dict[int, tuple[float, float]]:
    """Report index -> (d_temperature_c, d_rainfall_mm_7d) against the reference.

    Reports whose weather is missing on either side are left out; the model
    falls back to the peers' average for those.
    """
    out: dict[int, tuple[float, float]] = {}
    for k, rep in enumerate(history):
        t0, t1 = rep.earlier.temperature_c, rep.later.temperature_c
        if t0 is None or t1 is None:
            continue
        r0, r1 = rep.earlier.rainfall_mm_7d, rep.later.rainfall_mm_7d
        d_rain = 0.0 if (r0 is None or r1 is None) else float(r1) - float(r0)
        out[k] = (float(t1) - float(t0), d_rain)
    return out


def region_observations(history: list[ChangeReport], region_id: str) -> dict[int, tuple[float, float]]:
    """Report index -> (median_mm, lod_mm) for one region."""
    out: dict[int, tuple[float, float]] = {}
    for k, rep in enumerate(history):
        for rc in rep.regions:
            if rc.region_id == region_id:
                out[k] = (float(rc.median_mm), float(rc.lod_mm))
                break
    return out


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------
def fit_expected(history: list[ChangeReport], region_id: str,
                 exclude: frozenset[str] | set[str] = frozenset(),
                 include_self: bool = False) -> SeasonalFit:
    """Expected seasonal movement of `region_id` at every report, with a band.

    exclude       regions that must not be used as peers (already suspect).
    include_self  ONLY for the write-up's ablation: lets the judged region
                  into its own fit, to show how a failing region then hides
                  its own drift. evaluate() never sets this.
    """
    own = region_observations(history, region_id)
    if not own:
        return SeasonalFit(region_id=region_id, pool="none")

    ids = all_region_ids(history)
    peers, pool = choose_peers(region_id, ids, exclude)
    sources = list(peers)
    if include_self or pool == "self":
        sources.append(region_id)

    weather = weather_deltas(history)
    observations = {rid: region_observations(history, rid) for rid in sources}

    # ---- pooled least squares over every (peer, report) pair with weather
    rows, targets, lods = [], [], []
    for rid in sources:
        for k, (median, lod) in observations[rid].items():
            lods.append(lod)
            if k in weather:
                rows.append(weather[k])
                targets.append(median)

    noise_floor = (float(np.mean(lods)) if lods else float(np.mean([v[1] for v in own.values()]))) * LOD_TO_SIGMA
    beta = None
    xtx_inv = None
    noise = noise_floor
    if rows:
        X = np.asarray(rows, dtype=float)
        y = np.asarray(targets, dtype=float)
        # lstsq returns the minimum-norm answer when the fit is
        # under-determined (first captures), which at the training weather
        # reduces to "what the peers did" - exactly the behaviour we want.
        beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        dof = len(y) - int(rank)
        if dof >= MIN_DOF_FOR_EMPIRICAL_NOISE:
            rss = float(np.sum((y - X @ beta) ** 2))
            noise = max(noise_floor, math.sqrt(rss / dof))
        xtx_inv = np.linalg.pinv(X.T @ X)

    fit = SeasonalFit(
        region_id=region_id, pool=pool, peers=list(peers),
        coef_temp_mm_per_c=None if beta is None else float(beta[0]),
        coef_rain_mm_per_mm=None if beta is None else float(beta[1]),
        n_observations=len(rows), noise_sigma_mm=noise,
    )

    # ---- expectation and band at every report where the region appears
    spread = REL_COEFF_SPREAD[pool]
    for k, (_, lod) in own.items():
        if beta is not None and k in weather:
            x = np.asarray(weather[k], dtype=float)
            expected = float(x @ beta)
            fit_var = noise ** 2 * float(x @ xtx_inv @ x)
        else:
            # no weather for this capture: fall back to what the peers did
            peer_values = [observations[rid][k][0] for rid in peers if k in observations[rid]]
            expected = float(np.mean(peer_values)) if peer_values else 0.0
            fit_var = noise ** 2 / max(1, len(peer_values))
        measurement = max(lod * LOD_TO_SIGMA, noise if noise > noise_floor else 0.0)
        fit.expected[k] = expected
        fit.sigma[k] = math.sqrt(measurement ** 2 + fit_var + (spread * expected) ** 2)
    return fit


if __name__ == "__main__":
    from contracts.fakes import make_report_series

    reports, truth = make_report_series(seed=1, drift_region=None)
    print("SEASONAL MODEL - leave-one-region-out, pooled by height (seed 1, no drift)\n")
    print(f"  {'region':<14}{'fit mm/C':>10}{'true mm/C':>11}   peers")
    for rid in all_region_ids(reports):
        f = fit_expected(reports, rid)
        print(f"  {rid:<14}{f.coef_temp_mm_per_c:>10.2f}{truth.seasonal_coeff_mm_per_c[rid]:>11.2f}   "
              f"{', '.join(f.peers)}")

    print("\n  centre_upper: observed vs expected (band = 2 sigma)")
    f = fit_expected(reports, "centre_upper")
    obs = region_observations(reports, "centre_upper")
    for k, rep in enumerate(reports):
        print(f"    {rep.later.captured_on}  {rep.later.temperature_c:>5.1f} C   "
              f"observed {obs[k][0]:>6.1f}   expected {f.expected[k]:>6.1f} +/- {2 * f.sigma[k]:.1f}")
    print("\n(the 'true' column is printed for inspection only - evaluate() never sees it)")
