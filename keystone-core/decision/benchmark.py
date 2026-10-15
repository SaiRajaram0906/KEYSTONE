"""Benchmark - the numbers for the write-up.

    python -m decision.benchmark              # 200 synthetic years per experiment
    python -m decision.benchmark --seeds 50   # quicker

Every experiment evaluates after EVERY capture, the way the system would run
in service, and uses the ground truth from contracts/fakes.py only to SCORE
the alerts - never to make them.

    1. Naive threshold vs the seasonal model on years with NO structural drift
       -> false alarms caused by ordinary summer heat
    2. What each safeguard is worth (ablation)
       -> the judged region allowed into its own fit; peer screening off
    3. Detection speed vs drift rate
    4. The trade-off: how strict the persistence test is vs how early drift
       is caught and how many false alarms it costs

A "false alarm" is any ALERT or CRITICAL on a region that is not drifting, or
on the drifting region before the drift began.
"""
from __future__ import annotations

import argparse
import statistics
from typing import Callable

import decision.residuals as residuals_mod
import decision.severity as severity_mod
from contracts.fakes import WALL_REGIONS, make_report_series
from contracts.schema import SEVERITIES, ChangeReport
from decision.pipeline import analyse

RANK = {s: i for i, s in enumerate(SEVERITIES)}
NAIVE_LOD_MULTIPLE = 2.0      # the naive rule: alarm when movement exceeds twice the detection limit

Decider = Callable[[list[ChangeReport]], dict[str, str]]


# --------------------------------------------------------------------------
# the deciders being compared
# --------------------------------------------------------------------------
def naive_threshold(history: list[ChangeReport]) -> dict[str, str]:
    """ALERT whenever a region has moved more than NAIVE_LOD_MULTIPLE x its detection limit.

    No weather, no history: exactly the rule a first version would ship.
    """
    return {rc.region_id: ("ALERT" if abs(rc.median_mm) >= NAIVE_LOD_MULTIPLE * rc.lod_mm else "INFORMATIONAL")
            for rc in history[-1].regions}


def keystone(history: list[ChangeReport]) -> dict[str, str]:
    return {ra.region_id: ra.decision.severity for ra in analyse(history)}


def keystone_self_included(history: list[ChangeReport]) -> dict[str, str]:
    return {ra.region_id: ra.decision.severity for ra in analyse(history, leave_one_out=False)}


def keystone_no_screening(history: list[ChangeReport]) -> dict[str, str]:
    return {ra.region_id: ra.decision.severity for ra in analyse(history, screen_peers=False)}


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score(decider: Decider, seeds: int, drift_region: str | None,
          drift_mm_per_capture: float = 3.5, rotate_regions: bool = False) -> dict:
    """Run `seeds` synthetic years, deciding after every capture."""
    false_alarm_years = 0
    false_alarm_count = 0
    detected = 0
    delays: list[int] = []
    region_ids = [r.region_id for r in WALL_REGIONS]

    for seed in range(seeds):
        region = drift_region
        if drift_region and rotate_regions:
            region = region_ids[seed % len(region_ids)]
        reports, truth = make_report_series(seed=seed, drift_region=region,
                                            drift_mm_per_capture=drift_mm_per_capture)
        first_detection = None
        false_alarms_this_year = 0
        for n in range(1, len(reports) + 1):
            for rid, severity in decider(reports[:n]).items():
                if RANK[severity] < RANK["ALERT"]:
                    continue
                real = (rid == truth.drift_region and n - 1 >= truth.drift_onset_index)
                if real:
                    if first_detection is None:
                        first_detection = n - 1
                else:
                    false_alarms_this_year += 1
        false_alarm_count += false_alarms_this_year
        false_alarm_years += bool(false_alarms_this_year)
        if first_detection is not None:
            detected += 1
            delays.append(first_detection - truth.drift_onset_index + 1)

    return {
        "years": seeds,
        "false_alarm_years": false_alarm_years,
        "false_alarms": false_alarm_count,
        "detected": detected,
        "median_delay": statistics.median(delays) if delays else None,
        "mean_delay": statistics.fmean(delays) if delays else None,
    }


def _delay(result: dict) -> str:
    if result["median_delay"] is None:
        return "     -"
    return f"{result['mean_delay']:>6.1f}"


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------
def experiment_threshold_vs_model(seeds: int) -> None:
    print(f"\n1. SUMMER HEAT ONLY - {seeds} synthetic years, no structural drift anywhere")
    print(f"   {'method':<44}{'years with a false alarm':>26}{'false alarms':>15}")
    for name, decider in ((f"naive threshold (|movement| >= {NAIVE_LOD_MULTIPLE:.0f} x lod)", naive_threshold),
                          ("KEYSTONE seasonal model", keystone)):
        r = score(decider, seeds, drift_region=None)
        print(f"   {name:<44}{r['false_alarm_years']:>20} / {r['years']:<4}{r['false_alarms']:>14}")
    twenty = score(naive_threshold, 20, drift_region=None)
    print(f"   (first 20 years only, naive threshold: {twenty['false_alarms']} false alarms "
          f"in {twenty['false_alarm_years']} of 20 years)")


def experiment_ablation(seeds: int) -> None:
    print(f"\n2. WHAT EACH SAFEGUARD IS WORTH - {seeds} years, east_upper drifting 3.5 mm per capture")
    print(f"   {'method':<44}{'drift caught':>14}{'mean delay':>12}{'false-alarm years':>19}")
    for name, decider in (("naive threshold", naive_threshold),
                          ("judged region INCLUDED in its own fit", keystone_self_included),
                          ("leave-one-out, peer screening OFF", keystone_no_screening),
                          ("KEYSTONE (leave-one-out + screening)", keystone)):
        r = score(decider, seeds, drift_region="east_upper")
        print(f"   {name:<44}{r['detected']:>8} / {r['years']:<4}{_delay(r):>11}{r['false_alarm_years']:>13} / {r['years']}")
    print("   delay = captures from the first drifting capture to the first ALERT (1 = same capture)")

    print("\n   The drifting region's healthy same-height NEIGHBOURS, once drift is under way:")
    print(f"   {'method':<44}{'bias of expectation':>21}{'band (1 sigma)':>16}")
    for name, screen in (("leave-one-out, peer screening OFF", False),
                         ("KEYSTONE (leave-one-out + screening)", True)):
        bias, band = [], []
        for seed in range(min(seeds, 100)):
            reports, truth = make_report_series(seed=seed, drift_region="east_upper")
            for n in range(truth.drift_onset_index + 2, len(reports) + 1):
                for ra in analyse(reports[:n], screen_peers=screen):
                    if ra.region_id != truth.drift_region and ra.region_id.endswith("upper"):
                        bias.append(ra.residuals.latest_residual_mm)
                        band.append(ra.residuals.sigma[-1])
        print(f"   {name:<44}{statistics.fmean(bias):>18.1f} mm{statistics.fmean(band):>13.1f} mm")
    print("   without screening a failing wall makes its neighbours look as if they moved inward,\n"
          "   and widens their band so far that a second failure next to it would be missed")


def experiment_drift_rate(seeds: int) -> None:
    print(f"\n3. DETECTION SPEED vs DRIFT RATE - {seeds} years each, drifting region rotates over all six")
    print(f"   {'drift (mm per capture)':<26}{'caught by year end':>20}{'mean delay':>12}{'false-alarm years':>19}")
    for rate in (1.0, 1.5, 2.5, 3.5, 5.0):
        r = score(keystone, seeds, drift_region="east_upper", drift_mm_per_capture=rate, rotate_regions=True)
        print(f"   {rate:<26}{r['detected']:>14} / {r['years']:<4}{_delay(r):>11}{r['false_alarm_years']:>13} / {r['years']}")
    print("   drift starts at capture 5 of 10, so at most 5 drifting captures exist by year end")


def experiment_tradeoff(seeds: int) -> None:
    print(f"\n4. THE TRADE-OFF - strictness of the persistence test, {seeds} years each")
    print(f"   {'residual bar (sigma)':<24}{'false-alarm years':>19}{'drift caught':>16}{'mean delay':>12}")
    original = (residuals_mod.Z_STRONG, severity_mod.Z_STRONG_LOAD_BEARING)
    try:
        for bar in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
            residuals_mod.Z_STRONG = bar
            severity_mod.Z_STRONG_LOAD_BEARING = bar - 0.5
            quiet = score(keystone, seeds, drift_region=None)
            drift = score(keystone, seeds, drift_region="east_upper", rotate_regions=True)
            marker = "   <- shipped setting" if bar == original[0] else ""
            print(f"   {bar:<24}{quiet['false_alarm_years']:>13} / {quiet['years']:<4}"
                  f"{drift['detected']:>10} / {drift['years']:<4}{_delay(drift):>11}{marker}")
    finally:
        residuals_mod.Z_STRONG, severity_mod.Z_STRONG_LOAD_BEARING = original
    print("   false-alarm years are counted on drift-free years; load-bearing regions use a bar 0.5 lower")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=200, help="synthetic years per experiment (default 200)")
    args = ap.parse_args()

    print("=" * 78)
    print("KEYSTONE DECISION BENCHMARK - synthetic data from contracts/fakes.py")
    print("=" * 78)
    experiment_threshold_vs_model(args.seeds)
    experiment_ablation(args.seeds)
    experiment_drift_rate(args.seeds)
    experiment_tradeoff(args.seeds)
    print("\nAll figures describe SYNTHETIC walls. The seasonal coefficients in the fakes are not\n"
          "measurements of real stone; a real monument needs about a year of captures to calibrate.")


if __name__ == "__main__":
    main()
