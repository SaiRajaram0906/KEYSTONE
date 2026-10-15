"""Decision pipeline - the entry point the acceptance tests call.

    evaluate(history) -> list[Alert]

history is every ChangeReport for one monument, oldest first, all measured
against the same reference capture. The result is exactly one Alert per
region in the LATEST report, INFORMATIONAL included, so every region has a
recorded decision after every capture.

    for each region:   seasonal.fit_expected      what the weather explains
                       residuals.analyse_residuals what is left, and does it persist
                       severity.classify           which of the four bands
                       routing.route_for           who is told, and what is recommended

PEER SCREENING
--------------
The seasonal model judges each region against its neighbours at the same
height. That has one weakness: a neighbour that is itself failing would
distort "normal" for everyone else (a healthy wall would look as if it were
moving INWARD relative to a drifting one). So evaluate() runs in two passes.
Pass one fits every region against all its peers and finds the region that is
furthest outside its band. If it is far enough out, it is marked suspect and
pass two refits the others without it. The suspect is still judged, against
its healthy neighbours - it just is not allowed to define normal. At most a
minority of any height group can be marked suspect, so the majority always
defines the expectation.

evaluate() is pure: no files, no database, no clock. The decision date is the
latest capture's date, so the same history always gives the same alerts.
Use evaluate_and_archive() when the result should be stored.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from contracts.schema import Alert, ChangeReport, validate_alert
from decision.residuals import ResidualAnalysis, analyse_residuals
from decision.routing import Route, route_for
from decision.seasonal import SeasonalFit, all_region_ids, choose_peers, fit_expected, height_group
from decision.severity import CrackEvidence, SeverityDecision, classify, crack_evidence

__all__ = ["SCREEN_Z", "RegionAnalysis", "analyse", "evaluate", "evaluate_and_archive"]

SCREEN_Z = 3.0      # a peer this far outside its own band may not define normal for others


@dataclass
class RegionAnalysis:
    """Everything known about one region after the latest capture (used by the dashboard)."""

    region_id: str
    load_bearing: bool
    fit: SeasonalFit
    residuals: ResidualAnalysis
    crack: CrackEvidence
    decision: SeverityDecision
    route: Route
    alert: Alert


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _clean_history(history: list[ChangeReport]) -> list[ChangeReport]:
    """Keep the reports that belong with the latest one, oldest first.

    The contract promises one monument and one reference capture per history.
    If something else slips in, it is dropped rather than allowed to corrupt
    the comparison - cumulative movement only means something against a
    single reference.
    """
    if not history:
        return []
    ordered = sorted(history, key=lambda r: r.later.captured_on)
    latest = ordered[-1]
    return [r for r in ordered
            if r.monument_id == latest.monument_id
            and r.earlier.capture_id == latest.earlier.capture_id]


def _screen_suspects(history: list[ChangeReport], region_ids: list[str]) -> set[str]:
    """Pass one: find regions too far out to be trusted as anyone's peer."""
    groups: dict[str, list[str]] = {}
    for rid in region_ids:
        groups.setdefault(height_group(rid), []).append(rid)

    suspects: set[str] = set()
    for members in groups.values():
        allowed = (len(members) - 1) // 2          # always leave a clean majority
        group_suspects: set[str] = set()
        while len(group_suspects) < allowed:
            worst_id, worst_z = None, SCREEN_Z
            for rid in members:
                if rid in group_suspects:
                    continue
                a = analyse_residuals(history, fit_expected(history, rid, exclude=group_suspects))
                if a.z and abs(a.latest_z) >= worst_z:
                    worst_id, worst_z = rid, abs(a.latest_z)
            if worst_id is None:
                break
            group_suspects.add(worst_id)
        suspects |= group_suspects
    return suspects


def _reason(decision: SeverityDecision, fit: SeasonalFit, route: Route,
            latest: ChangeReport, n_captures: int, screened_out: list[str]) -> str:
    parts = ["; ".join(decision.findings).rstrip(".") + "."]
    parts.append(f"Seasonal expectation: {fit.describe()}.")
    if screened_out:
        parts.append("Not used as a reference because of its own unexplained movement: "
                     + ", ".join(screened_out) + ".")
    parts.append(route.action)
    parts.append(f"Evidence: capture {latest.later.capture_id} ({latest.later.captured_on}) compared with "
                 f"reference capture {latest.earlier.capture_id} ({latest.earlier.captured_on}); "
                 f"{n_captures} captures in the history.")
    return " ".join(parts)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def analyse(history: list[ChangeReport], *, leave_one_out: bool = True,
            screen_peers: bool = True) -> list[RegionAnalysis]:
    """Full analysis of every region in the latest report.

    leave_one_out and screen_peers exist ONLY so benchmark.py can switch each
    safeguard off and show what it is worth. evaluate() always uses both.
    """
    history = _clean_history(history)
    if not history:
        return []
    latest = history[-1]
    n_captures = len(history) + 1                  # the reference capture counts
    ids = all_region_ids(history)
    suspects = _screen_suspects(history, ids) if (screen_peers and leave_one_out) else set()

    out: list[RegionAnalysis] = []
    for rc in latest.regions:
        exclude = suspects - {rc.region_id}
        # never leave a region without same-height peers because of screening
        peers, pool = choose_peers(rc.region_id, ids, exclude)
        if pool != "same-height" and choose_peers(rc.region_id, ids)[1] == "same-height":
            exclude = set()
        fit = fit_expected(history, rc.region_id, exclude=exclude, include_self=not leave_one_out)
        residuals = analyse_residuals(history, fit)
        crack = crack_evidence(history, rc.region_id)
        decision = classify(residuals, n_captures, bool(rc.load_bearing), crack)
        route = route_for(decision.severity)
        screened_out = sorted(s for s in exclude
                              if height_group(s) == height_group(rc.region_id))
        alert = Alert(
            monument_id=latest.monument_id,
            region_id=rc.region_id,
            severity=decision.severity,
            recipient=route.recipient,
            reason=_reason(decision, fit, route, latest, n_captures, screened_out),
            reference_capture_id=latest.earlier.capture_id,
            latest_capture_id=latest.later.capture_id,
            decided_on=latest.later.captured_on,
        )
        validate_alert(alert)
        out.append(RegionAnalysis(rc.region_id, bool(rc.load_bearing), fit, residuals,
                                  crack, decision, route, alert))
    return out


def evaluate(history: list[ChangeReport]) -> list[Alert]:
    """One Alert per region in the latest report. See the module docstring."""
    return [ra.alert for ra in analyse(history)]


def evaluate_and_archive(history: list[ChangeReport], db_path: str | Path = "keystone.db") -> list[Alert]:
    """evaluate(), then store every report and the resulting alerts in SQLite."""
    from decision.store import Archive

    alerts = evaluate(history)
    with Archive(db_path) as archive:
        for report in history:
            archive.save_report(report)
        archive.save_alerts(alerts)
    return alerts


if __name__ == "__main__":
    from contracts.fakes import make_report_series

    reports, truth = make_report_series(seed=1)
    print(f"KEYSTONE DECISION - synthetic year, seed 1, true drift in {truth.drift_region} "
          f"from report {truth.drift_onset_index}\n")
    ids = all_region_ids(reports)
    print(f"  {'report':<8}{'date':<12}" + "".join(f"{r:>15}" for r in ids))
    for n in range(1, len(reports) + 1):
        by_id = {a.region_id: a.severity for a in evaluate(reports[:n])}
        print(f"  {n - 1:<8}{reports[n - 1].later.captured_on:<12}" + "".join(f"{by_id[r]:>15}" for r in ids))

    print("\nFINAL ALERTS")
    for a in evaluate(reports):
        print(f"\n  [{a.severity}] {a.region_id} -> {a.recipient}")
        print(f"    {a.reason}")
