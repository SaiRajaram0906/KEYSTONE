"""Acceptance tests for both KEYSTONE workstreams.

    python -m tests.contracts_test               # everything
    python -m tests.contracts_test --only G      # Geometry
    python -m tests.contracts_test --only D      # Decision

These check ANSWERS, not just shapes. Every scenario comes from
contracts/fakes.py with known ground truth, so a Geometry implementation
that returns a well-formed but wrong displacement fails, and a Decision
implementation that raises alarms on ordinary summer heat fails.

A workstream that has not created its package yet shows PENDING, not
FAIL, so both teams can run this from the first day.

ENTRY POINTS EACH TEAM MUST PROVIDE
-----------------------------------
Geometry:  geometry/pipeline.py
    build_change_report(earlier: CaptureInput, later: CaptureInput,
                        survey: dict[str, list[float]],
                        regions: list[Region]) -> ChangeReport

Decision:  decision/pipeline.py
    evaluate(history: list[ChangeReport]) -> list[Alert]
        history: every report for one monument, oldest first, all against
                 the same reference capture.
        returns: exactly one Alert per region in the LATEST report —
                 INFORMATIONAL included, so every region gets a recorded
                 decision.

The tolerances were checked against a deliberately crude reference
solution before these tests were written, so they are reachable. The
same check showed that a naive threshold ignoring temperature fails.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts.crack_adapter import from_crack_delta  # noqa: E402
from contracts.fakes import WALL_REGIONS, make_geometry_case, make_report_series  # noqa: E402
from contracts.schema import (  # noqa: E402
    SEVERITIES,
    Alert,
    ChangeReport,
    ContractError,
    CrackChange,
    validate_alert,
    validate_change_report,
)

RANK = {s: i for i, s in enumerate(SEVERITIES)}
results: list[tuple[str, str, str]] = []


def check(name: str, fn) -> None:
    t0 = time.time()
    try:
        fn()
        results.append((name, "PASS", ""))
        print(f"  PASS     {name}  ({time.time() - t0:.1f}s)")
    except Exception as exc:  # noqa: BLE001
        results.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))
        print(f"  FAIL     {name}\n           {type(exc).__name__}: {exc}")


def pending(name: str, why: str) -> None:
    results.append((name, "PENDING", why))
    print(f"  PENDING  {name}\n           {why}")


# ==========================================================================
# S — shared contract (always runs)
# ==========================================================================
def test_shared() -> None:
    print("\nSHARED CONTRACT")

    def reports_valid_and_roundtrip():
        reports, _ = make_report_series(seed=3)
        for rep in reports:
            validate_change_report(rep)
            assert ChangeReport.from_json(rep.to_json()) == rep, "JSON round-trip lost data"

    def reference_is_always_earlier():
        reports, _ = make_report_series(seed=3)
        refs = {rep.earlier.capture_id for rep in reports}
        assert len(refs) == 1, "every report must compare against the same reference capture"

    def crack_adapter_bridges_other_team():
        c = from_crack_delta({"d_max_width": 2.0, "new_crack_area": 290.0, "units": "mm"},
                             "east_upper", mm_per_px=0.5)
        assert isinstance(c, CrackChange) and c.width_uncertainty_mm == 0.5
        try:
            from_crack_delta({"d_max_width": 4.0, "new_crack_area": 1.0, "units": "px"},
                             "east_upper", 0.5)
        except ContractError:
            return
        raise AssertionError("pixel-unit crack deltas must be rejected")

    check("synthetic reports validate and survive JSON", reports_valid_and_roundtrip)
    check("reports share one reference capture", reference_is_always_earlier)
    check("crack-seg output converts through the adapter", crack_adapter_bridges_other_team)


# ==========================================================================
# G — geometry
# ==========================================================================
def test_geometry() -> None:
    print("\nWORKSTREAM G — geometry")
    try:
        from geometry.pipeline import build_change_report
    except ImportError as exc:
        pending("geometry.pipeline.build_change_report", f"not implemented yet ({exc})")
        return

    def run(seed, displaced=None):
        case = make_geometry_case(seed=seed, displaced=displaced)
        rep = build_change_report(case.earlier, case.later, case.survey, case.regions)
        return case, rep

    def output_is_a_valid_report():
        _, rep = run(0)
        validate_change_report(rep)
        ChangeReport.from_json(rep.to_json())

    def recovers_scale():
        for seed in (0, 1):
            case, rep = run(seed)
            for meta in (rep.earlier, rep.later):
                truth = case.truth_mm_per_unit[meta.capture_id]
                err = abs(meta.mm_per_unit - truth) / truth
                assert err < 0.01, (
                    f"seed {seed} capture {meta.capture_id}: mm_per_unit {meta.mm_per_unit:.4f} "
                    f"vs truth {truth:.4f} ({err:.2%} off, allowed 1%)")

    def measures_displacement():
        for seed in (0, 1):
            case, rep = run(seed)
            by_id = {r.region_id: r for r in rep.regions}
            for rid, truth in case.truth_displacement_mm.items():
                r = by_id[rid]
                if truth == 0:
                    assert abs(r.median_mm) < 2.0, (
                        f"seed {seed} {rid}: stable region reported {r.median_mm:+.2f} mm")
                else:
                    assert abs(r.median_mm - truth) < 2.5, (
                        f"seed {seed} {rid}: reported {r.median_mm:+.2f} mm, truth {truth:+.2f}")
                    assert r.significant_fraction > 0.5, (
                        f"seed {seed} {rid}: a {truth} mm move should be significant at most "
                        f"points, got significant_fraction {r.significant_fraction:.2f}")

    def lod_is_honest():
        case, rep = run(0)
        for r in rep.regions:
            if case.truth_displacement_mm[r.region_id] == 0:
                assert abs(r.median_mm) < r.lod_mm, (
                    f"{r.region_id}: stable region's |median| {abs(r.median_mm):.2f} exceeds its "
                    f"own lod {r.lod_mm:.2f} — the lod understates the noise")
                assert r.significant_fraction < 0.3, (
                    f"{r.region_id}: {r.significant_fraction:.0%} of points flagged on a stable region")

    def no_change_reads_zero():
        case, rep = run(2, displaced={})
        for r in rep.regions:
            assert abs(r.median_mm) < 2.0, f"{r.region_id} reported {r.median_mm:+.2f} mm with nothing moved"

    def registration_reported():
        _, rep = run(0)
        assert 0 < rep.registration_rmse_mm < 5.0, (
            f"registration_rmse_mm {rep.registration_rmse_mm} — expected a few mm at most")

    check("output is a valid ChangeReport", output_is_a_valid_report)
    check("recovers scale from markers within 1%", recovers_scale)
    check("measures a 12 mm displacement, stable regions near zero", measures_displacement)
    check("level of detection is honest on stable regions", lod_is_honest)
    check("identical walls report no change", no_change_reads_zero)
    check("registration error reported and plausible", registration_reported)


# ==========================================================================
# D — decision
# ==========================================================================
def test_decision() -> None:
    print("\nWORKSTREAM D — decision")
    try:
        from decision.pipeline import evaluate
    except ImportError as exc:
        pending("decision.pipeline.evaluate", f"not implemented yet ({exc})")
        return

    region_ids = [r.region_id for r in WALL_REGIONS]

    def by_region(alerts):
        return {a.region_id: a for a in alerts}

    def one_valid_alert_per_region():
        reports, _ = make_report_series(seed=0)
        alerts = evaluate(reports)
        assert all(isinstance(a, Alert) for a in alerts), "evaluate must return Alert objects"
        for a in alerts:
            validate_alert(a)
            assert Alert.from_json(a.to_json()) == a, "Alert JSON round-trip lost data"
        got = sorted(a.region_id for a in alerts)
        assert got == sorted(region_ids), f"expected one alert per region, got {got}"

    def alerts_are_traceable():
        reports, _ = make_report_series(seed=0)
        for a in evaluate(reports):
            assert a.reference_capture_id == reports[0].earlier.capture_id
            assert a.latest_capture_id == reports[-1].later.capture_id

    def no_alarm_before_three_captures():
        reports, _ = make_report_series(seed=0, drift_onset_capture=1, drift_mm_per_capture=15.0)
        for a in evaluate(reports[:1]):
            assert RANK[a.severity] < RANK["ALERT"], (
                f"{a.region_id} raised {a.severity} with only two captures — the rule is "
                "three captures minimum before ALERT")

    def summer_heat_is_not_an_alarm():
        """The core test. Seasonal movement alone must never reach ALERT."""
        for seed in range(10):
            reports, _ = make_report_series(seed=seed, drift_region=None)
            for n in range(1, len(reports) + 1):
                for a in evaluate(reports[:n]):
                    assert RANK[a.severity] < RANK["ALERT"], (
                        f"seed {seed}, after report {n - 1} ({reports[n - 1].later.captured_on}, "
                        f"{reports[n - 1].later.temperature_c} C): {a.region_id} raised "
                        f"{a.severity} on purely seasonal movement")

    def structural_drift_is_caught():
        missed = []
        for seed in range(10):
            reports, truth = make_report_series(seed=seed)
            final = by_region(evaluate(reports))
            if RANK[final[truth.drift_region].severity] < RANK["ALERT"]:
                missed.append(seed)
            for rid, a in final.items():
                if rid != truth.drift_region:
                    assert RANK[a.severity] <= RANK["WATCH"], (
                        f"seed {seed}: stable region {rid} raised {a.severity}")
        assert len(missed) <= 1, f"drift missed on seeds {missed} (at most 1 of 10 allowed)"

    def no_alarm_before_onset():
        for seed in range(10):
            reports, truth = make_report_series(seed=seed)
            for n in range(1, truth.drift_onset_index + 1):
                a = by_region(evaluate(reports[:n]))[truth.drift_region]
                assert RANK[a.severity] < RANK["ALERT"], (
                    f"seed {seed}: {truth.drift_region} raised {a.severity} before drift began")

    def load_bearing_escalates():
        reports, _ = make_report_series(seed=4, drift_region="centre_upper", drift_mm_per_capture=4.0)
        a = by_region(evaluate(reports))["centre_upper"]
        assert RANK[a.severity] >= RANK["ALERT"], (
            f"drift in a load-bearing pier only reached {a.severity}")

    def crack_widening_is_used():
        reports, _ = make_report_series(seed=5, drift_region=None)
        last = reports[-1]
        injected = dataclasses.replace(
            last, crack_changes=[CrackChange("west_lower", 3.0, 400.0, 0.5)])
        a = by_region(evaluate(reports[:-1] + [injected]))["west_lower"]
        assert RANK[a.severity] >= RANK["WATCH"], (
            f"a crack that widened 3 mm (6x its 0.5 mm uncertainty) produced only {a.severity}")

    check("one valid Alert per region, JSON-safe", one_valid_alert_per_region)
    check("alerts name the captures that caused them", alerts_are_traceable)
    check("no ALERT with fewer than three captures", no_alarm_before_three_captures)
    check("summer heat never reaches ALERT (10 seeds, every prefix)", summer_heat_is_not_an_alarm)
    check("structural drift caught, stable regions stay calm", structural_drift_is_caught)
    check("no ALERT before the drift begins", no_alarm_before_onset)
    check("drift in a load-bearing region escalates", load_bearing_escalates)
    check("crack widening from the crack-seg team is acted on", crack_widening_is_used)


# ==========================================================================
# P — capture and preservation
# ==========================================================================
def test_preservation() -> None:
    print("\nWORKSTREAM P — capture and preservation")
    try:
        from preservation.pipeline import (Archive, check_capture, extract_section,
                                           locate_markers)
    except ImportError as exc:
        pending("preservation.pipeline", f"not implemented yet ({exc})")
        return

    import tempfile

    import numpy as np

    from contracts.fakes_capture import (
        SCENARIOS,
        WALL_H,
        WALL_W,
        make_capture_cloud,
        make_monument,
        make_session,
        make_wall_cloud,
    )
    from contracts.preservation import (
        MonumentRecord,
        validate_archive_entry,
        validate_capture_qc,
        validate_monument_record,
        validate_section,
    )

    monument = make_monument()

    # ---- registry ----
    def registry_is_valid_and_feeds_geometry():
        validate_monument_record(monument)
        assert MonumentRecord.from_json(monument.to_json()) == monument, \
            "MonumentRecord JSON round-trip lost data"
        survey = monument.survey()
        assert len(survey) >= 3, "survey() must give Geometry at least 3 markers"
        for mid, pos in survey.items():
            assert len(pos) == 3, f"marker {mid} must be [x, y, z] in mm"

    # ---- the gate ----
    def qc_output_is_valid():
        for scenario in SCENARIOS:
            session, _ = make_session(scenario, seed=0, monument=monument)
            qc = check_capture(session, monument)
            validate_capture_qc(qc)
            assert qc.capture_id == session.capture_id
            from contracts.preservation import CaptureQC
            assert CaptureQC.from_json(qc.to_json()) == qc, "CaptureQC JSON round-trip lost data"

    def good_captures_are_not_rejected():
        """The false-positive test. Sending someone back to a remote site
        for no reason is how a monitoring programme gets abandoned."""
        wrong = []
        for seed in range(12):
            session, _ = make_session("good", seed=seed, monument=monument)
            qc = check_capture(session, monument)
            if qc.verdict != "PASS":
                wrong.append((seed, qc.verdict))
        assert not wrong, f"good captures rejected on seeds {wrong}"

    def every_defect_is_caught():
        missed = []
        for scenario in ("too_few", "blurry", "markers_missing", "overexposed", "thin_overlap"):
            for seed in range(6):
                session, truth = make_session(scenario, seed=seed, monument=monument)
                qc = check_capture(session, monument)
                if qc.verdict != "RESHOOT":
                    missed.append((scenario, seed, qc.verdict))
        assert not missed, (
            f"defects passed the gate: {missed[:6]}. Each scenario fails in a "
            "different way — count, resolution, sharpness, exposure, markers, "
            "redundancy — so one or two checks cannot catch them all."
        )

    def salvageable_capture_is_flagged_not_rejected():
        """dark_gap covers most of the wall. Rejecting it wastes a visit;
        passing it silently makes Geometry report a region it cannot see."""
        for seed in range(6):
            session, truth = make_session("dark_gap", seed=seed, monument=monument)
            qc = check_capture(session, monument)
            assert qc.verdict == "PASS_WITH_GAPS", (
                f"seed {seed}: dark_gap got {qc.verdict}; most of the wall is usable, "
                "but the uncovered region must be named")
            for rid in truth.uncovered_region_ids:
                assert rid in qc.weak_region_ids, (
                    f"seed {seed}: {rid} was never photographed but is not in "
                    f"weak_region_ids ({qc.weak_region_ids})")

    def reshoot_guidance_is_actionable():
        """The operator is still on site. 'Poor quality' sends them home
        none the wiser."""
        session, truth = make_session("blurry", seed=0, monument=monument)
        qc = check_capture(session, monument)
        assert qc.verdict == "RESHOOT"
        text = " ".join(qc.reshoot_guidance).lower()
        assert len(text) > 20, "guidance is too short to act on"
        named = sum(p.lower() in text for p in truth.bad_photo_ids)
        mentions_problem = any(w in text for w in ("blur", "sharp", "shake", "steady", "focus"))
        assert named or mentions_problem, (
            f"guidance does not identify what to re-shoot: {qc.reshoot_guidance}")

    # ---- archive ----
    def archive_round_trips_and_verifies():
        with tempfile.TemporaryDirectory() as tmp:
            arc = Archive(tmp)
            payload = b"point cloud bytes" * 500
            entry = arc.store(payload, monument_id="M-01", kind="pointcloud",
                              created_on="2026-01-10", capture_id="C000",
                              is_reference=True)
            validate_archive_entry(entry)
            assert arc.fetch(entry.artefact_id) == payload, "archive did not return what was stored"
            assert arc.verify(entry.artefact_id) is True

    def archive_detects_corruption():
        """An archive that cannot prove the bytes are unchanged is a folder,
        not a record."""
        with tempfile.TemporaryDirectory() as tmp:
            arc = Archive(tmp)
            entry = arc.store(b"A" * 4096, monument_id="M-01", kind="pointcloud",
                              created_on="2026-01-10", capture_id="C000")
            path = Path(arc.path_for(entry.artefact_id))
            assert path.exists(), "path_for must point at the stored bytes"
            data = bytearray(path.read_bytes())
            data[len(data) // 2] ^= 0xFF                 # flip one byte
            path.write_bytes(bytes(data))
            assert arc.verify(entry.artefact_id) is False, \
                "a flipped byte must fail verification"

    def retention_spares_the_reference():
        with tempfile.TemporaryDirectory() as tmp:
            arc = Archive(tmp)
            arc.store(b"R" * 200_000, monument_id="M-01", kind="pointcloud",
                      created_on="2020-01-10", capture_id="C000", is_reference=True)
            for i in range(1, 6):
                arc.store(bytes([i]) * 200_000, monument_id="M-01", kind="pointcloud",
                          created_on=f"20{19 + i}-06-10", capture_id=f"C{i:03d}")
            before = arc.stats("M-01")
            assert before.n_entries == 6
            assert before.total_bytes > 0

            reclaimed = arc.apply_retention("M-01")
            after = arc.stats("M-01")
            assert reclaimed > 0, (
                "retention reclaimed nothing. The archive only ever grows, so "
                "something must be decimated or tiered.")
            assert after.total_bytes < before.total_bytes

            for e in arc.entries("M-01"):
                validate_archive_entry(e)
                if e.is_reference:
                    assert e.density == 1.0, (
                        "the reference capture was decimated — every future "
                        "comparison is made against it")

    # ---- export ----
    def elevation_has_real_dimensions():
        """The claim that the archive 'guides restoration' rests on this:
        a slice must come out with the wall's true size, in millimetres."""
        cloud = make_wall_cloud(seed=0)
        sec = extract_section(cloud, monument, capture_id="C000",
                              kind="elevation", at_mm=0.0, thickness_mm=200.0)
        validate_section(sec)
        u0, v0, u1, v1 = sec.extent_mm
        assert abs((u1 - u0) - WALL_W) < 300.0, (
            f"elevation width {u1 - u0:.0f} mm, wall is {WALL_W:.0f} mm")
        assert abs((v1 - v0) - WALL_H) < 300.0, (
            f"elevation height {v1 - v0:.0f} mm, wall is {WALL_H:.0f} mm")
        assert sec.n_points > 0

    def plan_shows_the_buttress():
        """A horizontal cut must show the pier standing 400 mm proud —
        depth an elevation cannot show, which is why both views exist."""
        cloud = make_wall_cloud(seed=0, buttress=True)
        sec = extract_section(cloud, monument, capture_id="C000",
                              kind="plan", at_mm=1500.0, thickness_mm=200.0)
        validate_section(sec)
        depth = sec.extent_mm[3] - sec.extent_mm[1]
        assert depth > 300.0, (
            f"plan at z=1500 spans only {depth:.0f} mm in depth; the buttress "
            "stands 400 mm proud of the face and should appear")

    # ---- markers in the reconstructed cloud ----
    def markers_are_found_and_identified():
        """Geometry cannot start until somebody says where the surveyed
        markers are INSIDE the raw cloud. Nothing upstream provides that."""
        worst, wrong = 0.0, []
        for seed in range(5):
            cloud, truth = make_capture_cloud(monument, seed=seed)
            got = locate_markers(cloud, monument)
            missing = {m.marker_id for m in monument.markers} - set(got)
            if missing:
                wrong.append((seed, f"missing {sorted(missing)}"))
                continue
            for mid, pos in got.items():
                err_mm = float(np.linalg.norm(
                    np.asarray(pos) - truth.marker_positions_raw[mid])) * truth.mm_per_unit
                worst = max(worst, err_mm)
                if err_mm > 25.0:
                    wrong.append((seed, f"{mid} off by {err_mm:.0f} mm"))
        assert not wrong, (
            f"marker location failed: {wrong[:4]}. Targets are dense discs, so "
            "they stand out by local density; the arrangement is symmetric, so "
            "identity comes from target SIZE, not from position.")
        assert worst <= 25.0

    def located_markers_recover_the_true_scale():
        """What the markers are actually for. If these positions are wrong,
        every millimetre Geometry reports afterwards is wrong too."""
        for seed in range(3):
            cloud, truth = make_capture_cloud(monument, seed=seed)
            got = locate_markers(cloud, monument)
            ids = [m.marker_id for m in monument.markers if m.marker_id in got]
            assert len(ids) >= 3, "at least 3 markers are needed to fix scale"
            src = np.array([got[i] for i in ids])
            dst = np.array([m.position_mm for m in monument.markers if m.marker_id in got])
            # closed-form similarity scale between the two point sets
            s_c = src - src.mean(0)
            d_c = dst - dst.mean(0)
            _, sig, _ = np.linalg.svd(d_c.T @ s_c / len(src))
            est = float(sig.sum() / ((s_c ** 2).sum() / len(src)))
            err = abs(est - truth.mm_per_unit) / truth.mm_per_unit
            assert err < 0.01, (
                f"seed {seed}: scale from located markers is {est:.2f} mm/unit "
                f"against a true {truth.mm_per_unit:.2f} ({err:.1%} off, 1% allowed)")

    check("registry is valid and gives Geometry its survey", registry_is_valid_and_feeds_geometry)
    check("markers are found and identified in a raw cloud", markers_are_found_and_identified)
    check("located markers recover the true scale within 1%", located_markers_recover_the_true_scale)
    check("QC output is valid for every scenario, JSON-safe", qc_output_is_valid)
    check("good captures are never sent back (12 seeds)", good_captures_are_not_rejected)
    check("every defect type is caught (5 defects, 6 seeds each)", every_defect_is_caught)
    check("a salvageable capture is flagged, not rejected", salvageable_capture_is_flagged_not_rejected)
    check("reshoot guidance says what to re-shoot", reshoot_guidance_is_actionable)
    check("archive round-trips and verifies", archive_round_trips_and_verifies)
    check("archive detects a single flipped byte", archive_detects_corruption)
    check("retention reclaims space but spares the reference", retention_spares_the_reference)
    check("elevation comes out at the wall's real size", elevation_has_real_dimensions)
    check("plan view shows depth an elevation cannot", plan_shows_the_buttress)


# ==========================================================================
# I — integration (runs when both exist)
# ==========================================================================
def test_integration() -> None:
    print("\nINTEGRATION — G output into D")
    try:
        from decision.pipeline import evaluate
        from geometry.pipeline import build_change_report
    except ImportError:
        pending("end to end", "needs both geometry and decision implemented")
        return

    def geometry_feeds_decision_over_json():
        case = make_geometry_case(seed=0)
        rep = build_change_report(case.earlier, case.later, case.survey, case.regions)
        wire = ChangeReport.from_json(rep.to_json())        # cross the boundary as JSON
        alerts = evaluate([wire])
        for a in alerts:
            validate_alert(a)
        assert len(alerts) == len(case.regions)

    def registry_drives_geometry():
        """P's monument record is what Geometry runs on in the field —
        not the hardcoded survey in the fakes."""
        try:
            from contracts.fakes_capture import make_monument
            from contracts.schema import Region
        except ImportError:
            return
        monument = make_monument()
        case = make_geometry_case(seed=0)
        regions = [Region(r["region_id"], r["bounds_min"], r["bounds_max"],
                          r.get("load_bearing", False)) for r in monument.regions]
        rep = build_change_report(case.earlier, case.later, monument.survey(), regions)
        validate_change_report(rep)
        assert {r.region_id for r in rep.regions} == {r["region_id"] for r in monument.regions}

    check("a real Geometry report drives Decision across JSON", geometry_feeds_decision_over_json)
    check("P's registry drives Geometry in place of the fakes", registry_drives_geometry)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["S", "P", "G", "D", "I"])
    args = ap.parse_args()

    print("=" * 66)
    print("KEYSTONE CONTRACT AND ACCEPTANCE TESTS")
    print("=" * 66)
    for key, fn in (("S", test_shared), ("P", test_preservation),
                    ("G", test_geometry), ("D", test_decision),
                    ("I", test_integration)):
        if args.only in (None, key):
            fn()

    n_pass = sum(s == "PASS" for _, s, _ in results)
    n_fail = sum(s == "FAIL" for _, s, _ in results)
    n_pend = sum(s == "PENDING" for _, s, _ in results)
    print("\n" + "=" * 66)
    print(f"{n_pass} passed, {n_fail} failed, {n_pend} pending")
    if n_fail:
        sys.exit(1)
    print("everything implemented so far conforms" if n_pend else "all workstreams conform — ready to integrate")


if __name__ == "__main__":
    main()
