"""Does the console start, and do the numbers it shows match the planted truth?   python console/smoke_test.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from console import engine as E  # noqa: E402

geo = E.run_geometry(0, "east_upper", 12.0)
moved = {r.region_id: r.median_mm for r in geo.report.regions}
assert abs(moved["east_upper"] - 12.0) < 1.0, moved
assert all(abs(v) < 1.0 for k, v in moved.items() if k != "east_upper"), moved

a, b = E.example_masks(3, 7)
crack = E.crack_run(a, b, "east_upper", 0.5)
assert abs(crack.delta.d_max_width - 2.0) <= 0.5, crack.delta.d_max_width

report, alerts = E.chain_alerts(geo, crack)
assert {x.region_id: x.severity for x in alerts}["east_upper"] != "INFORMATIONAL"

year = E.year_run(1, "east_upper", 3.5)
sev = {a.region_id: a.alert.severity for a in year.analyses}
assert sev["east_upper"] in ("ALERT", "CRITICAL") and sum(v in ("ALERT", "CRITICAL") for v in sev.values()) == 1, sev

assert E.capture_check("good").qc.verdict == "PASS"
assert E.capture_check("blurry").qc.verdict == "RESHOOT"
rec = E.record_run(geo)
assert rec.verified and set(rec.sections) == {"plan", "elevation", "section"}

from streamlit.testing.v1 import AppTest  # noqa: E402
at = AppTest.from_file(str(ROOT / "console" / "app.py"), default_timeout=240).run()
assert not at.exception, [e.value for e in at.exception]
print("console smoke test passed")

# ---- temple (simulated year) ----
from console import monument as _MO, temple_fig as _TF
_tp = _MO.build_temple(0)
_res = _MO.run_year(_tp, 0)
assert _res.first_alert["tower_upper"] is not None and _res.first_alert["tower_upper"] <= 9, "tower lean not caught"
assert _res.first_alert["col_b_lower"] is not None, "settling column not caught"
assert _res.severity["cella_front_upper"][-1] == "INFORMATIONAL", "control wall should stay quiet"
assert _res.crack_series["hairline (control)"][-1] == _res.crack_series["hairline (control)"][0]
for _m in (0, 6, 12):
    _TF.temple_figure(_tp, _MO.month_state(_tp, _m, 0), 60)
print("temple smoke test passed")
