"""KEYSTONE dashboard - what the Decision workstream sees and decides.

    streamlit run decision/dashboard.py

For every region it shows the measured movement over time against the shaded
seasonal band (what the weather explains), and the current alerts with their
recipients and reasons. A slider replays the year capture by capture, so you
can watch a drift leave the band while summer heat stays inside it.

Data comes either from the synthetic generator in contracts/fakes.py or from
the SQLite archive written by decision/store.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run decision/dashboard.py` puts decision/ on sys.path, not the
# project root, so the imports below would fail without this line.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                      # noqa: E402
import plotly.graph_objects as go        # noqa: E402
import streamlit as st                   # noqa: E402

from contracts.fakes import WALL_REGIONS, make_report_series   # noqa: E402
from contracts.schema import SEVERITIES, ChangeReport          # noqa: E402
from decision.pipeline import RegionAnalysis, analyse          # noqa: E402
from decision.routing import ROUTES                            # noqa: E402
from decision.store import DEFAULT_DB, Archive                 # noqa: E402

# Status colours always travel with a text label - never colour alone.
SEVERITY_COLOUR = {
    "INFORMATIONAL": "#0ca30c",
    "WATCH": "#fab219",
    "ALERT": "#ec835a",
    "CRITICAL": "#d03b3b",
}
SEVERITY_MARK = {"INFORMATIONAL": "●", "WATCH": "▲", "ALERT": "◆", "CRITICAL": "■"}
OBSERVED_COLOUR = "#2a78d6"
EXPECTED_COLOUR = "#7a7974"
BAND_COLOUR = "rgba(122, 121, 116, 0.18)"
NAIVE_LOD_MULTIPLE = 2.0

st.set_page_config(page_title="KEYSTONE - Decision", page_icon="🏛️", layout="wide")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def synthetic_history(seed: int, drift_region: str | None, drift_rate: float) -> list[str]:
    reports, _ = make_report_series(seed=seed, drift_region=drift_region,
                                    drift_mm_per_capture=drift_rate)
    return [r.to_json() for r in reports]            # JSON keeps the cache hashable and honest


def load_archive(db_path: str, monument_id: str | None) -> tuple[list[ChangeReport], list[str]]:
    if not Path(db_path).exists():
        return [], []
    with Archive(db_path) as archive:
        monuments = archive.monuments()
        if not monuments:
            return [], []
        chosen = monument_id if monument_id in monuments else monuments[0]
        return archive.load_history(chosen), monuments


# --------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------
def region_figure(ra: RegionAnalysis, show_naive: bool, height: int = 300,
                  y_range: tuple[float, float] | None = None, legend: bool = True) -> go.Figure:
    r = ra.residuals
    upper = [e + 2 * s for e, s in zip(r.expected, r.sigma)]
    lower = [e - 2 * s for e, s in zip(r.expected, r.sigma)]
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=r.dates, y=upper, mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=r.dates, y=lower, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=BAND_COLOUR, name="seasonal band (±2σ)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=r.dates, y=r.expected, mode="lines", name="expected (seasonal)",
                             line=dict(color=EXPECTED_COLOUR, width=2, dash="dash"),
                             hovertemplate="expected %{y:.1f} mm<extra></extra>"))
    fig.add_trace(go.Scatter(x=r.dates, y=r.observed, mode="lines+markers", name="measured",
                             line=dict(color=OBSERVED_COLOUR, width=2),
                             marker=dict(size=8, color=OBSERVED_COLOUR, line=dict(width=2, color="white")),
                             hovertemplate="measured %{y:.1f} mm<extra></extra>"))
    if show_naive and r.latest_lod_mm:
        fig.add_hline(y=NAIVE_LOD_MULTIPLE * r.latest_lod_mm, line=dict(color="#d03b3b", width=1, dash="dot"),
                      annotation_text="naive threshold (2 × lod)", annotation_position="top left",
                      annotation_font=dict(size=11, color="#52514e"))

    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(size=11)),
        yaxis=dict(title="movement since reference (mm)", zeroline=True, zerolinecolor="rgba(0,0,0,0.25)",
                   gridcolor="rgba(128,128,128,0.15)", range=list(y_range) if y_range else None),
        xaxis=dict(gridcolor="rgba(128,128,128,0.10)"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def residual_figure(ra: RegionAnalysis) -> go.Figure:
    r = ra.residuals
    colours = [OBSERVED_COLOUR if abs(z) < 3 else SEVERITY_COLOUR["CRITICAL"] for z in r.z]
    fig = go.Figure(go.Bar(x=r.dates, y=r.residual, marker_color=colours, width=1000 * 3600 * 24 * 12,
                           customdata=r.z,
                           hovertemplate="unexplained %{y:.1f} mm (%{customdata:.1f}σ)<extra></extra>"))
    fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        yaxis=dict(title="unexplained movement (mm)", zeroline=True, zerolinecolor="rgba(0,0,0,0.4)",
                   gridcolor="rgba(128,128,128,0.15)"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def severity_badge(severity: str) -> str:
    return (f"<span style='color:{SEVERITY_COLOUR[severity]};font-size:1.05em'>{SEVERITY_MARK[severity]}</span> "
            f"<b>{severity}</b>")


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------
st.sidebar.title("KEYSTONE")
st.sidebar.caption("Workstream D - Decision")
source = st.sidebar.radio("Data source", ["Synthetic year", "Archive (SQLite)"])

truth_note = ""
if source == "Synthetic year":
    seed = st.sidebar.number_input("Seed (which synthetic year)", min_value=0, max_value=9999, value=1, step=1)
    region_options = ["none - seasonal movement only"] + [r.region_id for r in WALL_REGIONS]
    drift_choice = st.sidebar.selectbox("Region with structural drift", region_options, index=6)
    drift_region = None if drift_choice.startswith("none") else drift_choice
    drift_rate = st.sidebar.slider("Drift (mm per capture, from capture 5)", 1.0, 6.0, 3.5, 0.5,
                                   disabled=drift_region is None)
    full_history = [ChangeReport.from_json(j) for j in synthetic_history(int(seed), drift_region, float(drift_rate))]
    truth_note = (f"Ground truth of this synthetic year: **{drift_region}** drifts {drift_rate} mm per capture "
                  f"from capture 5 and never returns." if drift_region else
                  "Ground truth of this synthetic year: **no structural movement** - everything you see is weather.")
else:
    db_path = st.sidebar.text_input("Archive file", DEFAULT_DB)
    _, monuments = load_archive(db_path, None)
    monument = st.sidebar.selectbox("Monument", monuments) if monuments else None
    full_history, _ = load_archive(db_path, monument)

show_naive = st.sidebar.checkbox("Show the naive threshold (2 × lod)", value=True)

# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------
st.title("KEYSTONE - structural movement decisions")

if not full_history:
    st.info("The archive is empty. Switch the data source to **Synthetic year** and press "
            "**Save this history to the archive**, or run `python -m decision.store`.")
    st.stop()

n_reports = st.sidebar.slider("Replay: captures so far", 2, len(full_history) + 1, len(full_history) + 1,
                              help="Capture 1 is the reference. Move left to see what KEYSTONE decided earlier in the year.")
history = full_history[: n_reports - 1]
latest = history[-1]
analyses = analyse(history)

if source == "Synthetic year" and st.sidebar.button("Save this history to the archive"):
    with Archive(DEFAULT_DB) as archive:
        for k, report in enumerate(history):
            archive.save_report(report)
            archive.save_alerts([ra.alert for ra in analyse(history[: k + 1])])
        totals = archive.counts()
    st.sidebar.success(f"Saved. {DEFAULT_DB} now holds {totals['reports']} reports and {totals['alerts']} alerts.")

st.caption(
    f"Monument **{latest.monument_id}** · reference capture **{latest.earlier.capture_id}** "
    f"({latest.earlier.captured_on}) · latest capture **{latest.later.capture_id}** "
    f"({latest.later.captured_on}, {latest.later.temperature_c} °C, "
    f"{latest.later.rainfall_mm_7d} mm rain in 7 days) · {len(history) + 1} captures")
if truth_note:
    st.caption(truth_note + " KEYSTONE never sees this - it is shown so you can check the decisions.")

# ---- headline counts
counts = {s: sum(ra.decision.severity == s for ra in analyses) for s in SEVERITIES}
for column, severity in zip(st.columns(4), SEVERITIES):
    with column:
        st.markdown(severity_badge(severity), unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:2.2em;font-weight:600;line-height:1.1'>{counts[severity]}</div>"
                    f"<div style='color:gray;font-size:0.85em'>→ {ROUTES[severity].recipient}</div>",
                    unsafe_allow_html=True)

st.divider()
tab_regions, tab_alerts, tab_detail, tab_how = st.tabs(
    ["Regions", "Current alerts", "Region detail", "How it decides"])

# ---- all regions, small multiples on a shared layout
with tab_regions:
    st.markdown("Measured movement (blue) against what the weather explains (dashed, with the shaded "
                "seasonal band). **Inside the band is the season. Outside it, and staying outside, is the structure.**")
    st.caption("Blue line and dots: measured · dashed grey: expected from the weather · shaded: seasonal band (±2σ)"
               + (" · dotted red: where a naive 2 × lod threshold would alarm" if show_naive else "")
               + " · all six charts share one scale")
    lows = [min(min(ra.residuals.observed), min(e - 2 * s for e, s in zip(ra.residuals.expected, ra.residuals.sigma)))
            for ra in analyses]
    highs = [max(max(ra.residuals.observed), max(e + 2 * s for e, s in zip(ra.residuals.expected, ra.residuals.sigma)),
                 NAIVE_LOD_MULTIPLE * ra.residuals.latest_lod_mm if show_naive else 0.0) for ra in analyses]
    shared_range = (min(lows) - 1.5, max(highs) + 1.5)
    ordered = sorted(analyses, key=lambda ra: (ra.region_id.rsplit("_", 1)[-1] != "upper", ra.region_id))
    columns = st.columns(3)
    for i, ra in enumerate(ordered):
        with columns[i % 3]:
            st.markdown(f"**{ra.region_id}**{' · load-bearing' if ra.load_bearing else ''}<br>"
                        f"{severity_badge(ra.decision.severity)}", unsafe_allow_html=True)
            st.plotly_chart(region_figure(ra, show_naive, height=260, y_range=shared_range, legend=False),
                            key=f"grid-{ra.region_id}",
                            config={"displayModeBar": False})

# ---- alerts
with tab_alerts:
    ranked = sorted(analyses, key=lambda ra: -SEVERITIES.index(ra.decision.severity))
    table = pd.DataFrame([{
        "region": ra.region_id,
        "severity": f"{SEVERITY_MARK[ra.decision.severity]} {ra.decision.severity}",
        "recipient": ra.alert.recipient,
        "load-bearing": "yes" if ra.load_bearing else "no",
        "measured (mm)": round(ra.residuals.observed[-1], 1),
        "expected (mm)": round(ra.residuals.expected[-1], 1),
        "unexplained (mm)": round(ra.residuals.latest_residual_mm, 1),
        "captures outside band": ra.residuals.run_mild,
    } for ra in ranked])
    st.dataframe(table, hide_index=True)

    for ra in ranked:
        if ra.decision.severity == "INFORMATIONAL":
            continue
        with st.expander(f"{SEVERITY_MARK[ra.decision.severity]} {ra.decision.severity} · {ra.region_id} → {ra.alert.recipient}",
                         expanded=ra.decision.severity in ("ALERT", "CRITICAL")):
            st.write(ra.alert.reason)
            st.code(ra.alert.to_json(), language="json")
    if all(ra.decision.severity == "INFORMATIONAL" for ra in analyses):
        st.success("Every region is moving as the season predicts. All six decisions are recorded in the monument record.")

# ---- one region in depth
with tab_detail:
    chosen = st.selectbox("Region", [ra.region_id for ra in analyses],
                          index=max(range(len(analyses)),
                                    key=lambda i: SEVERITIES.index(analyses[i].decision.severity)))
    ra = next(a for a in analyses if a.region_id == chosen)
    st.markdown(f"{severity_badge(ra.decision.severity)} → {ra.alert.recipient}", unsafe_allow_html=True)
    st.plotly_chart(region_figure(ra, show_naive, height=380), key="detail-main", config={"displayModeBar": False})
    st.markdown("**Unexplained movement** - measured minus expected. Red bars are more than 3σ outside the band.")
    st.plotly_chart(residual_figure(ra), key="detail-residual", config={"displayModeBar": False})
    left, right = st.columns(2)
    with left:
        st.markdown("**Seasonal model for this region**")
        st.write(ra.fit.describe())
        st.write(f"Peers used: {', '.join(ra.fit.peers) or 'none'}")
    with right:
        st.markdown("**Persistence**")
        st.write(f"Captures in a row beyond 3σ: {ra.residuals.run_strong} · "
                 f"trend {ra.residuals.trend_mm_per_capture:+.1f} mm per capture · "
                 f"{'not returning' if ra.residuals.retained else 'returning towards the band'}")
        if ra.crack.significant:
            st.write(f"Crack widening {ra.crack.latest_width_mm:.1f} mm "
                     f"(uncertainty {ra.crack.latest_uncertainty_mm:.1f} mm)")
    with st.expander("Data table"):
        st.dataframe(pd.DataFrame({
            "date": ra.residuals.dates, "measured mm": ra.residuals.observed,
            "expected mm": ra.residuals.expected, "band 1σ mm": ra.residuals.sigma,
            "unexplained mm": ra.residuals.residual, "z": ra.residuals.z}).round(2), hide_index=True)

# ---- explanation
with tab_how:
    st.markdown("""
**1. Seasonal model.** For each region, KEYSTONE asks what the weather alone should have done to it:
`expected = a × Δtemperature + b × Δrainfall`. The coefficients are learned from the *other* regions at the same
height - never from the region being judged, so a failing region cannot hide its own drift inside "the season".

**2. Residual.** Measured minus expected. Seasonal movement leaves a residual that hovers around zero and changes
sign. Structural movement leaves one that grows and does not come back.

**3. Severity.** An ALERT needs the residual to be beyond 3σ on two captures running, in the same direction, not
returning, and larger than the region's own detection limit. One capture - however extreme - is at most a WATCH.
Load-bearing regions escalate faster, and crack widening from the crack-segmentation team counts as independent evidence.

**4. Routing.** Each severity goes to a different recipient, so the officer who took the photographs is not the
one told to go back:
""")
    st.table(pd.DataFrame([{"severity": f"{SEVERITY_MARK[r.severity]} {r.severity}", "recipient": r.recipient,
                            "recommended action": r.action} for r in ROUTES.values()]))
    st.caption("KEYSTONE recommends. It has no authority to close a monument - that decision rests with the custodian. "
               "The seasonal coefficients shown here come from synthetic data; a real monument needs about a year "
               "of captures to calibrate.")
