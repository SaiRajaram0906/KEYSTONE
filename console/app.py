"""KEYSTONE console: the whole chain on one screen.

    streamlit run console/app.py            (from the project root)

Capture check -> scaled model -> change in 3D -> cracks -> seasonal decision -> inspection order -> record.
Every number comes from the project's own code (console/engine.py). The wall is simulated with known
ground truth, so each stage can be checked against the truth. Real photos: the last tab.
"""
from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from console import engine as E  # noqa: E402
importlib.reload(E)
from console import monument as MO  # noqa: E402
from console import temple_fig as TF  # noqa: E402

st.set_page_config(page_title="KEYSTONE console", page_icon="🏛️", layout="wide")

INK, VERD, PLASTER, HAIR = "#1E2B33", "#2F7F79", "#ECEEE9", "#C5CBC3"
SEV_COLOUR = {"INFORMATIONAL": "#2f7f4f", "WATCH": "#b7791f", "ALERT": "#c4561d", "CRITICAL": "#b3261e"}
SEV_MARK = {"INFORMATIONAL": "●", "WATCH": "▲", "ALERT": "◆", "CRITICAL": "■"}
SEV_ORDER = {s: i for i, s in enumerate(["INFORMATIONAL", "WATCH", "ALERT", "CRITICAL"])}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500&display=swap');
html, body, .stApp, .stMarkdown, label, input, button, textarea {{ font-family:'IBM Plex Sans', system-ui, sans-serif; }}
[data-testid="stIconMaterial"], span[class*="material"] {{ font-family:'Material Symbols Rounded','Material Symbols Outlined' !important; }}
h1,h2,h3 {{ font-family:'Bricolage Grotesque','IBM Plex Sans',sans-serif; letter-spacing:-.015em; color:{INK}; }}
.block-container {{ max-width:1240px; padding-top:1.8rem; }}
header[data-testid="stHeader"] {{ background:transparent; }}
.ks-title {{ font:700 2.7rem/1.05 'Bricolage Grotesque',sans-serif; letter-spacing:-.03em; margin:0; color:{INK}; }}
.ks-sub {{ color:#4A5A63; margin:.3rem 0 .8rem 0; font-size:1.05rem; }}
.ks-sim {{ border-left:4px solid #b7791f; background:#E0E4DD; padding:.55rem .9rem; margin:.4rem 0 1rem 0; font-size:.92rem; }}
.ks-callout {{ border-left:4px solid {VERD}; background:#E0E4DD; padding:.7rem 1rem; margin:.5rem 0 1rem 0; border-radius:0 6px 6px 0; }}
.ks-callout.bad {{ border-left-color:#b3261e; }} .ks-callout.warn {{ border-left-color:#b7791f; }}
.ks-stage {{ border-top:2px solid {INK}; padding:.55rem 0 .3rem 0; min-height:6.4rem; }}
.ks-stage .n {{ font:700 .98rem 'Bricolage Grotesque',sans-serif; color:{INK}; }}
.ks-stage .v {{ font:700 1.35rem/1.15 'Bricolage Grotesque',sans-serif; color:{INK}; margin:.15rem 0; }}
.ks-stage .d {{ font-size:.82rem; color:#4A5A63; line-height:1.35; }}
.ks-row {{ display:flex; gap:.8rem; padding:.5rem 0; border-top:1px solid {HAIR}; align-items:baseline; }}
.stTabs [data-baseweb="tab-list"] {{ gap:1.3rem; border-bottom:1px solid {HAIR}; }}
.stTabs [data-baseweb="tab"] {{ font-family:'Bricolage Grotesque',sans-serif; font-weight:500; font-size:1.02rem; padding:.45rem 0; }}
.stTabs [aria-selected="true"] {{ color:{VERD}; }}
button:focus-visible, [role="tab"]:focus-visible {{ outline:3px solid {VERD} !important; outline-offset:2px; }}
</style>""", unsafe_allow_html=True)


def callout(text: str, kind: str = "") -> None:
    st.markdown(f'<div class="ks-callout {kind}">{text}</div>', unsafe_allow_html=True)


def badge(sev: str) -> str:
    return f"<span style='color:{SEV_COLOUR[sev]}'>{SEV_MARK[sev]}</span> <b>{sev}</b>"


# ------------------------------------------------------------------ cached runs
@st.cache_resource(show_spinner=False)
def c_capture(scenario, seed): return E.capture_check(scenario, seed)
@st.cache_resource(show_spinner=False)
def c_geometry(seed, region, mm): return E.run_geometry(seed, region, mm)
@st.cache_resource(show_spinner=False)
def c_year(seed, region, rate): return E.year_run(seed, region, rate)
@st.cache_resource(show_spinner=False)
def c_temple():
    tp = MO.build_temple(0)
    return tp, MO.run_year(tp, 0)
@st.cache_data(show_spinner=False, max_entries=3)
def c_temple_html(exag: float) -> str:
    return TF.animated_html(c_temple()[0], exag)
@st.cache_resource(show_spinner=False)
def c_record(seed, region, mm): return E.record_run(c_geometry(seed, region, mm))


# ----------------------------------------------------------------------- sidebar
DEFAULTS = dict(region="east_upper", mm=12.0, seed=0, w0=3, w1=7, mmpx=0.5,
                year_region="east_upper", rate=3.5, year_seed=1)
if "p" not in st.session_state:
    st.session_state["p"] = dict(DEFAULTS)
P = st.session_state["p"]

with st.sidebar:
    st.markdown("### Scenario")
    with st.form("scenario"):
        region = st.selectbox("Region that moves", E.REGION_IDS, index=E.REGION_IDS.index(P["region"]))
        mm = st.slider("Planted movement (mm outward)", 0.0, 40.0, float(P["mm"]), 1.0,
                       help="How far that region of the wall is pushed out between the two captures. 0 = nothing moved.")
        w0, w1 = st.slider("Crack width, before and after (pixels)", 1, 15, (int(P["w0"]), int(P["w1"])))
        mmpx = st.number_input("Millimetres per crack-photo pixel", 0.05, 5.0, float(P["mmpx"]), 0.05)
        st.markdown("**Year of captures**")
        year_region = st.selectbox("Region that drifts for good", ["(none)"] + E.REGION_IDS,
                                   index=(["(none)"] + E.REGION_IDS).index(P["year_region"]))
        rate = st.slider("Drift (mm per capture, from capture 5)", 0.5, 8.0, float(P["rate"]), 0.5)
        seed = st.number_input("Random seed", 0, 999, int(P["seed"]), 1)
        go_run = st.form_submit_button("Run the chain")
    if go_run:
        st.session_state["p"] = P = dict(region=region, mm=mm, seed=int(seed), w0=w0, w1=w1, mmpx=mmpx,
                                         year_region=year_region, rate=rate, year_seed=int(seed) + 1)
    if st.button("Reset to the demo scenario"):
        st.session_state["p"] = dict(DEFAULTS)
        st.rerun()

# ------------------------------------------------------------------------ header
st.markdown('<h1 class="ks-title">KEYSTONE</h1>'
            '<p class="ks-sub">A monument that measures itself, repeatedly, and raises its own alarm.</p>',
            unsafe_allow_html=True)
st.markdown('<div class="ks-sim"><b>Simulated wall.</b> A 6 m by 3 m masonry wall generated with known ground truth, so every '
            'stage below can be checked against the truth. The code is the real pipeline. Your own photos: last tab.</div>',
            unsafe_allow_html=True)

with st.spinner("Running capture check, scaling, alignment, change detection and decision..."):
    capture = c_capture("good", P["seed"])
    geo = c_geometry(P["seed"], P["region"], P["mm"])
    ma, mb = E.example_masks(P["w0"], P["w1"])
    crack = E.crack_run(ma, mb, P["region"], P["mmpx"])
    report, chain_alerts = E.chain_alerts(geo, crack)
    yr_region = None if P["year_region"] == "(none)" else P["year_region"]
    year = c_year(P["year_seed"], yr_region, P["rate"])
    rec = c_record(P["seed"], P["region"], P["mm"])

# ---------------------------------------------------------------------- figures
def change_figure(geo, exag: float, show_noise: bool, height: int = 560) -> go.Figure:
    d, core = geo.dist, geo.core
    sig = geo.valid & (np.abs(d) > geo.lod)
    quiet = geo.valid & ~sig
    cm = max(6.0, float(np.ceil(np.abs(d[sig]).max()))) if sig.any() else 6.0
    fig = go.Figure()
    if show_noise:
        fig.add_trace(go.Scatter3d(x=core[quiet, 0], y=d[quiet] * exag, z=core[quiet, 2], mode="markers",
                                   marker=dict(size=1.6, color="#b8c0b8"), name="within measurement noise",
                                   hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(
        x=core[sig, 0], y=d[sig] * exag, z=core[sig, 2], mode="markers", name="moved beyond noise",
        marker=dict(size=2.8, color=d[sig], colorscale="RdBu_r", cmin=-cm, cmax=cm,
                    colorbar=dict(title=dict(text="mm outward", side="right"), thickness=12, len=0.6)),
        customdata=d[sig], hovertemplate="%{customdata:+.1f} mm<extra></extra>"))
    med = {r.region_id: r.median_mm for r in geo.report.regions}
    for rid, med_mm in med.items():
        r = E.REGION_BY_ID[rid]
        (x0, _, z0), (x1, _, z1) = r.bounds_min, r.bounds_max
        fig.add_trace(go.Scatter3d(x=[x0, x1, x1, x0, x0], y=[0] * 5, z=[z0, z0, z1, z1, z0], mode="lines",
                                   line=dict(color=INK, width=2), hoverinfo="skip", showlegend=False))
        upper = z0 >= 1500
        fig.add_trace(go.Scatter3d(x=[(x0 + x1) / 2], y=[0], z=[z1 + 140 if upper else z0 - 140], mode="text",
                                   showlegend=False, hoverinfo="skip", text=[f"{rid}  {med_mm:+.1f} mm"],
                                   textposition="middle center", textfont=dict(size=13, color=INK)))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=11)),
        scene=dict(aspectmode="manual", aspectratio=dict(x=2.0, y=0.55, z=1.0),
                   camera=dict(eye=dict(x=0.55, y=-1.9, z=0.55)),
                   xaxis=dict(title="along the wall (mm)", showbackground=False, gridcolor=HAIR),
                   yaxis=dict(title=f"outward (x{exag:.0f})", showbackground=False, showticklabels=False, gridcolor=HAIR),
                   zaxis=dict(title="height (mm)", showbackground=False, gridcolor=HAIR)))
    return fig


def region_figure(ra, show_naive: bool, height: int = 260) -> go.Figure:
    r = ra.residuals
    up = [e + 2 * s for e, s in zip(r.expected, r.sigma)]
    lo = [e - 2 * s for e, s in zip(r.expected, r.sigma)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r.dates, y=up, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=r.dates, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(122,121,116,.2)", name="normal seasonal range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=r.dates, y=r.expected, mode="lines", name="expected from season",
                             line=dict(color="#7a7974", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=r.dates, y=r.observed, mode="lines+markers", name="measured",
                             line=dict(color="#2a78d6", width=2), marker=dict(size=7, color="#2a78d6", line=dict(width=2, color="white"))))
    if show_naive and r.latest_lod_mm:
        fig.add_hline(y=2 * r.latest_lod_mm, line=dict(color="#b3261e", width=1, dash="dot"),
                      annotation_text="fixed threshold (2 x noise)", annotation_position="top left",
                      annotation_font=dict(size=10))
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), showlegend=False, hovermode="x unified",
                      yaxis=dict(title="mm since reference", gridcolor="rgba(128,128,128,.15)"),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def section_figure(sec) -> go.Figure:
    pts = np.array(sec.polylines[0])
    x0, y0, x1, y1 = sec.extent_mm
    fig = go.Figure(go.Scattergl(x=pts[:, 0], y=pts[:, 1], mode="markers", marker=dict(size=2, color=INK)))
    names = {"plan": ("along the wall (mm)", "outward (mm)"), "elevation": ("along the wall (mm)", "height (mm)"),
             "section": ("outward (mm)", "height (mm)")}[sec.kind]
    yaxis = dict(title=names[1], gridcolor="#eee")
    if sec.kind == "plan":                      # a 6 m wall that is 0.4 m deep: stretch depth so the buttress is visible
        yaxis["range"] = [y0 - 100, y1 + 150]
    else:
        yaxis["scaleanchor"] = "x"
    fig.update_layout(height=300, margin=dict(l=8, r=8, t=8, b=8), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(title=names[0], gridcolor="#eee"), yaxis=yaxis)
    return fig


_all = st.tabs(["Whole chain", "Temple, one year", "Capture check", "Change in 3D", "Cracks",
                "Year of captures", "Orders and record", "Your photos"])
temple_tab = _all[1]
tabs = [_all[0]] + list(_all[2:])

# ------------------------------------------------------------------ temple, one year
with temple_tab:
    tp, tres = c_temple()
    st.markdown("### A temple, watched for a year")
    callout("<b>Simulated temple.</b> The stone temple, its cracks and its movement are made up, so the truth is known. "
            "Month 0 is the reference capture, then one capture a month. The decisions come from the project's real Decision code.")
    a, b, c1, c2 = st.columns([2, 2, 1.4, 1.6])
    month = a.slider("Month", 0, MO.N_MONTHS, MO.N_MONTHS, key="t_month")
    texag = b.slider("Exaggerate movement", 1, 150, 60, key="t_exag")
    tcracks = c1.checkbox("Show cracks", value=True, key="t_cr")
    tonly = c2.checkbox("Hide points inside the noise", value=False, key="t_only")
    stm = MO.month_state(tp, month, 0)
    st.plotly_chart(TF.temple_figure(tp, stm, float(texag), tonly, tcracks, 640), width="stretch", key="temple3d")
    st.caption("Grey ghost = the temple at month 0. Colour = how far each point moved (blue one way, red the other); "
               "grey points moved less than the noise floor. Dark red lines are cracks, labelled with their width. "
               f"Movement is drawn x{texag}; real movement is at most a few centimetres.")

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"#### What KEYSTONE says at month {month}")
        rows = []
        for rid in tp.region_ids:
            if month == 0:
                rows.append({"region": rid, "movement (mm)": 0.0, "KEYSTONE says": "reference capture"})
                continue
            sev = tres.severity[rid][month - 1]
            rows.append({"region": rid, "movement (mm)": round(tres.series[rid][month - 1], 1),
                         "noise floor (mm)": round(tp.lod[rid], 1),
                         "KEYSTONE says": f"{SEV_MARK[sev]} {sev}"})
        st.dataframe(rows, width="stretch", hide_index=True, height=460)
    with right:
        st.markdown("#### Crack widths")
        cf = go.Figure()
        for cid, ws in tres.crack_series.items():
            cf.add_trace(go.Scatter(x=list(range(MO.N_MONTHS + 1)), y=ws, mode="lines+markers", name=cid))
        cf.add_vline(x=month, line_color=INK, line_dash="dot")
        cf.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                         xaxis_title="month", yaxis_title="width (mm)", legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(cf, width="stretch", key="t_cracks")

    st.markdown("#### Did it catch the real problems, and when?")
    planted = {"tower_upper": "leans east from month 2", "col_b_lower": "settles from month 4",
               "lintel_upper": "sags with col_b", "plinth_lower": "sinks near col_b",
               "porch_roof_upper": "sags slightly with col_b"}
    cmp_rows = []
    for rid in tp.region_ids:
        fa, nf = tres.first_alert[rid], tres.naive_first[rid]
        cmp_rows.append({"region": rid, "planted": planted.get(rid, "nothing"),
                         "KEYSTONE first ALERT": f"month {fa}" if fa else "never",
                         "fixed 2 x noise threshold first fires": f"month {nf}" if nf else "never",
                         "KEYSTONE at month 12": tres.severity[rid][-1]})
    st.dataframe(cmp_rows, width="stretch", hide_index=True)
    struct = set(planted)
    false_k = [r for r in tp.region_ids if tres.first_alert[r] and r not in struct]
    false_n = [r for r in tp.region_ids if tres.naive_first[r] and r not in struct]
    msg = []
    if tres.first_alert["tower_upper"] and tres.naive_first["tower_upper"]:
        msg.append(f"The tower lean: KEYSTONE alerts at month {tres.first_alert['tower_upper']}, "
                   f"a fixed threshold at month {tres.naive_first['tower_upper']}.")
    miss_n = [r for r in ("lintel_upper", "plinth_lower") if tres.first_alert[r] and not tres.naive_first[r]]
    if miss_n:
        msg.append("The fixed threshold never fires on " + " and ".join(miss_n) +
                   " (their movement stays under it); KEYSTONE alerts because the persistent trend and crack widening count as evidence.")
    if false_k:
        msg.append("<b>Honest finding:</b> KEYSTONE raised a false ALERT on " + ", ".join(
            f"{r} (month {tres.first_alert[r]}, final {tres.severity[r][-1]})" for r in false_k) +
            ". The settling column is one of only a few peers at that height, so early on it drags the seasonal "
            "estimate (a year starting in winter makes drift and warming look alike). It cleared itself later.")
    if false_n:
        msg.append("The fixed threshold gave a false alarm on " + ", ".join(false_n) + ".")
    callout(" ".join(msg), "warn")

    ridsel = st.selectbox("Look at one region over the year", tp.region_ids, index=tp.region_ids.index("tower_upper"),
                          key="t_region")
    ra_last = next(x for x in tres.analyses[-1] if x.region_id == ridsel)
    st.plotly_chart(region_figure(ra_last, True, 300), width="stretch", key="t_regionfig")

    html = c_temple_html(float(texag))
    st.download_button("Download the whole year as an animated 3D viewer (one HTML file, about 15 MB)", html,
                       file_name="keystone_temple_year.html", mime="text/html")

# ------------------------------------------------------------------ whole chain
with tabs[0]:
    worst = max(year.analyses, key=lambda a: SEV_ORDER[a.alert.severity])
    moved = geo.report.regions
    planted = next((r for r in moved if r.region_id == P["region"]), None)
    sc = max(abs(geo.scale_found[k] / geo.scale_true[k] - 1) for k in geo.scale_found) * 100
    stages = [
        ("Capture check", capture.qc.verdict.replace("_", " ").title(), f"{capture.qc.coverage_fraction:.0%} of the face photographed, 7 checks"),
        ("Scaled model", f"{sc:.2f}% scale error", f"markers fix the units; registration error {geo.registration_rmse_mm:.2f} mm"),
        ("Change found", f"{planted.median_mm:+.1f} mm" if planted else "none",
         f"{P['region']}: planted {P['mm']:.0f} mm, noise floor {planted.lod_mm:.1f} mm" if planted else ""),
        ("Crack", f"{crack.delta.d_max_width:+.1f} mm", f"widest point, {crack.delta.earlier.max_width:.1f} to {crack.delta.later.max_width:.1f} mm"),
        ("Decision", worst.alert.severity.title(), f"{worst.region_id} over a year of captures"),
        ("Record", f"{rec.stats.n_entries} items, {'verified' if rec.verified else 'FAILED'}", "SHA-256 checked, reference kept in full"),
    ]
    cols = st.columns(6, gap="medium")
    for c, (n, v, d) in zip(cols, stages):
        c.markdown(f'<div class="ks-stage"><div class="n">{n}</div><div class="v">{v}</div><div class="d">{d}</div></div>',
                   unsafe_allow_html=True)
    st.plotly_chart(change_figure(geo, 40.0, True, 520), width="stretch", key="chain3d")
    st.caption("Each dot is a measured point on the wall. Grey dots moved less than the measurement noise. Coloured dots "
               "moved further than their own noise level, red outward and blue inward. Outward movement is exaggerated 40 times.")
    st.markdown("**What the system decided on this one comparison** (one report, so no seasonal history yet)")
    for a in sorted(chain_alerts, key=lambda a: -SEV_ORDER[a.severity]):
        st.markdown(f'<div class="ks-row"><div style="width:11rem">{a.region_id}</div><div style="width:10rem">{badge(a.severity)}</div>'
                    f'<div style="flex:1;font-size:.9rem">{a.reason}</div></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ capture check
with tabs[1]:
    st.markdown("### Is this capture good enough to reconstruct?")
    st.write("Before any 3D work, the gate checks the photos on site. Pick a kind of bad capture and see what the operator is told.")
    scen = st.selectbox("Capture session", E.CAPTURE_SCENARIOS, index=0,
                        format_func=lambda s: {"good": "good capture", "too_few": "too few photos", "blurry": "blurry photos",
                                               "dark_gap": "one area never photographed", "markers_missing": "scale markers missing",
                                               "overexposed": "overexposed photos", "thin_overlap": "photos barely overlap"}[s])
    cc = c_capture(scen, P["seed"])
    qc = cc.qc
    kind = {"PASS": "", "PASS_WITH_GAPS": "warn", "RESHOOT": "bad"}[qc.verdict]
    callout(f"<b>{qc.verdict.replace('_', ' ').title()}.</b> {qc.coverage_fraction:.0%} of the face is photographed."
            + (" Reconstruction is blocked until the fixes below are done." if qc.verdict == "RESHOOT" else ""), kind)
    l, r = st.columns([1.2, 1], gap="large")
    with l:
        for ch in qc.checks:
            st.markdown(f'<div class="ks-row"><div style="width:1.4rem">{"✔" if ch.passed else "✖"}</div>'
                        f'<div style="width:11rem"><b>{ch.name.replace("_", " ")}</b></div><div style="flex:1;font-size:.9rem">{ch.detail}</div></div>',
                        unsafe_allow_html=True)
        if qc.reshoot_guidance:
            st.markdown("**What to do on site**")
            for g in qc.reshoot_guidance:
                st.markdown(f"- {g}")
    with r:
        fig = go.Figure(go.Heatmap(z=cc.grid, colorscale=[[0, "#b3261e"], [1e-6, "#E0E4DD"], [1, VERD]], zmin=0, zmax=max(3, int(cc.grid.max())),
                                   colorbar=dict(title="photos", thickness=10), x0=0, dx=100, y0=0, dy=100))
        fig.update_layout(height=300, margin=dict(l=8, r=8, t=24, b=8), title=dict(text="How many photos see each part of the wall", font=dict(size=13)),
                          xaxis=dict(title="along the wall (mm)"), yaxis=dict(title="height (mm)", scaleanchor="x", range=[0, 3000]),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch", key="cov")
        st.caption(f"Truth for this simulated session: {'must be re-shot' if cc.truth.should_reshoot else 'usable'}. "
                   f"The gate said {qc.verdict.replace('_', ' ').lower()}.")

# ------------------------------------------------------------------ change in 3D
with tabs[2]:
    st.markdown("### Where did the wall move?")
    a, b = st.columns([1, 1])
    exag = a.slider("Exaggerate outward movement", 5, 120, 40, 5)
    noise = b.checkbox("Show points within measurement noise", value=True)
    st.plotly_chart(change_figure(geo, float(exag), noise, 600), width="stretch", key="detail3d")
    rows = []
    for rc in geo.report.regions:
        truth = geo.truth_mm[rc.region_id]
        rows.append({"region": rc.region_id, "measured (mm)": round(rc.median_mm, 1), "true (mm)": truth,
                     "error (mm)": round(rc.median_mm - truth, 1), "noise floor (mm)": round(rc.lod_mm, 1),
                     "points beyond noise": f"{rc.significant_fraction:.0%}", "load-bearing": "yes" if rc.load_bearing else ""})
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(f"The two captures were taken in different, unknown units and orientations. Surveyed markers fixed the scale "
               f"(found {geo.scale_found['C001']:.3f} mm per unit, truth {geo.scale_true['C001']:.3f}), then ICP aligned the stable parts of the wall.")
    viewer = change_figure(geo, float(exag), noise, 720).to_html(include_plotlyjs="cdn", full_html=True)
    st.download_button("Download this as a web viewer (one HTML file)", viewer, file_name="keystone_change_viewer.html", mime="text/html")

# ------------------------------------------------------------------ cracks
with tabs[3]:
    st.markdown("### Did the crack get wider?")
    st.write(f"Crack masks from the crack model, compared in millimetres. The widening is attached to **{P['region']}** "
             f"in the change report and travels to Decision with the movement.")
    c1, c2, c3 = st.columns(3, gap="large")
    for col, m, label in ((c1, crack.mask_a, "Earlier"), (c2, crack.mask_b, "Later")):
        col.image((255 - m * 255).astype(np.uint8), caption=label, clamp=True, width="stretch")
    d = crack.delta
    with c3:
        floor = P["mmpx"]
        st.metric("Widest point change", f"{d.d_max_width:+.2f} mm")
        st.metric("Average width change", f"{d.d_mean_width:+.2f} mm")
        st.metric("New crack area", f"{d.new_crack_area:,.0f} mm²")
    callout(f"Truth: the crack was drawn {P['w0']} px then {P['w1']} px wide, which is {(P['w1'] - P['w0']) * P['mmpx']:+.2f} mm at "
            f"{P['mmpx']} mm per pixel. Width is good to about 1 px ({floor:.2f} mm), so changes smaller than that are reported as noise.")
    st.caption("For real photos, use the crack app: `cd crack-seg` then `streamlit run app.py` (needs the trained weights).")

# ------------------------------------------------------------------ year
with tabs[4]:
    st.markdown("### A year of captures: heat or structure?")
    st.write("Walls move with the seasons. A fixed threshold shouts every summer. KEYSTONE learns the normal seasonal range from "
             "the wall's own history, so it only escalates movement that the season cannot explain.")
    show_naive = st.checkbox("Show the fixed threshold a simple system would use", value=True)
    ordered = sorted(year.analyses, key=lambda a: (-SEV_ORDER[a.alert.severity], a.region_id))
    for i in range(0, len(ordered), 3):
        cols = st.columns(3, gap="large")
        for col, ra in zip(cols, ordered[i:i + 3]):
            with col:
                st.markdown(f"**{ra.region_id}**{' · load-bearing' if ra.load_bearing else ''}<br>{badge(ra.alert.severity)}", unsafe_allow_html=True)
                st.plotly_chart(region_figure(ra, show_naive), width="stretch", key=f"yr_{ra.region_id}")
    naive = [ra.region_id for ra in year.analyses if ra.residuals.latest_lod_mm and max(ra.residuals.observed) > 2 * ra.residuals.latest_lod_mm]
    flagged = [ra.region_id for ra in year.analyses if SEV_ORDER[ra.alert.severity] >= SEV_ORDER["ALERT"]]
    truth_r = year.truth.drift_region
    callout(f"<b>Result.</b> The fixed threshold would have crossed in {len(naive)} region(s): {', '.join(naive) or 'none'}. "
            f"KEYSTONE escalated {len(flagged)}: {', '.join(flagged) or 'none'}. "
            f"The region that really drifted in this simulation: <b>{truth_r or 'none'}</b>.")

# ------------------------------------------------------------------ orders and record
with tabs[5]:
    st.markdown("### From decision to action")
    pool = year.analyses
    default = max(range(len(pool)), key=lambda i: SEV_ORDER[pool[i].alert.severity])
    pick = st.selectbox("Alert", range(len(pool)), index=default,
                        format_func=lambda i: f"{pool[i].alert.region_id}: {pool[i].alert.severity}")
    ra = pool[pick]
    last = year.reports[-1]
    rc = next((r for r in last.regions if r.region_id == ra.region_id), None)
    order_html = E.inspection_order_html(ra.alert, ra, rc)
    l, r = st.columns([1.1, 1], gap="large")
    with l:
        if hasattr(st, "iframe"):
            st.iframe(order_html, height=860)
        else:
            st.components.v1.html(order_html, height=860, scrolling=True)
    with r:
        st.download_button("Download inspection order (open, then print to PDF)", order_html,
                           file_name=f"inspection_order_{E.order_number(ra.alert)}.html", mime="text/html")
        st.markdown("**Who gets told what**")
        for sev, route in E.ROUTES.items():
            st.markdown(f'<div class="ks-row"><div style="width:9rem">{badge(sev)}</div>'
                        f'<div style="flex:1;font-size:.88rem"><b>{route.recipient}</b>'
                        f'{" · interrupts a person" if route.notify else ""}</div></div>', unsafe_allow_html=True)
        callout("KEYSTONE recommends. It has no authority to close a monument; the custodian decides.")

    st.markdown("### The permanent record")
    m1, m2, m3 = st.columns(3)
    m1.metric("Archived items", rec.stats.n_entries)
    m2.metric("Integrity check", "verified" if rec.verified else "FAILED")
    m3.metric("Kept in full (reference)", f"{rec.stats.reference_bytes / 1e3:.0f} kB")
    st.caption("The first capture is the reference and is never thinned. Older later captures can be reduced to save space, "
               "so the record that matters survives.")
    st.markdown("**Measured drawings cut from the 3D data** (a wall with a 400 mm buttress, so plan and elevation differ)")
    sc = st.columns(3, gap="large")
    for col, key, title in zip(sc, ("plan", "elevation", "section"), ("Plan at 1.5 m height (depth stretched)", "Elevation (the buttress stands proud of this cut)", "Section through the buttress")):
        with col:
            st.markdown(f"**{title}**")
            st.plotly_chart(section_figure(rec.sections[key]), width="stretch", key=f"sec_{key}")
            st.download_button(f"Download {key} (SVG)", E.section_svg(rec.sections[key]), file_name=f"keystone_{key}.svg",
                               mime="image/svg+xml", key=f"dl_{key}")

# ------------------------------------------------------------------ your photos
with tabs[6]:
    st.markdown("### Your own photos")
    st.write("Upload 20 to 40 overlapping photos of one textured wall. The app finds where each photo was taken from and "
             "builds a sparse 3D cloud (hundreds to a few thousand points, a skeleton of the scene).")
    callout("<b>What you get, and what you don't.</b> This is a sparse cloud in arbitrary units, not a measurement. To get millimetres "
            "you need printed markers of known size in the photos and the Geometry stage. Dense surfaces need a GPU.", "warn")
    ups = st.file_uploader("Photos (JPG or PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        tries = st.slider("Reconstruction attempts (the best is kept)", 1, 5, 3)
    with col_t2:
        max_size = st.select_slider(
            "Max image dimension (px)",
            options=[1200, 1600, 2000, 2400, 3200],
            value=2000,
            help="Limits image dimension during SIFT feature extraction. 2000px is optimal on CPU to prevent RAM exhaustion.",
        )
    if ups and st.button("Build 3D cloud from these photos"):
        if len(ups) < 6:
            st.markdown('<div class="ks-callout bad">Upload at least 6 photos. 20 to 40 works best.</div>', unsafe_allow_html=True)
        else:
            work = Path(tempfile.mkdtemp(prefix="keystone_photos_"))
            (work / "in").mkdir()
            for u in ups:
                (work / "in" / u.name).write_bytes(u.getvalue())
            with st.spinner("Matching photos and solving camera positions. This takes a few minutes..."):
                try:
                    pts, log = E.reconstruct_photos(str(work / "in"), str(work / "out"), tries, max_size)
                except TypeError:
                    pts, log = E.reconstruct_photos(str(work / "in"), str(work / "out"), tries)
            if pts is None:
                st.markdown('<div class="ks-callout bad">The reconstruction failed. If the log mentions pycolmap, run '
                            '<code>pip install pycolmap</code> and try again. Plain painted walls often fail: they need texture.</div>',
                            unsafe_allow_html=True)
            else:
                st.success(f"{len(pts):,} 3D points")
                lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
                keep = np.all((pts >= lo) & (pts <= hi), axis=1)
                q = pts[keep]
                fig = go.Figure(go.Scatter3d(x=q[:, 0], y=q[:, 1], z=q[:, 2], mode="markers",
                                             marker=dict(size=2, color=q[:, 2], colorscale="Viridis")))
                fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0), scene=dict(aspectmode="data"))
                st.plotly_chart(fig, width="stretch", key="photos3d")
            with st.container():
                st.text_area("Log", log, height=200)
