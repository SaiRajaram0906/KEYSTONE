"""Figures for the simulated temple (plotly only, no Streamlit)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from console import monument as M

INK, HAIR = "#1E2B33", "#C5CBC3"
CMAX = 25.0                      # colour scale limit, mm
CRACK = "#7a0d0d"


def _traces(tp: M.Temple, st: M.MonthState, exag: float, only_sig: bool, show_cracks: bool, ghost_idx):
    P = tp.points
    shown = P + exag * st.disp
    ghost = go.Scatter3d(x=P[ghost_idx, 0], y=P[ghost_idx, 1], z=P[ghost_idx, 2], mode="markers",
                         marker=dict(size=1.2, color="#9aa39a", opacity=0.22), name="reference (month 0)",
                         hoverinfo="skip")
    sig = st.significant
    quiet = ~sig
    q = go.Scatter3d(x=shown[quiet, 0], y=shown[quiet, 1], z=shown[quiet, 2], mode="markers",
                     marker=dict(size=1.5, color="#c9cfc9"), name="within measurement noise",
                     hoverinfo="skip", visible=True if not only_sig else "legendonly")
    s = go.Scatter3d(x=shown[sig, 0], y=shown[sig, 1], z=shown[sig, 2], mode="markers", name="moved beyond noise",
                     marker=dict(size=2.2, color=st.measured[sig], colorscale="RdBu_r", cmin=-CMAX, cmax=CMAX,
                                 colorbar=dict(title=dict(text="mm (signed)", side="right"), thickness=12, len=0.55)),
                     customdata=st.measured[sig], hovertemplate="%{customdata:+.1f} mm<extra></extra>")
    out = [ghost, q, s]
    labels_x, labels_y, labels_z, labels_t = [], [], [], []
    for c in st.cracks:
        poly = c["poly"] + exag * c["disp"]
        w = c["width"]
        vis = show_cracks
        out.append(go.Scatter3d(
            x=poly[:, 0], y=poly[:, 1] - 40, z=poly[:, 2], mode="lines",
            line=dict(color=CRACK, width=min(14, 3 + 2.2 * w)), name=c["id"], showlegend=False,
            hovertemplate=f"{c['id']}: {w:.1f} mm wide<extra></extra>", visible=vis))
        labels_x.append(poly[-1, 0]); labels_y.append(poly[-1, 1] - 40); labels_z.append(poly[-1, 2] + 160)
        labels_t.append(f"{w:.1f} mm")
    out.append(go.Scatter3d(x=labels_x, y=labels_y, z=labels_z, mode="text", text=labels_t, showlegend=False,
                            hoverinfo="skip", textfont=dict(size=12, color=CRACK), visible=show_cracks))
    return out


def temple_figure(tp: M.Temple, st: M.MonthState, exag: float, only_sig: bool = False,
                  show_cracks: bool = True, height: int = 640) -> go.Figure:
    rng = np.random.default_rng(3)
    ghost_idx = rng.choice(len(tp.points), size=min(4000, len(tp.points)), replace=False)
    fig = go.Figure(_traces(tp, st, exag, only_sig, show_cracks, ghost_idx))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.03, x=0, font=dict(size=11)),
        scene=dict(aspectmode="data", camera=dict(eye=dict(x=-1.15, y=-1.45, z=0.6)),
                   xaxis=dict(title="east-west (mm)", showbackground=False, gridcolor=HAIR),
                   yaxis=dict(title="depth (mm)", showbackground=False, gridcolor=HAIR),
                   zaxis=dict(title="height (mm)", showbackground=False, gridcolor=HAIR)))
    return fig


def animated_html(tp: M.Temple, exag: float, seed: int = 0) -> str:
    """One self-contained file: play the whole year, scrub the months."""
    rng = np.random.default_rng(3)
    ghost_idx = rng.choice(len(tp.points), size=min(2500, len(tp.points)), replace=False)
    frames = []
    for m in range(M.N_MONTHS + 1):
        st = M.month_state(tp, m, seed)
        frames.append(go.Frame(data=_traces(tp, st, exag, False, True, ghost_idx), name=str(m)))
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        height=760, margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text="SIMULATED temple: one year of captures (movement exaggerated x%d)" % exag, x=0.01),
        scene=dict(aspectmode="data", camera=dict(eye=dict(x=-1.15, y=-1.45, z=0.6))),
        updatemenus=[dict(type="buttons", x=0.01, y=0.02, buttons=[
            dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=900, redraw=True), fromcurrent=True)]),
            dict(label="Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
        sliders=[dict(active=0, x=0.12, len=0.85, currentvalue=dict(prefix="Month "),
                      steps=[dict(label=str(m), method="animate",
                                  args=[[str(m)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for m in range(M.N_MONTHS + 1)])])
    return fig.to_html(include_plotlyjs="cdn", full_html=True)
