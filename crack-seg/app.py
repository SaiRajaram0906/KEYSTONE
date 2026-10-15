"""Crack check: a browser front end for the crack-seg project.

    streamlit run app.py

Three jobs:
  Measure  one photo (or one mask)         -> crack length and width in mm
  Compare  two aligned captures of a wall  -> did the crack widen, by how much
  Results  the numbers the project reached -> and what they do not prove

The app calls the same functions the command line does (keystone/interface.py),
so a number here matches `python -m src.predict`.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from frontend.style import CSS, callout, plain_row, readout  # noqa: E402
from keystone.interface import compare_captures, measure_mask  # noqa: E402

st.set_page_config(page_title="Crack check", page_icon="🧱", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

CRACK_RGB = (229, 57, 31)      # detected crack
NEW_RGB = (255, 196, 0)        # crack pixels that appeared between captures
GONE_RGB = (30, 43, 51)        # crack pixels present before, gone now


# --------------------------------------------------------------------- helpers
def find_checkpoints() -> list[Path]:
    runs = ROOT / "runs"
    found = sorted(runs.glob("*/best.pt")) if runs.exists() else []
    # Best model first, if it is there.
    found.sort(key=lambda p: (p.parent.name != "loss_focal_tversky", p.parent.name))
    return found


@st.cache_resource(show_spinner="Loading model")
def load_segmenter(path: str, mtime: float, device: str):
    from keystone.interface import CrackSegmenter
    return CrackSegmenter(path, device=None if device == "auto" else device)


def decode_image(data: bytes) -> np.ndarray | None:
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return None if arr is None else cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def decode_mask(data: bytes) -> np.ndarray | None:
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    return None if arr is None else (arr > 127).astype(np.uint8)


def tint(image: np.ndarray, mask: np.ndarray, rgb=CRACK_RGB, alpha=0.85) -> np.ndarray:
    out = image.copy()
    m = mask > 0
    out[m] = (np.array(rgb) * alpha + out[m] * (1 - alpha)).astype(np.uint8)
    return out


def widen(mask: np.ndarray, px: int) -> np.ndarray:
    """Draw masks a little thicker for display only; measurements never use this."""
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, k)


def png_bytes(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def example_masks() -> tuple[np.ndarray, np.ndarray]:
    def one(w):
        m = np.zeros((400, 600), np.uint8)
        cv2.line(m, (40, 60), (560, 340), 1, thickness=w)
        return m
    return one(3), one(7)


def grey_canvas(mask: np.ndarray) -> np.ndarray:
    return np.full((*mask.shape, 3), 236, np.uint8)


# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Settings")
    mm_per_px = st.number_input(
        "Millimetres per pixel", min_value=0.0, value=0.5, step=0.05, format="%.3f",
        help="How much wall one pixel covers. Get it from a ruler or printed marker in the photo. "
             "0 keeps every result in pixels.")
    min_comp = st.number_input(
        "Ignore specks smaller than (pixels)", min_value=0, value=20, step=5,
        help="Connected blobs below this size are dropped before measuring.")

    st.markdown("### Model")
    checkpoints = find_checkpoints()
    options = [str(p.relative_to(ROOT)) for p in checkpoints] + ["Other path…"]
    choice = st.selectbox("Checkpoint", options, index=0,
                          help="Trained weights (best.pt). Not included in this zip, see README.")
    ckpt_path: Path | None = None
    if choice == "Other path…":
        typed = st.text_input("Path to best.pt", placeholder=r"C:\models\best.pt")
        ckpt_path = Path(typed) if typed else None
    elif checkpoints:
        ckpt_path = ROOT / choice
    device = st.selectbox("Run on", ["auto", "cpu", "cuda"], help="auto uses the GPU if there is one.")
    thr_override = st.checkbox("Set the threshold myself")
    thr_value = st.slider("Threshold", 0.05, 0.99, 0.50, 0.01) if thr_override else None

    if ckpt_path and ckpt_path.exists():
        st.success(f"Model ready: {ckpt_path.parent.name}")
    elif not checkpoints:
        st.info("No model found in runs/. You can still measure masks and compare masks.")
    else:
        st.warning("That path does not exist.")

units = "mm" if mm_per_px else "px"
scale = mm_per_px or None
floor = mm_per_px if mm_per_px else 1.0


def get_mask(image_bytes: bytes, is_mask: bool) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Return (rgb for display, binary mask, error message)."""
    if is_mask:
        m = decode_mask(image_bytes)
        if m is None:
            return None, None, "That file is not an image. Upload a PNG or JPG mask."
        return grey_canvas(m), m, None
    rgb = decode_image(image_bytes)
    if rgb is None:
        return None, None, "That file is not an image. Upload a JPG or PNG photo."
    if not (ckpt_path and ckpt_path.exists()):
        return rgb, None, ("No model is loaded, so a photo cannot be segmented yet. Put best.pt in "
                           "runs/loss_focal_tversky/ (see README), or tick 'already a mask'.")
    seg = load_segmenter(str(ckpt_path), ckpt_path.stat().st_mtime, device)
    if thr_value is not None:
        seg.threshold = float(thr_value)
    return rgb, seg.segment(rgb), None


# --------------------------------------------------------------------- header
st.markdown('<h1 class="cs-title">Crack check</h1>'
            '<p class="cs-sub">Find cracks in a wall photo and measure how long and how wide they are. '
            'Compare two photos of the same wall to see whether a crack has grown.</p>',
            unsafe_allow_html=True)

tab_measure, tab_compare, tab_results, tab_limits = st.tabs(
    ["Measure one photo", "Compare two photos", "Results", "What it can't do"])

# --------------------------------------------------------------------- measure
with tab_measure:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        up = st.file_uploader("Wall photo", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], key="m_up")
        as_mask = st.checkbox("This file is already a crack mask (white = crack)", key="m_mask")
    if not up:
        with left:
            st.markdown(callout("<b>Start here.</b> Upload one photo of a wall. The cracks are outlined in red "
                                "and measured. Set millimetres per pixel in the sidebar first, or results stay in pixels."),
                        unsafe_allow_html=True)
    else:
        rgb, mask, err = get_mask(up.getvalue(), as_mask)
        with left:
            if rgb is not None:
                st.image(rgb if mask is None else tint(rgb, widen(mask, 1)), width="stretch",
                         caption="Original" if mask is None else "Cracks found (drawn slightly thicker so you can see them)")
        with right:
            if err:
                st.markdown(callout(err, "warn"), unsafe_allow_html=True)
            elif mask is not None:
                s = measure_mask(mask, scale, int(min_comp))
                if s.n_components == 0:
                    st.markdown(callout("<b>No cracks found.</b> Either the wall is clean or the model missed them. "
                                        "Try lowering the threshold in the sidebar."), unsafe_allow_html=True)
                else:
                    st.markdown(readout("Average width", s.mean_width, units, floor,
                                        f"Measurements can be off by about 1 px, shown as the grey band ({floor:.2f} {units})."),
                                unsafe_allow_html=True)
                    st.markdown(readout("Widest point", s.max_width, units, floor), unsafe_allow_html=True)
                    st.markdown(plain_row("Cracks", f"{s.n_components}")
                                + plain_row("Total length", f"{s.total_length:,.1f} {units}")
                                + plain_row("Width, 95th percentile", f"{s.p95_width:.2f} {units}")
                                + plain_row("Wall area covered", f"{s.coverage_fraction:.2%}"),
                                unsafe_allow_html=True)
                    if s.mean_width and mm_per_px and s.mean_width / mm_per_px < 3:
                        st.markdown(callout("<b>Thin crack.</b> Under about 3 px wide, this model reads cracks about 1 px too "
                                            "wide and misses some length. Treat the width as an upper estimate.", "warn"),
                                    unsafe_allow_html=True)
                    st.download_button("Save marked-up photo (PNG)", png_bytes(tint(rgb, widen(mask, 1))),
                                       file_name="crack_overlay.png", mime="image/png")
                    st.download_button("Save measurements (JSON)",
                                       json.dumps({"units": units, "mm_per_px": mm_per_px, **s.as_dict()}, indent=2),
                                       file_name="crack_measurements.json", mime="application/json")

# --------------------------------------------------------------------- compare
with tab_compare:
    st.markdown(callout("<b>Before you compare.</b> The two photos must show the same wall from the same spot, at the "
                        "same size, and be lined up. The app does not align them; it only checks that the sizes match."),
                unsafe_allow_html=True)
    c_mask = st.checkbox("Both files are already crack masks (white = crack)", key="c_mask")
    a_col, b_col = st.columns(2, gap="large")
    with a_col:
        up_a = st.file_uploader("Earlier", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], key="c_a")
    with b_col:
        up_b = st.file_uploader("Later", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], key="c_b")
    use_example = st.button("Try an example (a drawn crack, 3 px then 7 px wide)")
    if use_example:
        st.session_state["example"] = True
    if up_a or up_b:
        st.session_state["example"] = False

    rgb_a = rgb_b = m_a = m_b = None
    err = None
    if st.session_state.get("example"):
        m_a, m_b = example_masks()
        rgb_a, rgb_b = grey_canvas(m_a), grey_canvas(m_b)
        mm_eff = 0.5
        st.caption("Example uses 0.5 mm per pixel, whatever the sidebar says. The crack is drawn at 3 px and 7 px wide, "
                   "so the true change is +2.0 mm.")
    elif up_a and up_b:
        rgb_a, m_a, err_a = get_mask(up_a.getvalue(), c_mask)
        rgb_b, m_b, err_b = get_mask(up_b.getvalue(), c_mask)
        err = err_a or err_b
        mm_eff = mm_per_px
    else:
        mm_eff = mm_per_px

    if err:
        st.markdown(callout(err, "warn"), unsafe_allow_html=True)
    elif m_a is not None and m_b is not None:
        if m_a.shape != m_b.shape:
            st.markdown(callout(f"<b>The photos are different sizes</b> ({m_a.shape[1]}×{m_a.shape[0]} and "
                                f"{m_b.shape[1]}×{m_b.shape[0]}). Crop or resize them to match, then upload again.", "bad"),
                        unsafe_allow_html=True)
        else:
            d = compare_captures(m_a, m_b, mm_per_px=mm_eff or None, min_component_px=int(min_comp))
            u = d.units
            fl = mm_eff if mm_eff else 1.0
            dm, dx = d.d_mean_width, d.d_max_width
            if d.earlier.n_components == 0 and d.later.n_components == 0:
                verdict = callout("<b>No cracks in either photo.</b>")
            elif max(abs(dm), abs(dx)) <= fl:
                verdict = callout(f"<b>No change you can trust.</b> Widths moved by at most {max(abs(dm), abs(dx)):.2f} {u}, "
                                  f"which is within the 1 px measurement step ({fl:.2f} {u}).")
            elif dx > fl or dm > fl:
                verdict = callout(f"<b>The crack got wider.</b> Widest point {dx:+.2f} {u}, average {dm:+.2f} {u}.", "bad")
            else:
                verdict = callout(f"<b>The crack got narrower.</b> Widest point {dx:+.2f} {u}, average {dm:+.2f} {u}. "
                                  "Check that the photos are aligned and the lighting is similar.", "warn")
            st.markdown(verdict, unsafe_allow_html=True)

            v1, v2, v3 = st.columns([1, 1, 1], gap="large")
            both = (m_a > 0) & (m_b > 0)
            new = (m_b > 0) & ~(widen(m_a, 1) > 0)
            gone = (m_a > 0) & ~(widen(m_b, 1) > 0)
            canvas = tint(rgb_b if rgb_b is not None else grey_canvas(m_b), widen(both.astype(np.uint8), 1))
            canvas = tint(canvas, widen(gone.astype(np.uint8), 1), GONE_RGB)
            canvas = tint(canvas, widen(new.astype(np.uint8), 1), NEW_RGB)
            with v1:
                st.image(tint(rgb_a, widen(m_a, 1)), caption="Earlier", width="stretch")
            with v2:
                st.image(tint(rgb_b, widen(m_b, 1)), caption="Later", width="stretch")
            with v3:
                st.image(canvas, width="stretch",
                         caption="Change: red = in both, yellow = new, dark = no longer there")

            r1, r2 = st.columns(2, gap="large")
            with r1:
                st.markdown(readout("Widest point, change", abs(dx), u, fl,
                                    f"{'Wider' if dx > 0 else 'Narrower' if dx < 0 else 'Same'}: {d.earlier.max_width:.2f} → {d.later.max_width:.2f} {u}"),
                            unsafe_allow_html=True)
            with r2:
                st.markdown(readout("Average width, change", abs(dm), u, fl,
                                    f"{'Wider' if dm > 0 else 'Narrower' if dm < 0 else 'Same'}: {d.earlier.mean_width:.2f} → {d.later.mean_width:.2f} {u}"),
                            unsafe_allow_html=True)
            st.markdown(plain_row("Total length", f"{d.earlier.total_length:,.1f} → {d.later.total_length:,.1f} {u}  ({d.d_total_length:+,.1f})")
                        + plain_row("Cracks", f"{d.earlier.n_components} → {d.later.n_components}")
                        + plain_row("New crack area", f"{d.new_crack_area:,.1f} {u}²"),
                        unsafe_allow_html=True)
            st.caption("KEYSTONE currently raises its crack alert on the widest-point change. Whether to use the average "
                       "instead is an open team decision (README, section 'Open decisions').")
            st.download_button("Save comparison (JSON)",
                               json.dumps({"units": u, "mm_per_px": mm_eff, **d.as_dict()}, indent=2, default=float),
                               file_name="crack_comparison.json", mime="application/json")
    elif not (up_a or up_b):
        st.markdown(callout("Upload an earlier and a later photo, or press the example button to see how the "
                            "comparison reads."), unsafe_allow_html=True)

# --------------------------------------------------------------------- results
with tab_results:
    csv_path = ROOT / "results" / "final_eval" / "results.csv"
    st.markdown("### Which loss function worked best")
    st.write("Six ways of scoring the model's mistakes were tried on the same 85 test images. "
             "Higher IoU is better (1.0 would be a perfect match with the hand-drawn cracks).")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["loss"] = df["run"].str.replace("loss_", "", regex=False)
        df = df.sort_values("iou", ascending=False)
        import plotly.graph_objects as go
        top = df["iou"].max()
        fig = go.Figure(go.Bar(
            x=df["iou"], y=df["loss"], orientation="h",
            marker_color=["#2F7F79" if v >= top - 0.007 else "#9BA8A2" for v in df["iou"]],
            text=[f"{v:.3f}" for v in df["iou"]], textposition="outside",
            hovertemplate="%{y}: IoU %{x:.4f}<extra></extra>"))
        fig.update_layout(height=300, margin=dict(l=0, r=40, t=10, b=30), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"),
                          xaxis=dict(range=[0.5, 0.7], title="IoU on test images"),
                          font=dict(family="IBM Plex Sans, Helvetica, Arial, sans-serif", color="#1E2B33"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Green bars are within 0.007 IoU of the best. Each loss was trained once, so those three are "
                   "effectively tied, not ranked.")
        show = df[["loss", "iou", "dice", "precision", "recall", "threshold"]].rename(
            columns={"iou": "IoU", "dice": "Dice", "precision": "Precision", "recall": "Recall", "threshold": "Threshold"})
        st.dataframe(show.style.format({"IoU": "{:.4f}", "Dice": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "Threshold": "{:.3f}"}),
                     width="stretch", hide_index=True)
    else:
        st.markdown(callout("results/final_eval/results.csv is missing."), unsafe_allow_html=True)

    st.markdown("### Measuring crack width")
    gt = ROOT / "results" / "final_eval" / "measure_vs_gt_summary.md"
    if gt.exists():
        if st.checkbox("Show the width accuracy table (against hand-drawn cracks)"):
            st.markdown(gt.read_text(encoding="utf-8"))
    st.markdown("### Brick walls")
    mj = ROOT / "results" / "final_eval" / "masonry_metrics.json"
    if mj.exists():
        data = json.loads(mj.read_text(encoding="utf-8"))
        src = data.get("all_images", data)
        flat = {k: v for k, v in src.items() if k in ("iou", "dice", "precision", "recall")}
        if flat:
            cols = st.columns(len(flat))
            for c, (k, v) in zip(cols, flat.items()):
                c.metric(k.upper() if k == "iou" else k.capitalize(), f"{v:.3f}")
        else:
            st.json(data)
    st.write("The model was trained on road and concrete. On brick it marks mortar joints as cracks, so precision falls "
             "and the score drops from 0.65 to about 0.46.")
    curves = ROOT / "results" / "b_loss_comparison" / "loss_curves.png"
    if curves.exists():
        if st.checkbox("Show training curves"):
            st.image(str(curves), width="stretch")

# --------------------------------------------------------------------- limits
with tab_limits:
    st.markdown("### Read this before trusting a number")
    st.markdown(callout("<b>Not yet tested on a real monument.</b> Every accuracy figure comes from road and concrete "
                        "images (DeepCrack) or from a drawn test crack. Brick and stone are the target; "
                        "performance there is worse."), unsafe_allow_html=True)
    st.markdown(callout("<b>Hairline cracks.</b> Cracks about 1 px wide are often lost during training, so the model may "
                        "miss them or read them too wide. This is an open decision for the team, not a fixed bug.", "warn"),
                unsafe_allow_html=True)
    st.markdown(callout("<b>Width is good to about 1 px.</b> Odd widths are exact; even widths read 1 px short. At "
                        "0.5 mm per pixel that is half a millimetre, and no comparison can see a smaller change.", "warn"),
                unsafe_allow_html=True)
    st.markdown(callout("<b>Scale comes from you.</b> Millimetres per pixel is only as good as the number you enter. "
                        "A wrong scale gives wrong millimetres with no warning."), unsafe_allow_html=True)
    st.markdown(callout("<b>No alignment.</b> Two photos taken from different positions will show 'change' that is just "
                        "camera movement. Compare only photos taken from the same fixed spot, or align them first.", "bad"),
                unsafe_allow_html=True)
