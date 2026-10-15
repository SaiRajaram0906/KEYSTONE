"""Look and feel for the crack-seg app: one stylesheet plus the mm ruler widget."""
from __future__ import annotations

import math

PLASTER = "#ECEEE9"
INK = "#1E2B33"
VERDIGRIS = "#2F7F79"
CRACK = "#E5391F"      # used only for detected cracks, so red always means "crack"
HAIRLINE = "#C5CBC3"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500&display=swap');
html, body, .stApp, .stMarkdown, label, input, button, textarea {{ font-family: 'IBM Plex Sans', system-ui, sans-serif; }}
[data-testid="stIconMaterial"], span[class*="material"] {{ font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important; }}
h1, h2, h3 {{ font-family: 'Bricolage Grotesque', 'IBM Plex Sans', sans-serif; letter-spacing: -0.015em; color: {INK}; }}
.block-container {{ max-width: 1120px; padding-top: 2.2rem; }}
header[data-testid="stHeader"] {{ background: transparent; }}

.cs-title {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 2.6rem;
  line-height: 1.05; letter-spacing: -0.03em; margin: 0 0 .35rem 0; color: {INK}; }}
.cs-sub {{ font-size: 1.02rem; color: #4A5A63; max-width: 58ch; margin: 0 0 1.4rem 0; line-height: 1.5; }}

/* the readout: a measured width drawn against a millimetre ruler */
.cs-read {{ border-top: 2px solid {INK}; padding: .7rem 0 1.0rem 0; }}
.cs-read .name {{ font-size: .92rem; color: #4A5A63; }}
.cs-read .val {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 2.0rem;
  letter-spacing: -0.02em; color: {INK}; line-height: 1.1; }}
.cs-read .unit {{ font-size: 1rem; font-weight: 500; color: #4A5A63; margin-left: .25rem; }}
.cs-read .note {{ font-size: .85rem; color: #4A5A63; margin-top: .25rem; }}
.cs-plain {{ border-top: 1px solid {HAIRLINE}; padding: .55rem 0; display: flex; justify-content: space-between; gap: 1rem; }}
.cs-plain span:first-child {{ color: #4A5A63; }}
.cs-plain span:last-child {{ font-weight: 500; }}

.cs-callout {{ border-left: 4px solid {VERDIGRIS}; background: #E0E4DD; padding: .8rem 1rem; border-radius: 0 6px 6px 0; margin: .6rem 0 1rem 0; }}
.cs-callout.warn {{ border-left-color: #B7791F; }}
.cs-callout.bad  {{ border-left-color: {CRACK}; }}
.cs-callout b {{ font-weight: 600; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid {HAIRLINE}; }}
.stTabs [data-baseweb="tab"] {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 500; font-size: 1.05rem; padding: .5rem 0; }}
.stTabs [aria-selected="true"] {{ color: {VERDIGRIS}; }}
button:focus-visible, [role="tab"]:focus-visible {{ outline: 3px solid {VERDIGRIS} !important; outline-offset: 2px; }}
@media (max-width: 640px) {{ .cs-title {{ font-size: 2rem; }} .block-container {{ padding: 1.2rem 16px; }} }}
</style>
"""


def ruler_svg(value_mm: float, unit: str = "mm", floor: float | None = None) -> str:
    """A bar of the measured length on a ruler with ticks, so a width reads as a size, not a number.

    `floor` marks the 1-pixel quantisation limit (the smallest difference the
    measurement can resolve) as a hatched band at the start of the ruler.
    """
    span = max(5, math.ceil(value_mm * 1.25))
    step = next(n for n in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 10**9) if span / n <= 8)
    w, h = 360, 34
    px = lambda v: 6 + (w - 12) * v / span
    ticks = []
    for t in range(0, span + 1, step):
        x = px(t)
        ticks.append(f'<line x1="{x:.1f}" y1="14" x2="{x:.1f}" y2="24" stroke="{INK}" stroke-width="1"/>')
        ticks.append(f'<text x="{x:.1f}" y="33" font-size="9" text-anchor="middle" fill="#4A5A63">{t}</text>')
    floor_band = ""
    if floor:
        floor_band = (f'<rect x="{px(0):.1f}" y="14" width="{px(floor) - px(0):.1f}" height="10" '
                      f'fill="{HAIRLINE}" opacity=".9"><title>1 px, the smallest step the measurement resolves</title></rect>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{value_mm:.2f} {unit} on a ruler">'
            f'{floor_band}'
            f'<rect x="{px(0):.1f}" y="4" width="{max(px(value_mm) - px(0), 1):.1f}" height="8" fill="{VERDIGRIS}"/>'
            f'<line x1="{px(0):.1f}" y1="24" x2="{px(span):.1f}" y2="24" stroke="{INK}" stroke-width="1.5"/>'
            f'{"".join(ticks)}</svg>')


def readout(name: str, value: float, unit: str, floor: float | None = None, note: str = "") -> str:
    note_html = f'<div class="note">{note}</div>' if note else ""
    return (f'<div class="cs-read"><div class="name">{name}</div>'
            f'<div class="val">{value:.2f}<span class="unit">{unit}</span></div>'
            f'{ruler_svg(value, unit, floor)}{note_html}</div>')


def plain_row(label: str, value: str) -> str:
    return f'<div class="cs-plain"><span>{label}</span><span>{value}</span></div>'


def callout(html: str, kind: str = "") -> str:
    return f'<div class="cs-callout {kind}">{html}</div>'
