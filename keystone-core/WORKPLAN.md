# KEYSTONE — three workstreams

Three people, three parts, one contract between them.

```
  photographs ─▶ [crack-seg team] ─▶ CrackDelta ──────────────┐  (separate project)
                                                               ▼
  [ P — CAPTURE ] ─▶ photos ─▶ [ G — GEOMETRY ] ─▶ ChangeReport ─▶ [ D — DECISION ] ─▶ alerts
    registry,          (off-the-    scale, align,      (JSON)        seasonal model,
    quality gate,       shelf recon) measure movement                severity, routing
    find markers
         ▲                                  │                              │
         └────────── [ P — ARCHIVE ] ◀──────┴──────────────────────────────┘
                     integrity, retention, export to drawings
```

Crack segmentation is **not** in scope for any of the three. Its output enters
through `contracts/crack_adapter.py`, which takes the crack team's plain
dict and produces a `CrackChange`. No shared code, no coupling.

---

## The split

| | P — Capture & Preservation | G — Geometry | D — Decision |
|---|---|---|---|
| **Question answered** | Is this capture good enough, and is the record safe? | How far did each part of the wall move? | Does that movement matter, and who needs to know? |
| **Input** | a monument, a folder of photos, and the raw cloud they reconstruct to | two point clouds + surveyed markers | a history of `ChangeReport`s |
| **Output** | `MonumentRecord`, `CaptureQC`, archive, `Section` | `ChangeReport` | one `Alert` per region |
| **Entry point** | `preservation/pipeline.py` | `geometry/pipeline.py::build_change_report` | `decision/pipeline.py::evaluate` |
| **Core skills** | image quality metrics, coverage geometry, storage engineering | 3D geometry, point clouds, linear algebra | time series, statistics, a dashboard |
| **Key library** | OpenCV | Open3D | pandas, scikit-learn, Streamlit |
| **Hardest part** | six defects need six different checks; finding the markers in an unscaled cloud | an honest level of detection | not alarming on summer heat |
| **Acceptance** | 13 checks, `--only P` | 6 checks, `--only G` | 8 checks, `--only D` |
| **Blocked by** | nobody | nobody | nobody |

Nobody waits for anybody. Each workstream develops against the fakes in
`contracts/` — `fakes.py` for G and D, `fakes_capture.py` for P — which
generate scenarios **with known ground truth**, so the tests check whether
your answer is right, not merely whether it is well-formed.

---

## Difficulty, verified rather than asserted

Before handing this out, every acceptance suite was run against a
deliberately crude throwaway implementation. All passed with margin, so the
tolerances are reachable. Naive versions were run too, and both failed:

- A Decision that ignores temperature raised ten false alarms across twenty
  synthetic years and failed three checks.
- A capture gate checking only photo count and coverage let four of five
  defect types through to reconstruction, and rejected the one capture that
  was actually salvageable.

So each workstream requires real thinking, and none requires research.

---

## Rules that make integration trivial

1. **Nobody edits `contracts/` alone.** All three agree, or it doesn't change.
2. **Millimetres across the boundary.** No pixels, metres, or raw units in a `ChangeReport`.
3. **Every report compares against the reference capture** — the monument's
   first — never the previous one. Decision depends on this.
4. **The boundary is JSON.** `ChangeReport.to_json()` / `from_json()`. If it
   round-trips, it integrates.
5. **Run `python -m tests.contracts_test --only <G|D>` before every push.**

---

## Integration checkpoints

| When | What must be true |
|---|---|
| **1 — day one** | All three run the suite; shared checks pass, own section shows PENDING. Environments work. |
| **2 — midpoint** | Each person's own section fully green against the fakes. |
| **3 — integration** | The `INTEGRATION` section goes green: P's registry drives Geometry, and a real Geometry report drives Decision across JSON. |
| **4 — real data** | Optional: photograph one wall twice with a phone, run it through P's gate and the whole pipeline. |

Checkpoint 3 is a single command, not a day:

```bash
python -m tests.contracts_test
```

---

## After both finish

The full KEYSTONE pipeline:

```python
from contracts.crack_adapter import from_crack_delta
from decision.pipeline import evaluate
from geometry.pipeline import build_change_report
from preservation.pipeline import Archive, check_capture

# P: is this capture even usable?
qc = check_capture(session, monument)
if not qc.usable:
    raise SendThemBackToTheSite(qc.reshoot_guidance)

# G: measure it, using P's registry
report = build_change_report(reference, latest, monument.survey(), regions)
report.crack_changes = [from_crack_delta(d, region_id, mm_per_px) for ...]   # crack team

# D: decide
alerts = evaluate(history + [report])

# P: keep all of it
Archive("archive/").store(report.to_json().encode(), monument_id=monument.monument_id,
                          kind="report", created_on=report.later.captured_on)
```

Four teams, four repositories, one contract.
