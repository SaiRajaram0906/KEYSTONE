# HANDOVER — KEYSTONE Workstream D (Decision)

*Paste everything below into a new chat, and attach the `keystone-core` folder (zipped).*

---

I'm on a team building **KEYSTONE**, a system for autonomous structural monitoring of heritage monuments. I own **Workstream D — Decision**. I need help building my part. The shared contract and tests already exist; I'm attaching the project folder.

## The project in one paragraph

India has 3,698 centrally protected monuments, and most are inspected by eye — a conservator looks at a wall and writes "minor cracking". Nothing is measured, so slow structural movement goes unnoticed until it's obvious. KEYSTONE photographs a monument repeatedly, measures how far each part of the wall has moved since the first capture, and decides — autonomously — whether that movement matters and who should be told.

## The three teams

- **Geometry (another team):** turns point clouds into a `ChangeReport` — how many millimetres each region of the wall has moved since the reference capture.
- **Decision (me):** reads the history of those reports and decides.
- **Crack segmentation (a separate project):** measures crack widening from photos. Its results arrive inside my `ChangeReport` as `crack_changes`, already converted by an adapter.

## The actual problem

**Stone moves with the seasons.** Heat and monsoon moisture push walls out and let them back, by roughly as much as we can measure. A wall that has moved 9 mm in May is probably fine. A wall that has moved 9 mm and hasn't come back by January is not.

If I just threshold displacement, I'll raise alarms every summer. This was tested: a naive threshold raised **ten false alarms across twenty synthetic years**. My job is to tell **reversible seasonal movement** from **persistent structural movement**.

## My job

Implement one function in a new file `decision/pipeline.py`:

```python
evaluate(history: list[ChangeReport]) -> list[Alert]
```

- `history` — every report for one monument, oldest first, all compared against the same reference capture. So each region's `median_mm` is **cumulative** movement since the reference.
- Returns **exactly one `Alert` per region** in the latest report — `INFORMATIONAL` included, so every region gets a recorded decision.

**Each report gives me, per region:** `median_mm` (movement), `p95_abs_mm`, `lod_mm` (per-point detection limit), `significant_fraction`, and `load_bearing`. Each capture carries `temperature_c` and `rainfall_mm_7d`. Plus `crack_changes` with widening and its uncertainty.

**The components I need to build:**

1. **Seasonal model** — predict how much each region *should* move given temperature and rainfall. Per-region coefficients are hard to estimate from few captures, so pool across regions at similar height, and exclude the region being judged so a failing region can't hide its own drift.
2. **Residual analysis** — movement minus seasonal expectation. Structural movement shows up as a residual that keeps growing and doesn't reverse.
3. **Severity** — map to `INFORMATIONAL` / `WATCH` / `ALERT` / `CRITICAL`. Rules the tests enforce:
   - no `ALERT` with fewer than three captures (two reports)
   - seasonal movement alone never reaches `ALERT`
   - a persistent drift reaches `ALERT`
   - drift in a load-bearing region escalates
   - crack widening well beyond its uncertainty raises at least `WATCH`
4. **Routing** — each severity goes to a *different* recipient. The officer who took the photos shouldn't be the one told to go back. Suggested: monument record → capture schedule → regional office priority list → structural engineer plus site closure recommendation.
5. **Archive** — store every report and alert in SQLite so decisions are traceable.
6. **Dashboard** — Streamlit: per-region displacement over time with the seasonal band shaded, and the current alerts. This is what makes the project visible.

## Rules I must follow

- Do not edit anything in `contracts/` — the other team depends on it.
- Severity strings must be exactly `INFORMATIONAL`, `WATCH`, `ALERT`, `CRITICAL`.
- Every alert names the reference and latest capture ids, so a decision traces back to its evidence.
- My system **recommends** restricting access — it has no legal authority to close a monument. Word the `reason` accordingly.

## How I know I'm done

```bash
python -m tests.contracts_test --only D
```

Eight checks against synthetic years with known truth from `contracts/fakes.py`. The central one runs 10 simulated years and evaluates after **every** capture — summer heat must never reach `ALERT`. At most 1 drift in 10 may be missed.

A crude reference passed all eight. A naive threshold failed three. So this needs real modelling, not research.

## Setup (Windows)

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-decision.txt
python -m contracts.fakes
python -m tests.contracts_test
```

`python -m contracts.fakes` prints a synthetic year — look at how `centre_upper` hits about 9 mm in summer and returns, while `east_upper` keeps climbing. That's the whole problem in one table.

## Suggested structure

```
decision/
  __init__.py
  seasonal.py    pooled temperature/rainfall model, leave-one-region-out
  residuals.py   residual series and persistence test
  severity.py    the four bands and escalation rules
  routing.py     severity -> recipient
  store.py       SQLite archive of reports and alerts
  pipeline.py    evaluate — wires the above together
  dashboard.py   streamlit run decision/dashboard.py
```

## What to write up

- Why a threshold fails, shown with the false-alarm count.
- The seasonal model, and why leave-one-region-out matters.
- A precision/recall picture: over many synthetic seeds, how early is drift caught, and how many false alarms occur — that trade-off is the honest result.
- The limitation: the seasonal coefficients in the fakes are synthetic. Real monuments would need a year of captures to calibrate.

Start by explaining the seasonal model approach and giving me a complete `decision/seasonal.py`.
