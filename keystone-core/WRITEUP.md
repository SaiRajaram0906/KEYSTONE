# KEYSTONE — Workstream D (Decision): write-up

All numbers below were produced by `python -m decision.benchmark` (200 synthetic years per experiment, deciding after every capture) on the synthetic data in `contracts/fakes.py`. Re-run the command to reproduce them. Ground truth is used only to score alerts, never to make them.

## 1. Why a simple threshold fails

The obvious rule is "raise an alarm when a region has moved more than twice its detection limit". It ignores weather and history. On years with **no structural movement at all**, that rule raised **10 false alarms across the first 20 synthetic years** (in 5 of the 20 years), and **97 false alarms across 200 years** — 62 of the 200 years had at least one.

All 97 fall on the mid-April, late-May and mid-July captures, and all in upper regions. The upper regions move about 0.6 mm per °C and the year swings about 10 °C, so they are 6–9 mm out by late May. The detection limit is 3–4.5 mm, so "twice the detection limit" sits right inside the ordinary seasonal swing. Raising the threshold does not fix this: a threshold high enough to clear summer is also high enough to ignore the first 10 mm of a real failure. The problem is not the threshold's value, it is that size alone cannot tell reversible movement from persistent movement.

On the same 200 drift-free years, the seasonal model raised **0 false alarms**. On a separate run of 1,000 drift-free years (54,000 individual decisions) it raised 1.

## 2. The seasonal model, and why the judged region is left out

For each region, the model predicts what the weather alone should have done:

    expected_mm = a × (temperature − temperature at reference) + b × (7-day rainfall − rainfall at reference)

There is no intercept, because every report is measured against the reference capture. The coefficients are fitted by least squares on the *other* regions at the same height (the top of a wall moves roughly twice as much as its base, so heights are not mixed). The prediction carries a band made of measurement noise (from `lod_mm`), the regression's own prediction variance, and a 20% allowance for neighbours not responding identically. Because the fit includes what the neighbours did at the latest capture, the model never extrapolates to weather it has not seen — the first monsoon week is explained by what the neighbours did that same week.

**Leaving the judged region out is what makes the residual honest.** When the judged region is allowed into its own fit, its drift is partly absorbed into the "seasonal" coefficients and the residual that should expose it shrinks. Measured on 200 years with `east_upper` drifting 3.5 mm per capture:

| Method | Drift caught | Mean captures to first ALERT | False-alarm years |
|---|---|---|---|
| Naive threshold | 200 / 200 | 2.2 | 62 / 200 |
| Judged region **included** in its own fit | 200 / 200 | **4.3** | 0 / 200 |
| Leave-one-out (shipped) | 200 / 200 | **2.8** | 0 / 200 |

Including the region costs 1.5 captures — about 68 days at a 45-day interval — before anyone is told. With a slower drift or a shorter history, that delay becomes a miss.

The same idea is applied one step further (**peer screening**): a region that is already far outside its own band is not allowed to define "normal" for its neighbours. Without screening, a drifting wall made its healthy neighbours' expectation wrong by −1.9 mm on average and widened their band from 1.6 mm to 4.5 mm (1σ), which would hide a second failure next to the first. With screening the bias is 0.0 mm.

## 3. Detection speed versus false alarms

An ALERT requires the residual to be beyond 3σ on two consecutive captures, in the same direction, not returning, and larger than the region's detection limit. Load-bearing regions use a bar 0.5σ lower. Varying that bar gives the trade-off:

| Residual bar (σ) | False-alarm years (drift-free) | Drift caught | Mean captures to ALERT |
|---|---|---|---|
| 1.5 | 12 / 200 | 200 / 200 | 2.1 |
| 2.0 | 6 / 200 | 200 / 200 | 2.3 |
| 2.5 | 1 / 200 | 200 / 200 | 2.5 |
| **3.0 (shipped)** | **0 / 200** | **200 / 200** | **2.7** |
| 3.5 | 0 / 200 | 200 / 200 | 2.9 |
| 4.0 | 0 / 200 | 200 / 200 | 3.1 |

Loosening the bar from 3.0 to 1.5 buys about half a capture of speed and costs 12 false-alarm years in 200. Because the rule demands two consecutive captures, the floor on delay is 2 captures whatever the bar; the honest way to detect faster is to capture more often, which is exactly what a WATCH recommends.

Sensitivity to drift rate (drifting region rotated over all six regions, drift beginning at capture 5 of 10, so at most 5 drifting captures exist by year end):

| Drift (mm per capture) | Caught by year end | Mean captures to ALERT | False-alarm years |
|---|---|---|---|
| 1.0 | 87 / 200 | 4.8 | 0 / 200 |
| 1.5 | 191 / 200 | 4.4 | 0 / 200 |
| 2.5 | 200 / 200 | 3.1 | 0 / 200 |
| 3.5 | 200 / 200 | 2.7 | 0 / 200 |
| 5.0 | 200 / 200 | 2.3 | 0 / 200 |

A drift of 1 mm per capture accumulates only 5 mm by year end, close to the detection limit, so under half are caught within the year. Those are not missed for ever — the residual keeps growing — but they need a longer history than one year.

## 4. Limitations

**The seasonal coefficients are synthetic.** The 0.6 mm/°C response, the 2:1 upper-to-lower ratio, the 0.03 mm per mm rain term and the noise level of `lod/3` are properties of `contracts/fakes.py`, chosen to make the problem hard — they are not measurements of real stone, and must not be quoted as such. A real monument needs **about a year of captures** (one full temperature and monsoon cycle) before its own coefficients are calibrated; until then the band should be treated as provisional and ALERTs reviewed by a person.

Other limits worth stating: the model needs at least one healthy neighbour at the same height — a monument reported as a single region falls back to fitting its own history, which is precisely the case leave-one-out exists to avoid, and a drift there can hide. Height is read from the region id suffix (`_upper`, `_lower`), which is a naming convention rather than geometry. If a majority of one height group fails together, the majority defines "normal" and the failure is hidden; crack evidence is the only independent check in that case. The fakes contain no reference-capture error, no sensor bias and no one-off shocks (an earthquake, a vehicle strike), so the false-alarm figures are a best case.

KEYSTONE recommends; it has no authority to close a monument. A CRITICAL decision is worded as a recommendation to the custodian to consider restricting access, and is routed to a structural engineer.
