# KEYSTONE + crack-seg: handoff file

Written 2026-10-10. Everything needed to continue this work in a brand-new chat, with no access to the old one.
Open a new Claude chat, attach this zip (or this folder's files), and paste the **starter prompt** below.

---

## 1. Starter prompt (paste this into the new chat)

> I am continuing a project from an earlier chat that I can no longer open. Read `HANDOFF.md` in the attached
> project first, then `README.md`. Do not trust any number in a message I paste from other chats: check it
> against the files, or run it. I write short messages and want the direct answer first.
> Project: Team NeuraCoders, Rajalakshmi Engineering College (AI & DS, third year).
> KEYSTONE is our Intellect Hack 2026 Track 1 entry (autonomous structural monitoring of heritage monuments).
> crack-seg is a separate U-Net crack-segmentation project that also feeds KEYSTONE.
> Start by telling me, in five lines, the state of both projects and what is still open. Then wait.

---

## 2. What the two projects are

**KEYSTONE** (Intellect Hack 2026, Track 1): "Measure it before we lose it." A monument is photographed from a phone
or a sub-Rs 1 lakh drone, reconstructed to a scaled 3D point cloud (COLMAP, OpenMVS, scaled with surveyed markers),
archived, re-captured months later, differenced (ICP alignment, M3C2 change detection), and the system acts on its own:
four severity levels (informational, watch, alert, critical) with inspection orders or access restrictions.
Problem source: CAG audit (3,698 protected monuments, no complete physical survey since Independence, 92 untraceable of
1,655 inspected, 546 encroachments found vs 249 reported). Accuracy claimed: 1 to 3 cm with ground control points.
Stated limits: surfaces only, seasonal movement is a confound, coverage gaps flagged. The submitted deck is in
`docs/originals/KEYSTONE_pitch_deck.pdf`.

**crack-seg**: U-Net trained from scratch (base 32, depth 4, 7.85M parameters, bilinear upsampling) on DeepCrack
(road/concrete cracks), six loss functions compared for the 1 to 5% class imbalance, plus width/length measurement by
skeleton and distance transform, plus a Streamlit app. It feeds KEYSTONE through `keystone-core/contracts/crack_adapter.py`.

## 3. Teams and ownership

KEYSTONE build, three workstreams (details in `keystone-core/WORKPLAN.md`):
- **P** Capture & Preservation (capture quality gate, marker finding, archive, plan/elevation/section export)
- **G** Geometry (scale, align, per-region movement with an honest level of detection)
- **D** Decision (seasonal vs structural, severity, routing, alerts, dashboard). The user (REC) owns D.

crack-seg workstreams: **A** data, **B** model/training, **C** measurement/integration. The user owns A and C
(this has been stated both ways in different chats; B's loss table and checkpoints came from another team member).
Members work in separate Claude chats. The user relays messages by hand: there is no messaging connector.

Rule in both repos: you may rewrite anything you own; do not change `keystone-core/contracts/schema.py` or
`crack-seg/src/contracts.py` unless every owner agrees.

## 4. State of the work

| Part | State | Evidence |
|---|---|---|
| KEYSTONE contracts + P + G + D | Built; 32 acceptance checks pass | `python run_tests.py` |
| Integration (Geometry to crack adapter to Decision) | Works end to end on synthetic data | `python run_demo.py`: planted 12 mm displacement recovered as 11.84 mm; only the drifting region escalates over a simulated year |
| crack-seg code | 22/22 checks pass | `python tests/contracts_test.py` in `crack-seg/` |
| crack-seg six-loss comparison | Done, verified | table below |
| Width measurement | Validated against ground truth | below |
| Streamlit app | Works, tested on a dummy checkpoint and on real weights with the user's own wall photo | `crack-seg/app.py`, `RUN_APP.md` |
| Updated deck (12 slides) | Done | `docs/CrackSegmentation_Project_updated.pptx` |
| Report (workstreams A and C) | Done, with two decisions pending | `docs/CrackSeg_Report_Workstreams_A_C.docx` |

### Verified crack-seg numbers
85 test images, threshold tuned on validation only, applied once to test. Checkpoints come from the Drive export
`crack-runs-20261001T074007Z` (user's copy: `Downloads\crack-runs-20261001T074007Z-1-001\crack-runs\<run>\best.pt`).

| Loss | IoU | Dice | Precision | Recall | Threshold |
|---|---|---|---|---|---|
| focal_tversky | 0.6506 | 0.7711 | 0.7365 | 0.8507 | 0.96 |
| tversky | 0.6484 | 0.7703 | 0.7303 | 0.8456 | 0.996 |
| dice | 0.6440 | 0.7636 | 0.7310 | 0.8366 | 0.90 |
| bce_dice | 0.6355 | 0.7552 | 0.7180 | 0.8363 | 0.95 |
| focal | 0.6318 | 0.7489 | 0.7244 | 0.8058 | 0.55 |
| bce | 0.6012 | 0.7173 | 0.7042 | 0.7813 | 0.40 |

Top three are within 0.007 IoU and each loss was trained once, so treat them as tied. Do-nothing model pixel accuracy
on the test set: 0.9657.
- Width vs ground truth (focal_tversky, 320 input): mean width median error 0.98 px, 51% of images within 1 px; widest
  point within 1 px on 27%; a perfect model at 320 would reach 99%. Cracks under 3 px (17 images) read about 0.97 px too
  wide, length 58 px short.
- Width formula `2d - 1`: exact for odd widths, 1 px short for even.
- Brick masonry (Dais et al., 240 images, GPL-3.0, never trained on): IoU 0.4553, Dice 0.5875, precision 0.5452,
  recall 0.7698. Mortar joints are marked as cracks. No crack-free images, so the false-alarm rate is unmeasured.
- 1 px cracks fade out of training images in about 90% of draws (resize/crop, not blur). Harmless on DeepCrack (median
  crack 5.9 px), a risk for hairline cracks.
- `ablation_bilinear` re-run with a wider threshold grid (0.6624 val at 0.95) was reported by teammates and not verified.

## 5. Not proven (do not overclaim)
- Nothing has run on a real monument. Geometry and Decision accuracy come from synthetic walls.
- No real photo set with printed scale markers has been through Geometry. A 17-photo wall test gave a sparse cloud with no
  mm scale (16/17 registered, about 870 points, 0.70 px).
- No viewable textured 3D model exists. The system measures movement of surveyed regions. Dense reconstruction needs a GPU.
- The crack model is trained on road/concrete. On the user's plaster-corner wall photo it marked the dark gap behind a door
  frame as a crack and missed hairline cracks. Millimetre values need the user's real mm-per-pixel.

## 6. Open decisions (not yours to make alone)
1. **Alert signal:** `crack_adapter.py` and the crack-seg README alert on `d_max_width`; mean width is more reliable against
   ground truth. Switching touches the shared contract (`schema.py`, `crack_adapter.py`, `severity.py`,
   `tests/contracts_test.py`): needs the Decision owner and all owners to agree, then rerun both suites (about an hour).
2. **Thin-crack fade:** retrain on native-resolution crops without blur, or record it as a limitation. Recommendation:
   limitation, since there is no hairline masonry data to validate a fix before the deadline.
3. **Which `transforms.py` trained the checkpoints:** blur 0.10 (in this zip) or 0.25. Evidence points to 0.25. Only
   Workstream B can confirm.
4. **Best use of remaining time:** capture one real structure with printed scale markers and run it through Geometry.
   This is the biggest gap between the deck's claims and what can be shown.

## 6b. What was done in the earlier chat (so nothing is repeated)
1. Checked every claim relayed from teammates' chats against real files or by running code. Several were wrong and were corrected.
2. Combined KEYSTONE and crack-seg into one runnable project with one environment, tests and an end-to-end demo.
3. Verified B's six-loss results by re-evaluating the real checkpoints; extended the threshold grid; found the top three tied.
4. Validated crack width against ground truth (about 1 px, stated limits) and quantified the brick-wall domain gap (IoU 0.455).
5. Ran the augmentation experiment: 1 px cracks fade mainly because of resize/crop, not blur.
6. Reviewed the A/C report: the mean-width alert only works on the crack-seg side; KEYSTONE reads max width only.
7. Wrote `tools/sparse_reconstruct.py` (photos to sparse point cloud) and tested it on 17 real wall photos.
8. Built the crack-seg Streamlit app and ran it on real weights with the user's own photo.
9. Updated the crack deck to 12 slides with real numbers; summarised KEYSTONE's idea, novelty and feasibility; built this handoff.
Status of the original 4-week plan: week 1 (concept, deck) done; weeks 2 to 4 are done only on synthetic data.

## 6c. Future plan (proposed, in priority order)
**Before the deadline**
1. Ask B two things: which `transforms.py` trained the checkpoints (0.10 or 0.25 blur), and whether to retrain for hairline cracks. Recommended answer: record it as a limitation.
2. Team decision on the alert signal (max vs mean crack width). If mean is approved, change `schema.py`, `crack_adapter.py`, `severity.py` and `tests/contracts_test.py` together, then rerun both suites.
3. **One real capture, the most valuable item.** Pick one accessible wall or structure. Fix printed scale markers of known length in view. Take 150 to 300 overlapping photos, then a second set after a known deliberate change (move a marked object by a measured amount). Run: COLMAP sparse (CPU, `tools/sparse_reconstruct.py`) or dense on a free Colab GPU, then Geometry, then Decision. Report the actual error against the taped value. This is the deck's own demo success criterion.
4. Rehearse the 3-minute demo: capture, flag, action. Prepare the hostile Q&A answers from the "not proven" list.
5. Finalise the report and deck: remove "pending" markers, fill in the transforms version, and add one "claims vs proof" slide.

**After the deadline, if the project continues**
6. Fine-tune the crack model on the 240 masonry images (Colab GPU) and measure the false-alarm rate on crack-free walls.
7. Dense reconstruction, meshing and texturing, plus a web viewer (OpenMVS, Potree), for a viewable archival model.
8. Repeat captures over seasons to feed the Decision model with real seasonal data instead of synthetic data.
9. Add crack-photo alignment (registration) so two hand-held photos can be compared.

## 7. How to run (Windows PowerShell, from this folder)
```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
Unblock-File .\setup.ps1
.\setup.ps1                       # 5 to 15 minutes, PyTorch download is silent
.\.venv\Scripts\Activate.ps1
python run_tests.py               # 32 + 22 checks
python run_demo.py                # whole chain on synthetic data
cd crack-seg
streamlit run app.py              # the crack app
```
Trained weights are NOT in this zip. Copy `best.pt` into `crack-seg\runs\loss_focal_tversky\best.pt`, then restart the app.
Do not use `runs\fake` or `runs\smoke` checkpoints: they come from the test suite.
The KEYSTONE dashboard: `cd keystone-core` then `streamlit run decision/dashboard.py`.
More commands (training, evaluation, photos to point cloud) are in `README.md`.

## 8. Folder map
- `keystone-core/` KEYSTONE code: `contracts/`, `preservation/`, `geometry/`, `decision/`, `tests/`, handover docs per workstream
- `crack-seg/` crack project: `src/`, `keystone/interface.py` (the only module KEYSTONE imports), `app.py` + `frontend/`,
  `results/` (loss table, width accuracy, masonry metrics), `scripts/`, `RUN_APP.md`
- `tools/sparse_reconstruct.py` photos to sparse point cloud via pycolmap (retries 5 times and keeps the best, because
  COLMAP mapping is random on flat walls)
- `docs/` updated 12-slide crack deck, report, app screenshot; `docs/originals/` the submitted KEYSTONE pitch deck
- `run_tests.py`, `run_demo.py`, `setup.ps1`, `setup.sh`, `requirements-all.txt`

## 9. How the user likes to work
- Very short messages; give the direct answer first, then detail.
- Verify, do not trust: relayed numbers from other chats were wrong several times (impossible 1.00 threshold, wrong
  fade percentages). Read the real files or run the real code.
- Step-by-step terminal instructions, written for Windows PowerShell and VS Code.
- Complete files, not snippets.
- No chat can message teammates: draft the message and the user sends it.

## 10. Update 2026-10-15: the KEYSTONE console (the thing to present)
The Decision dashboard alone is one workstream on synthetic data. `console/` is one app that runs the whole chain on screen:
`streamlit run console/app.py` (tests: `python console/smoke_test.py`). Tabs: Whole chain, Capture check, Change in 3D
(with a downloadable web viewer), Cracks, Year of captures, Orders and record (printable inspection order, archive
integrity check, plan/elevation/section drawings), Your photos (real photos to sparse 3D cloud; tested on 17 wall photos,
855 points).
It uses the real pipeline code on a **simulated wall with known truth** (default: planted 12 mm, recovered 11.8 mm).
Still NOT implemented or proven: a real monument capture with printed scale markers through Geometry, dense/textured
reconstruction (needs GPU), VR, the FastAPI/PostGIS/React stack, any validation on real data. Say "simulated" when presenting.

### Temple tab (added later the same day)
`console/monument.py` simulates a stone temple (12k points, 12 regions, 4 cracks) over 12 monthly captures: the tower leans
east from month 2, column col_b settles from month 4 (sagging the beam and plinth, opening two cracks), one hairline crack is a
control that never changes. Monthly change reports go through the real `decision.pipeline.analyse`.
Result with the default seed: tower alert month 7 (fixed 2 x noise threshold: month 10); beam and plinth alerted (fixed threshold never);
col_b alert month 9 (fixed threshold: month 8, one month earlier). **Honest finding:** a false ALERT on col_d_lower at month 4-5 in
7 of 8 seeds, because the settling column is a peer of col_d and drags the seasonal estimate in the first winter-to-summer months.
It clears later. A fixed threshold gave a false alarm in 2 of 8 seeds. Not tuned away on purpose; fix idea: weight peers by distance or
exclude peers with a rising trend earlier. All simulated, nothing measured on a real monument.
