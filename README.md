# KEYSTONE + crack-seg — combined project

Two projects, one folder, one Python environment.

| Folder | What it is | Workstreams |
|---|---|---|
| `keystone-core/` | KEYSTONE: structural monitoring of heritage monuments. Capture QC, geometry change measurement, seasonal-vs-structural decision | P (preservation), G (geometry), D (decision) |
| `crack-seg/` | U-Net crack segmentation and width measurement, standalone deep-learning project | A (data), B (model/training), C (measurement) |
| `tools/` | Optional: photos to sparse 3D point cloud | — |
| `docs/` | Crack-seg report (Workstreams A and C) | — |

They connect through one function: `keystone-core/contracts/crack_adapter.py` takes
crack-seg's `CrackDelta.as_dict()` and returns a `CrackChange` that rides inside
KEYSTONE's `ChangeReport`. Neither codebase imports the other.

```
photos ──► [P] capture gate ──► point clouds ──► [G] movement in mm ──┐
                                                                       ├─► [D] alerts
photos ──► [crack-seg] masks ──► width change in mm ──► adapter ───────┘
```

---

## 0. The console (start here for a demo)

The **Temple, one year** tab shows a simulated stone temple over 12 monthly captures: tower lean, a settling column, cracks that widen, and KEYSTONE's real Decision code raising alerts. Code: `console/monument.py`.

One screen that runs the whole chain with the project's real code: capture check, scaled model, change in 3D,
crack widening, a year of seasonal decisions, an inspection order, and the permanent record.

```
streamlit run console/app.py          # from this folder, with the environment active
python console/smoke_test.py          # checks the numbers against the planted truth
```
The wall is simulated with known ground truth (labelled on screen). The last tab builds a sparse 3D cloud from your own
photos (`pip install pycolmap`). Downloads: inspection order (HTML, print to PDF), measured plan/elevation/section (SVG),
and a one-file web viewer of the change map.

---

## 1. Setup (once)

**Windows PowerShell**, from this folder:
```
.\setup.ps1
.\.venv\Scripts\Activate.ps1
```
If PowerShell blocks the script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then run it again.
The PyTorch step can sit for 5 to 15 minutes. Leave it.

**Linux / macOS:**
```
bash setup.sh
source .venv/bin/activate
```

**By hand** (any OS): `python -m venv .venv`, activate it, then `pip install -r requirements-all.txt`.
Python 3.10 to 3.13.

## 2. Check everything works

```
python run_tests.py
```
Expect `32 passed` for KEYSTONE and `22/22 checks passed` for crack-seg.

```
python run_demo.py
```
Runs the whole chain on synthetic data: Geometry → crack measurement → adapter → Decision.

## 3. Run each part

### KEYSTONE (run from `keystone-core/`)
```
cd keystone-core
python -m tests.contracts_test --only P     # or G, D, I
python -m contracts.fakes                   # the Geometry and Decision scenarios
python -m contracts.fakes_capture           # seven capture scenarios with known defects
python -m decision.benchmark --seeds 50     # seasonal model vs naive threshold, for the write-up
streamlit run decision/dashboard.py         # interactive: replay a year of captures
python -m preservation.cli PHOTO_FOLDER     # capture-quality gate on a folder of photos
```

### crack-seg (run from `crack-seg/`)
```
cd crack-seg
python tests/contracts_test.py              # 22 checks
python src/losses.py                        # why BCE fails on 1-5% crack pixels
python keystone/interface.py                # width measurement against known widths
python scripts/aug_survival_experiment.py   # does augmentation erase 1 px cracks? (about 3 min)
```

Train and evaluate on real data (needs the DeepCrack dataset, see below):
```
python scripts/prepare_data.py --src data/raw/DeepCrack/dataset/DeepCrack --layout deepcrack --clean
python scripts/inspect_data.py --root data/processed --sheet data/contact_sheet.png
python -m src.train --config configs/default.yaml
python -m src.evaluate --checkpoint runs/default/best.pt
python -m src.predict --checkpoint runs/default/best.pt --input data/test_images --mm-per-px 0.8
```
Compare two captures of the same wall:
```
python -m src.predict --checkpoint runs/default/best.pt --input later.jpg --compare-to earlier.jpg --mm-per-px 0.8
```
After several runs exist under `runs/`:
```
python scripts/collect_results.py
python scripts/measure_vs_gt.py --checkpoint runs/loss_focal_tversky/best.pt
python scripts/eval_external.py --checkpoint runs/loss_focal_tversky/best.pt --images <masonry images> --masks <masonry masks> --name masonry
```

### Photos to a point cloud (optional)
```
pip install pycolmap
python tools/sparse_reconstruct.py PHOTO_FOLDER out/
```
Sparse, CPU only. Output is `out/sparse_points.ply`, in arbitrary units. COLMAP's mapping step is random on a
flat, low-texture wall (one run in five fell from about 870 points to about 240), so the tool tries 5 times and
keeps the best (`--tries N` to change). Tested on 17 real photos of a wall crack: 16 registered, 0.70 px error.

## 4. What is not in this folder (and where to get it)

| Missing | Why | How to get it |
|---|---|---|
| DeepCrack dataset | Licence, size | github.com/yhlleo/DeepCrack, Download ZIP, extract into `crack-seg/data/raw/DeepCrack`. Full steps in `crack-seg/RUNBOOK_A_C.md` |
| Trained checkpoints (`best.pt`) | Live on Workstream B's Google Drive, `crack-runs/` | Put each at `crack-seg/runs/<run name>/best.pt`, e.g. `runs/loss_focal_tversky/best.pt` |
| Masonry test set | Licence (GPL-3.0) | github.com/dimitrisdais/crack_detection_CNN_masonry, folders `crack_detection_224_images` and `crack_detection_224_masks` |
| Real monument captures | Nobody has shot a monument with printed scale markers yet | See "What is not proven" |

## 5. Results already in the folder

`crack-seg/results/b_loss_comparison/` is Workstream B's original six-loss table.
`crack-seg/results/final_eval/` is the re-evaluation with the threshold grid extended past 0.95:

| Loss | Test IoU | Threshold |
|---|---|---|
| focal_tversky | 0.6506 | 0.96 |
| tversky | 0.6484 | 0.996 |
| dice | 0.6440 | 0.90 |
| bce_dice | 0.6355 | 0.95 |
| focal | 0.6318 | 0.55 |
| bce | 0.6012 | 0.40 |

The top three are within 0.007 IoU and each loss was trained once, so treat them as tied.
Masonry (Dais et al.), same checkpoint, no re-tuning: IoU 0.455, precision 0.545, recall 0.770.

## 6. What is proven and what is not

**Proven, on synthetic data with known ground truth:** the three KEYSTONE workstreams meet
their contracts and work together (32 checks, including Geometry feeding Decision over JSON);
crack-seg's measurement is exact on odd widths and one pixel short on even ones.

**Not proven:**
- Nothing here has run on a real monument. The Geometry accuracy figures come from synthetic walls.
- No real photo set with printed scale markers has been through Geometry. A 17-photo test
  produced a sparse cloud with no millimetre scale.
- The crack model was trained on road and concrete. On brick masonry precision falls to 0.55:
  it marks mortar joints as cracks. Fine-tuning on the 240 masonry images is possible and not done.
- Cracks 1 px wide fade out of the image in about 90% of training draws (resize and crop,
  not blur). Irrelevant on DeepCrack (median crack 5.9 px), relevant for hairline masonry cracks.
- This is not a full 3D model of a monument. KEYSTONE measures movement of surveyed regions
  between two captures. A viewable, textured model would need dense reconstruction (GPU),
  meshing and texturing, none of which exist here.

## 7. Open decisions

1. **Alert signal.** `crack_adapter.py` and the crack-seg README alert on `d_max_width`.
   Measured against ground truth, mean width is more reliable (51% of images within 1 px,
   against 27% for max width). Changing it touches the shared contract, so it needs sign-off
   from the Decision side. `CrackDelta` already carries `d_mean_width`.
2. **Thin-crack fade.** Retrain on native-resolution crops without blur, or document it as a limitation.
3. **`transforms.py` version.** `crack-seg/src/data/transforms.py` is the latest one sent by the team
   (blur probability 0.10, kernel 3). The report's Table A2 was measured with the earlier version
   (0.25, kernel 3 to 5). Re-running `aug_survival_experiment.py` on this version still gives about
   90 to 97% fade for 1 px cracks. Which version the checkpoints were trained with is unconfirmed.
4. `ablation_bilinear` still needs re-evaluating with the extended threshold grid.

## 8. Who owns what

`keystone-core/WORKPLAN.md` and `crack-seg/WORKPLAN.md` have the full split.
Rule in both: you may rewrite anything you own; do not change `contracts/schema.py` or
`crack-seg/src/contracts.py` without every owner agreeing.
