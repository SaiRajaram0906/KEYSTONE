# Running the front end

## Windows (PowerShell), from this folder
```
.\setup.ps1
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```
Your browser opens at http://localhost:8501. If PowerShell blocks the script:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then run it again.
The PyTorch download can take 5 to 15 minutes.

## Linux / macOS
```
bash setup.sh
source .venv/bin/activate
streamlit run app.py
```

## Add the trained model
The weights are not in this zip. Copy Workstream B's `best.pt` files from Google Drive (`crack-runs/`) to:
```
runs/loss_focal_tversky/best.pt     (the best one, the app picks it first)
runs/loss_tversky/best.pt           (any other run appears in the Checkpoint list)
```
Without a model the app still works for **masks**: tick "already a crack mask" to measure or compare masks,
and the example button on the Compare tab shows a full comparison.

## What each tab does
| Tab | Does | Needs |
|---|---|---|
| Measure one photo | Outlines cracks, reports length and width in mm | model, or a mask |
| Compare two photos | Widening, new crack area, change image | two aligned captures of one wall, same size |
| Results | The six-loss table, masonry numbers, width accuracy | nothing |
| What it can't do | Limits, in plain words | nothing |

Set **Millimetres per pixel** in the sidebar before trusting any mm figure. The app does not know the scale of your photo.

## Check it
```
python tests/contracts_test.py       # 22 checks, the project itself
python tests/app_smoke_test.py       # the app starts and the example comparison works
```

## Honest limits
Trained on road and concrete (DeepCrack). On brick it marks mortar joints as cracks (IoU 0.455). Hairline cracks (about 1 px)
are often lost. Width is good to about 1 px. Photos are not aligned for you. See the "What it can't do" tab and README.md.
