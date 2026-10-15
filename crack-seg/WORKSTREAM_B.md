# Workstream B — Model & Training: full process

**You own:** `src/models/unet.py`, `src/losses.py`, `src/train.py`, `configs/`,
`scripts/collect_results.py`

**You must not change:** `src/contracts.py` — A and C build against it.

**Good news:** the model, the six losses and the training loop are already
written and verified. Your job is to **run the experiments, read them
correctly, and report what they show.** The loss comparison you produce is
the core result of the whole project — it fills slide 7 of the deck.

---

## Step 0 — Local setup (Windows, VS Code, PowerShell)

Run these one line at a time.

```powershell
cd "path\to\crack-seg"
```
```powershell
py -3.13 -m venv .venv
```
```powershell
.venv\Scripts\activate
```
If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.
```powershell
pip install -r requirements.txt
```

This installs the CPU build of PyTorch — fine for checks, too slow for real
training. Real training happens on Colab (Step 3).

---

## Step 1 — Verify what you own

```powershell
python src/models/unet.py
```
Expect: two lines with `shapes OK` and `all shape checks passed`.

```powershell
python src/losses.py
```
Expect a table. **Read the `all-empty` column** — it is the whole argument of
your results section. BCE scores a model that predicts nothing at about 0.31
(mediocre, not failing); Dice scores it at about 0.99 (failing). That is why
plain BCE under-segments cracks.

```powershell
python tests/contracts_test.py --only B
```
Expect: `7/7 checks passed`.

---

## Step 2 — Prove training runs, on fake data

```powershell
python -c "import sys; sys.path.insert(0,'.'); from tests.fakes import make_fake_dataset; make_fake_dataset('data/processed', n=60, size=160)"
```

Make a quick CPU config: copy `configs/default.yaml` to `configs/smoke.yaml`
and change `epochs: 2`, `image_size: 128`, `base: 8`, `num_workers: 0`.

```powershell
python -m src.train --config configs/smoke.yaml
```
Expect two epoch lines, then `checkpoints and history written`.

Then delete the fake data so it can't mix with real data:
```powershell
Remove-Item -Recurse -Force data
```

---

## Step 3 — Move training to Colab (free GPU)

Training six models on a laptop CPU would take days. A free Colab T4 does
each in minutes to tens of minutes.

1. Go to colab.research.google.com → New notebook
2. **Runtime → Change runtime type → T4 GPU → Save**
3. Upload `crack-seg.zip` using the folder icon on the left
4. Run these cells:

```python
!unzip -q crack-seg.zip && cd crack-seg && pip install -q albumentations scikit-image PyYAML
```
```python
%cd crack-seg
import torch; print(torch.cuda.is_available())   # must print True
```

**Save results to Google Drive**, because free sessions disconnect:
```python
from google.colab import drive; drive.mount('/content/drive')
```
Then add `--out /content/drive/MyDrive/crack-runs/<name>` to each train
command, or copy `runs/` to Drive after each run.

---

## Step 4 — Get real data

Member A owns this. If A has delivered `data/processed/`, upload it. If not,
don't wait — do it in Colab yourself:

```python
!git clone -q https://github.com/yhlleo/DeepCrack.git data/raw/DeepCrack
!ls data/raw/DeepCrack/dataset
```
```python
!python scripts/prepare_data.py --src data/raw/DeepCrack/dataset --layout deepcrack --clean
!python scripts/inspect_data.py --root data/processed
```

**Do not train until `inspect_data.py` prints `DATASET LOOKS USABLE`.**
Tell A what you found.

---

## Step 5 — Baseline run

```python
!python -m src.train --config configs/default.yaml
!python -m src.evaluate --checkpoint runs/default/best.pt
```

`evaluate` does two important things: it picks the decision threshold on the
**validation** split (never test), and it prints your model next to a
"predict nothing" baseline.

### How to read the result — diagnostics

| What you see | What it means | What to do |
|---|---|---|
| IoU equals the share of empty test images (e.g. 0.23 with 3/13 empty) | the model predicts nothing at all | train longer, check `pos_weight`, try Dice or Tversky |
| precision high, recall very low | under-segmenting — missing thin cracks | expected for BCE; that's your finding |
| recall high, precision low | over-segmenting — shadows and joints read as cracks | look at overlays from `src.predict` |
| val loss falling but val IoU flat | loss and metric disagree | normal on imbalanced data — trust IoU, which is how checkpoints are chosen |
| loss becomes `nan` | learning rate too high, or AMP instability | set `lr: 0.0001` or `amp: false` |
| CUDA out of memory | batch too big | `batch_size: 4` |

---

## Step 6 — The loss comparison (your core result)

Six configs already exist. Each is identical to `default.yaml` except the
loss, so any difference is caused by the loss alone.

```python
for L in ["bce", "dice", "bce_dice", "focal", "tversky", "focal_tversky"]:
    !python -m src.train --config configs/loss_{L}.yaml
    !python -m src.evaluate --checkpoint runs/loss_{L}/best.pt
```

Then collect everything into one table:
```python
!python scripts/collect_results.py --pattern loss_
```

This prints a table sorted by IoU and saves `runs/results.md` — paste it
into slide 7.

**Expected story** (check it, don't assume it): BCE has high precision and
low recall because it under-segments. Dice-family losses recover recall.
Tversky, with β > α, should have the highest recall at some cost to
precision. That's the right trade for structural monitoring — missing a crack
matters more than a false alarm.

**If the results don't match the expected story, report what you got.** A
surprising result that's honestly reported is worth more than an expected
one that's forced.

---

## Step 7 — Ablations (if time allows)

Change one line of `default.yaml` per run and give each its own `out_dir`.

| Ablation | Change | Question it answers |
|---|---|---|
| Upsampling | `bilinear: false` | does learned upsampling beat interpolation? |
| Capacity | `base: 16`, then `base: 64` | where does more capacity stop helping? |
| Resolution | `image_size: 512` | do thin cracks need higher resolution? |
| Augmentation | `aug_strength: light` / `heavy` | does heavier augmentation close the train/val gap? |

Pick the two that interest you most. Two done properly beat four done badly.

---

## Step 8 — Training curves

Each run writes `runs/<name>/history.json`. Plot `train_loss`, `val_loss` and
`val_iou` against epoch for the best and worst loss. One figure, two runs,
side by side — that goes in the report.

---

## What you write up

| Section | Content |
|---|---|
| Architecture | U-Net from scratch, skip connections, why logits not probabilities |
| The imbalance problem | the `all-empty` table from `python src/losses.py` |
| Loss comparison | your table from Step 6, and what it shows |
| Training | optimiser, learning rate schedule, early stopping on IoU, curves |
| Ablations | whichever you ran |
| Limitation | trained on road/concrete; masonry mortar joints will be flagged as cracks |

**Every number in the report must come from a run you actually did.** Where
you haven't run something, say so.

---

## Definition of done

- `python tests/contracts_test.py --only B` passes
- All six loss runs trained and evaluated
- `scripts/collect_results.py` table produced and pasted into slide 7
- At least one training-curve figure
- Results shared with C, who does evaluation and failure analysis on your best checkpoint
