# Crack Segmentation

A U-Net that finds cracks in masonry photographs, pixel by pixel, and
converts the result into a physical measurement.

Built standalone so it can be marked as a deep learning project on its
own, with a clean integration surface (`keystone/interface.py`) so it can
later drop into KEYSTONE without rewriting anything.

---

## Front end

A browser app for measuring and comparing cracks: `streamlit run app.py`. Setup, model placement and tab guide are in `RUN_APP.md`.

## Why this is a real project and not a tutorial

Cracks occupy roughly 1–5% of the pixels in a typical image. A model that
predicts "no crack" everywhere scores about 98% pixel accuracy and is
useless. Everything interesting here follows from that single fact:

- **Loss choice matters more than architecture.** `src/losses.py`
  implements six losses so the comparison can be measured rather than
  asserted. Run `python src/losses.py` to see BCE score a do-nothing model
  at 0.31 while Dice scores it 0.99.
- **Pixel accuracy is reported only to show how misleading it is.**
  `src/metrics.py` handles the empty-mask edge case explicitly instead of
  averaging NaNs into a flattering number.
- **Segmentation is not measurement.** A mask says *which pixels*. Turning
  that into "this crack widened 3 mm" needs skeletonisation, a distance
  transform, and a scale reference. `keystone/interface.py` does all three
  and documents its ±1 px quantisation error rather than hiding it.

---

## Setup

```bash
pip install -r requirements.txt
```

Get a dataset (none download automatically — see the notes in
`scripts/prepare_data.py` for links and licences), unzip it under
`data/raw/`, then convert it:

```bash
python scripts/prepare_data.py --src data/raw/DeepCrack --layout deepcrack
```

This writes `data/processed/images/` and `data/processed/masks/` and
prints the class balance.

---

## Running it

```bash
# train
python -m src.train --config configs/default.yaml

# evaluate on the held-out test split, tuning the threshold on validation
python -m src.evaluate --checkpoint runs/default/best.pt

# run on new images, with a scale so widths come out in millimetres
python -m src.predict --checkpoint runs/default/best.pt \
    --input captures/ --mm-per-px 0.8

# compare two aligned captures of the same wall
python -m src.predict --checkpoint runs/default/best.pt \
    --input captures/2026-09/north.jpg \
    --compare-to captures/2026-03/north.jpg --mm-per-px 0.8
```

Every module also runs standalone as a self-test:

```bash
python src/models/unet.py       # shape checks, both upsampling modes
python src/losses.py            # loss behaviour on a degenerate model
python src/metrics.py           # metrics, including the empty-mask case
python src/data/dataset.py      # loader and deterministic splitting
python src/data/transforms.py   # augmentations keep masks binary
python keystone/interface.py    # width calibration against ground truth
```

---

## Experiments worth running

These are the results that make a report rather than a demo.

| Experiment | Vary | What to report |
|---|---|---|
| Loss comparison | `loss.name` across all six | IoU, Dice, precision, recall per loss. Expect BCE to under-segment and Tversky (β>α) to win on recall. |
| Class weighting | `loss.pos_weight`: 1, 10, auto | Precision/recall trade-off curve |
| Threshold | swept automatically by `evaluate.py` | IoU vs threshold; show 0.5 is not optimal |
| Augmentation | `data.aug_strength` | Train/val gap — does heavy augmentation close it? |
| Upsampling | `model.bilinear` true/false | IoU and parameter count |
| Capacity | `model.base`: 16, 32, 64 | IoU vs parameters — where does it saturate? |
| Resolution | `data.image_size`: 256, 320, 512 | Recall on thin cracks specifically |

Ablations change one line in the config. Copy `configs/default.yaml`,
change one value, point `--out` somewhere new.

---

## The limitation to state in your report

Every public crack dataset is predominantly **road and concrete**.
KEYSTONE's target is **stone masonry**, where mortar joints look a great
deal like cracks — dark, linear, high-contrast.

Expect the model to flag mortar joints. Do not present road-surface
metrics as if they transfer to masonry. The honest options are:

1. Report the domain gap and show failure cases on masonry images.
2. Hand-label 50–100 masonry images and fine-tune, reporting both.

Option 2 is a strong addition if you have time. Option 1 is acceptable and
takes a paragraph.

---

## Integration with KEYSTONE

KEYSTONE imports one module:

```python
from keystone.interface import CrackSegmenter, compare_captures

seg = CrackSegmenter("runs/default/best.pt")

mask_march = seg.segment("capture_2026_03/north_wall.jpg")
mask_sept  = seg.segment("capture_2026_09/north_wall.jpg")

delta = compare_captures(mask_march, mask_sept, mm_per_px=0.8)
if delta.d_max_width > 1.0:
    raise_alert(delta)
```

Two things KEYSTONE must supply:

- **`mm_per_px`** — from the surveyed scale markers in the photograph.
  Without it every number is in pixels and is not comparable between
  captures.
- **Alignment** — `compare_captures` assumes the two masks are spatially
  registered. In KEYSTONE that is free, because both captures are
  registered to the same ground control points during reconstruction.

---

## Layout

```
crack-seg/
├── configs/default.yaml        all hyperparameters
├── scripts/prepare_data.py     public datasets -> images/ + masks/
├── src/
│   ├── models/unet.py          U-Net, written from scratch
│   ├── data/dataset.py         loading, deterministic splits, pos_weight
│   ├── data/transforms.py      augmentation tuned for thin structures
│   ├── losses.py               six losses for the imbalance problem
│   ├── metrics.py              IoU, Dice, precision, recall
│   ├── train.py                training loop, AMP, early stopping
│   ├── evaluate.py             test metrics + threshold tuning
│   └── predict.py              inference, overlays, comparison
└── keystone/interface.py       the only module KEYSTONE imports
```

---

## Notes on design decisions

**The model returns logits, never probabilities.** Every loss applies its
own sigmoid. This avoids the double-sigmoid bug, which trains without
error and quietly halves your effective learning signal.

**Splits are hashed from filenames, not shuffled.** An image lands in the
same split every run, even if files are added later. A re-shuffle that
leaks test images into training is very hard to spot from the metrics.

**The threshold is tuned on validation and applied once to test.** Tuning
on test and then reporting test is leakage. `evaluate.py` writes the
chosen threshold back into the checkpoint so inference picks it up.

**`pos_weight` is computed from the training split only.** Computing it
over the whole dataset leaks test statistics.
