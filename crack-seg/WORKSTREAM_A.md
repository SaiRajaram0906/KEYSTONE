# Workstream A — Data: full process

Everything you own, in order. Roughly a day of real work spread over a
few sittings, most of it waiting for downloads.

**You own:** `src/data/dataset.py`, `src/data/transforms.py`,
`scripts/prepare_data.py`, `scripts/inspect_data.py`

**You must not change:** `src/contracts.py` — B and C build against it.

**You are on the critical path.** B and C can start against fake data, but
nobody produces a real result until you deliver. Do Step 2 on day one,
even if you do nothing else.

---

## Step 0 — Environment

```bash
cd crack-seg
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Confirm it works:

```bash
python src/data/transforms.py
```

You should see four lines ending `crack survived 12/12 draws`. If this
fails, fix it before anything else — it means albumentations or OpenCV
did not install cleanly.

---

## Step 1 — Prove the pipeline runs before you have real data

Do not wait for a download to find out your environment is broken.

```bash
python -c "
import sys; sys.path.insert(0,'.')
from tests.fakes import make_fake_dataset
make_fake_dataset('data/processed', n=40, size=180)
"
python scripts/inspect_data.py --root data/processed
python tests/contracts_test.py --only A
```

Expect `DATASET LOOKS USABLE` and `6/6 checks passed`.

Now delete the fake data so it cannot get mixed with the real thing:

```bash
rm -rf data/processed
```

---

## Step 2 — Get the real dataset

Start with **DeepCrack**: 537 images, clean pixel-level labels, varied
surfaces. It is the best trade-off between size and annotation quality.

```bash
mkdir -p data/raw
cd data/raw
git clone https://github.com/yhlleo/DeepCrack.git
cd ../..
```

The images live in `DeepCrack/dataset/`. Unzip anything that arrives
zipped, then look at the folder names — the converter expects
`train_img` / `train_lab` and `test_img` / `test_lab`.

```bash
ls data/raw/DeepCrack/dataset/
```

**If the layout differs**, do not edit the converter blindly. Find the two
folders holding images and masks, then use the generic mode:

```bash
python scripts/prepare_data.py --src data/raw/DeepCrack/dataset \
    --layout pairs --images-dir <image folder> --masks-dir <mask folder>
```

**Alternatives** if DeepCrack is unavailable:

| Dataset | Size | Notes |
|---|---|---|
| Crack500 | 3,368 | Larger, noisier. Good second experiment. |
| CrackForest | 118 | Small; masks are `.mat` files, use `--layout crackforest` |
| CrackTree260 | 260 | Very thin cracks — useful stress test later |

---

## Step 3 — Convert

```bash
python scripts/prepare_data.py --src data/raw/DeepCrack/dataset \
    --layout deepcrack --clean
```

This writes `data/processed/images/` and `data/processed/masks/` with
matched stems, and prints the crack pixel fraction.

**Expect roughly 1–5% crack pixels.** If you see 50% or 99%, the masks are
inverted and Step 4 will tell you so.

---

## Step 4 — Inspect. This is the real job.

```bash
python scripts/inspect_data.py --root data/processed --sheet data/contact_sheet.png
```

It checks for the six failures that silently ruin segmentation projects:
inverted masks, empty masks, over-full masks, size mismatches, non-binary
masks, and duplicate images. Duplicates matter more than they sound — the
same photo under two names leaks across the train/test split and inflates
every number you report.

It exits non-zero if anything fails, so you can use it in a script.

**Then open `data/contact_sheet.png` and actually look at it.** The red
overlay must sit exactly on the dark lines in the photograph. Five minutes
of eyeballing catches misalignment that no statistic will.

Common problems and what they mean:

| Symptom | Cause | Fix |
|---|---|---|
| ~99% crack pixels | masks inverted | invert them in the converter |
| overlay offset from the crack | mask/image mismatch | wrong folder pairing |
| almost all masks empty | wrong mask folder | recheck `--masks-dir` |
| grey values in masks | JPEG-compressed labels | harmless, thresholding handles it |
| duplicate groups | dataset ships copies | delete the duplicates |

Do not proceed until this prints `DATASET LOOKS USABLE`.

---

## Step 5 — Confirm the contract

```bash
python tests/contracts_test.py --only A
```

All six must pass. This is what guarantees B and C can use your output
without a shape mismatch on integration day.

If a check fails, the message names what is wrong. The two you are most
likely to hit:

- *mask missing channel dim* — the dataset returned `(H, W)` instead of
  `(1, H, W)`
- *masks must contain only 0.0 and 1.0* — a geometric transform used
  bilinear interpolation on the mask instead of nearest-neighbour

---

## Step 6 — Tell B and C

Post three things in the group:

1. The crack pixel fraction and the suggested `pos_weight` from Step 4.
   B needs the second number for the loss configuration.
2. The contact sheet image, so everyone has seen the data.
3. That `python tests/contracts_test.py --only A` passes.

B can now train on real data. Your critical-path obligation is discharged.

---

## Step 7 — The experiment that is yours to write up

This is what turns your part from plumbing into a results section.

**Question.** How often does each augmentation strength destroy a thin
crack while leaving its mask intact?

That failure mode is specific and dangerous: the image no longer contains
a crack, the mask still says there is one, and the model learns that blank
wall means crack.

```bash
python src/data/transforms.py
```

The self-test already reports crack survival per strength. Extend it: for
each of `light`, `medium`, `heavy`, run a few hundred draws on real images
and measure

- how often the crack becomes invisible in the image,
- how often the mask area changes by more than, say, 20%,
- the distribution of resulting crack widths in pixels.

Report a small table. Then recommend a strength, and say why.

A second experiment if you have time: **how thin is too thin?** Filter the
dataset by crack width and report how many cracks fall below 2 pixels at
each `image_size` setting (256, 320, 512). That number directly bounds
what the model can possibly detect, and C will want it for the limitations
section.

---

## What you write in the report

| Section | Content |
|---|---|
| Dataset | Source, licence, image count, resolution range, how it was converted |
| Class imbalance | Crack pixel fraction, per-image distribution, the resulting `pos_weight` |
| Quality control | What `inspect_data.py` checks and what it found; include the contact sheet as a figure |
| Splits | Why hashing filenames rather than shuffling — leak prevention |
| Augmentation study | The survival table from Step 7, and your recommendation |
| Domain note | Every public dataset is road and concrete; the target is stone masonry, where mortar joints look like cracks |

That last row is shared with B and C, but you own the evidence for it —
if you can find even a handful of masonry photographs, put them in the
contact sheet next to the road images. The visual comparison makes the
point faster than a paragraph.

---

## Definition of done

```bash
python src/data/transforms.py                 # crack survives every draw
python scripts/inspect_data.py --root data/processed   # DATASET LOOKS USABLE
python tests/contracts_test.py --only A       # 6/6 checks passed
```

All three green, contact sheet inspected by eye, numbers posted to the
group. That is your workstream complete.
