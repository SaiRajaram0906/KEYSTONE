# Work Split — three parallel workstreams

Three people, three parts, one integration. The parts are separated along
the boundaries defined in `src/contracts.py`, so each can be built and
tested without waiting for the other two.

**The rule that makes this work:** you may rewrite anything you own. You
may not change `src/contracts.py` without all three agreeing, because
changing it breaks the other two by definition.

---

## The split

```
   [A] DATA  ─────────▶  [B] MODEL  ─────────▶  [C] MEASURE  ────▶ KEYSTONE
    what goes in          what learns            what comes out
```

| | Workstream A | Workstream B | Workstream C |
|---|---|---|---|
| **Owns** | `src/data/dataset.py`<br>`src/data/transforms.py`<br>`scripts/prepare_data.py` | `src/models/unet.py`<br>`src/losses.py`<br>`src/train.py`<br>`configs/` | `src/metrics.py`<br>`src/evaluate.py`<br>`src/predict.py`<br>`keystone/interface.py` |
| **Delivers** | batches that satisfy A→B | logits + checkpoints that satisfy B→C | masks + measurements that satisfy C→KEYSTONE |
| **Tests with** | `--only A` | `--only B` | `--only C` |
| **Blocked by** | nobody | nobody (uses `make_fake_dataset`) | nobody (uses `save_fake_checkpoint`) |

Nobody is blocked. That is the point of `tests/fakes.py`.

---

## Workstream A — data

**The job.** Get real crack data into the pipeline and make sure what
comes out the other end is correct.

1. Download a public dataset (links and licence notes are in
   `scripts/prepare_data.py`). DeepCrack is the best starting point.
2. Convert it: `python scripts/prepare_data.py --src data/raw/DeepCrack --layout deepcrack`
3. Confirm the class balance it prints. If crack pixels are above 10% or
   below 0.1%, something is wrong — masks may be inverted.
4. Tune the augmentation. Cracks are thin; the failure mode is an
   augmentation that erases a one-pixel crack while leaving the mask
   intact, teaching the model that blank wall means crack.

**How you know you're done**

```bash
python tests/contracts_test.py --only A
python src/data/transforms.py      # cracks must survive all 12 draws
```

**The interesting part to write up.** Measure how often each augmentation
strength destroys a thin crack. That is a real experiment and nobody
expects it.

---

## Workstream B — model and training

**The job.** Train a model, and find out which loss actually handles the
class imbalance.

1. Verify the architecture: `python src/models/unet.py`
2. Verify the losses: `python src/losses.py` — the all-empty column is
   the whole story.
3. Train against fake data first so you are not blocked:
   ```python
   from tests.fakes import make_fake_dataset
   make_fake_dataset("data/processed", n=200, size=160)
   ```
   Then `python -m src.train --config configs/default.yaml`
4. Once A delivers real data, run the loss comparison. Six configs, one
   changed line each.

**How you know you're done**

```bash
python tests/contracts_test.py --only B
```

**The interesting part to write up.** The loss comparison table is the
core result of the whole project. Report IoU, Dice, precision and recall
for all six. The expected story: BCE under-segments badly, Dice fixes it,
Tversky with β>α buys recall at the cost of precision.

---

## Workstream C — measurement and integration

**The job.** Turn a mask into a number, and make KEYSTONE able to call it.

1. Build against a fake checkpoint from day one:
   ```python
   from tests.fakes import save_fake_checkpoint
   save_fake_checkpoint("runs/fake/best.pt")
   ```
2. Verify the width measurement is calibrated:
   `python keystone/interface.py` — odd widths must come out exact.
3. Build the threshold sweep in `evaluate.py`. Tune on validation, apply
   once to test. Tuning on test and reporting test is leakage.
4. Build the overlay output in `predict.py`. Looking at overlays catches
   failures that IoU hides — shadows read as cracks, mortar joints read
   as cracks.

**How you know you're done**

```bash
python tests/contracts_test.py --only C
```

**The interesting part to write up.** The ±1 px quantisation error in the
width measurement, and what it means for the minimum detectable change.
At 0.5 mm/px you cannot honestly claim to detect anything below about
1 mm of widening. Stating that is worth more than hiding it.

---

## Integration

Because the contracts are tested continuously, integration is not a day —
it is a command.

```bash
python tests/contracts_test.py        # all 22 checks
python -m src.train --config configs/default.yaml
python -m src.evaluate --checkpoint runs/default/best.pt
python -m src.predict --checkpoint runs/default/best.pt --input data/processed/images --mm-per-px 0.8
```

If all three sections passed independently, this sequence works. If it
does not, the contract test that should have caught it is missing — add
it, rather than patching the symptom.

### Integration checkpoints

Meet at these three points rather than at the end.

| When | What must be true |
|---|---|
| **Checkpoint 1** | All three run `contracts_test.py --only <yours>` green, using fakes. Nobody has real data yet. This proves the boundaries hold. |
| **Checkpoint 2** | A has real data converted. B trains on it and reports a first IoU. C evaluates that checkpoint and produces overlays. |
| **Checkpoint 3** | Loss comparison complete, threshold tuned, measurements calibrated, README results filled in. |

---

## Splitting the report the same way

| Section | Owner |
|---|---|
| Dataset, class imbalance analysis, augmentation study | A |
| Architecture, loss comparison, training curves, ablations | B |
| Metrics, threshold analysis, measurement calibration, failure cases | C |
| Limitations and the domain-gap discussion | all three — it is the most important section |

---

## The limitation none of you should paper over

Every public crack dataset is **road and concrete**. KEYSTONE's target is
**stone masonry**, where mortar joints look exactly like cracks: dark,
linear, high contrast.

Expect the model to flag mortar joints. Do not present road-surface
metrics as though they transfer. Either show the failure cases honestly,
or hand-label 50–100 masonry images and fine-tune, reporting both.

The second option is a strong addition if you have time. The first takes
a paragraph and is still the right thing to do.

---

## Later: plugging into KEYSTONE

KEYSTONE imports one module and calls two functions.

```python
from keystone.interface import CrackSegmenter, compare_captures

seg = CrackSegmenter("runs/default/best.pt")
mask_march = seg.segment("capture_2026_03/north_wall.jpg")
mask_sept  = seg.segment("capture_2026_09/north_wall.jpg")

delta = compare_captures(mask_march, mask_sept, mm_per_px=0.8)
if delta.d_max_width > 1.0:
    raise_alert(delta)
```

KEYSTONE must supply two things, and both come free from its
photogrammetry stage:

- **`mm_per_px`**, from the surveyed scale markers in the photograph.
  Without it every number is in pixels and is not comparable between
  captures.
- **Aligned masks.** `compare_captures` assumes the two are spatially
  registered. In KEYSTONE both captures are registered to the same ground
  control points during reconstruction, so this is already true.

Workstream C owns that boundary. Keep it stable and the eventual
integration is an import, not a rewrite.
