# Changes on top of the original crack-seg

Nothing in `src/contracts.py` or `tests/` was changed. All 22 contract checks
still pass.

## Bug fixes

### `scripts/prepare_data.py` (Workstream A): DeepCrack lost 237 of 537 images
With `--layout deepcrack`, `train_img` and `test_img` were each numbered from
`0000`, so the test images overwrote the first train images. The script still
printed "537 pairs written". Only 300 files ended up on disk.

Fix: `convert_pairs` takes a `start` index and the DeepCrack loop passes the
running count. The script now also exits with an error if the number of files
on disk differs from the number it reports.

Workstream B hit the same bug on Colab and fixed it the same way (test
numbering continues after train). Both fixes name the 537 files `0000`-`0536`,
which hash to the split B trained on: 386 train / 66 val / 85 test. Keep one
copy of this file in the shared repo.

### `src/predict.py` (Workstream C): `--compare-to` crashed
`cv2.COLOR_BGR2BGR` does not exist in OpenCV, so the before/after comparison
raised `AttributeError`. The image was already BGR, so the conversion is
removed: `vis = cv2.imread(str(later_path), cv2.IMREAD_COLOR)`.

### `src/evaluate.py` + `src/metrics.py` (Workstream C): threshold sweep stopped at 0.95
The validation sweep covered 0.05-0.95. For `loss_focal_tversky` validation
IoU was still rising at 0.95 (0.6697 at 0.90, 0.6717 at 0.95), so its best
threshold lay outside the grid. Three of the six losses (focal_tversky,
tversky, bce_dice) selected exactly 0.95. The grid now adds 0.96, 0.97, 0.98,
0.99 (`metrics.THRESHOLD_GRID`), `evaluate.py` takes `--thresholds` to
override it, and prints a warning if the selected threshold sits on either
edge of the grid. The threshold is still tuned on validation only and applied
once to test. Re-evaluating with the wider grid changes B's reported numbers;
the original table is kept in `results/b_loss_comparison/`.

## Added

- `scripts/aug_survival_experiment.py`: Workstream A write-up experiment
  (WORKSTREAM_A.md Step 7). Measures how often each augmentation fades a thin
  crack out of the image while the mask still marks it.
- `SETUP_WINDOWS.md`: setup commands for Windows PowerShell.
- `scripts/measure_vs_gt.py`: Workstream C. Measures crack width, length and
  count on the test split from the model's mask, the ground-truth mask, and
  the ground truth passed through the model's input grid (a perfect model at
  that resolution), and reports the errors in pixels.
- `scripts/eval_external.py`: Workstream C. Scores a checkpoint on another
  labelled dataset (used for the Dais et al. masonry set) exactly as
  `src.evaluate` scores DeepCrack, with the checkpoint's own threshold, and
  lists the worst images. Normalises 0/1 and 0/255 label files.
- `RUNBOOK_A_C.md`: step-by-step order for Workstreams A and C, from setup to
  report.
- `results/b_loss_comparison/`: Workstream B's six-loss comparison
  (`results.md`, `results.csv`, `loss_curves.png`) from the Colab run on the
  full 537-image DeepCrack set. Checkpoints are not included (they live in
  B's Google Drive, `crack-runs/`).
