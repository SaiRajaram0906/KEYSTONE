# Workstreams A and C: what to do, in order

Owner of A (data) and C (measurement): REC. B (model) results are in
`results/b_loss_comparison/`.

Windows PowerShell, from the `crack-seg` folder, one line at a time.

---

## Phase 1: Setup (once)

```
python -m venv .venv
```
```
.\.venv\Scripts\Activate.ps1
```
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
```
pip install -r requirements.txt
```
```
python tests/contracts_test.py
```
Expect `22/22 checks passed`.

## Phase 2: Real data (Workstream A)

1. https://github.com/yhlleo/DeepCrack -> Code -> Download ZIP -> extract.
2. Copy the extracted folder to `data\raw\` and rename it `DeepCrack`.
   Inside `dataset\`, extract any `.zip` in place.
3. Find the folder that holds `train_img`:
```
Get-ChildItem -Recurse -Directory -Filter train_img data\raw | Select-Object FullName
```
4. Convert. Use that folder, without the `\train_img` part:
```
python scripts/prepare_data.py --src data/raw/DeepCrack/dataset/DeepCrack --layout deepcrack --clean
```
Expect `537 pairs written`.
5. Check the split matches B's training run:
```
python -c "from src.data.dataset import discover_pairs, split_pairs; s=split_pairs(discover_pairs('data/processed')); print({k:len(v) for k,v in s.items()})"
```
Must print `{'train': 386, 'val': 66, 'test': 85}`.
6. Inspect:
```
python scripts/inspect_data.py --root data/processed --sheet data/contact_sheet.png
```
Expect `pairs: 537` and `DATASET LOOKS USABLE`. Open `data/contact_sheet.png`:
red must sit on the dark crack lines.
7. ```
   python tests/contracts_test.py --only A
   ```
8. Write-up experiment (a few minutes):
```
python scripts/aug_survival_experiment.py
```
Table in `runs/aug_experiment/results.md`.

## Phase 3: Real model (Workstream C)

1. Get B's checkpoints from their Google Drive (`crack-runs/`) and place them at
   `runs\loss_focal_tversky\best.pt` and `runs\loss_dice\best.pt`.
2. Reproduce B's score:
```
python -m src.evaluate --checkpoint runs/loss_focal_tversky/best.pt --data-root data/processed
```
Test IoU should be about 0.649 (B's table: 0.6493). Note: this re-tunes the
threshold on val and writes it back into `best.pt`; B's Drive copy is the backup.
3. Copy the 85 test images to their own folder:
```
python -c "import shutil,os; from src.data.dataset import discover_pairs, split_pairs; os.makedirs('data/test_images',exist_ok=True); [shutil.copy(i,'data/test_images') for i,m in split_pairs(discover_pairs('data/processed'))['test']]"
```
4. Overlays and measurements:
```
python -m src.predict --checkpoint runs/loss_focal_tversky/best.pt --input data/test_images --outdir predictions/focal_tversky --mm-per-px 0.8
```
Expect `85 image(s) processed`. Open the `*_overlay.png` files and note where
the prediction is thicker than the crack, or fires on non-cracks.

## Phase 4: Still open after Phase 3

| Item | Owner | Notes |
|---|---|---|
| Threshold sweep stops at 0.95 | C | Fixed: grid now reaches 0.99. Re-run `src.evaluate` on all six runs, then `scripts/collect_results.py`, then `scripts/measure_vs_gt.py`. Tell B: it changes the reported numbers. |
| Thin-crack fade | A + B | `aug_survival_experiment.py` shows 1 px cracks fade in ~93% of medium-pipeline draws, mostly from resizing to 320 and blur. Fixing it changes `image_size` (B's config). Team decision. |
| Masonry failure cases | C | Done with `scripts/eval_external.py` on the Dais et al. masonry set (github.com/dimitrisdais/crack_detection_CNN_masonry, GPL-3.0): `--images` = `crack_detection_224_images`, `--masks` = `crack_detection_224_masks`, `--name masonry`. Overlays from `src.predict --outdir predictions/masonry`. |
| Report sections | A, C | A: dataset, class imbalance, QC (contact sheet), splits, augmentation study. C: metrics, threshold analysis, width calibration (+/-1 px), failure cases. |
