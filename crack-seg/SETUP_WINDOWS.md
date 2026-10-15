# Setup on Windows (PowerShell, VS Code / Antigravity)

Run each command on its own line, from the `crack-seg` folder.

## 1. Environment

```
python -m venv .venv
```
```
.\.venv\Scripts\Activate.ps1
```
If PowerShell says running scripts is disabled, run this once, then activate again:
```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
```
python -m pip install --upgrade pip
```

## 2. PyTorch

No NVIDIA GPU (`nvidia-smi` is "not recognized"):
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
With a GPU, use the command from the selector at https://pytorch.org instead.
The last step of this install can sit for 5-15 minutes on Windows while it
unpacks files. Leave it alone.

## 3. Everything else

```
pip install -r requirements.txt
```

## 4. Check it works

```
python tests/contracts_test.py
```
Expect `22/22 checks passed`.

```
python src/data/transforms.py
```
```
python keystone/interface.py
```
```
python src/losses.py
```

## 5. Workstream A experiment (added)

```
python scripts/aug_survival_experiment.py
```
Takes a few minutes and prints nothing per draw until each test finishes.
Writes `runs/aug_experiment/results.md`. See the docstring at the top of the
script for what it measures and why `transforms.py`'s own 12/12 check does
not catch the same thing.

## Git not installed?

Wherever the docs say `git clone`, open the repository page in a browser and
use **Code -> Download ZIP** instead, then extract it to the same place.
Where they say `rm -rf data/processed`, delete the folder in the file
explorer.
