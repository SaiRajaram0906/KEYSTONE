# Windows PowerShell. Run from the keystone-full folder:
#     .\setup.ps1
# If PowerShell blocks it, run once:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    py -3.13 -m venv .venv 2>$null
    if ($LASTEXITCODE -ne 0) { python -m venv .venv }
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip

# CPU-only PyTorch first (small download). If this fails, the next step
# installs the default build from PyPI instead.
& .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { Write-Host "CPU wheel index failed, falling back to PyPI torch" }

& .\.venv\Scripts\python.exe -m pip install -r requirements-all.txt
Write-Host ""
Write-Host "Done. Activate with:  .\.venv\Scripts\Activate.ps1"
Write-Host "Then:                 python run_tests.py"
