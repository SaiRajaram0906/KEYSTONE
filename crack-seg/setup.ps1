# Windows PowerShell, run from this folder:  .\setup.ps1
# If PowerShell blocks it, run once:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
$ErrorActionPreference = "Stop"
if (-not (Test-Path ".venv")) {
    py -3.13 -m venv .venv 2>$null
    if ($LASTEXITCODE -ne 0) { python -m venv .venv }
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { Write-Host "CPU wheel index failed, falling back to PyPI torch" }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
Write-Host "`nDone. Activate:  .\.venv\Scripts\Activate.ps1`nThen:  streamlit run app.py"
