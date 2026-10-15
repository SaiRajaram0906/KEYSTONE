#!/usr/bin/env bash
# Linux / macOS. Run from the keystone-full folder:   bash setup.sh
set -euo pipefail

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-all.txt

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Then:                 python run_tests.py"
