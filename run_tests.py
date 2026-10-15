"""Run both acceptance suites with the right working directory.

    python run_tests.py

KEYSTONE : 32 checks (contracts, Preservation, Geometry, Decision, integration)
crack-seg: 22 checks (Data, Model, Measurement boundaries)

The two projects each have a top-level `tests` package, so they are run as
separate processes from their own folders instead of from one interpreter.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SUITES = (
    ("KEYSTONE  (P + G + D)", ROOT / "keystone-core", ["-m", "tests.contracts_test"]),
    ("crack-seg (A + B + C)", ROOT / "crack-seg", ["tests/contracts_test.py"]),
)


def main() -> int:
    failed = []
    for name, cwd, args in SUITES:
        print(f"\n{'#' * 70}\n# {name}\n{'#' * 70}")
        rc = subprocess.run([sys.executable, *args], cwd=cwd).returncode
        if rc != 0:
            failed.append(name)

    print("\n" + "=" * 70)
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    print("both suites passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
