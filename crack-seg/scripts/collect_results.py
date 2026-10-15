"""Collect every run's test metrics into one comparison table.

    python scripts/collect_results.py                 # all runs under runs/
    python scripts/collect_results.py --pattern loss_ # just the loss comparison

Reads runs/<name>/test_metrics.json, which src/evaluate.py writes. Prints a
table sorted by IoU and saves it as runs/results.md and runs/results.csv —
the numbers for slide 7 of the project deck.

A run without test_metrics.json is listed as NOT EVALUATED rather than
silently skipped, so a missing experiment is obvious.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

COLS = ("iou", "dice", "precision", "recall")


def load(run_dir: Path) -> dict | None:
    p = run_dir / "test_metrics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    m = d.get("model", {})
    base = d.get("all_background_baseline", {})
    return {
        "run": run_dir.name,
        "threshold": d.get("threshold"),
        **{c: m.get(c) for c in COLS},
        "n_test": m.get("n_images"),
        "baseline_pixacc": base.get("pixel_accuracy"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--pattern", default="", help="only runs whose name contains this")
    args = ap.parse_args()

    root = Path(args.runs)
    if not root.is_dir():
        raise SystemExit(f"{root} not found — train something first")

    rows, missing = [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and args.pattern in p.name):
        r = load(d)
        (rows if r else missing).append(r or d.name)

    if not rows:
        raise SystemExit("no evaluated runs found — run src.evaluate on each checkpoint")

    rows.sort(key=lambda r: r["iou"] or 0.0, reverse=True)

    head = f"{'run':<24}{'IoU':>8}{'Dice':>8}{'Prec':>8}{'Recall':>8}{'thr':>6}{'n':>6}"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['run']:<24}{r['iou']:>8.4f}{r['dice']:>8.4f}{r['precision']:>8.4f}"
              f"{r['recall']:>8.4f}{r['threshold']:>6.2f}{r['n_test']:>6}")
    for m in missing:
        print(f"{m:<24}{'NOT EVALUATED':>30}")

    best = rows[0]
    top_recall = max(rows, key=lambda r: r["recall"] or 0.0)
    print(f"\nbest IoU     : {best['run']} ({best['iou']:.4f})")
    print(f"best recall  : {top_recall['run']} ({top_recall['recall']:.4f})")
    if rows[0].get("baseline_pixacc") is not None:
        print(f"do-nothing baseline pixel accuracy: {rows[0]['baseline_pixacc']:.4f} "
              "(the number that shows why accuracy is not reported)")

    md = root / "results.md"
    with md.open("w") as f:
        f.write("| Run | IoU | Dice | Precision | Recall |\n|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['run']} | {r['iou']:.4f} | {r['dice']:.4f} | "
                    f"{r['precision']:.4f} | {r['recall']:.4f} |\n")
    with (root / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {md} and {root / 'results.csv'}")


if __name__ == "__main__":
    main()
