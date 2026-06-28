"""
Aggregate slice-fusion results into a single table.

Scans {results}/{variant}/summary.json and writes:
    results/ablation_table.csv   -- machine-readable
    results/ablation_table.md    -- paste-ready Markdown for the paper

Run from the project root:
    python analysis/make_tables.py
    python analysis/make_tables.py --results results
"""
import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def _row_from_summary(variant, s):
    def mean(key):
        return s.get(key, {}).get("mean") if isinstance(s.get(key), dict) else s.get(key)

    return {
        "variant": variant,
        "dice": mean("dice"),
        "dice_fg": mean("dice_fg"),
        "hd95": mean("hd95"),
        "num_slices": s.get("num_slices"),
        "best_epoch": (s.get("training", {}) or {}).get("best_epoch"),
    }


def collect(results_dir: Path):
    rows = []
    for summary_path in sorted(results_dir.glob("*/summary.json")):
        variant = summary_path.parent.name
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                s = json.load(f)
            rows.append(_row_from_summary(variant, s))
        except Exception as e:
            print(f"  skip {summary_path}: {e}")
    return rows


def write_csv(rows, path):
    if not rows:
        return
    fields = ["variant", "dice", "dice_fg", "hd95", "num_slices", "best_epoch"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_md(rows, path):
    if not rows:
        return
    header = "| Variant | Dice | Dice_FG | HD95 | #Slices | Best Epoch |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {_fmt(r['dice'])} | {_fmt(r['dice_fg'])} | "
            f"{_fmt(r['hd95'], 2)} | {r['num_slices']} | {r['best_epoch']} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate slice-fusion results")
    parser.add_argument("--results", type=str, default="results")
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir

    rows = collect(results_dir)
    if not rows:
        print(f"No summary.json found under {results_dir}/*/. "
              f"Run training/ablation first.")
        sys.exit(0)

    # Sort by variant name: S0, S1, S2 fall into natural order.
    rows.sort(key=lambda r: r["variant"])

    csv_path = results_dir / "ablation_table.csv"
    md_path = results_dir / "ablation_table.md"
    write_csv(rows, csv_path)
    write_md(rows, md_path)

    print("\n".join(open(md_path, encoding="utf-8").read().splitlines()))
    print(f"\nWrote:\n  {csv_path}\n  {md_path}")


if __name__ == "__main__":
    main()