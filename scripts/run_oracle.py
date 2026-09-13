"""Recoverability and localization of gold values without gold hints.

Usage: python -m scripts.run_oracle [--split dev]
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from sdsbench import annotations
from sdsbench.oracle import analyze, summarize

OUTPUT_PATH = Path("data/results/oracle.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", help="dev (default), locked, or all")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    gold = annotations.load_gold(split=args.split)
    rows = analyze(gold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print("=" * 78)
    print("ORACLE: value recoverability and localization without gold hints")
    print("=" * 78)
    print("\n".join(summarize(rows)))
    print()
    print(f"wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
