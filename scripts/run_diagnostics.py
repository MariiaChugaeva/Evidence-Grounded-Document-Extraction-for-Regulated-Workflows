"""Gold-value / gold-evidence / gold-verify diagnostics for the rules system.

Usage: python -m scripts.run_diagnostics [--layer native|ocr] [--split dev]
       [--predictions data/predictions/rules-native-rows.json]
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from sdsbench import annotations, diagnostics, textlayer
from sdsbench.schema import PredictionSet

OUTPUT_DIR = Path("data/results")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", choices=[textlayer.NATIVE, textlayer.OCR], default=textlayer.NATIVE)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--allow-locked", action="store_true")
    parser.add_argument("--predictions", type=Path, default=None,
                        help="pipeline predictions for the reference column (default: rules-<layer>-rows)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from scripts.evaluate import guard_split

    gold = annotations.load_gold(split=args.split)
    guard_split(gold, args.allow_locked)

    predictions_path = args.predictions or Path(f"data/predictions/rules-{args.layer}-rows.json")
    pipeline = PredictionSet.load(predictions_path) if predictions_path.exists() else None

    results = diagnostics.run_probes(gold, args.layer, pipeline)
    suffix = "" if args.split == "dev" else f"-{args.split}"
    output = args.output or OUTPUT_DIR / f"diagnostics-{args.layer}{suffix}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))

    print("=" * 78)
    print(f"DIAGNOSTICS: rules system, layer {args.layer}, split {args.split}")
    print("=" * 78)
    print("\n".join(diagnostics.summarize(results)))
    print()
    print(f"wrote {len(results)} rows -> {output}")


if __name__ == "__main__":
    main()
