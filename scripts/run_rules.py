"""Run the rules baseline.

Usage: python -m scripts.run_rules [--layer native|ocr] [--layout rows|linear] [--split dev]

The prediction set records the extractor version and the git commit so that a
run against the locked split can be traced to the exact rules it used.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from sdsbench import annotations, textlayer
from sdsbench.extractors import rules
from sdsbench.schema import PredictionSet

OUTPUT_DIR = Path("data/predictions")


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(documents: list[str], layer: str, layout: str, split: str = "dev") -> PredictionSet:
    predictions = []
    for document in documents:
        try:
            predictions.append(rules.extract_document(document, layer=layer, layout=layout))
        except Exception as error:
            predictions.append(
                rules.DocumentPrediction(document=document, fields={}, error=str(error))
            )
    return PredictionSet(
        system=f"rules-{layer}-{layout}",
        configuration={
            "layer": layer,
            "layout": layout,
            "extractor": rules.EXTRACTOR_VERSION,
            "commit": git_commit(),
            "split": split,
        },
        documents=predictions,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", choices=[textlayer.NATIVE, textlayer.OCR])
    parser.add_argument("--layout", choices=[rules.LAYOUT_ROWS, rules.LAYOUT_LINEAR])
    parser.add_argument("--split", default="dev", help="dev (default), locked, or all")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    gold = annotations.load_gold(split=args.split)
    documents = [item.document for item in gold]

    layers = [args.layer] if args.layer else [textlayer.NATIVE, textlayer.OCR]
    layouts = [args.layout] if args.layout else [rules.LAYOUT_ROWS, rules.LAYOUT_LINEAR]

    for layer in layers:
        for layout in layouts:
            prediction_set = run(documents, layer, layout, args.split)
            suffix = "" if args.split == "dev" else f"-{args.split}"
            path = args.output / f"{prediction_set.system}{suffix}.json"
            prediction_set.save(path)
            print(f"{prediction_set.system}: {len(prediction_set.documents)} documents -> {path}")


if __name__ == "__main__":
    main()
