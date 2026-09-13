"""Resolve source annotations into data/annotations/gold.json."""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from sdsbench import annotations
from sdsbench.ontology import FieldState


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    documents, problems = annotations.build_gold()

    print(f"resolved {len(documents)} documents -> {annotations.GOLD_PATH}")

    if args.report:
        states: Counter[str] = Counter()
        derivations: Counter[str] = Counter()
        ambiguous_spans = 0
        ocr_missing = 0
        total_spans = 0
        premise_spans = 0
        unverifiable: list[str] = []
        for annotation in documents:
            for name, field in annotation.fields.items():
                states[field.state.value] += 1
                derivations[field.derivation.value] += 1
                for resolved in field.resolved_evidence + field.resolved_premises:
                    total_spans += 1
                    if resolved.occurrences_on_page > 1:
                        ambiguous_spans += 1
                    if resolved.ocr_bbox is None:
                        ocr_missing += 1
                        print(
                            f"  no OCR anchor: {annotation.document}/{name} "
                            f"p{resolved.page} {resolved.text[:60]!r}"
                        )
                premise_spans += len(field.resolved_premises)
                if annotations.premise_verifiable(field) is False:
                    unverifiable.append(f"{annotation.document}/{name} ({field.rule})")
        print("\nstate distribution:")
        for state in FieldState:
            print(f"  {state.value:<16} {states.get(state.value, 0)}")
        print("\nderivation distribution:")
        for kind, count in sorted(derivations.items()):
            print(f"  {kind:<20} {count}")
        print(f"\nevidence spans: {total_spans} (of which premises of derived fields: {premise_spans})")
        print(f"  spans whose text occurs more than once on its page: {ambiguous_spans}")
        print(f"  spans with no OCR anchor: {ocr_missing}")
        print(f"\nderived fields whose gold premises the rule registry cannot verify: {len(unverifiable)}")
        for item in unverifiable:
            print(f"  {item}")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print("\nall annotations resolve cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
