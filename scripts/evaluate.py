"""Score prediction files.

Usage: python -m scripts.evaluate data/predictions/*.json [--split dev] [--by-field]
       [--confusion] [--ingredients] [--by-template] [--allow-locked]
       [--expect-id ID]

A prediction file is only scored against the gold it was produced on: the
split it records has to be the split being requested and its document set has
to be exactly the gold document set, so a locked-split run cannot be reported
as a development number by passing the wrong flag. The locked split is refused
without --allow-locked, and every run against it is appended to
docs/locked-runs.csv with its date, commit and prediction id.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from datetime import date
from pathlib import Path

from sdsbench import annotations, manifest
from sdsbench.annotations import SPLIT_LOCKED, DocumentAnnotation
from sdsbench.evaluator import (
    AXES,
    NOT_APPLICABLE,
    PASSING,
    UNDECIDABLE,
    IngredientSummary,
    Summary,
    evaluate,
    evaluate_ingredients,
)
from sdsbench.ontology import PRODUCT_FIELD_NAMES, FieldState
from sdsbench.schema import PredictionSet

OUTPUT_DIR = Path("data/results")
LOCKED_RUN_LOG = Path("docs/locked-runs.csv")
LOCKED_RUN_COLUMNS = (
    "date",
    "split",
    "system",
    "extractor",
    "commit",
    "prediction_id",
    "documents",
    "prediction_file",
)

STAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("state", ("state",)),
    ("+ value", ("state", "value")),
    ("+ page", ("state", "value", "page")),
    ("+ span", ("state", "value", "page", "span")),
    ("+ bbox", ("state", "value", "page", "span", "bbox")),
    ("+ derivation", ("state", "value", "page", "span", "bbox", "derivation")),
    ("+ support (strict joint)", AXES),
)


class LockedSplitError(RuntimeError):
    pass


class PredictionSetError(RuntimeError):
    """The prediction file does not describe the gold it is about to be scored on."""


def guard_split(documents: list[DocumentAnnotation], allow_locked: bool) -> None:
    locked = [item.document for item in documents if item.split == SPLIT_LOCKED]
    if locked and not allow_locked:
        raise LockedSplitError(
            f"{len(locked)} document(s) belong to the locked split; pass --allow-locked "
            "to score them and record the run"
        )


def guard_predictions(
    documents: list[DocumentAnnotation],
    predictions: PredictionSet,
    requested_split: str,
    allow_locked: bool,
    expect_id: str | None = None,
) -> None:
    """Refuse to score a prediction file against gold it was not produced on."""
    guard_split(documents, allow_locked)

    if expect_id is not None and predictions.prediction_id != expect_id:
        raise PredictionSetError(
            f"{predictions.system}: prediction id is {predictions.prediction_id}, "
            f"expected {expect_id}"
        )

    if requested_split == "all":
        print(
            f"note: --split all scores {predictions.system} against every annotated "
            "document; the split and document-set checks do not apply"
        )
        return

    declared = predictions.split
    if declared is None:
        raise PredictionSetError(
            f"{predictions.system}: the file records no split in its configuration; "
            "re-run the extractor so the run can be traced"
        )
    if declared != requested_split:
        raise PredictionSetError(
            f"{predictions.system}: predictions were produced on split {declared!r} "
            f"but the gold requested is {requested_split!r}"
        )

    expected = tuple(sorted(item.document for item in documents))
    actual = predictions.document_names
    if actual != expected:
        missing = [item for item in expected if item not in set(actual)]
        extra = [item for item in actual if item not in set(expected)]
        raise PredictionSetError(
            f"{predictions.system}: document set does not match split {requested_split!r} "
            f"({len(actual)} predicted, {len(expected)} in gold); "
            f"missing {missing or '-'}, unexpected {extra or '-'}"
        )


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


def record_locked_run(
    predictions: PredictionSet, path: Path, log: Path = LOCKED_RUN_LOG
) -> None:
    """Append one locked-split evaluation to the run log."""
    log.parent.mkdir(parents=True, exist_ok=True)
    exists = log.exists()
    with open(log, "a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(LOCKED_RUN_COLUMNS)
        writer.writerow(
            [
                date.today().isoformat(),
                predictions.split or "",
                predictions.system,
                predictions.configuration.get("extractor", ""),
                predictions.configuration.get("commit") or git_commit(),
                predictions.prediction_id,
                len(predictions.documents),
                path.as_posix(),
            ]
        )
    print(f"recorded locked run {predictions.prediction_id} in {log}")


def _format(rate: float | None) -> str:
    return "  n/a" if rate is None else f"{rate:5.1%}"


def print_summary(summary: Summary, predictions: PredictionSet | None = None) -> None:
    total = len(summary.instances)
    evidence_bearing = summary.evidence_bearing

    print("=" * 78)
    print(f"SYSTEM: {summary.system}")
    print("=" * 78)
    if predictions is not None:
        print(
            f"split: {predictions.split}   prediction id: {predictions.prediction_id}   "
            f"extractor: {predictions.configuration.get('extractor', '?')}"
        )
    print(f"instances: {total}   with gold evidence or premises to locate: {len(evidence_bearing)}")
    multi_role = [item for item in summary.evidence_bearing if item.target_groups > 1]
    if multi_role:
        complete = sum(item.axes["span"] in PASSING for item in multi_role)
        print(
            f"instances whose gold premise set needs more than one role: {len(multi_role)}   "
            f"fully located: {complete}"
        )
    undescribed = [
        item
        for item in summary.instances
        if item.gold_requires_evidence and not item.premise_roles_described
    ]
    if undescribed:
        print(
            f"derived instances whose gold premises the registry cannot group by role: "
            f"{len(undescribed)} (scored as one group, listed for annotation review)"
        )
    print()

    print(f"{'axis':<12}{'rate':>8}{'satisfied':>12}{'undecidable':>13}{'applicable':>12}")
    print("-" * 57)
    for axis in AXES:
        stats = summary.axis(axis)
        print(
            f"{axis:<12}{_format(stats.rate):>8}{stats.satisfied:>12}"
            f"{stats.undecidable:>13}{stats.applicable:>12}"
        )
    print()

    print(f"{'cumulative conjunction':<30}{'all instances':>16}{'evidence-bearing':>18}")
    print("-" * 64)
    for label, axes in STAGES:
        print(
            f"  {label:<28}{summary.conjunction_rate(axes):>15.1%}"
            f"{summary.conjunction_rate(axes, evidence_bearing):>18.1%}"
        )
    print()

    verbatim, cited = summary.verbatim_rate()
    print(f"strict state agreement             {summary.axis('state').rate:6.1%}")
    print(f"lenient state agreement            {summary.lenient_state_rate():6.1%}")
    print(f"joint, all instances               {summary.joint_rate():6.1%}")
    print(f"joint, evidence-bearing only       {summary.joint_rate(evidence_bearing):6.1%}")
    print(f"instances with an undecidable axis {summary.undecidable_count():6d}")
    print(f"mean span coverage (evidence set)  {summary.mean('span_coverage'):6.3f}")
    print(f"mean span IoU (evidence set)       {summary.mean('span_iou'):6.3f}")
    print(f"mean span dilution (evidence set)  {summary.mean('span_dilution'):6.2f}x")
    print(f"mean box coverage (evidence set)   {summary.mean('bbox_coverage'):6.3f}")
    print(f"mean box IoU (evidence set)        {summary.mean('bbox_iou'):6.3f}")
    print(f"mean box area dilution             {summary.mean('bbox_dilution'):6.2f}x")
    print(f"citations verbatim in the document {verbatim}/{cited}")
    print()


def print_by_field(summary: Summary) -> None:
    columns = ("state", "value", "page", "span", "bbox", "derivation", "support")
    print(f"{'field':<22}" + "".join(f"{name[:6]:>8}" for name in columns) + f"{'joint':>8}")
    print("-" * (22 + 8 * (len(columns) + 1)))
    for field in PRODUCT_FIELD_NAMES:
        subset = [item for item in summary.instances if item.field == field]
        if not subset:
            continue
        cells = [_format(summary.axis(axis, subset).rate) for axis in columns]
        joint = summary.joint_rate(subset)
        print(f"{field:<22}" + "".join(f"{cell:>8}" for cell in cells) + f"{joint:>8.1%}")
    print()


def print_state_confusion(summary: Summary) -> None:
    states = list(FieldState)
    print("state confusion (rows: gold, columns: predicted)")
    header = f"{'':<18}" + "".join(f"{state.value[:12]:>14}" for state in states) + f"{'none':>8}"
    print(header)
    for gold_state in states:
        row = [item for item in summary.instances if item.gold_state is gold_state]
        cells = [
            sum(1 for item in row if item.predicted_state is predicted)
            for predicted in states
        ]
        missing = sum(1 for item in row if item.predicted_state is None)
        print(
            f"{gold_state.value:<18}"
            + "".join(f"{cell:>14}" for cell in cells)
            + f"{missing:>8}"
        )
    print()


def print_by_template(summary: Summary, entries: dict[str, manifest.ManifestEntry]) -> None:
    print("joint rate by template family (documents in this evaluation)")
    print(f"{'template family':<24}{'documents':>10}{'instances':>10}{'joint':>8}")
    print("-" * 52)
    families = manifest.template_families(entries)
    documents_here = {item.document for item in summary.instances}
    for family, documents in sorted(families.items()):
        selected = [document for document in documents if document in documents_here]
        if not selected:
            continue
        subset = [item for item in summary.instances if item.document in selected]
        print(
            f"{family:<24}{len(selected):>10}{len(subset):>10}{summary.joint_rate(subset):>8.1%}"
        )
    print()


def print_ingredients(scores: IngredientSummary) -> None:
    print("ingredient records (name + CAS or CAS state + concentration)")
    print(f"  gold rows {scores.gold_rows}, predicted rows {scores.predicted_rows}, "
          f"matched by CAS or name {scores.matched}")
    print(f"  key recall (row found at all)        {_format(scores.key_recall)}")
    for attribute, label in (
        ("name_ok", "name correct"),
        ("cas_ok", "CAS / CAS state correct"),
        ("concentration_ok", "concentration correct"),
        ("evidence_ok", "row evidence located"),
        ("verbatim", "row citation verbatim"),
    ):
        print(f"  among matched: {label:<24}{_format(scores.attribute_rate(attribute))}")
    print(f"  complete records                     {scores.records_exact}/{scores.gold_rows} gold, "
          f"{scores.records_exact}/{scores.predicted_rows} predicted")
    print(f"  record precision {_format(scores.precision)}   record recall {_format(scores.recall)}")
    print(f"  complete records with located evidence: {scores.records_with_evidence}")
    agreed, documents = scores.state_agreement
    print(f"  ingredient block state agreement     {agreed}/{documents} documents")
    print()


def write_instances(summary: Summary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "document",
                "field",
                "gold_state",
                "predicted_state",
                "gold_value",
                "predicted_value",
                "gold_derivation",
                "predicted_derivation",
                "gold_rule",
                "predicted_rule",
                "evidence_role",
                "gold_pages",
                "predicted_page",
                "gold_evidence",
                "predicted_evidence",
                *AXES,
                "span_coverage",
                "span_iou",
                "span_dilution",
                "bbox_coverage",
                "bbox_iou",
                "bbox_dilution",
                "verbatim",
                "verbatim_fuzzy_score",
                "joint_correct",
                "confidence",
                "gold_requires_evidence",
                "gold_ambiguous",
                "bbox_provenance",
                "target_groups",
                "groups_located",
                "premise_roles_described",
                "support_note",
                "note",
            ]
        )
        for item in summary.instances:
            writer.writerow(
                [
                    item.document,
                    item.field,
                    item.gold_state.value,
                    item.predicted_state.value if item.predicted_state else "",
                    item.gold_value or "",
                    item.predicted_value or "",
                    item.gold_derivation,
                    item.predicted_derivation,
                    item.gold_rule or "",
                    item.predicted_rule or "",
                    item.evidence_role,
                    ",".join(str(page) for page in item.gold_pages),
                    item.predicted_page or "",
                    item.gold_evidence.replace("\n", " "),
                    item.predicted_evidence.replace("\n", " "),
                    *[item.axes[axis] for axis in AXES],
                    f"{item.span_coverage:.4f}",
                    f"{item.span_iou:.4f}",
                    f"{item.span_dilution:.2f}",
                    f"{item.bbox_coverage:.4f}",
                    f"{item.bbox_iou:.4f}",
                    f"{item.bbox_dilution:.2f}",
                    item.verbatim,
                    f"{item.verbatim_fuzzy_score:.1f}",
                    item.joint_correct,
                    item.confidence,
                    item.gold_requires_evidence,
                    item.gold_ambiguous,
                    item.bbox_provenance,
                    item.target_groups,
                    item.groups_located,
                    item.premise_roles_described,
                    item.support_note,
                    item.note,
                ]
            )


def write_ingredient_records(scores: IngredientSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "document",
                "gold_index",
                "predicted_index",
                "key",
                "gold_name",
                "predicted_name",
                "gold_concentration",
                "predicted_concentration",
                "name_ok",
                "cas_ok",
                "concentration_ok",
                "evidence_ok",
                "verbatim",
                "record_ok",
                "record_with_evidence_ok",
            ]
        )
        for match in scores.matches:
            writer.writerow(
                [
                    match.document,
                    match.gold_index,
                    match.predicted_index,
                    match.key,
                    match.gold_name or "",
                    match.predicted_name or "",
                    match.gold_concentration or "",
                    match.predicted_concentration or "",
                    match.name_ok,
                    match.cas_ok,
                    match.concentration_ok,
                    match.evidence_ok,
                    match.verbatim,
                    match.record_ok,
                    match.record_with_evidence_ok,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+", type=Path)
    parser.add_argument("--split", default="dev", help="dev (default), locked, or all")
    parser.add_argument("--allow-locked", action="store_true")
    parser.add_argument("--expect-id", default=None, help="refuse a prediction file with another id")
    parser.add_argument("--by-field", action="store_true")
    parser.add_argument("--confusion", action="store_true")
    parser.add_argument("--ingredients", action="store_true")
    parser.add_argument("--by-template", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    gold = annotations.load_gold(split=args.split)
    guard_split(gold, args.allow_locked)
    entries = manifest.load_manifest() if args.by_template else {}

    for path in args.predictions:
        prediction_set = PredictionSet.load(path)
        guard_predictions(gold, prediction_set, args.split, args.allow_locked, args.expect_id)
        if prediction_set.split == SPLIT_LOCKED:
            record_locked_run(prediction_set, path)
        summary = evaluate(gold, prediction_set)
        print_summary(summary, prediction_set)
        if args.by_field:
            print_by_field(summary)
        if args.confusion:
            print_state_confusion(summary)
        if args.by_template:
            print_by_template(summary, entries)
        if args.ingredients:
            scores = evaluate_ingredients(gold, prediction_set)
            print_ingredients(scores)
            write_ingredient_records(scores, args.output / f"{summary.system}_ingredients.csv")

        write_instances(summary, args.output / f"{summary.system}_instances.csv")


if __name__ == "__main__":
    main()
