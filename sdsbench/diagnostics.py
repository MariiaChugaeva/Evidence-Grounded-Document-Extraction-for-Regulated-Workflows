"""Diagnostics: which stage breaks evidence-grounded extraction?

Three probes give the system one part of the answer and measure the rest,
building a normal ``FieldPrediction`` and scoring it with the standard
evaluator so the numbers line up with the pipeline's axes.

gold-value      the value (or state, or product form) is known: can the
                system locate evidence for it? Scored on page / span / bbox
                and gold-free support.
gold-evidence   the gold span is known: can the system read state and value
                off it? Scored on state / value and support, citing the gold
                span itself.
gold-verify     the gold pair is known: does the gold-free verifier accept
                it? Scored on support only; an ``undecidable`` here is a gap
                in the cue lexicon or the rule registry, not a system error.

The rules system is probed through the same helpers the extractor uses
(label lists, value-slot parsers, cue lexicons), without changing
``extractors/rules.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import textlayer
from .annotations import DocumentAnnotation, FieldAnnotation
from .derivation import RULE_COMPOSITION_FORM, premise_form
from .evaluator import (
    DEFAULT_THRESHOLDS,
    NOT_APPLICABLE,
    PASSING,
    SATISFIED,
    InstanceResult,
    Thresholds,
    evaluate_instance,
)
from .extractors import rules
from .ontology import (
    PRODUCT_FIELD_NAMES,
    FieldState,
    ProductForm,
    classify_form_cue,
)
from .schema import EvidenceSpan, FieldPrediction, PredictionSet


@dataclass(frozen=True)
class FieldRules:
    labels: tuple[str, ...]
    prior: float
    parser: Callable[[str], str | None]
    cues: tuple[tuple[str, FieldState], ...] = ()
    trim: Callable[[list[textlayer.Token]], list[textlayer.Token]] | None = None


FIELD_RULES: dict[str, FieldRules] = {
    "product_name": FieldRules(rules.PRODUCT_NAME_LABELS, 0.9, rules._parse_free_text),
    "supplier_name": FieldRules(
        rules.SUPPLIER_LABELS, 0.85, rules._parse_company, trim=rules._trim_company_name
    ),
    "product_cas_number": FieldRules(rules.CAS_LABELS, 0.9, rules._parse_cas),
    "chemical_formula": FieldRules(rules.FORMULA_LABELS, 0.85, rules._parse_formula),
    "molecular_weight": FieldRules(rules.WEIGHT_LABELS, 0.9, rules._parse_weight),
    "flash_point": FieldRules(
        rules.FLASH_POINT_LABELS, 0.9, rules._parse_flash_point, rules.FLASH_POINT_CUES
    ),
}

IDENTIFIER_FIELDS = ("product_cas_number", "chemical_formula", "molecular_weight")


# ---------------------------------------------------------------------------
# Citation helpers
# ---------------------------------------------------------------------------


def span_from_char_range(page: textlayer.Page, start: int, end: int) -> EvidenceSpan:
    tokens = page.tokens_in_char_range(start, end)
    box = textlayer.union_bbox([token.bbox for token in tokens])
    return EvidenceSpan(
        page=page.page_number,
        text=page.text[start:end].strip(),
        bbox=list(box) if box else None,
        bbox_provenance="system",
    )


def _line_bounds(text: str, position: int) -> tuple[int, int]:
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    return start, (len(text) if end == -1 else end)


def cite_line_around(page: textlayer.Page, char_start: int, char_end: int) -> EvidenceSpan:
    """The reading-order line holding a hit, plus the preceding line when it is a bare label."""
    start, end = _line_bounds(page.text, char_start)
    end = max(end, _line_bounds(page.text, max(char_end - 1, char_start))[1])
    if start > 0:
        previous_start, previous_end = _line_bounds(page.text, start - 1)
        previous = page.text[previous_start:previous_end].strip()
        if previous.endswith(":") and ":" not in page.text[start:end]:
            start = previous_start
    return span_from_char_range(page, start, end)


def locate_value(document_text: textlayer.DocumentText, values: list[str]) -> EvidenceSpan | None:
    """First exact occurrence of any accepted value, cited with its line. No gold hints."""
    for page in document_text.pages:
        for value in values:
            if not value:
                continue
            anchors = textlayer.find_all_in_page(page, value)
            if anchors:
                anchor = anchors[0]
                return cite_line_around(page, anchor.char_start, anchor.char_end)
    return None


def locate_form_declaration(
    document_text: textlayer.DocumentText, form: ProductForm
) -> EvidenceSpan | None:
    for page in document_text.pages:
        for row in textlayer.rows(page):
            if classify_form_cue(row.text) is form:
                return rules._row_evidence(page.page_number, row)
    return None


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def _derived_prediction(
    field_name: str, gold: FieldAnnotation, form: ProductForm, premises: list[EvidenceSpan]
) -> FieldPrediction:
    if field_name == "substance_or_mixture":
        return FieldPrediction(
            state=FieldState.PRESENT,
            value=form.value,
            derivation="DERIVED",
            rule=gold.rule,
            premises=premises,
        )
    return FieldPrediction(
        state=FieldState.NOT_APPLICABLE,
        derivation="DERIVED",
        rule=gold.rule,
        premises=premises,
    )


def probe_gold_value(
    annotation: DocumentAnnotation,
    field_name: str,
    gold: FieldAnnotation,
    document_text: textlayer.DocumentText,
) -> FieldPrediction | None:
    """Value / state / form known; the system must find evidence for it."""
    if gold.state is FieldState.NOT_STATED:
        return None

    if gold.derivation.is_derived:
        premise = locate_form_declaration(document_text, annotation.product_form)
        premises = [premise] if premise else []
        if not premises and gold.rule == RULE_COMPOSITION_FORM:
            composition = rules.extract_ingredients(document_text)
            premises = [item.evidence for item in composition.items[:2] if item.evidence]
        if not premises:
            return None
        return _derived_prediction(field_name, gold, annotation.product_form, premises)

    if field_name == "substance_or_mixture":
        span = locate_form_declaration(document_text, ProductForm(gold.value))
        if span is None:
            return None
        return FieldPrediction(state=FieldState.PRESENT, value=gold.value, evidence=span)

    field_rules = FIELD_RULES[field_name]
    if gold.state is FieldState.PRESENT:
        span = locate_value(document_text, gold.all_values)
        if span is None:
            return None
        return FieldPrediction(state=FieldState.PRESENT, value=gold.value, evidence=span)

    for candidate in rules.find_candidates(document_text, field_rules.labels, rules.LAYOUT_ROWS):
        if field_rules.trim is not None:
            candidate.value_tokens = field_rules.trim(candidate.value_tokens)
        prediction = rules._from_candidate(
            candidate, field_rules.prior, field_rules.parser, field_rules.cues
        )
        if prediction is not None and prediction.state is gold.state:
            return prediction
    return None


def _gold_span_on_layer(
    resolved, document_text: textlayer.DocumentText, layer: str
) -> tuple[textlayer.Page, int, int, str] | None:
    """(page, char_start, char_end, text) of a gold span in the requested layer."""
    if layer == textlayer.OCR:
        if resolved.ocr_page is None or resolved.ocr_char_start is None:
            return None
        page = document_text.page(resolved.ocr_page)
        if page is None:
            return None
        start, end = resolved.ocr_char_start, resolved.ocr_char_end
        return page, start, end, page.text[start:end]
    page = document_text.page(resolved.page)
    if page is None:
        return None
    return page, resolved.char_start, resolved.char_end, resolved.text


def probe_gold_evidence(
    annotation: DocumentAnnotation,
    field_name: str,
    gold: FieldAnnotation,
    document_text: textlayer.DocumentText,
    layer: str,
) -> FieldPrediction | None:
    """Gold span known; the system must read state and value off it."""
    if gold.state is FieldState.NOT_STATED or not gold.gold_targets:
        return None

    if gold.derivation.is_derived:
        premises: list[EvidenceSpan] = []
        for resolved in gold.resolved_premises:
            located = _gold_span_on_layer(resolved, document_text, layer)
            if located is None:
                continue
            page, start, end, text = located
            premises.append(span_from_char_range(page, start, end))
        if not premises:
            return None
        form = premise_form([span.text for span in premises])
        if form is None:
            return None
        if field_name in IDENTIFIER_FIELDS and form not in (ProductForm.MIXTURE, ProductForm.ARTICLE):
            return None
        return _derived_prediction(field_name, gold, form, premises)

    located = _gold_span_on_layer(gold.resolved_evidence[0], document_text, layer)
    if located is None:
        return None
    page, start, end, text = located
    citation = span_from_char_range(page, start, end)

    if field_name == "substance_or_mixture":
        form = classify_form_cue(text)
        if form is None:
            return None
        return FieldPrediction(state=FieldState.PRESENT, value=form.value, evidence=citation)

    field_rules = FIELD_RULES[field_name]
    tokens = page.tokens_in_char_range(start, end)
    label_tokens: list[textlayer.Token] = []
    value_tokens = list(tokens)
    for position in range(len(tokens)):
        hit = None
        for label in field_rules.labels:
            length = rules._phrase_matches(tokens, position, label)
            if length:
                hit = length
                break
        if hit:
            label_tokens = tokens[position : position + hit]
            value_tokens = rules._value_slot(tokens, position + hit)
            break
    if field_rules.trim is not None:
        value_tokens = field_rules.trim(value_tokens)
    candidate = rules.Candidate(page.page_number, label_tokens, value_tokens)
    prediction = rules._from_candidate(
        candidate, field_rules.prior, field_rules.parser, field_rules.cues
    )
    if prediction is None:
        return None
    prediction.evidence = citation
    return prediction


def prediction_from_gold(
    gold: FieldAnnotation, document_text: textlayer.DocumentText, layer: str
) -> FieldPrediction | None:
    """The gold pair itself as a prediction, for the verifier probe."""
    if gold.state is FieldState.NOT_STATED:
        return FieldPrediction(state=FieldState.NOT_STATED)
    if gold.derivation.is_derived:
        premises = []
        for resolved in gold.resolved_premises:
            located = _gold_span_on_layer(resolved, document_text, layer)
            if located is not None:
                page, start, end, _ = located
                premises.append(span_from_char_range(page, start, end))
        if not premises:
            return None
        return FieldPrediction(
            state=gold.state,
            value=gold.value,
            derivation="DERIVED",
            rule=gold.rule,
            premises=premises,
        )
    if not gold.resolved_evidence:
        return None
    located = _gold_span_on_layer(gold.resolved_evidence[0], document_text, layer)
    if located is None:
        return None
    page, start, end, _ = located
    return FieldPrediction(
        state=gold.state, value=gold.value, evidence=span_from_char_range(page, start, end)
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    document: str
    field: str
    gold_state: str
    gold_derivation: str
    value_cited: bool = False
    value_page: str = NOT_APPLICABLE
    value_span: str = NOT_APPLICABLE
    value_bbox: str = NOT_APPLICABLE
    value_support: str = NOT_APPLICABLE
    value_localized: bool = False
    evidence_available: bool = False
    evidence_read: bool = False
    evidence_state: str = NOT_APPLICABLE
    evidence_value: str = NOT_APPLICABLE
    evidence_support: str = NOT_APPLICABLE
    verify_available: bool = False
    verify_support: str = NOT_APPLICABLE
    verify_note: str = ""
    pipeline_joint: str = ""


def _passes(result: InstanceResult, axes: tuple[str, ...]) -> bool:
    return all(result.axes[axis] in PASSING for axis in axes)


def run_probes(
    gold_documents: list[DocumentAnnotation],
    layer: str = textlayer.NATIVE,
    pipeline: PredictionSet | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> list[ProbeResult]:
    pipeline_by_document = pipeline.by_document() if pipeline else {}
    results: list[ProbeResult] = []
    for annotation in gold_documents:
        document_text = textlayer.load(annotation.document, layer)
        pipeline_document = pipeline_by_document.get(annotation.document)
        for field_name in PRODUCT_FIELD_NAMES:
            gold = annotation.fields.get(field_name)
            if gold is None or gold.state is FieldState.NOT_STATED:
                continue
            probe = ProbeResult(
                document=annotation.document,
                field=field_name,
                gold_state=gold.state.value,
                gold_derivation=gold.derivation.value,
            )

            prediction = probe_gold_value(annotation, field_name, gold, document_text)
            if prediction is not None:
                probe.value_cited = True
                scored = evaluate_instance(annotation, field_name, gold, prediction, thresholds, layer)
                probe.value_page = scored.axes["page"]
                probe.value_span = scored.axes["span"]
                probe.value_bbox = scored.axes["bbox"]
                probe.value_support = scored.axes["support"]
                probe.value_localized = _passes(scored, ("page", "span", "bbox"))

            prediction = probe_gold_evidence(annotation, field_name, gold, document_text, layer)
            probe.evidence_available = any(
                _gold_span_on_layer(item, document_text, layer) is not None
                for item in gold.gold_targets
            )
            if prediction is not None:
                probe.evidence_read = True
                scored = evaluate_instance(annotation, field_name, gold, prediction, thresholds, layer)
                probe.evidence_state = scored.axes["state"]
                probe.evidence_value = scored.axes["value"]
                probe.evidence_support = scored.axes["support"]

            prediction = prediction_from_gold(gold, document_text, layer)
            if prediction is not None:
                probe.verify_available = True
                scored = evaluate_instance(annotation, field_name, gold, prediction, thresholds, layer)
                probe.verify_support = scored.axes["support"]
                probe.verify_note = scored.support_note

            if pipeline_document is not None:
                scored = evaluate_instance(
                    annotation,
                    field_name,
                    gold,
                    pipeline_document.fields.get(field_name),
                    thresholds,
                    layer,
                )
                probe.pipeline_joint = "yes" if scored.joint_correct else "no"
            results.append(probe)
    return results


def summarize(results: list[ProbeResult]) -> list[str]:
    lines: list[str] = []

    def rate(count: int, total: int) -> str:
        return f"{count / total:6.1%} ({count}/{total})" if total else "   n/a"

    total = len(results)
    lines.append(f"instances with something to locate or verify: {total}")
    lines.append("")
    lines.append("gold-value: the value / state / form is known, find the evidence")
    cited = [item for item in results if item.value_cited]
    lines.append(f"  cited anything            {rate(len(cited), total)}")
    lines.append(f"  located (page+span+bbox)  {rate(sum(item.value_localized for item in results), total)}")
    lines.append(f"  support satisfied         {rate(sum(item.value_support == SATISFIED for item in results), total)}")
    lines.append("")
    lines.append("gold-evidence: the gold span is known, read state and value off it")
    available = [item for item in results if item.evidence_available]
    lines.append(f"  gold span present in layer {rate(len(available), total)}")
    lines.append(f"  produced a claim          {rate(sum(item.evidence_read for item in available), len(available))}")
    lines.append(f"  state correct             {rate(sum(item.evidence_state == SATISFIED for item in available), len(available))}")
    lines.append(f"  state and value correct   {rate(sum(item.evidence_state == SATISFIED and item.evidence_value == SATISFIED for item in available), len(available))}")
    lines.append(f"  support satisfied         {rate(sum(item.evidence_support == SATISFIED for item in available), len(available))}")
    lines.append("")
    lines.append("gold-verify: the gold pair is known, does the gold-free verifier accept it")
    verifiable = [item for item in results if item.verify_available]
    lines.append(f"  satisfied                 {rate(sum(item.verify_support == SATISFIED for item in verifiable), len(verifiable))}")
    lines.append(f"  undecidable               {rate(sum(item.verify_support == 'undecidable' for item in verifiable), len(verifiable))}")
    lines.append(f"  violated                  {rate(sum(item.verify_support == 'violated' for item in verifiable), len(verifiable))}")
    with_pipeline = [item for item in results if item.pipeline_joint]
    if with_pipeline:
        lines.append("")
        lines.append("full pipeline (same instances, for reference)")
        lines.append(f"  joint correct             {rate(sum(item.pipeline_joint == 'yes' for item in with_pipeline), len(with_pipeline))}")
    lines.append("")
    lines.append(f"{'field':<22}{'value-loc':>11}{'ev-state':>10}{'ev-value':>10}{'verify':>9}{'pipeline':>10}")
    lines.append("-" * 72)
    for field_name in PRODUCT_FIELD_NAMES:
        subset = [item for item in results if item.field == field_name]
        if not subset:
            continue
        available = [item for item in subset if item.evidence_available]
        verifiable = [item for item in subset if item.verify_available]
        piped = [item for item in subset if item.pipeline_joint]
        cells = [
            f"{sum(item.value_localized for item in subset) / len(subset):.0%}",
            f"{sum(item.evidence_state == SATISFIED for item in available) / len(available):.0%}" if available else "n/a",
            f"{sum(item.evidence_state == SATISFIED and item.evidence_value == SATISFIED for item in available) / len(available):.0%}" if available else "n/a",
            f"{sum(item.verify_support == SATISFIED for item in verifiable) / len(verifiable):.0%}" if verifiable else "n/a",
            f"{sum(item.pipeline_joint == 'yes' for item in piped) / len(piped):.0%}" if piped else "n/a",
        ]
        lines.append(f"{field_name:<22}" + "".join(f"{cell:>10}" for cell in cells))
    return lines
