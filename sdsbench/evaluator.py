"""Score state, value, page, span, bbox, derivation and support separately, then their conjunction.

Statuses
--------
``satisfied`` and ``n/a`` pass; ``violated`` and ``undecidable`` fail. An
undecidable axis means the claim could not be verified from what the system
cited, and an unverifiable claim is not an acceptable one. ``n/a`` arises only
where there is nothing to check: the localization, derivation and support
axes when the gold field is NOT_STATED, and the derivation and support axes
when the prediction itself is NOT_STATED (it claims nothing).

Observed vs derived
-------------------
An observed claim cites direct evidence. A derived claim cites premise spans
and names a rule. Localization of a derived claim is scored against the gold
*premises*; its support is the rule check on the cited premise texts; the
derivation axis asks whether the system derived the field the way the gold
does (same kind, same rule). Citing the premise of a derived field as if it
were direct evidence fails the derivation axis and, having no state cue, the
support axis.

Localization
------------
``page``: the cited page is a gold page. ``span``: exact token overlap with
the gold span, coverage >= 0.8 and IoU >= 0.5 (a short fragment fails
coverage; a paragraph fails IoU). ``bbox``: the cited box covers >= 0.8 of the
gold box and has IoU >= 0.5 with it (a box that is the right height but the
page's width fails IoU). Every threshold is a parameter of ``Thresholds``.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Iterable

from . import derivation, textlayer
from .annotations import DocumentAnnotation, FieldAnnotation, Ingredient, ResolvedEvidence
from .matching import (
    canonical_value,
    concentrations_equal,
    names_equal,
    normalize_cas,
    span_overlap,
    value_supported_by,
    values_equal,
)
from .ontology import (
    FIELD_SPECS,
    PRODUCT_FIELD_NAMES,
    FieldState,
    ValueKind,
    classify_form_cue,
    classify_state_cue,
    value_part,
)
from .schema import (
    DocumentPrediction,
    EvidenceSpan,
    FieldPrediction,
    IngredientPrediction,
    PredictionSet,
)

SATISFIED = "satisfied"
VIOLATED = "violated"
UNDECIDABLE = "undecidable"
NOT_APPLICABLE = "n/a"

PASSING = frozenset({SATISFIED, NOT_APPLICABLE})

AXES = ("state", "value", "page", "span", "bbox", "derivation", "support")
LOCALIZATION_AXES = ("page", "span", "bbox")


@dataclass(frozen=True)
class Thresholds:
    span_coverage: float = 0.8
    span_iou: float = 0.5
    bbox_coverage: float = 0.8
    bbox_iou: float = 0.5
    ingredient_bbox_coverage: float = 0.8
    ingredient_max_height_ratio: float = 2.5


DEFAULT_THRESHOLDS = Thresholds()


@dataclass
class InstanceResult:
    document: str
    field: str
    gold_state: FieldState
    predicted_state: FieldState | None
    gold_value: str | None
    predicted_value: str | None
    axes: dict[str, str]
    gold_derivation: str = "EXPLICIT"
    predicted_derivation: str = ""
    gold_rule: str | None = None
    predicted_rule: str | None = None
    evidence_role: str = "direct"
    predicted_page: int | None = None
    predicted_evidence: str = ""
    gold_pages: tuple[int, ...] = ()
    gold_evidence: str = ""
    span_coverage: float = 0.0
    span_dilution: float = 0.0
    span_iou: float = 0.0
    bbox_coverage: float = 0.0
    bbox_dilution: float = 0.0
    bbox_iou: float = 0.0
    verbatim: bool = True
    verbatim_fuzzy_score: float = 0.0
    confidence: float = 0.0
    gold_requires_evidence: bool = False
    gold_ambiguous: bool = False
    state_correct_lenient: bool = False
    bbox_provenance: str = "none"
    support_note: str = ""
    note: str = ""

    @property
    def joint_correct(self) -> bool:
        return all(status in PASSING for status in self.axes.values())

    def conjunction_of(self, axes: Iterable[str]) -> bool:
        return all(self.axes[axis] in PASSING for axis in axes)

    @property
    def has_undecidable(self) -> bool:
        return any(status == UNDECIDABLE for status in self.axes.values())


@dataclass(frozen=True, order=True)
class _Localization:
    page_match: bool
    span_coverage: float = 0.0
    span_iou: float = 0.0
    box_coverage: float = 0.0
    box_iou: float = 0.0
    span_dilution: float = 0.0
    box_dilution: float = 0.0


def _score_against_gold(cited: EvidenceSpan, gold: ResolvedEvidence) -> _Localization:
    if cited.page != gold.page:
        return _Localization(page_match=False)
    overlap = span_overlap(cited.text, gold.text)
    cited_box = tuple(cited.bbox) if cited.bbox else None
    gold_box = tuple(gold.bbox) if gold.bbox else None
    box_coverage, box_dilution = textlayer.bbox_containment(cited_box, gold_box)
    return _Localization(
        page_match=True,
        span_coverage=overlap.coverage,
        span_iou=overlap.iou,
        box_coverage=box_coverage,
        box_iou=textlayer.bbox_iou(cited_box, gold_box),
        span_dilution=overlap.dilution,
        box_dilution=box_dilution,
    )


def _support_status(
    field_name: str,
    prediction: FieldPrediction,
    cited: list[EvidenceSpan],
    verbatim: bool,
) -> tuple[str, str]:
    """Gold-free: does what the system cited back the system's own claim?"""
    if prediction.state is FieldState.NOT_STATED:
        return (VIOLATED, "NOT_STATED with a citation") if cited else (NOT_APPLICABLE, "")
    if not cited or any(not span.text.strip() for span in cited):
        return VIOLATED, "no citation"
    if not verbatim:
        return VIOLATED, "citation is not verbatim"

    if prediction.derivation == "DERIVED":
        check = derivation.check_rule(
            prediction.rule,
            field_name,
            prediction.state,
            prediction.value,
            [span.text for span in cited],
        )
        return check.status, check.note

    span_text = cited[0].text
    if prediction.state is FieldState.PRESENT:
        if not prediction.value:
            return VIOLATED, "PRESENT without a value"
        kind = FIELD_SPECS[field_name].value_kind
        if kind is ValueKind.CATEGORICAL:
            declared = classify_form_cue(span_text)
            if declared is None:
                return UNDECIDABLE, "cited span declares no product form"
            claimed = canonical_value(prediction.value, kind)
            if declared.value == claimed:
                return SATISFIED, ""
            return VIOLATED, f"cited span declares {declared.value}"
        if value_supported_by(prediction.value, span_text, kind):
            return SATISFIED, ""
        return VIOLATED, "value does not occur in the cited span"

    cue = classify_state_cue(value_part(span_text)) or classify_state_cue(span_text)
    if cue is None:
        return UNDECIDABLE, "cited span carries no state cue"
    if cue is prediction.state:
        return SATISFIED, ""
    return VIOLATED, f"cited span reads {cue.value}"


def evaluate_instance(
    annotation: DocumentAnnotation,
    field_name: str,
    gold: FieldAnnotation,
    prediction: FieldPrediction | None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    layer: str = textlayer.NATIVE,
) -> InstanceResult:
    spec = FIELD_SPECS[field_name]
    gold_targets = gold.gold_targets
    gold_requires_evidence = gold.state.requires_evidence and bool(gold_targets)
    cited = prediction.cited_spans if prediction is not None else []

    axes = {axis: NOT_APPLICABLE for axis in AXES}
    result = InstanceResult(
        document=annotation.document,
        field=field_name,
        gold_state=gold.state,
        predicted_state=prediction.state if prediction else None,
        gold_value=gold.value,
        predicted_value=prediction.value if prediction else None,
        axes=axes,
        gold_derivation=gold.derivation.value,
        predicted_derivation=prediction.derivation if prediction else "",
        gold_rule=gold.rule,
        predicted_rule=prediction.rule if prediction else None,
        evidence_role="premise" if gold.derivation.is_derived else "direct",
        gold_requires_evidence=gold_requires_evidence,
        gold_ambiguous=gold.ambiguous,
        confidence=prediction.confidence if prediction else 0.0,
        predicted_page=cited[0].page if cited else None,
        predicted_evidence=" | ".join(span.text for span in cited),
        gold_pages=tuple(sorted({item.page for item in gold_targets})),
        gold_evidence=" | ".join(item.text for item in gold_targets),
        bbox_provenance=cited[0].bbox_provenance if cited else "none",
    )

    if prediction is None:
        axes["state"] = VIOLATED
        axes["value"] = VIOLATED
        if gold_requires_evidence:
            for axis in (*LOCALIZATION_AXES, "derivation", "support"):
                axes[axis] = VIOLATED
        result.note = "no prediction"
        return result

    axes["state"] = SATISFIED if prediction.state is gold.state else VIOLATED
    result.state_correct_lenient = prediction.state is gold.state or (
        gold.alternative_state is not None and prediction.state is gold.alternative_state
    )

    if gold.state is FieldState.PRESENT:
        if not prediction.value:
            axes["value"] = VIOLATED
        else:
            axes["value"] = (
                SATISFIED
                if any(
                    values_equal(
                        prediction.value, candidate, spec.value_kind, spec.numeric_tolerance
                    )
                    for candidate in gold.all_values
                )
                else VIOLATED
            )
    else:
        axes["value"] = VIOLATED if prediction.value else SATISFIED

    verbatim = True
    if cited:
        document_text = textlayer.load(annotation.document, layer)
        for span in cited:
            page = document_text.page(span.page)
            if page is not None and textlayer.occurs_on_page(page, span.text):
                continue
            verbatim = False
            if page is not None:
                result.verbatim_fuzzy_score = max(
                    result.verbatim_fuzzy_score,
                    textlayer.fuzzy_occurrence_score(page, span.text),
                )
        result.verbatim = verbatim
        if not verbatim:
            result.note = "cited quote does not occur verbatim on the cited page"

    if not gold_requires_evidence:
        for axis in LOCALIZATION_AXES:
            axes[axis] = NOT_APPLICABLE
    elif not cited:
        for axis in LOCALIZATION_AXES:
            axes[axis] = VIOLATED
    else:
        best = max(
            (_score_against_gold(span, target) for span in cited for target in gold_targets),
            default=_Localization(page_match=False),
        )
        result.span_coverage = best.span_coverage
        result.span_dilution = best.span_dilution
        result.span_iou = best.span_iou
        result.bbox_coverage = best.box_coverage
        result.bbox_dilution = best.box_dilution
        result.bbox_iou = best.box_iou
        axes["page"] = SATISFIED if best.page_match else VIOLATED
        axes["span"] = (
            SATISFIED
            if best.span_coverage >= thresholds.span_coverage
            and best.span_iou >= thresholds.span_iou
            else VIOLATED
        )
        axes["bbox"] = (
            SATISFIED
            if best.box_coverage >= thresholds.bbox_coverage
            and best.box_iou >= thresholds.bbox_iou
            else VIOLATED
        )

    if gold.state is FieldState.NOT_STATED or prediction.state is FieldState.NOT_STATED:
        axes["derivation"] = NOT_APPLICABLE
    else:
        gold_kind = "DERIVED" if gold.derivation.is_derived else "DIRECT"
        if prediction.derivation != gold_kind:
            axes["derivation"] = VIOLATED
            result.note = (result.note + "; " if result.note else "") + (
                f"gold is {gold_kind.lower()}, prediction is {prediction.derivation.lower()}"
            )
        elif gold_kind == "DERIVED" and prediction.rule != gold.rule:
            axes["derivation"] = VIOLATED
            result.note = (result.note + "; " if result.note else "") + (
                f"gold rule {gold.rule}, prediction rule {prediction.rule}"
            )
        else:
            axes["derivation"] = SATISFIED

    axes["support"], result.support_note = _support_status(
        field_name, prediction, cited, verbatim
    )
    return result


@dataclass
class AxisSummary:
    applicable: int = 0
    satisfied: int = 0
    undecidable: int = 0

    @property
    def rate(self) -> float | None:
        return self.satisfied / self.applicable if self.applicable else None


@dataclass
class Summary:
    system: str
    instances: list[InstanceResult] = dataclass_field(default_factory=list)

    def axis(self, name: str, subset: Iterable[InstanceResult] | None = None) -> AxisSummary:
        summary = AxisSummary()
        for instance in self.instances if subset is None else subset:
            status = instance.axes[name]
            if status == NOT_APPLICABLE:
                continue
            summary.applicable += 1
            summary.satisfied += status == SATISFIED
            summary.undecidable += status == UNDECIDABLE
        return summary

    @property
    def evidence_bearing(self) -> list[InstanceResult]:
        return [item for item in self.instances if item.gold_requires_evidence]

    def joint_rate(self, subset: Iterable[InstanceResult] | None = None) -> float:
        items = list(self.instances if subset is None else subset)
        if not items:
            return 0.0
        return sum(item.joint_correct for item in items) / len(items)

    def conjunction_rate(
        self, axes: Iterable[str], subset: Iterable[InstanceResult] | None = None
    ) -> float:
        axes = list(axes)
        items = list(self.instances if subset is None else subset)
        if not items:
            return 0.0
        return sum(item.conjunction_of(axes) for item in items) / len(items)

    def mean(self, attribute: str) -> float:
        subset = self.evidence_bearing
        if not subset:
            return 0.0
        return sum(getattr(item, attribute) for item in subset) / len(subset)

    def verbatim_rate(self) -> tuple[int, int]:
        cited = [item for item in self.instances if item.predicted_evidence]
        return sum(item.verbatim for item in cited), len(cited)

    def lenient_state_rate(self) -> float:
        if not self.instances:
            return 0.0
        return sum(item.state_correct_lenient for item in self.instances) / len(self.instances)

    def undecidable_count(self) -> int:
        return sum(item.has_undecidable for item in self.instances)


def evaluate(
    gold_documents: list[DocumentAnnotation],
    predictions: PredictionSet,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> Summary:
    predicted_by_document: dict[str, DocumentPrediction] = predictions.by_document()
    summary = Summary(system=predictions.system)
    layer = predictions.configuration.get("layer", textlayer.NATIVE)

    for annotation in gold_documents:
        document_prediction = predicted_by_document.get(annotation.document)
        for field_name in PRODUCT_FIELD_NAMES:
            gold = annotation.fields.get(field_name)
            if gold is None:
                continue
            prediction = (
                document_prediction.fields.get(field_name)
                if document_prediction
                else None
            )
            summary.instances.append(
                evaluate_instance(annotation, field_name, gold, prediction, thresholds, layer)
            )
    return summary


# ---------------------------------------------------------------------------
# Ingredient records: name + CAS (or CAS state) + concentration + row evidence.
# ---------------------------------------------------------------------------


@dataclass
class IngredientRecordResult:
    document: str
    gold_index: int
    predicted_index: int
    key: str
    name_ok: bool
    cas_ok: bool
    concentration_ok: bool
    evidence_ok: bool
    verbatim: bool
    gold_name: str | None = None
    predicted_name: str | None = None
    gold_concentration: str | None = None
    predicted_concentration: str | None = None

    @property
    def record_ok(self) -> bool:
        return self.name_ok and self.cas_ok and self.concentration_ok

    @property
    def record_with_evidence_ok(self) -> bool:
        return self.record_ok and self.evidence_ok


@dataclass
class IngredientDocumentScore:
    document: str
    gold_state: FieldState
    predicted_state: FieldState | None
    gold_rows: int
    predicted_rows: int
    matches: list[IngredientRecordResult] = dataclass_field(default_factory=list)

    @property
    def state_ok(self) -> bool:
        return self.predicted_state is self.gold_state


@dataclass
class IngredientSummary:
    documents: list[IngredientDocumentScore] = dataclass_field(default_factory=list)

    @property
    def matches(self) -> list[IngredientRecordResult]:
        return [match for item in self.documents for match in item.matches]

    @property
    def gold_rows(self) -> int:
        return sum(item.gold_rows for item in self.documents)

    @property
    def predicted_rows(self) -> int:
        return sum(item.predicted_rows for item in self.documents)

    @property
    def matched(self) -> int:
        return len(self.matches)

    @property
    def records_exact(self) -> int:
        return sum(match.record_ok for match in self.matches)

    @property
    def records_with_evidence(self) -> int:
        return sum(match.record_with_evidence_ok for match in self.matches)

    def attribute_rate(self, attribute: str) -> float | None:
        matches = self.matches
        if not matches:
            return None
        return sum(getattr(match, attribute) for match in matches) / len(matches)

    @property
    def precision(self) -> float | None:
        return self.records_exact / self.predicted_rows if self.predicted_rows else None

    @property
    def recall(self) -> float | None:
        return self.records_exact / self.gold_rows if self.gold_rows else None

    @property
    def key_recall(self) -> float | None:
        return self.matched / self.gold_rows if self.gold_rows else None

    @property
    def state_agreement(self) -> tuple[int, int]:
        return sum(item.state_ok for item in self.documents), len(self.documents)


def _effective_cas_state(item: IngredientPrediction | Ingredient) -> FieldState:
    if item.cas_number:
        return FieldState.PRESENT
    return item.cas_state if item.cas_state is not FieldState.PRESENT else FieldState.NOT_STATED


def _cas_key(item: IngredientPrediction | Ingredient) -> str | None:
    if item.cas_number and _effective_cas_state(item) is FieldState.PRESENT:
        return normalize_cas(item.cas_number)
    return None


def _match_rows(
    gold_items: list[Ingredient], predicted_items: list[IngredientPrediction]
) -> list[tuple[int, int, str]]:
    """One-to-one matching: CAS number first, then normalized name."""
    pairs: list[tuple[int, int, str]] = []
    free_predicted = set(range(len(predicted_items)))
    for gold_index, gold_item in enumerate(gold_items):
        key = _cas_key(gold_item)
        if key is None:
            continue
        for predicted_index in sorted(free_predicted):
            if _cas_key(predicted_items[predicted_index]) == key:
                pairs.append((gold_index, predicted_index, "cas"))
                free_predicted.discard(predicted_index)
                break
    matched_gold = {gold_index for gold_index, _, _ in pairs}
    for gold_index, gold_item in enumerate(gold_items):
        if gold_index in matched_gold or not gold_item.name:
            continue
        for predicted_index in sorted(free_predicted):
            if names_equal(predicted_items[predicted_index].name, gold_item.name):
                pairs.append((gold_index, predicted_index, "name"))
                free_predicted.discard(predicted_index)
                break
    return pairs


def _row_evidence_ok(
    predicted: IngredientPrediction,
    gold: Ingredient,
    document: str,
    layer: str,
    thresholds: Thresholds,
) -> tuple[bool, bool]:
    """(evidence located, citation verbatim).

    Gold row evidence is often the bare CAS token, so the predicted row box is
    required to cover it and to be at most ``ingredient_max_height_ratio``
    times its height; the width is the row's and is not constrained.
    """
    if predicted.evidence is None:
        return False, False
    page = textlayer.load(document, layer).page(predicted.evidence.page)
    verbatim = page is not None and textlayer.occurs_on_page(page, predicted.evidence.text)
    if not verbatim:
        return False, False
    cited_box = tuple(predicted.evidence.bbox) if predicted.evidence.bbox else None
    for resolved in gold.resolved_evidence:
        if predicted.evidence.page != resolved.page:
            continue
        gold_box = tuple(resolved.bbox) if resolved.bbox else None
        coverage, _ = textlayer.bbox_containment(cited_box, gold_box)
        height_ratio = textlayer.bbox_height_ratio(cited_box, gold_box)
        if (
            coverage >= thresholds.ingredient_bbox_coverage
            and height_ratio <= thresholds.ingredient_max_height_ratio
        ):
            return True, True
    return False, True


def evaluate_ingredients(
    gold_documents: list[DocumentAnnotation],
    predictions: PredictionSet,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> IngredientSummary:
    predicted_by_document = predictions.by_document()
    layer = predictions.configuration.get("layer", textlayer.NATIVE)
    summary = IngredientSummary()

    for annotation in gold_documents:
        gold_items = annotation.ingredients.items
        document_prediction = predicted_by_document.get(annotation.document)
        predicted_block = document_prediction.ingredients if document_prediction else None
        predicted_items = predicted_block.items if predicted_block else []

        score = IngredientDocumentScore(
            document=annotation.document,
            gold_state=annotation.ingredients.state,
            predicted_state=predicted_block.state if predicted_block else None,
            gold_rows=len(gold_items),
            predicted_rows=len(predicted_items),
        )
        for gold_index, predicted_index, key in _match_rows(gold_items, predicted_items):
            gold_item = gold_items[gold_index]
            predicted_item = predicted_items[predicted_index]
            gold_cas_state = _effective_cas_state(gold_item)
            predicted_cas_state = _effective_cas_state(predicted_item)
            if gold_cas_state is FieldState.PRESENT:
                cas_ok = _cas_key(predicted_item) == _cas_key(gold_item)
            else:
                cas_ok = predicted_cas_state is gold_cas_state
            evidence_ok, verbatim = _row_evidence_ok(
                predicted_item, gold_item, annotation.document, layer, thresholds
            )
            score.matches.append(
                IngredientRecordResult(
                    document=annotation.document,
                    gold_index=gold_index,
                    predicted_index=predicted_index,
                    key=key,
                    name_ok=names_equal(predicted_item.name, gold_item.name),
                    cas_ok=cas_ok,
                    concentration_ok=concentrations_equal(
                        predicted_item.concentration, gold_item.concentration
                    ),
                    evidence_ok=evidence_ok,
                    verbatim=verbatim,
                    gold_name=gold_item.name,
                    predicted_name=predicted_item.name,
                    gold_concentration=gold_item.concentration,
                    predicted_concentration=predicted_item.concentration,
                )
            )
        summary.documents.append(score)
    return summary
