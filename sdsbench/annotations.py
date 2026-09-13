"""Gold annotations: hand-written (page, text, occurrence) resolved against the PDF.

Observed fields carry direct evidence. Derived fields (derivation ONTOLOGY or
COMPOSITION_IMPLIED) carry *premises* and a rule: the spans that establish
the product form are premises of "product CAS not applicable", never direct
evidence for it. Premises resolve into ``resolved_premises``; the direct
``resolved_evidence`` of a derived field stays empty.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dataclass_field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

from . import derivation, textlayer
from .ontology import (
    FIELD_SPECS,
    PRODUCT_FIELD_NAMES,
    FieldState,
    ProductForm,
    field_is_defined_for,
)

SOURCE_DIR = Path("data/annotations/source")
GOLD_PATH = Path("data/annotations/gold.json")

SPLIT_DEV = "dev"
SPLIT_LOCKED = "locked"


class AnnotationError(Exception):
    pass


class Derivation(str, Enum):
    EXPLICIT = "EXPLICIT"
    COMPOSITION_IMPLIED = "COMPOSITION_IMPLIED"
    ONTOLOGY = "ONTOLOGY"
    ABSENT = "ABSENT"

    @property
    def is_derived(self) -> bool:
        return self in (Derivation.COMPOSITION_IMPLIED, Derivation.ONTOLOGY)


@dataclass
class EvidenceRef:
    page: int
    text: str
    occurrence: int = 1
    case_sensitive: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceRef":
        return cls(
            page=int(data["page"]),
            text=data["text"],
            occurrence=int(data.get("occurrence", 1)),
            case_sensitive=bool(data.get("case_sensitive", False)),
        )

    @property
    def key(self) -> tuple[int, str, int]:
        return (self.page, self.text, self.occurrence)


@dataclass
class ResolvedEvidence:
    page: int
    text: str
    char_start: int
    char_end: int
    bbox: list[float]
    occurrences_on_page: int
    ocr_page: int | None = None
    ocr_bbox: list[float] | None = None
    ocr_char_start: int | None = None
    ocr_char_end: int | None = None
    ocr_exact: bool = False
    ocr_score: float = 0.0


@dataclass
class FieldAnnotation:
    state: FieldState
    derivation: Derivation
    value: str | None = None
    accepted_values: list[str] = dataclass_field(default_factory=list)
    evidence: list[EvidenceRef] = dataclass_field(default_factory=list)
    rule: str | None = None
    resolved_evidence: list[ResolvedEvidence] = dataclass_field(default_factory=list)
    resolved_premises: list[ResolvedEvidence] = dataclass_field(default_factory=list)
    ambiguous: bool = False
    alternative_state: FieldState | None = None
    note: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FieldAnnotation":
        alternative = data.get("alternative_state")
        kind = Derivation(data.get("derivation", "EXPLICIT"))
        rule = data.get("rule")
        if rule is None and kind.is_derived:
            rule = derivation.default_rule_for("", kind.value)
        return cls(
            state=FieldState(data["state"]),
            derivation=kind,
            value=data.get("value"),
            accepted_values=list(data.get("accepted_values", [])),
            evidence=[EvidenceRef.from_dict(item) for item in data.get("evidence", [])],
            rule=rule,
            ambiguous=bool(data.get("ambiguous", False)),
            alternative_state=FieldState(alternative) if alternative else None,
            note=data.get("note"),
        )

    @property
    def all_values(self) -> list[str]:
        values = [self.value] if self.value is not None else []
        return values + [item for item in self.accepted_values if item != self.value]

    @property
    def gold_targets(self) -> list[ResolvedEvidence]:
        """What a system must locate: premises for a derived field, else direct evidence."""
        return self.resolved_premises if self.derivation.is_derived else self.resolved_evidence


@dataclass
class Ingredient:
    name: str | None
    cas_number: str | None
    cas_state: FieldState
    concentration: str | None = None
    evidence: list[EvidenceRef] = dataclass_field(default_factory=list)
    resolved_evidence: list[ResolvedEvidence] = dataclass_field(default_factory=list)
    note: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Ingredient":
        return cls(
            name=data.get("name"),
            cas_number=data.get("cas_number"),
            cas_state=FieldState(data.get("cas_state", "PRESENT")),
            concentration=data.get("concentration"),
            evidence=[EvidenceRef.from_dict(item) for item in data.get("evidence", [])],
            note=data.get("note"),
        )


@dataclass
class IngredientAnnotation:
    state: FieldState
    evidence: list[EvidenceRef] = dataclass_field(default_factory=list)
    resolved_evidence: list[ResolvedEvidence] = dataclass_field(default_factory=list)
    items: list[Ingredient] = dataclass_field(default_factory=list)
    note: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IngredientAnnotation":
        return cls(
            state=FieldState(data["state"]),
            evidence=[EvidenceRef.from_dict(item) for item in data.get("evidence", [])],
            items=[Ingredient.from_dict(item) for item in data.get("items", [])],
            note=data.get("note"),
        )


@dataclass
class DocumentAnnotation:
    document: str
    split: str
    product_form: ProductForm
    fields: dict[str, FieldAnnotation]
    ingredients: IngredientAnnotation
    note: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentAnnotation":
        return cls(
            document=data["document"],
            split=data.get("split", SPLIT_DEV),
            product_form=ProductForm(data["product_form"]),
            fields={
                name: FieldAnnotation.from_dict(payload)
                for name, payload in data["fields"].items()
            },
            ingredients=IngredientAnnotation.from_dict(data["ingredients"]),
            note=data.get("note"),
        )


def load_source(source_dir: Path = SOURCE_DIR) -> list[DocumentAnnotation]:
    documents = []
    for path in sorted(source_dir.glob("*.json")):
        with open(path, encoding="utf-8") as handle:
            documents.append(DocumentAnnotation.from_dict(json.load(handle)))
    return documents


def _resolve_reference(
    reference: EvidenceRef,
    native: textlayer.DocumentText,
    ocr: textlayer.DocumentText | None,
    where: str,
) -> ResolvedEvidence:
    page = native.page(reference.page)
    if page is None:
        raise AnnotationError(f"{where}: page {reference.page} does not exist")

    anchors = textlayer.find_all_in_page(
        page, reference.text, casefold=not reference.case_sensitive
    )
    if not anchors:
        raise AnnotationError(
            f"{where}: span not found verbatim on page {reference.page}: "
            f"{reference.text!r}"
        )
    if reference.occurrence > len(anchors):
        raise AnnotationError(
            f"{where}: occurrence {reference.occurrence} requested but only "
            f"{len(anchors)} found on page {reference.page} for {reference.text!r}"
        )
    anchor = anchors[reference.occurrence - 1]

    resolved = ResolvedEvidence(
        page=reference.page,
        text=anchor.text,
        char_start=anchor.char_start,
        char_end=anchor.char_end,
        bbox=list(anchor.bbox) if anchor.bbox else [],
        occurrences_on_page=len(anchors),
    )
    if not resolved.bbox:
        raise AnnotationError(f"{where}: span resolved to no tokens, cannot build a box")

    if ocr is not None:
        ocr_page = ocr.page(reference.page)
        ocr_anchor = (
            textlayer.find_in_page(ocr_page, reference.text, min_score=85.0)
            if ocr_page
            else None
        )
        if ocr_anchor is not None:
            resolved.ocr_page = ocr_anchor.page_number
            resolved.ocr_bbox = list(ocr_anchor.bbox) if ocr_anchor.bbox else None
            resolved.ocr_char_start = ocr_anchor.char_start
            resolved.ocr_char_end = ocr_anchor.char_end
            resolved.ocr_exact = ocr_anchor.exact
            resolved.ocr_score = ocr_anchor.score
    return resolved


def _check_consistency(annotation: DocumentAnnotation) -> list[str]:
    problems: list[str] = []
    for name in PRODUCT_FIELD_NAMES:
        if name not in annotation.fields:
            problems.append(f"{annotation.document}: missing field {name}")
    for name, field in annotation.fields.items():
        where = f"{annotation.document}/{name}"
        if name not in FIELD_SPECS:
            problems.append(f"{where}: unknown field")
            continue
        if field.state is FieldState.PRESENT and not field.value:
            problems.append(f"{where}: PRESENT requires a value")
        if field.state is not FieldState.PRESENT and field.value:
            problems.append(f"{where}: {field.state.value} must not carry a value")
        if field.state.requires_evidence and not field.evidence:
            problems.append(f"{where}: {field.state.value} requires evidence")
        if not field.state.requires_evidence and field.evidence:
            problems.append(f"{where}: NOT_STATED must not carry evidence")
        if field.state is FieldState.NOT_STATED and field.derivation is not Derivation.ABSENT:
            problems.append(f"{where}: NOT_STATED must use derivation ABSENT")
        if field.derivation is Derivation.ONTOLOGY and field.state is not FieldState.NOT_APPLICABLE:
            problems.append(f"{where}: ONTOLOGY derivation only applies to NOT_APPLICABLE")
        if (
            field.derivation is Derivation.ONTOLOGY
            and field_is_defined_for(name, annotation.product_form)
        ):
            problems.append(
                f"{where}: ONTOLOGY derivation but the field is defined for "
                f"{annotation.product_form.value}"
            )
        if (
            field.state is FieldState.PRESENT
            and not field_is_defined_for(name, annotation.product_form)
        ):
            problems.append(
                f"{where}: PRESENT but the ontology rules the field out for "
                f"{annotation.product_form.value}"
            )
        if field.derivation.is_derived:
            if not derivation.rule_applies(field.rule, name, field.state):
                problems.append(
                    f"{where}: derived field needs a registered rule for "
                    f"{name}/{field.state.value}, got {field.rule!r}"
                )
        elif field.rule is not None:
            problems.append(f"{where}: {field.derivation.value} field must not name a rule")

    declared = annotation.fields.get("substance_or_mixture")
    if declared and declared.state is FieldState.PRESENT:
        if declared.value != annotation.product_form.value:
            problems.append(
                f"{annotation.document}: product_form {annotation.product_form.value} "
                f"disagrees with substance_or_mixture value {declared.value}"
            )
    elif annotation.product_form is not ProductForm.UNDETERMINED:
        problems.append(
            f"{annotation.document}: product_form is {annotation.product_form.value} "
            "but substance_or_mixture is not PRESENT"
        )
    return problems


def premise_references(annotation: DocumentAnnotation, name: str) -> list[EvidenceRef]:
    """Accepted premises of a derived field.

    The rule "identifiers are undefined for a mixture / article" rests on the
    product form, so every span that establishes the form is an accepted
    premise, whether the annotator listed it on the derived field or on
    ``substance_or_mixture``. Composition-implied form claims keep only the
    spans listed on the field itself.
    """
    field = annotation.fields[name]
    references = list(field.evidence)
    known = {item.key for item in references}
    if field.derivation is Derivation.ONTOLOGY:
        form_field = annotation.fields.get("substance_or_mixture")
        if form_field is not None:
            for reference in form_field.evidence:
                if reference.key not in known:
                    references.append(reference)
                    known.add(reference.key)
    return references


def _resolve_all(
    references: list[EvidenceRef],
    native: textlayer.DocumentText,
    ocr: textlayer.DocumentText | None,
    where: str,
    problems: list[str],
) -> list[ResolvedEvidence]:
    resolved: list[ResolvedEvidence] = []
    for index, reference in enumerate(references):
        try:
            resolved.append(_resolve_reference(reference, native, ocr, f"{where}[{index}]"))
        except AnnotationError as error:
            problems.append(str(error))
    return resolved


def resolve_document(annotation: DocumentAnnotation) -> list[str]:
    problems = _check_consistency(annotation)
    native = textlayer.load(annotation.document, textlayer.NATIVE)
    try:
        ocr = textlayer.load(annotation.document, textlayer.OCR)
    except FileNotFoundError:
        ocr = None

    for name, field in annotation.fields.items():
        where = f"{annotation.document}/{name}"
        if field.derivation.is_derived:
            field.resolved_evidence = []
            field.resolved_premises = _resolve_all(
                premise_references(annotation, name), native, ocr, where, problems
            )
        else:
            field.resolved_premises = []
            field.resolved_evidence = _resolve_all(field.evidence, native, ocr, where, problems)

    ingredients = annotation.ingredients
    ingredients.resolved_evidence = _resolve_all(
        ingredients.evidence, native, ocr, f"{annotation.document}/ingredients", problems
    )
    for position, item in enumerate(ingredients.items):
        item.resolved_evidence = _resolve_all(
            item.evidence, native, ocr, f"{annotation.document}/ingredient[{position}]", problems
        )
    return problems


def premise_verifiable(field: FieldAnnotation) -> bool | None:
    """Whether the rule registry can establish the premise from the gold premise texts.

    None for observed fields. A False here is a gold/registry disagreement
    worth reviewing, not an evaluation error.
    """
    if not field.derivation.is_derived:
        return None
    return derivation.premise_form(item.text for item in field.resolved_premises) is not None


def _serialize(value: Any) -> Any:
    if isinstance(value, (FieldState, ProductForm, Derivation)):
        return value.value
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


def build_gold(
    source_dir: Path = SOURCE_DIR, output: Path = GOLD_PATH
) -> tuple[list[DocumentAnnotation], list[str]]:
    documents = load_source(source_dir)
    problems: list[str] = []
    for annotation in documents:
        problems.extend(resolve_document(annotation))

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "documents": [_serialize(asdict(annotation)) for annotation in documents],
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return documents, problems


def load_gold(path: Path = GOLD_PATH, split: str | None = None) -> list[DocumentAnnotation]:
    """Load and re-resolve the gold. ``split`` keeps only that split ("all" keeps everything)."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    documents = [
        DocumentAnnotation.from_dict(item) for item in payload["documents"]
    ]
    if split is not None and split != "all":
        documents = [item for item in documents if item.split == split]
    for annotation in documents:
        problems = resolve_document(annotation)
        if problems:
            raise AnnotationError(
                "gold annotations no longer resolve:\n  " + "\n  ".join(problems)
            )
    return documents
