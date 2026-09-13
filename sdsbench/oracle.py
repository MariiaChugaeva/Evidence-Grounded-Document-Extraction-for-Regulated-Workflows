"""Recoverability of gold values without gold hints, and gold-span recoverability, kept apart.

One row per (document, field). Columns:

value_exact_*        an accepted gold value occurs exactly (after normalization) in the layer
value_typed_*        typed containment on page or row text (290 °F found as 143.3 °C); no location
value_fuzzy_*        partial-ratio >= 90 somewhere in the layer; diagnostic only
value_located_*      an exact value hit that resolves to a box, searching every page in
                     document order with no knowledge of the gold page
value_located_pages_*  the pages of those hits
value_on_gold_page_* at least one located page is a gold evidence page
span_exact_*         a gold span (direct evidence, or a premise for a derived field) occurs exactly
span_fuzzy_ocr       the gold span aligns in OCR at >= 90; diagnostic only

Nothing from the gold evidence feeds the value_* columns: localization by
value is measured from the value alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from . import textlayer
from .annotations import Derivation, DocumentAnnotation, FieldAnnotation
from .matching import value_supported_by
from .ontology import FIELD_SPECS, PRODUCT_FIELD_NAMES, FieldState, ValueKind


@dataclass
class OracleRow:
    document: str
    field: str
    gold_state: str
    gold_derivation: str
    gold_value: str
    value_exact_native: str = ""
    value_exact_ocr: str = ""
    value_typed_native: str = ""
    value_typed_ocr: str = ""
    value_fuzzy_native: str = ""
    value_fuzzy_ocr: str = ""
    value_located_native: str = ""
    value_located_ocr: str = ""
    value_located_pages_native: str = ""
    value_located_pages_ocr: str = ""
    value_on_gold_page_native: str = ""
    value_on_gold_page_ocr: str = ""
    span_exact_native: str = ""
    span_exact_ocr: str = ""
    span_fuzzy_ocr: str = ""
    note: str = ""


def locate_value_exact(
    layer: textlayer.DocumentText, values: Iterable[str]
) -> list[textlayer.Anchor]:
    """Every exact occurrence of any value that resolves to a box, all pages, no hints."""
    anchors: list[textlayer.Anchor] = []
    for page in layer.pages:
        for value in values:
            if not value:
                continue
            anchors.extend(
                anchor
                for anchor in textlayer.find_all_in_page(page, value)
                if anchor.bbox is not None
            )
    return anchors


def _value_typed(layer: textlayer.DocumentText, values: Iterable[str], kind: ValueKind) -> bool:
    for value in values:
        if not value:
            continue
        for page in layer.pages:
            if value_supported_by(value, page.text, kind) or value_supported_by(
                value, page.row_text, kind
            ):
                return True
    return False


def _value_fuzzy(layer: textlayer.DocumentText, values: Iterable[str]) -> bool:
    return any(
        value and textlayer.find_in_document(layer, value, min_score=90.0) is not None
        for value in values
    )


def _span_exact(layer: textlayer.DocumentText, texts: Iterable[str]) -> bool:
    return any(
        textlayer.find_all_in_page(page, text)
        for text in texts
        if text
        for page in layer.pages
    )


def _span_fuzzy(layer: textlayer.DocumentText, texts: Iterable[str]) -> bool:
    return any(
        text and textlayer.find_in_document(layer, text, min_score=90.0) is not None
        for text in texts
    )


def _flag(value: bool | None) -> str:
    if value is None:
        return ""
    return "yes" if value else "no"


def analyze_field(
    document: str,
    field_name: str,
    gold: FieldAnnotation,
    native: textlayer.DocumentText,
    ocr: textlayer.DocumentText | None,
    kind: ValueKind,
) -> OracleRow:
    values = [value for value in gold.all_values if value]
    has_value = gold.state is FieldState.PRESENT and bool(values)
    targets = gold.gold_targets
    span_texts = [item.text for item in targets] or [item.text for item in gold.evidence]
    gold_pages = {item.page for item in targets} or {item.page for item in gold.evidence}
    has_span = bool(span_texts)

    row = OracleRow(
        document=document,
        field=field_name,
        gold_state=gold.state.value,
        gold_derivation=gold.derivation.value,
        gold_value=gold.value or "",
    )
    notes: list[str] = []

    for suffix, layer in (("native", native), ("ocr", ocr)):
        if layer is None:
            continue
        if has_value:
            anchors = locate_value_exact(layer, values)
            pages = sorted({anchor.page_number for anchor in anchors})
            setattr(row, f"value_exact_{suffix}", _flag(bool(anchors)))
            setattr(row, f"value_typed_{suffix}", _flag(_value_typed(layer, values, kind)))
            setattr(row, f"value_fuzzy_{suffix}", _flag(_value_fuzzy(layer, values)))
            setattr(row, f"value_located_{suffix}", _flag(bool(anchors)))
            setattr(row, f"value_located_pages_{suffix}", ";".join(str(page) for page in pages))
            setattr(
                row,
                f"value_on_gold_page_{suffix}",
                _flag(bool(set(pages) & gold_pages)) if anchors else "no",
            )
            if not anchors:
                notes.append(f"value not locatable in {suffix}")
        if has_span:
            setattr(row, f"span_exact_{suffix}", _flag(_span_exact(layer, span_texts)))
            if suffix == "ocr":
                row.span_fuzzy_ocr = _flag(_span_fuzzy(layer, span_texts))
            if getattr(row, f"span_exact_{suffix}") == "no":
                notes.append(f"gold span missing from {suffix}")

    row.note = "; ".join(notes)
    return row


def analyze(documents: list[DocumentAnnotation]) -> list[OracleRow]:
    rows: list[OracleRow] = []
    for annotation in documents:
        native = textlayer.load(annotation.document, textlayer.NATIVE)
        try:
            ocr = textlayer.load(annotation.document, textlayer.OCR)
        except FileNotFoundError:
            ocr = None

        for field_name in PRODUCT_FIELD_NAMES:
            gold = annotation.fields.get(field_name)
            if gold is None:
                continue
            rows.append(
                analyze_field(
                    annotation.document,
                    field_name,
                    gold,
                    native,
                    ocr,
                    FIELD_SPECS[field_name].value_kind,
                )
            )

        for item in annotation.ingredients.items:
            if item.cas_state is FieldState.PRESENT and item.cas_number:
                ingredient_gold = FieldAnnotation(
                    state=FieldState.PRESENT,
                    derivation=Derivation.EXPLICIT,
                    value=item.cas_number,
                    evidence=list(item.evidence),
                    resolved_evidence=list(item.resolved_evidence),
                )
                rows.append(
                    analyze_field(
                        annotation.document,
                        f"ingredient_cas:{item.cas_number}",
                        ingredient_gold,
                        native,
                        ocr,
                        ValueKind.IDENTIFIER,
                    )
                )
    return rows


def summarize(rows: list[OracleRow]) -> list[str]:
    lines: list[str] = []

    def rate(subset: list[OracleRow], attribute: str) -> str:
        decided = [item for item in subset if getattr(item, attribute)]
        yes = sum(1 for item in decided if getattr(item, attribute) == "yes")
        if not decided:
            return "  n/a"
        return f"{yes / len(decided):6.1%}  ({yes}/{len(decided)})"

    present = [item for item in rows if item.gold_state == FieldState.PRESENT.value]
    product_present = [item for item in present if not item.field.startswith("ingredient_")]
    ingredient_present = [item for item in present if item.field.startswith("ingredient_")]
    spans = [item for item in rows if item.span_exact_native]
    direct_spans = [item for item in spans if item.gold_derivation in ("EXPLICIT",)]
    premise_spans = [item for item in spans if item.gold_derivation in ("ONTOLOGY", "COMPOSITION_IMPLIED")]

    lines.append("Gold PRESENT product values, searched by the value alone (no gold hints)")
    lines.append(f"  exact occurrence, native     {rate(product_present, 'value_exact_native')}")
    lines.append(f"  exact occurrence, OCR        {rate(product_present, 'value_exact_ocr')}")
    lines.append(f"  typed containment, native    {rate(product_present, 'value_typed_native')}")
    lines.append(f"  typed containment, OCR       {rate(product_present, 'value_typed_ocr')}")
    lines.append(f"  fuzzy >= 90 (diagnostic), native {rate(product_present, 'value_fuzzy_native')}")
    lines.append(f"  fuzzy >= 90 (diagnostic), OCR    {rate(product_present, 'value_fuzzy_ocr')}")
    lines.append(f"  located to a box, native     {rate(product_present, 'value_located_native')}")
    lines.append(f"    of which on a gold page    {rate(product_present, 'value_on_gold_page_native')}")
    lines.append(f"  located to a box, OCR        {rate(product_present, 'value_located_ocr')}")
    lines.append(f"    of which on a gold page    {rate(product_present, 'value_on_gold_page_ocr')}")
    lines.append("")
    lines.append("Gold ingredient CAS numbers, searched by the value alone")
    lines.append(f"  exact occurrence, native     {rate(ingredient_present, 'value_exact_native')}")
    lines.append(f"  exact occurrence, OCR        {rate(ingredient_present, 'value_exact_ocr')}")
    lines.append(f"  located to a box, native     {rate(ingredient_present, 'value_located_native')}")
    lines.append(f"  located to a box, OCR        {rate(ingredient_present, 'value_located_ocr')}")
    lines.append("")
    lines.append("Gold spans recoverable (reported apart from value localization)")
    lines.append(f"  direct evidence, exact native   {rate(direct_spans, 'span_exact_native')}")
    lines.append(f"  direct evidence, exact OCR      {rate(direct_spans, 'span_exact_ocr')}")
    lines.append(f"  direct evidence, fuzzy OCR      {rate(direct_spans, 'span_fuzzy_ocr')}")
    lines.append(f"  premises, exact native          {rate(premise_spans, 'span_exact_native')}")
    lines.append(f"  premises, exact OCR             {rate(premise_spans, 'span_exact_ocr')}")
    lines.append(f"  ingredient rows, exact native   {rate(ingredient_present, 'span_exact_native')}")
    lines.append(f"  ingredient rows, exact OCR      {rate(ingredient_present, 'span_exact_ocr')}")

    for suffix, label in (("native", "native text"), ("ocr", "OCR")):
        missing = [
            item for item in product_present if getattr(item, f"value_located_{suffix}") == "no"
        ]
        if missing:
            lines.append("")
            lines.append(f"PRESENT product values not locatable by value in {label}:")
            for item in missing:
                typed = getattr(item, f"value_typed_{suffix}")
                lines.append(
                    f"  {item.document}/{item.field}: {item.gold_value!r}"
                    f" (typed containment: {typed or 'n/a'})"
                )
    return lines
