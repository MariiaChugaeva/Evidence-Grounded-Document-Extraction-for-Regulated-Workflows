"""Rule-based baseline. Configurable by text layer (native/ocr) and layout (rows/linear).

rules-v3: claims that follow from the product form are emitted as DERIVED
claims (premises + rule) instead of citing the form declaration as if it
were direct evidence; a product form inferred from the composition table is
itself a derived claim; ingredient rows are split into name, CAS and
concentration.

The label lists and the form-declaration phrases were written while looking
at the development sheets (sds_01-20). They must not be tuned on a locked
split.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dataclass_field

from .. import textlayer
from ..derivation import RULE_COMPOSITION_FORM, RULE_IDENTIFIER_UNDEFINED
from ..matching import CAS_PATTERN, cas_checksum_valid, normalize_cas, parse_concentration
from ..ontology import (
    FIELD_SPECS,
    FieldState,
    ProductForm,
    ValueKind,
    classify_state_cue,
    field_is_defined_for,
)
from ..schema import (
    DocumentPrediction,
    EvidenceSpan,
    FieldPrediction,
    IngredientPrediction,
    IngredientsPrediction,
)

EXTRACTOR_VERSION = "rules-v3"

LAYOUT_ROWS = "rows"
LAYOUT_LINEAR = "linear"


@dataclass
class Candidate:
    """A label hit and the tokens that fill its value slot."""

    page_number: int
    label_tokens: list[textlayer.Token]
    value_tokens: list[textlayer.Token]

    @property
    def raw_value_text(self) -> str:
        return " ".join(token.text for token in self.value_tokens).strip()

    @property
    def value_text(self) -> str:
        return self.raw_value_text.strip(" :;–-")

    @property
    def span_text(self) -> str:
        return " ".join(
            token.text for token in self.label_tokens + self.value_tokens
        ).strip()

    @property
    def bbox(self) -> list[float] | None:
        box = textlayer.union_bbox(
            [token.bbox for token in self.label_tokens + self.value_tokens]
        )
        return list(box) if box else None

    @property
    def ocr_confidence(self) -> float | None:
        scores = [
            token.confidence
            for token in self.label_tokens + self.value_tokens
            if token.confidence is not None and token.confidence >= 0
        ]
        return sum(scores) / len(scores) / 100.0 if scores else None


def _normalize(text: str) -> str:
    return " ".join(textlayer.fold(text).lower().split())


def _phrase_matches(tokens: list[textlayer.Token], start: int, phrase: str) -> int:
    words = phrase.split()
    if start + len(words) > len(tokens):
        return 0
    window = _normalize(" ".join(token.text for token in tokens[start : start + len(words)]))
    return len(words) if window.rstrip(":") == phrase else 0


_LINE_PREFIX = re.compile(r"^(\d+(\.\d+)*\.?|[•·*\-–])$")
_COLUMN_GAP = 0.015
_MAX_LINEAR_VALUE_TOKENS = 12


def _starts_line(tokens: list[textlayer.Token], start: int) -> bool:
    return all(_LINE_PREFIX.match(token.text) for token in tokens[:start])


def _separator_follows(tokens: list[textlayer.Token], end: int) -> bool:
    if end > 0 and tokens[end - 1].text.rstrip().endswith(":"):
        return True
    if end >= len(tokens):
        return True
    if tokens[end].text in {":", "="}:
        return True
    return tokens[end].bbox[0] - tokens[end - 1].bbox[2] >= _COLUMN_GAP


def _value_slot(
    tokens: list[textlayer.Token], end: int
) -> list[textlayer.Token]:
    rest = list(tokens[end:])
    while rest and rest[0].text in {":", "="}:
        rest = rest[1:]
    return rest


def _row_candidates(page: textlayer.Page, labels: tuple[str, ...]) -> list[Candidate]:
    found: list[Candidate] = []
    page_rows = textlayer.rows(page)
    for index, row in enumerate(page_rows):
        for label in labels:
            hit = None
            for start in range(len(row.tokens)):
                length = _phrase_matches(row.tokens, start, label)
                if (
                    length
                    and _starts_line(row.tokens, start)
                    and _separator_follows(row.tokens, start + length)
                ):
                    hit = (start, length)
                    break
            if hit is None:
                continue
            start, length = hit
            rest = _value_slot(row.tokens, start + length)
            if not rest and index + 1 < len(page_rows):
                following = page_rows[index + 1]
                if not _row_holds_label(following, labels):
                    rest = following.tokens
            found.append(
                Candidate(
                    page_number=page.page_number,
                    label_tokens=row.tokens[start : start + length],
                    value_tokens=rest,
                )
            )
            break
    return found


def _row_holds_label(row: textlayer.Row, labels: tuple[str, ...]) -> bool:
    return any(
        _phrase_matches(row.tokens, position, label) and _starts_line(row.tokens, position)
        for position in range(len(row.tokens))
        for label in labels
    )


def _line_start_indices(page: textlayer.Page) -> set[int]:
    starts = set()
    for index, (span_start, _) in enumerate(page.token_spans):
        if span_start == 0 or page.text[span_start - 1] == "\n":
            starts.add(index)
    return starts


def _linear_candidates(page: textlayer.Page, labels: tuple[str, ...]) -> list[Candidate]:
    found: list[Candidate] = []
    tokens = page.tokens
    line_starts = _line_start_indices(page)
    for start in range(len(tokens)):
        at_line_start = start in line_starts or (
            start - 1 in line_starts and bool(_LINE_PREFIX.match(tokens[start - 1].text))
        )
        if not at_line_start:
            continue
        for label in labels:
            length = _phrase_matches(tokens, start, label)
            if not length:
                continue
            end = start + length
            if not (
                end >= len(tokens)
                or tokens[end - 1].text.rstrip().endswith(":")
                or tokens[end].text in {":", "="}
                or end in line_starts
            ):
                continue
            value_tokens = _value_slot(tokens, end)[:_MAX_LINEAR_VALUE_TOKENS]
            for offset in range(len(value_tokens)):
                if any(
                    _phrase_matches(tokens, end + offset, other) for other in labels
                ):
                    value_tokens = value_tokens[:offset]
                    break
            found.append(
                Candidate(
                    page_number=page.page_number,
                    label_tokens=tokens[start:end],
                    value_tokens=value_tokens,
                )
            )
            break
    return found


def find_candidates(
    document_text: textlayer.DocumentText, labels: tuple[str, ...], layout: str
) -> list[Candidate]:
    finder = _row_candidates if layout == LAYOUT_ROWS else _linear_candidates
    return [
        candidate
        for page in document_text.pages
        for candidate in finder(page, labels)
    ]


PRODUCT_NAME_LABELS = (
    "ghs product identifier",
    "product identifier",
    "product description",
    "identification of the substance",
    "product name",
    "material name",
    "trade name",
    "product",
)

SUPPLIER_LABELS = (
    "manufacturer/importer/supplier/distributor information",
    "details of the supplier of the safety data sheet",
    "details of the supplier of the data sheet",
    "uk entity/business name",
    "manufacturer's name",
    "supplier's details",
    "company name",
    "contact address",
    "manufactured by",
    "manufacturer",
    "distributor",
    "supplier",
    "company",
)

FLASH_POINT_LABELS = (
    "closed-cup flash point",
    "flash point [method]",
    "flash point",
    "flashpoint",
    "flash pt",
)

FORMULA_LABELS = (
    "chemical formula",
    "molecular formula",
    "formula",
)

WEIGHT_LABELS = (
    "average molecular weight",
    "molecular weight",
    "molecular mass",
    "molar mass",
)

CAS_LABELS = (
    "cas registry number",
    "cas number",
    "c.a.s. no.",
    "cas no.",
    "cas no",
    "cas #",
    "cas#",
    "cas",
)

# Direct product-form declarations. A row containing one of these phrases is
# cited as direct evidence of the form; the evaluator re-checks the citation
# with the gold-free form lexicon (ontology.classify_form_cue).
FORM_DECLARATIONS: tuple[tuple[str, ProductForm], ...] = (
    ("defined by osha as an", ProductForm.ARTICLE),
    ("substance/mixture : substance", ProductForm.SUBSTANCE),
    ("substance/mixture: substance", ProductForm.SUBSTANCE),
    ("3.1. substances not applicable", ProductForm.MIXTURE),
    ("3.1 substances not applicable", ProductForm.MIXTURE),
    ("substance: non-applicable", ProductForm.MIXTURE),
    ("substances not applicable", ProductForm.MIXTURE),
    ("regulated as a mixture", ProductForm.MIXTURE),
    ("defined as a mixture", ProductForm.MIXTURE),
    ("mixture of substances", ProductForm.MIXTURE),
    ("chemical description: mixture", ProductForm.MIXTURE),
    ("substance or preparation: preparation", ProductForm.MIXTURE),
    ("mixture identification", ProductForm.MIXTURE),
    ("3.1 n/a", ProductForm.MIXTURE),
    ("3.2 mixtures", ProductForm.MIXTURE),
    ("3.2. mixtures", ProductForm.MIXTURE),
)

_ADDRESS_STOPWORDS = re.compile(
    r"^(apartado|address|tel|tel\.|telephone|phone|fax|e-?mail|www|http|street|str\.|"
    r"\d{1,5}$)",
    re.IGNORECASE,
)


def _trim_company_name(tokens: list[textlayer.Token]) -> list[textlayer.Token]:
    kept: list[textlayer.Token] = []
    for token in tokens:
        if _ADDRESS_STOPWORDS.match(token.text) and kept:
            break
        kept.append(token)
        if len(kept) >= 8:
            break
    return kept


def _evidence(candidate: Candidate) -> EvidenceSpan:
    return EvidenceSpan(
        page=candidate.page_number,
        text=candidate.span_text,
        bbox=candidate.bbox,
        bbox_provenance="system",
    )


def _confidence(candidate: Candidate, prior: float) -> float:
    ocr = candidate.ocr_confidence
    return round(prior if ocr is None else 0.5 * prior + 0.5 * ocr, 4)


def _not_stated() -> FieldPrediction:
    return FieldPrediction(state=FieldState.NOT_STATED, value=None, evidence=None)


def _from_candidate(
    candidate: Candidate,
    prior: float,
    parse_value,
    extra_cues: tuple[tuple[str, FieldState], ...] = (),
) -> FieldPrediction | None:
    raw = candidate.raw_value_text
    if not raw:
        return None
    normalized = _normalize(raw)
    cue = next((state for phrase, state in extra_cues if phrase in normalized), None)
    if cue is None:
        cue = classify_state_cue(raw)
    if cue is not None:
        return FieldPrediction(
            state=cue,
            value=None,
            evidence=_evidence(candidate),
            confidence=_confidence(candidate, prior),
        )
    value = parse_value(candidate.value_text)
    if value is None:
        return None
    return FieldPrediction(
        state=FieldState.PRESENT,
        value=value,
        evidence=_evidence(candidate),
        confidence=_confidence(candidate, prior),
    )


def _first_usable(
    candidates: list[Candidate],
    prior: float,
    parse_value,
    extra_cues: tuple[tuple[str, FieldState], ...] = (),
) -> FieldPrediction | None:
    for candidate in candidates:
        prediction = _from_candidate(candidate, prior, parse_value, extra_cues)
        if prediction is not None:
            return prediction
    return None


def _parse_free_text(raw: str) -> str | None:
    cleaned = raw.strip(" :;,.")
    return cleaned or None


def _parse_company(raw: str) -> str | None:
    cleaned = raw.strip(" :;,.")
    return cleaned or None


def _parse_formula(raw: str) -> str | None:
    cleaned = raw.strip(" :;,.")
    if not cleaned or len(cleaned) > 40:
        return None
    if not re.fullmatch(r"[A-Za-z0-9\[\]\(\)\.·\*\s₀-₉]+", cleaned):
        return None
    if not re.search(r"[A-Z]", cleaned):
        return None
    if not re.search(r"\d", cleaned) and len(cleaned.replace(" ", "")) > 3:
        return None
    if cleaned.replace(" ", "").isalpha() and cleaned.islower():
        return None
    return cleaned


def _parse_weight(raw: str) -> str | None:
    cleaned = raw.strip(" :;,.")
    return cleaned if re.search(r"\d", cleaned) else None


FLASH_POINT_CUES: tuple[tuple[str, FieldState], ...] = (
    ("does not sustain combustion", FieldState.NOT_APPLICABLE),
    ("does not flash", FieldState.NOT_APPLICABLE),
    ("non-combustible", FieldState.NOT_APPLICABLE),
    ("noncombustible", FieldState.NOT_APPLICABLE),
    ("not combustible", FieldState.NOT_APPLICABLE),
)


def _parse_flash_point(raw: str) -> str | None:
    cleaned = raw.strip(" :;,")
    return cleaned if re.search(r"\d", cleaned) else None


def _parse_cas(raw: str) -> str | None:
    for token in raw.split():
        cleaned = normalize_cas(token)
        if CAS_PATTERN.match(cleaned):
            return cleaned
    return None


def _row_evidence(page_number: int, row: textlayer.Row) -> EvidenceSpan:
    box = textlayer.union_bbox([token.bbox for token in row.tokens])
    return EvidenceSpan(
        page=page_number,
        text=row.text.strip(),
        bbox=list(box) if box else None,
        bbox_provenance="system",
    )


@dataclass
class FormFinding:
    """How the product form was established: a direct declaration or a derivation."""

    form: ProductForm
    evidence: EvidenceSpan | None = None
    premises: list[EvidenceSpan] = dataclass_field(default_factory=list)
    rule: str | None = None
    confidence: float = 0.0

    @property
    def derived(self) -> bool:
        return self.rule is not None

    @property
    def cited(self) -> list[EvidenceSpan]:
        if self.derived:
            return list(self.premises)
        return [self.evidence] if self.evidence is not None else []


def determine_product_form(document_text: textlayer.DocumentText) -> FormFinding:
    for phrase, form in FORM_DECLARATIONS:
        for page in document_text.pages:
            for row in textlayer.rows(page):
                if phrase in _normalize(row.text):
                    return FormFinding(
                        form=form, evidence=_row_evidence(page.page_number, row), confidence=0.8
                    )

    composition = extract_ingredients(document_text)
    if len(composition.items) >= 2:
        premises = [item.evidence for item in composition.items[:2] if item.evidence]
        return FormFinding(
            form=ProductForm.MIXTURE,
            premises=premises,
            rule=RULE_COMPOSITION_FORM,
            confidence=0.5,
        )
    if len(composition.items) == 1:
        only = composition.items[0]
        parsed = parse_concentration(only.concentration)
        full = parsed is not None and parsed.low is not None and parsed.low >= 90.0
        if full and only.evidence is not None:
            return FormFinding(
                form=ProductForm.SUBSTANCE,
                premises=[only.evidence],
                rule=RULE_COMPOSITION_FORM,
                confidence=0.5,
            )
    return FormFinding(form=ProductForm.UNDETERMINED)


_SECTION_3 = re.compile(
    r"^(section\s*)?0?3[\.:)\s]|^composition\b|^3\.\s*composition", re.IGNORECASE
)
_SECTION_4 = re.compile(
    r"^(section\s*)?0?4[\.:)\s]|^first[\s-]?aid", re.IGNORECASE
)


def _composition_rows(
    document_text: textlayer.DocumentText,
) -> list[tuple[int, textlayer.Row]]:
    collected: list[tuple[int, textlayer.Row]] = []
    inside = False
    for page in document_text.pages:
        for row in textlayer.rows(page):
            heading = row.text.strip()
            if _SECTION_4.match(heading):
                if inside:
                    return collected
                continue
            if _SECTION_3.match(heading):
                inside = True
            if inside:
                collected.append((page.page_number, row))
    return collected


_CONCENTRATION_TOKEN = re.compile(
    r"^(?:<=|>=|[<>≤≥~≈])?\d+(?:[.,]\d+)?%?$"
    r"|^\d+(?:[.,]\d+)?%?\s*[-–~]\s*<?\d+(?:[.,]\d+)?%?$"
    r"|^%$|^(?:[-–—~]|to)$|^(?:<=|>=|[<>≤≥≈~])$|^(?:nmt|nlt|max\.?|min\.?|approx\.?|ca\.?)$",
    re.IGNORECASE,
)
_EC_TOKEN = re.compile(r"^\d{3}-\d{3}-\d$")
_LABEL_TOKEN = re.compile(
    r"^(?:cas|cas[:#]|cas-?no\.?:?|c\.a\.s\.?|no\.?:?|number:?|nr\.?:?|ec|ec-?no\.?:?|einecs|"
    r"index|reach|reg\.?|registration:?|#|:|\||-)$",
    re.IGNORECASE,
)


def _concentration_run(tokens: list[textlayer.Token]) -> tuple[int, int] | None:
    """Indices [start, end) of the token run most likely to be the concentration cell."""
    best: tuple[int, int] | None = None
    best_numbers = 0
    index = 0
    while index < len(tokens):
        if not _CONCENTRATION_TOKEN.match(tokens[index].text):
            index += 1
            continue
        end = index
        while end < len(tokens) and _CONCENTRATION_TOKEN.match(tokens[end].text):
            end += 1
        numbers = sum(bool(re.search(r"\d", token.text)) for token in tokens[index:end])
        if numbers and numbers >= best_numbers:
            best, best_numbers = (index, end), numbers
        index = end
    return best


def _split_ingredient_row(
    tokens: list[textlayer.Token], cas_index: int
) -> tuple[str | None, str | None]:
    """(name, concentration) from a composition row whose CAS sits at cas_index."""
    run = _concentration_run(tokens)
    excluded = {cas_index}
    if run is not None:
        excluded.update(range(run[0], run[1]))
    name_tokens = [
        token.text
        for position, token in enumerate(tokens)
        if position not in excluded
        and not _EC_TOKEN.match(normalize_cas(token.text))
        and not _LABEL_TOKEN.match(token.text)
    ]
    name = " ".join(name_tokens).strip(" :;,|-") or None
    concentration = (
        " ".join(token.text for token in tokens[run[0] : run[1]]).strip() if run else None
    )
    return name, concentration


def extract_ingredients(
    document_text: textlayer.DocumentText,
) -> IngredientsPrediction:
    items: list[IngredientPrediction] = []
    seen: set[str] = set()
    header: EvidenceSpan | None = None

    for page_number, row in _composition_rows(document_text):
        if header is None and re.search(r"(?<![a-z])cas(?![a-z])", row.text.lower()):
            header = _row_evidence(page_number, row)
        for position, token in enumerate(row.tokens):
            cleaned = normalize_cas(token.text)
            if not CAS_PATTERN.match(cleaned) or cleaned in seen:
                continue
            seen.add(cleaned)
            name, concentration = _split_ingredient_row(row.tokens, position)
            items.append(
                IngredientPrediction(
                    name=name,
                    cas_number=cleaned,
                    cas_state=FieldState.PRESENT,
                    concentration=concentration,
                    evidence=_row_evidence(page_number, row),
                )
            )

    return IngredientsPrediction(
        state=FieldState.PRESENT if items else FieldState.NOT_STATED,
        items=items,
        evidence=header,
    )


def extract_document(
    document: str,
    layer: str = textlayer.NATIVE,
    layout: str = LAYOUT_ROWS,
) -> DocumentPrediction:
    document_text = textlayer.load(document, layer)
    finding = determine_product_form(document_text)
    form = finding.form

    fields: dict[str, FieldPrediction] = {}

    fields["product_name"] = (
        _first_usable(
            find_candidates(document_text, PRODUCT_NAME_LABELS, layout), 0.9, _parse_free_text
        )
        or _not_stated()
    )

    supplier_candidates = find_candidates(document_text, SUPPLIER_LABELS, layout)
    for candidate in supplier_candidates:
        candidate.value_tokens = _trim_company_name(candidate.value_tokens)
    fields["supplier_name"] = (
        _first_usable(supplier_candidates, 0.85, _parse_company) or _not_stated()
    )

    if form is ProductForm.UNDETERMINED or not finding.cited:
        fields["substance_or_mixture"] = _not_stated()
    elif finding.derived:
        fields["substance_or_mixture"] = FieldPrediction(
            state=FieldState.PRESENT,
            value=form.value,
            derivation="DERIVED",
            rule=finding.rule,
            premises=finding.premises,
            confidence=finding.confidence,
            rationale="composition table",
        )
    else:
        fields["substance_or_mixture"] = FieldPrediction(
            state=FieldState.PRESENT,
            value=form.value,
            evidence=finding.evidence,
            confidence=finding.confidence,
        )

    conditional = {
        "product_cas_number": (CAS_LABELS, 0.9, _parse_cas),
        "chemical_formula": (FORMULA_LABELS, 0.85, _parse_formula),
        "molecular_weight": (WEIGHT_LABELS, 0.9, _parse_weight),
    }
    for name, (labels, prior, parser) in conditional.items():
        explicit = _first_usable(find_candidates(document_text, labels, layout), prior, parser)
        if explicit is not None and explicit.state is not FieldState.PRESENT:
            fields[name] = explicit
            continue
        if not field_is_defined_for(name, form) and finding.cited:
            fields[name] = FieldPrediction(
                state=FieldState.NOT_APPLICABLE,
                value=None,
                derivation="DERIVED",
                rule=RULE_IDENTIFIER_UNDEFINED,
                premises=finding.cited,
                confidence=finding.confidence,
                rationale=f"derived: identifiers are undefined for {form.value}",
            )
            continue
        fields[name] = explicit or _not_stated()

    fields["flash_point"] = (
        _first_usable(
            find_candidates(document_text, FLASH_POINT_LABELS, layout),
            0.9,
            _parse_flash_point,
            FLASH_POINT_CUES,
        )
        or _not_stated()
    )

    for name, prediction in fields.items():
        if (
            prediction.state is FieldState.PRESENT
            and FIELD_SPECS[name].value_kind is ValueKind.IDENTIFIER
            and prediction.value
            and not cas_checksum_valid(prediction.value)
        ):
            prediction.rationale = "CAS check digit failed"

    return DocumentPrediction(
        document=document,
        fields=fields,
        ingredients=extract_ingredients(document_text),
    )
