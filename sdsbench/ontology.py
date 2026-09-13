"""Field ontology: states, product forms, comparison kinds, and the cue lexicons.

Two gold-free lexicons live here because both the extractor and the evaluator
need them: state cues (a value slot that says "not applicable" / "no data")
and product-form declarations (a span that says the product is a substance,
a mixture or an article). Neither lexicon may be tuned on a locked split.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class FieldState(str, Enum):
    PRESENT = "PRESENT"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NO_DATA = "NO_DATA"
    NOT_STATED = "NOT_STATED"

    @property
    def requires_evidence(self) -> bool:
        return self is not FieldState.NOT_STATED


class ValueKind(str, Enum):
    NAME = "NAME"
    IDENTIFIER = "IDENTIFIER"
    FORMULA = "FORMULA"
    QUANTITY = "QUANTITY"
    TEMPERATURE = "TEMPERATURE"
    CATEGORICAL = "CATEGORICAL"


class ProductForm(str, Enum):
    SUBSTANCE = "SUBSTANCE"
    MIXTURE = "MIXTURE"
    ARTICLE = "ARTICLE"
    UNDETERMINED = "UNDETERMINED"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    value_kind: ValueKind
    description: str
    typical_sections: tuple[str, ...] = ()
    defined_for_forms: tuple[ProductForm, ...] | None = None
    allowed_values: tuple[str, ...] = ()
    numeric_tolerance: float = 0.0
    canonical_unit: str | None = None


PRODUCT_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        name="product_name",
        value_kind=ValueKind.NAME,
        description="Trade name / product identifier.",
        typical_sections=("1.1",),
    ),
    FieldSpec(
        name="supplier_name",
        value_kind=ValueKind.NAME,
        description="Supplier, manufacturer or distributor named on the SDS.",
        typical_sections=("1.3",),
    ),
    FieldSpec(
        name="substance_or_mixture",
        value_kind=ValueKind.CATEGORICAL,
        description="Whether the product is a substance, mixture or article.",
        typical_sections=("3.1", "3.2"),
        allowed_values=tuple(form.value for form in ProductForm),
    ),
    FieldSpec(
        name="product_cas_number",
        value_kind=ValueKind.IDENTIFIER,
        description="CAS of the product itself. Defined only for a substance.",
        typical_sections=("1.1", "3.1"),
        defined_for_forms=(ProductForm.SUBSTANCE,),
    ),
    FieldSpec(
        name="chemical_formula",
        value_kind=ValueKind.FORMULA,
        description="Molecular formula of the product. Defined only for a substance.",
        typical_sections=("1.1", "3.1", "9.1"),
        defined_for_forms=(ProductForm.SUBSTANCE,),
    ),
    FieldSpec(
        name="molecular_weight",
        value_kind=ValueKind.QUANTITY,
        description="Molecular weight in g/mol. Defined only for a substance.",
        typical_sections=("3.1", "9.1", "9.2"),
        defined_for_forms=(ProductForm.SUBSTANCE,),
        numeric_tolerance=0.05,
        canonical_unit="g/mol",
    ),
    FieldSpec(
        name="flash_point",
        value_kind=ValueKind.TEMPERATURE,
        description="Flash point, compared in degrees Celsius.",
        typical_sections=("9.1",),
        numeric_tolerance=0.6,
        canonical_unit="C",
    ),
)

INGREDIENT_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("ingredient_name", ValueKind.NAME, "Declared ingredient name.", ("3.2",)),
    FieldSpec(
        "ingredient_cas_number",
        ValueKind.IDENTIFIER,
        "CAS of a declared ingredient.",
        ("3.2",),
    ),
    FieldSpec(
        "ingredient_concentration",
        ValueKind.QUANTITY,
        "Declared concentration or range.",
        ("3.2",),
    ),
)

FIELD_SPECS: dict[str, FieldSpec] = {spec.name: spec for spec in PRODUCT_FIELDS}
INGREDIENT_FIELD_SPECS: dict[str, FieldSpec] = {
    spec.name: spec for spec in INGREDIENT_FIELDS
}
PRODUCT_FIELD_NAMES: tuple[str, ...] = tuple(spec.name for spec in PRODUCT_FIELDS)


def field_is_defined_for(field_name: str, form: ProductForm | None) -> bool:
    spec = FIELD_SPECS[field_name]
    if spec.defined_for_forms is None:
        return True
    if form is None or form is ProductForm.UNDETERMINED:
        return True
    return form in spec.defined_for_forms


# ---------------------------------------------------------------------------
# State cues. Word-boundary matching: "nap" must not fire inside "naphthalene".
# The combustion phrases are flash-point specific: a sheet that says the
# product does not sustain combustion is stating that the property does not
# exist for it, which is NOT_APPLICABLE, not NO_DATA.
# ---------------------------------------------------------------------------

NOT_APPLICABLE_CUES: tuple[str, ...] = (
    "not applicable",
    "non-applicable",
    "non applicable",
    "not relevant",
    "n/a",
    "nap",
    "does not apply",
    "does not sustain combustion",
    "does not flash",
    "non-combustible",
    "noncombustible",
    "not combustible",
    "no flash point",
)

NO_DATA_CUES: tuple[str, ...] = (
    "no applicable data available",
    "no data available",
    "no data is available",
    "no data",
    "not available",
    "no information available",
    "not determined",
    "not established",
    "not specified",
    "not known",
    "unknown",
    "nav",
    "n/d",
    "no information",
)

_EMPTY_VALUE_SLOT = re.compile(r"[\s:]*[-–—]\s*$")


def _cue_matches(cue: str, text: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(cue)}(?![a-z0-9])", text) is not None


def looks_like_empty_value_slot(text: str) -> bool:
    return bool(_EMPTY_VALUE_SLOT.search(text))


def classify_state_cue(text: str) -> FieldState | None:
    """Map a value slot to NOT_APPLICABLE / NO_DATA, or None if undecidable.

    NO_DATA is tested first and longest-first so "no applicable data available"
    is not read as NOT_APPLICABLE.
    """
    normalized = " ".join(text.lower().split()).strip(" .:;")
    if not normalized:
        return None
    for cue in sorted(NO_DATA_CUES, key=len, reverse=True):
        if _cue_matches(cue, normalized):
            return FieldState.NO_DATA
    for cue in sorted(NOT_APPLICABLE_CUES, key=len, reverse=True):
        if _cue_matches(cue, normalized):
            return FieldState.NOT_APPLICABLE
    if looks_like_empty_value_slot(normalized):
        return FieldState.NO_DATA
    return None


def value_part(text: str) -> str:
    """The part of a span after its last ':' (the value slot), or the whole span.

    "Substance/mixture : Mixture" has both form words in the label; only the
    value slot says what the product is.
    """
    normalized = " ".join(text.lower().split())
    if ":" in normalized:
        _, _, tail = normalized.rpartition(":")
        tail = tail.strip(" .;")
        if tail:
            return tail
    return normalized.strip(" .;")


# ---------------------------------------------------------------------------
# Product-form declarations.
#
# A span declares the product form when its value slot names the form. The
# SDS format's own convention is recognised too: REACH Annex II and OSHA HCS
# Appendix D split Section 3 into "3.1 Substances" and "3.2 Mixtures", so a
# substances entry marked not applicable / N/A declares a mixture.
# ---------------------------------------------------------------------------

FORM_CUES: dict[ProductForm, tuple[str, ...]] = {
    ProductForm.ARTICLE: ("article", "articles"),
    ProductForm.MIXTURE: ("mixture", "mixtures", "preparation", "preparations"),
    ProductForm.SUBSTANCE: ("substance", "substances", "mono-constituent"),
}

_SUBSTANCE_ENTRY_NEGATED = re.compile(
    r"(?:(?<![a-z0-9])3\.1\.?\s*)?(?<![a-z])substances?(?![a-z])\W{0,3}"
    r"(?:not applicable|non-applicable|non applicable|n/a|not relevant|does not apply)"
    r"|(?<![a-z0-9])3\.1\.?\s*n/a(?![a-z0-9])"
)

_NEGATED_FORM = re.compile(r"\bnot\s+(?:a\s+|an\s+)?(?:hazardous\s+)?(?:substance|mixture|article)\b")

# "3.2 Mixtures Composition comments: ..." starts with the section heading; the
# plural heading word is the declaration. "Substances or 3.2 Mixtures" offers
# both headings and declares nothing.
_BOTH_HEADINGS = re.compile(
    r"^(?:3\.1\.?\s*)?substances\s+(?:or|/|and)\s+(?:3\.2\.?\s*)?mixtures(?![a-z])"
)
_SECTION_HEADING = re.compile(r"^(?:3\.[12]\.?\s*)?(mixtures|substances)(?![a-z])")


def classify_form_cue(text: str) -> ProductForm | None:
    """Which product form the span declares, or None when it declares nothing.

    Gold-free. Used by the evaluator to verify a categorical claim and by the
    rule registry to check the premise of a derived claim.
    """
    normalized = " ".join(text.lower().split())
    if not normalized:
        return None
    if _SUBSTANCE_ENTRY_NEGATED.search(normalized):
        return ProductForm.MIXTURE
    if _BOTH_HEADINGS.match(normalized):
        return None
    heading = _SECTION_HEADING.match(normalized)
    if heading:
        return ProductForm.MIXTURE if heading.group(1) == "mixtures" else ProductForm.SUBSTANCE
    part = value_part(normalized)
    if _NEGATED_FORM.search(part):
        return None
    best: tuple[int, ProductForm] | None = None
    for form, cues in FORM_CUES.items():
        for cue in cues:
            match = re.search(rf"(?<![a-z0-9]){re.escape(cue)}(?![a-z0-9])", part)
            if match and (best is None or match.start() < best[0]):
                best = (match.start(), form)
    return best[1] if best else None
