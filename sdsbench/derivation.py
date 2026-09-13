"""Derivation rules: a derived claim cites premise spans plus the rule that licenses it.

An *observed* claim is stated by the cited span itself ("Flash point: 61 °C").
A *derived* claim follows from the document through one of the rules below
("the product is a mixture, therefore a product-level CAS number does not
apply"). A derived claim is scored on three things: its premises are located
(page / span / bbox against the gold premises), it names the rule the gold
names (derivation axis), and the rule actually licenses the conclusion from
the cited premise texts (support axis). Every check here is gold-free: it
reads only the texts the system cited.

Sources of the rules:

- REACH Annex II (Regulation (EU) 2020/878) splits Section 3 into 3.1
  Substances and 3.2 Mixtures, and asks for a product identifier (CAS / EC
  number, molecular formula) only for a substance; a mixture is identified
  through its constituents. OSHA HCS 29 CFR 1910.1200 Appendix D keeps the
  same split. An article is not a chemical product at all.
- A composition table with several constituents, or a single constituent at
  100 %, is the conventional way a sheet shows the product form when the
  words substance / mixture are never used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .ontology import FieldState, ProductForm, classify_form_cue

RULE_IDENTIFIER_UNDEFINED = "identifier_undefined_for_form"
RULE_COMPOSITION_FORM = "composition_implies_form"

SATISFIED = "satisfied"
VIOLATED = "violated"
UNDECIDABLE = "undecidable"


@dataclass(frozen=True)
class DerivationRule:
    rule_id: str
    fields: tuple[str, ...]
    conclusion_state: FieldState
    description: str


RULES: dict[str, DerivationRule] = {
    RULE_IDENTIFIER_UNDEFINED: DerivationRule(
        rule_id=RULE_IDENTIFIER_UNDEFINED,
        fields=("product_cas_number", "chemical_formula", "molecular_weight"),
        conclusion_state=FieldState.NOT_APPLICABLE,
        description=(
            "If the premises establish that the product is a mixture or an "
            "article, the product-level CAS number, molecular formula and "
            "molecular weight are not applicable."
        ),
    ),
    RULE_COMPOSITION_FORM: DerivationRule(
        rule_id=RULE_COMPOSITION_FORM,
        fields=("substance_or_mixture",),
        conclusion_state=FieldState.PRESENT,
        description=(
            "A composition table with several constituents implies a mixture; "
            "a single constituent at 100 % implies a substance."
        ),
    ),
}


@dataclass(frozen=True)
class RuleCheck:
    status: str
    note: str = ""
    form: ProductForm | None = None


_CAS_WORD = r"(?<![a-z])(?:cas|c\.a\.s\.?)(?![a-z])"
_COMPOSITION_HEADER = re.compile(
    rf"{_CAS_WORD}.*?(?:%|(?<![a-z])wt(?![a-z])|weight|concentration|(?<![a-z])conc(?![a-z])|content)"
    rf"|(?:%|(?<![a-z])wt(?![a-z])|weight|concentration|content).*?{_CAS_WORD}"
)
_CAS_NUMBER = re.compile(r"(?<![\d-])\d{2,7}-\d{2}-\d(?![\d-])")
_FULL_CONTENT = re.compile(r"(?<![\d.,])100(?:[.,]0+)?\s*%?(?![\d.,])")


def form_from_composition(text: str) -> ProductForm | None:
    """The form a composition span establishes, or None."""
    normalized = " ".join(text.lower().split())
    if not normalized:
        return None
    if _CAS_NUMBER.search(normalized) and _FULL_CONTENT.search(normalized):
        return ProductForm.SUBSTANCE
    if _COMPOSITION_HEADER.search(normalized):
        return ProductForm.MIXTURE
    return None


def form_established_by(text: str) -> ProductForm | None:
    """Direct declaration first, composition convention second."""
    return classify_form_cue(text) or form_from_composition(text)


def premise_form(texts: Iterable[str]) -> ProductForm | None:
    """The single product form the premise texts establish, or None.

    Two or more distinct CAS numbers across the premises are composition rows
    of several constituents, which establishes a mixture. Conflicting
    premises (one says substance, another mixture) establish nothing.
    """
    texts = list(texts)
    forms = {
        form
        for form in (form_established_by(text) for text in texts)
        if form is not None and form is not ProductForm.UNDETERMINED
    }
    cas_numbers = {
        match.group(0)
        for text in texts
        for match in _CAS_NUMBER.finditer(" ".join(text.lower().split()))
    }
    if len(cas_numbers) >= 2:
        forms.add(ProductForm.MIXTURE)
    if len(forms) != 1:
        return None
    return next(iter(forms))


def default_rule_for(field_name: str, derivation: str) -> str | None:
    """The rule a gold derivation kind implies when the annotation names none."""
    if derivation == "ONTOLOGY":
        return RULE_IDENTIFIER_UNDEFINED
    if derivation == "COMPOSITION_IMPLIED":
        return RULE_COMPOSITION_FORM
    return None


def rule_applies(rule_id: str | None, field_name: str, state: FieldState) -> bool:
    rule = RULES.get(rule_id or "")
    if rule is None:
        return False
    return field_name in rule.fields and state is rule.conclusion_state


def check_rule(
    rule_id: str | None,
    field_name: str,
    state: FieldState,
    value: str | None,
    premise_texts: Iterable[str],
) -> RuleCheck:
    """Does the named rule license (field, state, value) from the cited premises?"""
    rule = RULES.get(rule_id or "")
    if rule is None:
        return RuleCheck(VIOLATED, f"unknown rule {rule_id!r}")
    if field_name not in rule.fields:
        return RuleCheck(VIOLATED, f"rule {rule_id} does not apply to {field_name}")
    if state is not rule.conclusion_state:
        return RuleCheck(VIOLATED, f"rule {rule_id} concludes {rule.conclusion_state.value}, not {state.value}")

    texts = [text for text in premise_texts if text and text.strip()]
    if not texts:
        return RuleCheck(VIOLATED, "no premise cited")
    form = premise_form(texts)
    if form is None:
        return RuleCheck(UNDECIDABLE, "cited premises do not establish a product form")

    if rule.rule_id == RULE_IDENTIFIER_UNDEFINED:
        if form in (ProductForm.MIXTURE, ProductForm.ARTICLE):
            return RuleCheck(SATISFIED, f"premise establishes {form.value}", form)
        return RuleCheck(VIOLATED, f"premise establishes {form.value}, identifiers stay defined", form)

    if rule.rule_id == RULE_COMPOSITION_FORM:
        claimed = " ".join((value or "").split()).upper().replace(" ", "_")
        if claimed == form.value:
            return RuleCheck(SATISFIED, f"composition establishes {form.value}", form)
        return RuleCheck(VIOLATED, f"composition establishes {form.value}, claim says {claimed or 'nothing'}", form)

    return RuleCheck(VIOLATED, "rule has no checker")
