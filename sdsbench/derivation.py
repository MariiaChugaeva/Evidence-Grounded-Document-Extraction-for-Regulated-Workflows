"""Derivation rules: a derived claim cites a *complete* premise set plus the rule.

An *observed* claim is stated by the cited span itself ("Flash point: 61 °C").
A *derived* claim follows from the document through one of the rules below
("the product is a mixture, therefore a product-level CAS number does not
apply"). A derived claim is scored on three things: it names the rule the gold
names (derivation axis), the premises it cites are located (page / span / bbox
against the gold premises, one hit per required premise role), and the rule
actually licenses the conclusion from the cited premise texts (support axis).
Every check here is gold-free: it reads only the texts the system cited.

Each rule declares the *roles* its premises must fill and how many spans each
role needs. A single well-chosen span is not a premise set: a claim that fills
only part of a role list is reported as an incomplete derivation, not as a
near miss.

What the rules deliberately do **not** do
-----------------------------------------
The product form is never inferred from the number of composition rows, nor
from one constituent holding a large share of the product. A substance may
legally declare stabilisers and impurities alongside itself (REACH Art. 3(1)
counts additives needed to preserve stability and impurities as part of the
substance), so "two rows" does not mean mixture and "one row at 90 %" does not
mean substance. Diethyl ether stabilised with butyl hydroxytoluene is the
standard counterexample: two rows, one substance. A composition table
therefore licenses a form only in the two shapes below, and an explicit
declaration always takes precedence over both.

Sources of the rules:

- REACH Annex II (Regulation (EU) 2020/878) splits Section 3 into 3.1
  Substances and 3.2 Mixtures, and asks for a product identifier (CAS / EC
  number, molecular formula) only for a substance; a mixture is identified
  through its constituents. OSHA HCS 29 CFR 1910.1200 Appendix D keeps the
  same split. An article is not a chemical product at all.
- REACH Art. 3(1)/3(2): a substance may contain additives and impurities; a
  mixture is a mixture of two or more substances. A composition table shows a
  substance when it declares a single constituent at 100 % and rules out any
  further constituent, and a mixture when the product's own composition
  header is followed by at least two constituents none of which can account
  for the whole product.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

from .matching import Concentration, parse_concentration
from .ontology import FieldState, ProductForm, classify_form_cue

RULE_IDENTIFIER_UNDEFINED = "identifier_undefined_for_form"
RULE_COMPOSITION_FORM = "composition_implies_form"

SATISFIED = "satisfied"
VIOLATED = "violated"
UNDECIDABLE = "undecidable"


class PremiseRole(str, Enum):
    """What a premise span contributes to a derivation."""

    FORM_DECLARATION = "form_declaration"
    COMPOSITION_HEADER = "composition_header"
    SOLE_CONSTITUENT = "sole_constituent"
    PARTIAL_CONSTITUENT = "partial_constituent"
    COMPLETENESS_STATEMENT = "completeness_statement"


ROLE_DESCRIPTIONS: dict[PremiseRole, str] = {
    PremiseRole.FORM_DECLARATION: "a span whose value slot names the product form",
    PremiseRole.COMPOSITION_HEADER: "the composition header naming the CAS and share columns",
    PremiseRole.SOLE_CONSTITUENT: "a constituent row declared at exactly 100 %",
    PremiseRole.PARTIAL_CONSTITUENT: (
        "a constituent row whose declared share stays strictly below 100 %"
    ),
    PremiseRole.COMPLETENESS_STATEMENT: (
        "a span ruling out any further constituent (impurity, stabiliser, additive)"
    ),
}


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
            "A composition table that declares a single constituent at 100 % "
            "and rules out any further constituent implies a substance; a "
            "composition header followed by at least two constituents none of "
            "which can account for the whole product implies a mixture. "
            "Row count alone implies nothing."
        ),
    ),
}


@dataclass(frozen=True)
class PremiseRequirement:
    """The complete premise set one conclusion needs: role -> how many spans."""

    roles: tuple[tuple[PremiseRole, int], ...]
    description: str

    def missing(self, audit: "PremiseAudit") -> tuple[tuple[PremiseRole, int, int], ...]:
        """(role, required, available) for every role the premises under-fill."""
        return tuple(
            (role, needed, audit.count(role))
            for role, needed in self.roles
            if audit.count(role) < needed
        )


COMPOSITION_REQUIREMENTS: dict[ProductForm, PremiseRequirement] = {
    ProductForm.SUBSTANCE: PremiseRequirement(
        roles=(
            (PremiseRole.SOLE_CONSTITUENT, 1),
            (PremiseRole.COMPLETENESS_STATEMENT, 1),
        ),
        description=(
            "one constituent declared at exactly 100 % together with a span "
            "ruling out any further constituent"
        ),
    ),
    ProductForm.MIXTURE: PremiseRequirement(
        roles=(
            (PremiseRole.COMPOSITION_HEADER, 1),
            (PremiseRole.PARTIAL_CONSTITUENT, 2),
        ),
        description=(
            "the composition header together with at least two constituent "
            "rows whose declared share stays strictly below 100 %"
        ),
    ),
}

FORM_DECLARATION_REQUIREMENT = PremiseRequirement(
    roles=((PremiseRole.FORM_DECLARATION, 1),),
    description="a span that declares the product form",
)


# ---------------------------------------------------------------------------
# Reading roles off a premise span. Gold-free.
# ---------------------------------------------------------------------------

_CAS_WORD = r"(?<![a-z])(?:cas|c\.a\.s\.?)(?![a-z])"
_SHARE_WORD = (
    r"%|(?<![a-z])wt(?![a-z])|weight|concentration|(?<![a-z])conc(?![a-z])|content"
)
_COMPOSITION_HEADER = re.compile(
    rf"{_CAS_WORD}.*?(?:{_SHARE_WORD})|(?:{_SHARE_WORD}).*?{_CAS_WORD}"
)
_CAS_NUMBER = re.compile(r"(?<![\d-])\d{2,7}-\d{2}-\d(?![\d-])")
_EC_NUMBER = re.compile(r"(?<![\d-])\d{3}-\d{3}-\d(?![\d-])")

# A span that rules out any constituent beyond the one declared. Without such
# a statement a single row at 100 % is only a rounded share.
_COMPLETENESS = re.compile(
    r"no (?:other |further )?(?:impurit(?:y|ies)|stabili[sz]ers?|additives?)"
    r"|no (?:other|further) (?:ingredient|component|constituent|substance)s?"
    r"|contains? no (?:other|further)"
    r"|100\s*%\s*(?:pure|purity)"
    r"|purity\s*(?:of\s*)?100\s*%"
)

_FULL_SHARE_TOLERANCE = 1e-6


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _share_of(segment: str) -> Concentration | None:
    """Parse a declared share, ignoring EC numbers that look like ranges."""
    return parse_concentration(_EC_NUMBER.sub(" ", segment))


def constituent_shares(text: str) -> list[tuple[str, Concentration | None]]:
    """(CAS number, declared share) for every constituent the span names.

    The share is read from the text that follows the CAS number up to the next
    one, and from the text in front of it when the row prints the share first.
    """
    normalized = _normalize(text)
    matches = list(_CAS_NUMBER.finditer(normalized))
    shares: list[tuple[str, Concentration | None]] = []
    for index, match in enumerate(matches):
        head_start = matches[index - 1].end() if index else 0
        tail_end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        share = _share_of(normalized[match.end() : tail_end]) or _share_of(
            normalized[head_start : match.start()]
        )
        shares.append((match.group(0), share))
    return shares


def is_whole_product_share(share: Concentration | None) -> bool:
    """The share is exactly 100 %."""
    if share is None or share.low is None or share.high is None:
        return False
    return (
        abs(share.low - 100.0) < _FULL_SHARE_TOLERANCE
        and abs(share.high - 100.0) < _FULL_SHARE_TOLERANCE
    )


def is_partial_share(share: Concentration | None) -> bool:
    """The share has an upper bound strictly below 100 %.

    "80 - <100 %" qualifies (open at 100), ">= 90 - <= 100" does not: the
    latter leaves the constituent room to be the entire product, which is what
    a stabilised substance looks like.
    """
    if share is None or share.high is None:
        return False
    if share.high < 100.0 - _FULL_SHARE_TOLERANCE:
        return True
    return abs(share.high - 100.0) < _FULL_SHARE_TOLERANCE and not share.high_inclusive


def premise_roles(text: str) -> frozenset[PremiseRole]:
    """Every role a single span can fill."""
    normalized = _normalize(text)
    if not normalized:
        return frozenset()
    roles: set[PremiseRole] = set()
    form = classify_form_cue(text)
    if form is not None and form is not ProductForm.UNDETERMINED:
        roles.add(PremiseRole.FORM_DECLARATION)
    if _COMPOSITION_HEADER.search(normalized):
        roles.add(PremiseRole.COMPOSITION_HEADER)
    if _COMPLETENESS.search(normalized):
        roles.add(PremiseRole.COMPLETENESS_STATEMENT)
    for _, share in constituent_shares(text):
        if is_whole_product_share(share):
            roles.add(PremiseRole.SOLE_CONSTITUENT)
        elif is_partial_share(share):
            roles.add(PremiseRole.PARTIAL_CONSTITUENT)
    return frozenset(roles)


@dataclass(frozen=True)
class PremiseAudit:
    """How many spans fill each role, and which forms the spans declare."""

    counts: dict[PremiseRole, int]
    declared_forms: frozenset[ProductForm]

    def count(self, role: PremiseRole) -> int:
        return self.counts.get(role, 0)

    @property
    def filled_roles(self) -> tuple[PremiseRole, ...]:
        return tuple(role for role in PremiseRole if self.count(role) > 0)


def audit_premises(texts: Iterable[str]) -> PremiseAudit:
    """Count role coverage across a premise set.

    Constituent roles are counted by distinct CAS number, so quoting the same
    row twice does not turn one constituent into two.
    """
    counts: dict[PremiseRole, int] = {}
    declared: set[ProductForm] = set()
    whole: set[str] = set()
    partial: set[str] = set()
    for text in texts:
        normalized = _normalize(text)
        if not normalized:
            continue
        form = classify_form_cue(text)
        if form is not None and form is not ProductForm.UNDETERMINED:
            declared.add(form)
            counts[PremiseRole.FORM_DECLARATION] = (
                counts.get(PremiseRole.FORM_DECLARATION, 0) + 1
            )
        if _COMPOSITION_HEADER.search(normalized):
            counts[PremiseRole.COMPOSITION_HEADER] = (
                counts.get(PremiseRole.COMPOSITION_HEADER, 0) + 1
            )
        if _COMPLETENESS.search(normalized):
            counts[PremiseRole.COMPLETENESS_STATEMENT] = (
                counts.get(PremiseRole.COMPLETENESS_STATEMENT, 0) + 1
            )
        for cas, share in constituent_shares(text):
            if is_whole_product_share(share):
                whole.add(cas)
            elif is_partial_share(share):
                partial.add(cas)
    if whole:
        counts[PremiseRole.SOLE_CONSTITUENT] = len(whole)
    if partial:
        counts[PremiseRole.PARTIAL_CONSTITUENT] = len(partial)
    return PremiseAudit(counts=counts, declared_forms=frozenset(declared))


# ---------------------------------------------------------------------------
# Establishing the product form from a premise set.
# ---------------------------------------------------------------------------

DECLARED = "declared"
COMPOSITION = "composition"


@dataclass(frozen=True)
class FormBasis:
    """The form a premise set establishes and how it establishes it."""

    form: ProductForm | None
    basis: str = ""
    note: str = ""
    missing: tuple[tuple[PremiseRole, int, int], ...] = ()
    audit: PremiseAudit | None = None

    @property
    def missing_roles(self) -> tuple[PremiseRole, ...]:
        return tuple(role for role, _, _ in self.missing)


def _describe_missing(missing: tuple[tuple[PremiseRole, int, int], ...]) -> str:
    return ", ".join(
        f"{role.value} ({have}/{needed}: {ROLE_DESCRIPTIONS[role]})"
        for role, needed, have in missing
    )


def establish_form(texts: Iterable[str]) -> FormBasis:
    """The single product form a premise set establishes, or no form at all.

    An explicit declaration is preferred; the composition shapes are consulted
    only when no cited span declares the form. Conflicting premises establish
    nothing.
    """
    texts = [text for text in texts if text and text.strip()]
    audit = audit_premises(texts)

    if len(audit.declared_forms) > 1:
        names = ", ".join(sorted(form.value for form in audit.declared_forms))
        return FormBasis(None, "", f"premises declare conflicting forms: {names}", (), audit)
    if len(audit.declared_forms) == 1:
        form = next(iter(audit.declared_forms))
        return FormBasis(form, DECLARED, f"premise declares {form.value}", (), audit)

    satisfied = [
        form
        for form, requirement in COMPOSITION_REQUIREMENTS.items()
        if not requirement.missing(audit)
    ]
    if len(satisfied) > 1:
        names = ", ".join(sorted(form.value for form in satisfied))
        return FormBasis(
            None, "", f"composition premises support conflicting forms: {names}", (), audit
        )
    if satisfied:
        form = satisfied[0]
        return FormBasis(
            form,
            COMPOSITION,
            f"composition establishes {form.value}: "
            f"{COMPOSITION_REQUIREMENTS[form].description}",
            (),
            audit,
        )

    closest = min(
        (
            (sum(needed - have for _, needed, have in requirement.missing(audit)), form)
            for form, requirement in COMPOSITION_REQUIREMENTS.items()
        ),
        default=(0, ProductForm.MIXTURE),
    )
    missing = COMPOSITION_REQUIREMENTS[closest[1]].missing(audit)
    return FormBasis(
        None,
        "",
        "cited premises establish no product form: no span declares one and the "
        f"composition premise set is incomplete for {closest[1].value} "
        f"[missing {_describe_missing(missing)}]",
        missing,
        audit,
    )


def premise_form(texts: Iterable[str]) -> ProductForm | None:
    """The single product form the premise texts establish, or None."""
    return establish_form(texts).form


# ---------------------------------------------------------------------------
# Which gold spans a prediction has to locate.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PremiseGroup:
    """Premise spans that fill one role; a prediction must locate ``minimum`` of them."""

    role: str
    indices: tuple[int, ...]
    minimum: int = 1


def premise_groups(
    rule_id: str | None, value: str | None, texts: Sequence[str]
) -> tuple[PremiseGroup, ...] | None:
    """Group a gold premise set by the roles its rule requires.

    Returns None when the registry cannot describe the set - the gold premises
    do not fill the rule's roles - so the caller can fall back to "locate any
    one premise" and report the annotation for review.
    """
    rule = RULES.get(rule_id or "")
    if rule is None or not texts:
        return None

    if rule.rule_id == RULE_COMPOSITION_FORM:
        claimed = _claimed_form(value)
        requirement = COMPOSITION_REQUIREMENTS.get(claimed) if claimed else None
    else:
        basis = establish_form(texts)
        if basis.basis == COMPOSITION:
            requirement = COMPOSITION_REQUIREMENTS[basis.form] if basis.form else None
        elif basis.basis == DECLARED:
            requirement = FORM_DECLARATION_REQUIREMENT
        else:
            requirement = None
    if requirement is None:
        return None

    roles_per_span = [premise_roles(text) for text in texts]
    groups: list[PremiseGroup] = []
    for role, needed in requirement.roles:
        indices = tuple(
            index for index, roles in enumerate(roles_per_span) if role in roles
        )
        if len(indices) < needed:
            return None
        groups.append(PremiseGroup(role=role.value, indices=indices, minimum=needed))
    return tuple(groups)


def _claimed_form(value: str | None) -> ProductForm | None:
    claimed = " ".join((value or "").split()).upper().replace(" ", "_")
    try:
        return ProductForm(claimed)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Rule checks.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleCheck:
    status: str
    note: str = ""
    form: ProductForm | None = None
    missing: tuple[PremiseRole, ...] = ()


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


def required_roles(rule_id: str | None, value: str | None) -> tuple[tuple[PremiseRole, int], ...]:
    """The premise roles a claim under this rule has to fill."""
    rule = RULES.get(rule_id or "")
    if rule is None:
        return ()
    if rule.rule_id == RULE_COMPOSITION_FORM:
        claimed = _claimed_form(value)
        requirement = COMPOSITION_REQUIREMENTS.get(claimed) if claimed else None
        return requirement.roles if requirement else ()
    return FORM_DECLARATION_REQUIREMENT.roles


def check_rule(
    rule_id: str | None,
    field_name: str,
    state: FieldState,
    value: str | None,
    premise_texts: Iterable[str],
) -> RuleCheck:
    """Does the named rule license (field, state, value) from a complete premise set?"""
    rule = RULES.get(rule_id or "")
    if rule is None:
        return RuleCheck(VIOLATED, f"unknown rule {rule_id!r}")
    if field_name not in rule.fields:
        return RuleCheck(VIOLATED, f"rule {rule_id} does not apply to {field_name}")
    if state is not rule.conclusion_state:
        return RuleCheck(
            VIOLATED,
            f"rule {rule_id} concludes {rule.conclusion_state.value}, not {state.value}",
        )

    texts = [text for text in premise_texts if text and text.strip()]
    if not texts:
        return RuleCheck(VIOLATED, "no premise cited")

    if rule.rule_id == RULE_COMPOSITION_FORM:
        return _check_composition_form(value, texts)
    return _check_identifier_undefined(texts)


def _check_identifier_undefined(texts: list[str]) -> RuleCheck:
    basis = establish_form(texts)
    if basis.form is None:
        return RuleCheck(UNDECIDABLE, basis.note, None, basis.missing_roles)
    if basis.form in (ProductForm.MIXTURE, ProductForm.ARTICLE):
        return RuleCheck(SATISFIED, basis.note, basis.form)
    return RuleCheck(
        VIOLATED,
        f"{basis.note}; identifiers stay defined for {basis.form.value}",
        basis.form,
    )


def _check_composition_form(value: str | None, texts: list[str]) -> RuleCheck:
    claimed = _claimed_form(value)
    if claimed is None or claimed is ProductForm.UNDETERMINED:
        return RuleCheck(VIOLATED, f"claim names no product form (value {value!r})")

    audit = audit_premises(texts)
    if audit.count(PremiseRole.FORM_DECLARATION):
        declared = ", ".join(sorted(form.value for form in audit.declared_forms))
        return RuleCheck(
            VIOLATED,
            f"the cited premises declare the form explicitly ({declared}); a declared "
            "form is direct evidence, not a composition derivation",
        )

    requirement = COMPOSITION_REQUIREMENTS.get(claimed)
    if requirement is None:
        return RuleCheck(VIOLATED, f"no composition rule concludes {claimed.value}")

    missing = requirement.missing(audit)
    if missing:
        return RuleCheck(
            UNDECIDABLE,
            f"incomplete premise set for {claimed.value}: missing "
            f"{_describe_missing(missing)}",
            None,
            tuple(role for role, _, _ in missing),
        )

    other = [
        form
        for form, candidate in COMPOSITION_REQUIREMENTS.items()
        if form is not claimed and not candidate.missing(audit)
    ]
    if other:
        names = ", ".join(sorted(form.value for form in other))
        return RuleCheck(
            VIOLATED,
            f"the composition premises also establish {names}, so they do not "
            f"single out {claimed.value}",
        )
    return RuleCheck(
        SATISFIED,
        f"composition establishes {claimed.value}: {requirement.description}",
        claimed,
    )
