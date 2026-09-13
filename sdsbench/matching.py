"""Typed value normalization and comparison. Nothing here is fuzzy.

Span overlap is exact on tokens: a token either matches or it does not, and
the score is how many gold tokens the citation covers in one contiguous run.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

from .ontology import ValueKind

_SUBSCRIPTS = {
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
}

_TRADEMARKS = "™®©"


def _fold_degree_marks(value: str) -> str:
    """Map masculine-ordinal º to degree before NFKC, which would otherwise turn it into 'o'."""
    return value.replace("º", "°")


def normalize_text(value: str) -> str:
    """Casefold, strip trademark marks and collapse whitespace."""
    text = unicodedata.normalize("NFKC", _fold_degree_marks(value))
    text = "".join(char for char in text if char not in _TRADEMARKS)
    return " ".join(text.split()).strip().lower()


def normalize_name(value: str) -> str:
    text = normalize_text(value)
    text = text.strip(" .,:;-–—")
    text = re.sub(r"\b(inc|ltd|co|corp|llc|gmbh|sa|bv|plc|kg|pty)\.", r"\1", text)
    return " ".join(text.split())


def names_equal(predicted: str | None, expected: str | None) -> bool:
    """Normalized name equality; a hyphenation break ("pseudotuber- culosis") is folded."""
    if not predicted or not expected:
        return False
    fold = lambda text: re.sub(r"(?<=[a-z])- (?=[a-z])", "", normalize_name(text))  # noqa: E731
    return fold(predicted) == fold(expected)


def normalize_formula(value: str) -> str:
    """Strip whitespace and fold subscripts. Case is kept (Co vs CO)."""
    text = unicodedata.normalize("NFKC", value)
    text = "".join(_SUBSCRIPTS.get(char, char) for char in text)
    text = text.replace("·", "*").replace("•", "*")
    return re.sub(r"\s+", "", text)


CAS_PATTERN = re.compile(r"^(\d{2,7})-(\d{2})-(\d)$")


def normalize_cas(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    text = re.sub(r"\s+", "", text)
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    return text.strip(" .,:;")


def cas_checksum_valid(value: str) -> bool:
    match = CAS_PATTERN.match(normalize_cas(value))
    if not match:
        return False
    digits = match.group(1) + match.group(2)
    checksum = sum(
        int(digit) * position for position, digit in enumerate(reversed(digits), start=1)
    )
    return checksum % 10 == int(match.group(3))


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str | None = None
    comparator: str = "="

    def __str__(self) -> str:
        prefix = "" if self.comparator == "=" else self.comparator
        return f"{prefix}{self.value:g}{' ' + self.unit if self.unit else ''}"


_COMPARATOR_WORDS = (
    (r">=|≥|at\s+least|not\s+less\s+than|nlt", ">="),
    (r"<=|≤|not\s+more\s+than|nmt|at\s+most", "<="),
    (r">|greater\s+than|more\s+than|above|over|higher\s+than", ">"),
    (r"<|less\s+than|below|under|lower\s+than", "<"),
)

_NUMBER = r"[-+]?\d+(?:[.,]\d+)?"


def _parse_number(text: str) -> float:
    return float(text.replace(",", "."))


def parse_temperature(value: str) -> Quantity | None:
    """Parse a flash-point string to a comparator plus °C. Fahrenheit is converted."""
    text = unicodedata.normalize("NFKC", _fold_degree_marks(value))
    lowered = text.lower()

    celsius_match = re.search(rf"({_NUMBER})\s*°?\s*c\b", lowered)
    fahrenheit_match = re.search(rf"({_NUMBER})\s*°?\s*f\b", lowered)

    if celsius_match:
        number = _parse_number(celsius_match.group(1))
        prefix = lowered[: celsius_match.start()]
    elif fahrenheit_match:
        number = (_parse_number(fahrenheit_match.group(1)) - 32.0) * 5.0 / 9.0
        prefix = lowered[: fahrenheit_match.start()]
    else:
        return None

    comparator = "="
    for pattern, symbol in _COMPARATOR_WORDS:
        if re.search(pattern, prefix):
            comparator = symbol
            break
    return Quantity(value=number, unit="C", comparator=comparator)


def parse_quantity(value: str, unit_pattern: str = r"g\s*/\s*mol(?:e)?") -> Quantity | None:
    text = unicodedata.normalize("NFKC", value).lower()
    match = re.search(rf"({_NUMBER})\s*(?:{unit_pattern})?", text)
    if not match:
        return None
    comparator = "="
    prefix = text[: match.start()]
    for pattern, symbol in _COMPARATOR_WORDS:
        if re.search(pattern, prefix):
            comparator = symbol
            break
    return Quantity(value=_parse_number(match.group(1)), unit="g/mol", comparator=comparator)


def quantities_match(
    predicted: Quantity | None, expected: Quantity | None, tolerance: float
) -> bool:
    if predicted is None or expected is None:
        return False
    if predicted.comparator != expected.comparator:
        return False
    return math.isclose(predicted.value, expected.value, abs_tol=tolerance)


def canonical_value(value: str, kind: ValueKind) -> str:
    if kind is ValueKind.FORMULA:
        return normalize_formula(value)
    if kind is ValueKind.IDENTIFIER:
        return normalize_cas(value)
    if kind is ValueKind.CATEGORICAL:
        return normalize_text(value).upper().replace(" ", "_")
    if kind is ValueKind.TEMPERATURE:
        parsed = parse_temperature(value)
        return str(parsed) if parsed else normalize_text(value)
    if kind is ValueKind.QUANTITY:
        parsed = parse_quantity(value)
        return str(parsed) if parsed else normalize_text(value)
    return normalize_name(value)


def values_equal(
    predicted: str,
    expected: str,
    kind: ValueKind,
    tolerance: float = 0.0,
) -> bool:
    if kind is ValueKind.TEMPERATURE:
        left, right = parse_temperature(predicted), parse_temperature(expected)
        if left is not None and right is not None:
            return quantities_match(left, right, tolerance)
        if left is None and right is None:
            return normalize_text(predicted) == normalize_text(expected)
        return False
    if kind is ValueKind.QUANTITY:
        left, right = parse_quantity(predicted), parse_quantity(expected)
        if left is not None and right is not None:
            return quantities_match(left, right, tolerance)
        if left is None and right is None:
            return normalize_text(predicted) == normalize_text(expected)
        return False
    return canonical_value(predicted, kind) == canonical_value(expected, kind)


def value_supported_by(value: str, evidence_text: str, kind: ValueKind) -> bool:
    """Whether the cited span contains the asserted value (gold-free)."""
    if not value or not evidence_text:
        return False
    if kind is ValueKind.TEMPERATURE:
        parsed_value = parse_temperature(value)
        parsed_span = parse_temperature(evidence_text)
        if parsed_value and parsed_span:
            return quantities_match(parsed_span, parsed_value, tolerance=0.6)
        return normalize_text(value) in normalize_text(evidence_text)
    if kind is ValueKind.QUANTITY:
        parsed_value = parse_quantity(value)
        parsed_span = parse_quantity(evidence_text)
        if parsed_value and parsed_span:
            return quantities_match(parsed_span, parsed_value, tolerance=0.05)
        return normalize_text(value) in normalize_text(evidence_text)
    if kind is ValueKind.FORMULA:
        return normalize_formula(value) in normalize_formula(evidence_text)
    if kind is ValueKind.IDENTIFIER:
        return normalize_cas(value) in normalize_cas(evidence_text)
    return normalize_name(value) in normalize_name(evidence_text)


# ---------------------------------------------------------------------------
# Exact span overlap.
# ---------------------------------------------------------------------------

_TOKEN_PUNCTUATION = ".,:;()[]{}\"'"


def _search_form(text: str) -> str:
    folded = unicodedata.normalize("NFKC", _fold_degree_marks(text))
    folded = folded.replace("–", "-").replace("—", "-")
    return " ".join(folded.split()).lower()


def span_tokens(text: str) -> list[str]:
    """Whitespace tokens with surrounding punctuation stripped; empty tokens dropped."""
    tokens = []
    for piece in _search_form(text).split():
        stripped = piece.strip(_TOKEN_PUNCTUATION)
        if stripped:
            tokens.append(stripped)
    return tokens


def _longest_common_run(gold: list[str], predicted: list[str]) -> int:
    """Length of the longest run of gold tokens that appears contiguously in predicted."""
    best = 0
    previous = [0] * (len(predicted) + 1)
    for gold_token in gold:
        current = [0] * (len(predicted) + 1)
        for index, predicted_token in enumerate(predicted, start=1):
            if predicted_token == gold_token:
                current[index] = previous[index - 1] + 1
                best = max(best, current[index])
        previous = current
    return best


@dataclass(frozen=True)
class SpanOverlap:
    coverage: float
    dilution: float
    iou: float
    matched: int
    gold_tokens: int
    predicted_tokens: int


def span_overlap(predicted_text: str, gold_text: str) -> SpanOverlap:
    """Exact contiguous token overlap between a citation and a gold span.

    coverage = matched / gold tokens, dilution = predicted / gold tokens,
    iou = matched / (gold + predicted - matched). A citation that is a short
    fragment of the gold span scores low coverage; one that quotes a whole
    paragraph scores low iou. No similarity scoring is involved.
    """
    gold = span_tokens(gold_text)
    predicted = span_tokens(predicted_text)
    if not gold or not predicted:
        return SpanOverlap(0.0, 0.0, 0.0, 0, len(gold), len(predicted))
    matched = _longest_common_run(gold, predicted)
    union = len(gold) + len(predicted) - matched
    return SpanOverlap(
        coverage=matched / len(gold),
        dilution=len(predicted) / len(gold),
        iou=matched / union if union else 0.0,
        matched=matched,
        gold_tokens=len(gold),
        predicted_tokens=len(predicted),
    )


# ---------------------------------------------------------------------------
# Ingredient concentrations.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Concentration:
    """A closed/open interval in percent. Exact values have low == high."""

    low: float | None
    high: float | None
    low_inclusive: bool = True
    high_inclusive: bool = True


_CONCENTRATION_NOISE = re.compile(r"%|w/w|v/v|(?<![a-z])wt\.?(?![a-z])|weight|percent|by\s+wt", re.IGNORECASE)
_RANGE = re.compile(
    rf"({_NUMBER})\s*(?:-|~|to|\.\.)\s*(<=|<|≤)?\s*({_NUMBER})"
)
_SINGLE = re.compile(
    rf"(<=|>=|≤|≥|<|>|~|≈|nmt|nlt|not\s+more\s+than|not\s+less\s+than|"
    rf"at\s+most|at\s+least|max(?:imum)?\.?|min(?:imum)?\.?|approx\.?|about|ca\.?)?\s*({_NUMBER})"
)


def parse_concentration(text: str | None) -> Concentration | None:
    if not text:
        return None
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = normalized.replace("–", "-").replace("—", "-").replace("−", "-")
    normalized = _CONCENTRATION_NOISE.sub(" ", normalized)
    normalized = " ".join(normalized.split())

    match = _RANGE.search(normalized)
    if match:
        low = _parse_number(match.group(1))
        high = _parse_number(match.group(3))
        return Concentration(low, high, True, match.group(2) != "<")

    match = _SINGLE.search(normalized)
    if not match:
        return None
    value = _parse_number(match.group(2))
    comparator = (match.group(1) or "").strip()
    if comparator in ("<=", "≤", "nmt", "at most") or comparator.startswith(("not more", "max")):
        return Concentration(None, value, True, True)
    if comparator == "<":
        return Concentration(None, value, True, False)
    if comparator in (">=", "≥", "nlt", "at least") or comparator.startswith(("not less", "min")):
        return Concentration(value, None, True, True)
    if comparator == ">":
        return Concentration(value, None, False, True)
    return Concentration(value, value, True, True)


def _close(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(left, right, abs_tol=1e-6)


def concentrations_equal(predicted: str | None, expected: str | None) -> bool:
    if not predicted and not expected:
        return True
    if not predicted or not expected:
        return False
    left, right = parse_concentration(predicted), parse_concentration(expected)
    if left is None and right is None:
        return normalize_text(predicted) == normalize_text(expected)
    if left is None or right is None:
        return False
    return (
        _close(left.low, right.low)
        and _close(left.high, right.high)
        and left.low_inclusive == right.low_inclusive
        and left.high_inclusive == right.high_inclusive
    )
