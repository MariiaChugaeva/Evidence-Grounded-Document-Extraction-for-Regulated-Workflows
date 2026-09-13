"""LLM diagnostic baseline: no fine-tuning, same text layer, same schema, same evaluator.

The model receives the page-tagged text of one sheet (native or OCR layer, in
reading order or visual-row order), the field ontology and the derivation
rules, and must return state, value, page and a verbatim quote for every
field, plus premises and a rule id for derived claims, plus the ingredient
rows. Quotes are resolved against the same text layer with the evaluator's
exact-match verbatim function; a quote that does not occur is kept as cited
(no box) so the evaluator can fail it.

Default backend is a local Ollama server (``qwen2.5:7b``, JSON-schema output,
``num_ctx`` 24576). ``claude-*`` model names still go through the official
Anthropic SDK. Raw responses are cached under ``data/llm_cache`` so a re-run
is free and reproducible. Token usage, latency and an estimated cost (API
models only) are recorded per document.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from .. import textlayer
from ..derivation import RULE_COMPOSITION_FORM, RULE_IDENTIFIER_UNDEFINED, RULES
from ..ontology import PRODUCT_FIELD_NAMES, FieldState
from ..schema import (
    DocumentPrediction,
    EvidenceSpan,
    FieldPrediction,
    IngredientPrediction,
    IngredientsPrediction,
)

DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_NUM_CTX = 24576
CACHE_DIR = Path("data/llm_cache")

# USD per million tokens (input, output); Anthropic first-party rates, 2026-06.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5-1": (10.0, 50.0),
}

StateName = Literal["PRESENT", "NOT_APPLICABLE", "NO_DATA", "NOT_STATED"]
RuleName = Literal["identifier_undefined_for_form", "composition_implies_form"]


# ---------------------------------------------------------------------------
# The model's output contract (kept flat and explicit for structured outputs)
# ---------------------------------------------------------------------------


class LlmCitation(BaseModel):
    page: int
    quote: str


class LlmField(BaseModel):
    state: StateName
    value: str | None
    derivation: Literal["DIRECT", "DERIVED"]
    rule: RuleName | None
    evidence: LlmCitation | None
    premises: list[LlmCitation]


class LlmIngredient(BaseModel):
    name: str | None
    cas_number: str | None
    cas_state: StateName
    concentration: str | None
    citation: LlmCitation | None


class LlmOutput(BaseModel):
    product_name: LlmField
    supplier_name: LlmField
    substance_or_mixture: LlmField
    product_cas_number: LlmField
    chemical_formula: LlmField
    molecular_weight: LlmField
    flash_point: LlmField
    ingredients_state: StateName
    ingredients: list[LlmIngredient]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You extract a fixed set of fields from one safety data sheet (SDS) and cite verbatim evidence for every claim.

Input: the sheet's text layer, page by page, marked "=== PAGE n ===". The text is in {order} order. It may contain OCR noise; quote it as it is.

Product fields: product_name (trade name / product identifier), supplier_name (supplier, manufacturer or distributor named in Section 1), substance_or_mixture (SUBSTANCE, MIXTURE or ARTICLE), product_cas_number, chemical_formula, molecular_weight, flash_point.

States: PRESENT = the sheet states a usable value; NOT_APPLICABLE = the sheet says the property does not apply, or a rule below rules it out; NO_DATA = the sheet says the value is not available, not determined, not specified, or leaves the slot empty with a dash; NOT_STATED = the sheet is silent. Only NOT_STATED carries no citation.

Citations: page number plus a quote copied exactly from that page's text: contiguous, same spelling, punctuation and case, no paraphrase, no ellipsis. It may run across a line break. Prefer the label together with the value ("Flash point: 12 °C (53.6 °F)"); when the label and the value are not adjacent in the text, quote the value alone.

Values: copy them as printed, with units, comparators and methods ("> 61 °C", ">=290 ºF [Test Method:Tagliabue Closed Cup]"). substance_or_mixture value is one of SUBSTANCE, MIXTURE, ARTICLE ("preparation" means MIXTURE; a "3.1 Substances: not applicable" entry means MIXTURE).

Derived claims. A claim is DERIVED when it follows from the sheet through a rule rather than being stated:
- identifier_undefined_for_form: product_cas_number, chemical_formula and molecular_weight are NOT_APPLICABLE for a MIXTURE or an ARTICLE. Set derivation DERIVED, rule identifier_undefined_for_form, premises = the quote(s) that establish the product form, evidence null. If the sheet itself says "Molecular weight: not applicable", that is a DIRECT NOT_APPLICABLE with that quote instead.
- composition_implies_form: when the sheet never says substance or mixture but the composition table shows several constituents (MIXTURE) or a single constituent at 100 % (SUBSTANCE), set substance_or_mixture DERIVED with rule composition_implies_form and premises = the composition rows quoted.
For a SUBSTANCE the identifier fields are PRESENT when stated (CAS in Section 1 or 3, formula and molecular weight in Section 3 or 9) and NOT_STATED when not.

Ingredients: one item per row of the Section 3 composition table with name, CAS number (or cas_state NOT_APPLICABLE / NOT_STATED when the row has none), concentration as printed, and a quote of the row. ingredients_state is NOT_APPLICABLE when the sheet says no component needs disclosure, NOT_STATED when there is no composition information, else PRESENT.

Return only the structured output."""


def page_texts(document_text: textlayer.DocumentText, layout: str) -> str:
    pieces = []
    for page in document_text.pages:
        text = page.row_text if layout == "rows" else page.text
        pieces.append(f"=== PAGE {page.page_number} ===\n{text}")
    return "\n\n".join(pieces)


def build_messages(document: str, document_text: textlayer.DocumentText, layout: str) -> tuple[str, str]:
    order = "visual row" if layout == "rows" else "reading"
    system = SYSTEM_PROMPT.format(order=order)
    user = f"Document: {document}\n\n{page_texts(document_text, layout)}"
    return system, user


# ---------------------------------------------------------------------------
# Citation resolution
# ---------------------------------------------------------------------------


def _row_page(page: textlayer.Page) -> textlayer.Page:
    rows = textlayer.rows(page)
    return textlayer._build_page(page.page_number, [row.tokens for row in rows], page.width, page.height)


def resolve_citation(
    document_text: textlayer.DocumentText, citation: LlmCitation | None, layout: str
) -> EvidenceSpan | None:
    """Exact match of the quote on the cited page; otherwise the quote is kept without a box."""
    if citation is None or not citation.quote.strip():
        return None
    page = document_text.page(citation.page)
    if page is None:
        return EvidenceSpan(page=max(1, citation.page), text=citation.quote, bbox=None)
    anchors = textlayer.find_all_in_page(page, citation.quote)
    if not anchors:
        anchors = textlayer.find_all_in_page(_row_page(page), citation.quote)
    if anchors:
        anchor = anchors[0]
        return EvidenceSpan(
            page=page.page_number,
            text=anchor.text,
            bbox=list(anchor.bbox) if anchor.bbox else None,
            bbox_provenance="system",
        )
    return EvidenceSpan(page=page.page_number, text=citation.quote, bbox=None)


def _field_prediction(
    raw: LlmField, document_text: textlayer.DocumentText, layout: str
) -> tuple[FieldPrediction, str | None]:
    """Convert one model field to the benchmark schema, repairing contract violations."""
    note = None
    state = FieldState(raw.state)
    value = raw.value if state is FieldState.PRESENT and raw.value else None
    premises = [
        span
        for span in (resolve_citation(document_text, item, layout) for item in raw.premises)
        if span is not None
    ]
    evidence = resolve_citation(document_text, raw.evidence, layout)

    if state is FieldState.NOT_STATED:
        return FieldPrediction(state=state), None

    if raw.derivation == "DERIVED" and raw.rule in RULES and premises:
        return (
            FieldPrediction(
                state=state, value=value, derivation="DERIVED", rule=raw.rule, premises=premises
            ),
            None,
        )
    if raw.derivation == "DERIVED":
        note = "derived claim without a registered rule or premises; downgraded to direct"
        evidence = evidence or (premises[0] if premises else None)
    try:
        return FieldPrediction(state=state, value=value, evidence=evidence, rationale=note), note
    except ValidationError as error:
        return FieldPrediction(state=FieldState.NOT_STATED, rationale=f"invalid output: {error}"), str(error)


def to_document_prediction(
    document: str,
    output: LlmOutput,
    document_text: textlayer.DocumentText,
    layout: str,
) -> DocumentPrediction:
    fields: dict[str, FieldPrediction] = {}
    notes: list[str] = []
    for name in PRODUCT_FIELD_NAMES:
        prediction, note = _field_prediction(getattr(output, name), document_text, layout)
        fields[name] = prediction
        if note:
            notes.append(f"{name}: {note}")
    items = [
        IngredientPrediction(
            name=item.name,
            cas_number=item.cas_number,
            cas_state=FieldState(item.cas_state),
            concentration=item.concentration,
            evidence=resolve_citation(document_text, item.citation, layout),
        )
        for item in output.ingredients
    ]
    return DocumentPrediction(
        document=document,
        fields=fields,
        ingredients=IngredientsPrediction(state=FieldState(output.ingredients_state), items=items),
        error="; ".join(notes) or None,
    )


# ---------------------------------------------------------------------------
# Model calls with caching and accounting
# ---------------------------------------------------------------------------


@dataclass
class CallRecord:
    document: str
    model: str
    cached: bool
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    stop_reason: str = ""
    error: str | None = None


@dataclass
class LlmRun:
    model: str
    layer: str
    layout: str
    documents: list[DocumentPrediction] = dataclass_field(default_factory=list)
    calls: list[CallRecord] = dataclass_field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return sum(call.cost_usd for call in self.calls)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = PRICES.get(model)
    if prices is None:
        return 0.0
    return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000


def cache_key(model: str, system: str, user: str) -> str:
    digest = hashlib.sha256()
    for piece in (model, system, user):
        digest.update(piece.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _cached_response(key: str, cache_dir: Path) -> dict | None:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _store_response(key: str, payload: dict, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / f"{key}.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def is_anthropic_model(model: str) -> bool:
    return model.startswith("claude-")


def ollama_host(host: str | None = None) -> str:
    return (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")


def _parse_json_content(content: str) -> dict | None:
    text = content.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.rstrip().endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _call_ollama(
    model: str, system: str, user: str, max_tokens: int, num_ctx: int, host: str
) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": LlmOutput.model_json_schema(),
        "options": {
            "temperature": 0,
            "seed": 0,
            "num_ctx": num_ctx,
            "num_predict": max_tokens,
        },
    }
    request = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(f"Ollama is not reachable at {host}: {error}") from error
    latency = time.perf_counter() - started
    message = payload.get("message") or {}
    parsed = _parse_json_content(message.get("content") or "")
    return {
        "model": payload.get("model", model),
        "stop_reason": payload.get("done_reason", ""),
        "output": parsed,
        "usage": {
            "input_tokens": int(payload.get("prompt_eval_count") or 0),
            "output_tokens": int(payload.get("eval_count") or 0),
            "cache_read_input_tokens": 0,
        },
        "latency_seconds": latency,
        "request_id": None,
        "backend": "ollama",
        "num_ctx": num_ctx,
    }


def _call_anthropic(model: str, system: str, user: str, max_tokens: int) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    started = time.perf_counter()
    response = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=LlmOutput,
    )
    latency = time.perf_counter() - started
    parsed = response.parsed_output
    usage = response.usage
    return {
        "model": model,
        "stop_reason": response.stop_reason,
        "output": parsed.model_dump() if parsed is not None else None,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        },
        "latency_seconds": latency,
        "request_id": getattr(response, "_request_id", None),
        "backend": "anthropic",
    }


def call_model(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 16000,
    num_ctx: int = DEFAULT_NUM_CTX,
    host: str | None = None,
) -> dict:
    """One structured-output request. Returns the parsed output plus usage as a plain dict."""
    if is_anthropic_model(model):
        return _call_anthropic(model, system, user, max_tokens)
    return _call_ollama(model, system, user, max_tokens, num_ctx, ollama_host(host))


def extract_document(
    document: str,
    model: str = DEFAULT_MODEL,
    layer: str = textlayer.NATIVE,
    layout: str = "rows",
    cache_dir: Path = CACHE_DIR,
    use_cache: bool = True,
    num_ctx: int = DEFAULT_NUM_CTX,
    host: str | None = None,
) -> tuple[DocumentPrediction, CallRecord]:
    document_text = textlayer.load(document, layer)
    system, user = build_messages(document, document_text, layout)
    key = cache_key(model, system, user)
    record = CallRecord(document=document, model=model, cached=False)

    payload = _cached_response(key, cache_dir) if use_cache else None
    if payload is not None:
        record.cached = True
    else:
        try:
            payload = call_model(model, system, user, num_ctx=num_ctx, host=host)
        except Exception as error:  # noqa: BLE001 - recorded per document, run continues
            record.error = f"{type(error).__name__}: {error}"
            return DocumentPrediction(document=document, fields={}, error=record.error), record
        _store_response(key, payload, cache_dir)

    usage = payload.get("usage", {})
    record.input_tokens = int(usage.get("input_tokens", 0))
    record.output_tokens = int(usage.get("output_tokens", 0))
    record.cache_read_tokens = int(usage.get("cache_read_input_tokens", 0))
    record.latency_seconds = float(payload.get("latency_seconds", 0.0))
    record.stop_reason = str(payload.get("stop_reason", ""))
    record.cost_usd = 0.0 if record.cached else estimate_cost(model, record.input_tokens, record.output_tokens)

    if payload.get("output") is None:
        record.error = f"no structured output (stop_reason={record.stop_reason})"
        return DocumentPrediction(document=document, fields={}, error=record.error), record

    try:
        output = LlmOutput.model_validate(payload["output"])
    except ValidationError as error:
        record.error = f"output failed validation: {error}"
        return DocumentPrediction(document=document, fields={}, error=record.error), record
    return to_document_prediction(document, output, document_text, layout), record
