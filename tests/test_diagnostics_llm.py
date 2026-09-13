"""Tests for the diagnostics probes and the LLM baseline plumbing (no API calls)."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from sdsbench import diagnostics, textlayer
from sdsbench.annotations import (
    Derivation,
    DocumentAnnotation,
    FieldAnnotation,
    IngredientAnnotation,
    ResolvedEvidence,
)
from sdsbench.derivation import RULE_IDENTIFIER_UNDEFINED
from sdsbench.evaluator import SATISFIED
from sdsbench.extractors import llm
from sdsbench.ontology import FieldState, ProductForm
from sdsbench.textlayer import DocumentText, Token


def _tokens(words: list[str], y: float) -> list[Token]:
    return [
        Token(text=word, bbox=(0.10 + index * 0.07, y, 0.16 + index * 0.07, y + 0.012))
        for index, word in enumerate(words)
    ]


def _page(lines: list[tuple[float, list[str]]]) -> textlayer.Page:
    return textlayer._build_page(1, [_tokens(words, y) for y, words in lines], 595.0, 842.0)


def _doc(page: textlayer.Page) -> DocumentText:
    return DocumentText(document="synthetic.pdf", layer=textlayer.NATIVE, pages=[page])


def _resolved(page: textlayer.Page, text: str) -> ResolvedEvidence:
    anchor = textlayer.find_all_in_page(page, text)[0]
    return ResolvedEvidence(
        page=1,
        text=anchor.text,
        char_start=anchor.char_start,
        char_end=anchor.char_end,
        bbox=list(anchor.bbox),
        occurrences_on_page=1,
    )


def _annotation(field_name: str, gold: FieldAnnotation, form: ProductForm = ProductForm.MIXTURE) -> DocumentAnnotation:
    return DocumentAnnotation(
        document="synthetic.pdf",
        split="dev",
        product_form=form,
        fields={field_name: gold},
        ingredients=IngredientAnnotation(state=FieldState.NOT_STATED),
    )


class TestGoldValueProbe(unittest.TestCase):
    def test_value_on_its_own_line_is_cited_with_the_label_line_above(self):
        page = _page([(0.40, ["·", "Flash", "point:"]), (0.42, ["12", "°C", "(53.6", "°F)"])])
        span = diagnostics.locate_value(_doc(page), ["12 °C (53.6 °F)"])
        self.assertIsNotNone(span)
        self.assertIn("Flash point:", span.text)
        self.assertIn("53.6", span.text)
        self.assertIsNotNone(span.bbox)

    def test_probe_scores_localization_against_gold(self):
        page = _page([(0.40, ["Flash", "point:", "61", "°C"]), (0.42, ["Boiling", "point:", "100", "°C"])])
        gold = FieldAnnotation(
            state=FieldState.PRESENT,
            derivation=Derivation.EXPLICIT,
            value="61 °C",
            resolved_evidence=[_resolved(page, "Flash point: 61 °C")],
        )
        annotation = _annotation("flash_point", gold)
        with patch("sdsbench.diagnostics.textlayer.load", return_value=_doc(page)), patch(
            "sdsbench.evaluator.textlayer.load", return_value=_doc(page)
        ):
            results = diagnostics.run_probes([annotation])
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].value_cited)
        self.assertTrue(results[0].value_localized)
        self.assertEqual(results[0].value_support, SATISFIED)
        self.assertEqual(results[0].evidence_state, SATISFIED)
        self.assertEqual(results[0].evidence_value, SATISFIED)
        self.assertEqual(results[0].verify_support, SATISFIED)

    def test_derived_field_is_probed_through_its_premise(self):
        page = _page([(0.40, ["Substance/mixture", ":", "Mixture"])])
        gold = FieldAnnotation(
            state=FieldState.NOT_APPLICABLE,
            derivation=Derivation.ONTOLOGY,
            rule=RULE_IDENTIFIER_UNDEFINED,
            resolved_premises=[_resolved(page, "Substance/mixture : Mixture")],
        )
        annotation = _annotation("product_cas_number", gold)
        with patch("sdsbench.diagnostics.textlayer.load", return_value=_doc(page)), patch(
            "sdsbench.evaluator.textlayer.load", return_value=_doc(page)
        ):
            results = diagnostics.run_probes([annotation])
        self.assertTrue(results[0].value_localized)
        self.assertEqual(results[0].evidence_state, SATISFIED)
        self.assertEqual(results[0].verify_support, SATISFIED)


class TestLlmPlumbing(unittest.TestCase):
    def _page(self):
        return _page([(0.40, ["Flash", "point:", "12", "°C"]), (0.42, ["Substance/mixture", ":", "Mixture"])])

    def test_exact_quote_gets_a_box_and_inexact_quote_keeps_no_box(self):
        doc = _doc(self._page())
        exact = llm.resolve_citation(doc, llm.LlmCitation(page=1, quote="flash point: 12 °C"), "rows")
        self.assertIsNotNone(exact.bbox)
        self.assertEqual(exact.text, "Flash point: 12 °C")
        inexact = llm.resolve_citation(doc, llm.LlmCitation(page=1, quote="Flash point: 13 °C"), "rows")
        self.assertIsNone(inexact.bbox)
        self.assertEqual(inexact.text, "Flash point: 13 °C")
        self.assertIsNone(llm.resolve_citation(doc, None, "rows"))

    def _output(self, **overrides) -> llm.LlmOutput:
        silent = {"state": "NOT_STATED", "value": None, "derivation": "DIRECT", "rule": None, "evidence": None, "premises": []}
        premise = {"page": 1, "quote": "Substance/mixture : Mixture"}
        fields = {
            "product_name": silent,
            "supplier_name": silent,
            "substance_or_mixture": {"state": "PRESENT", "value": "MIXTURE", "derivation": "DIRECT", "rule": None, "evidence": premise, "premises": []},
            "product_cas_number": {"state": "NOT_APPLICABLE", "value": None, "derivation": "DERIVED", "rule": "identifier_undefined_for_form", "evidence": None, "premises": [premise]},
            "chemical_formula": {"state": "NOT_APPLICABLE", "value": None, "derivation": "DERIVED", "rule": None, "evidence": None, "premises": [premise]},
            "molecular_weight": silent,
            "flash_point": {"state": "PRESENT", "value": "12 °C", "derivation": "DIRECT", "rule": None, "evidence": {"page": 1, "quote": "Flash point: 12 °C"}, "premises": []},
            "ingredients_state": "NOT_STATED",
            "ingredients": [],
        }
        fields.update(overrides)
        return llm.LlmOutput.model_validate(fields)

    def test_output_converts_to_benchmark_schema_and_repairs_violations(self):
        doc = _doc(self._page())
        prediction = llm.to_document_prediction("synthetic.pdf", self._output(), doc, "rows")
        cas = prediction.fields["product_cas_number"]
        self.assertEqual(cas.derivation, "DERIVED")
        self.assertEqual(cas.rule, "identifier_undefined_for_form")
        self.assertEqual(len(cas.premises), 1)
        self.assertIsNotNone(cas.premises[0].bbox)
        formula = prediction.fields["chemical_formula"]
        self.assertEqual(formula.derivation, "DIRECT")
        self.assertIn("downgraded", formula.rationale)
        self.assertIsNotNone(formula.evidence)
        self.assertEqual(prediction.fields["flash_point"].value, "12 °C")
        self.assertEqual(prediction.fields["product_name"].state, FieldState.NOT_STATED)
        self.assertIn("chemical_formula", prediction.error or "")

    def test_prompt_states_the_rules_and_the_order(self):
        doc = _doc(self._page())
        system, user = llm.build_messages("synthetic.pdf", doc, "rows")
        self.assertIn("identifier_undefined_for_form", system)
        self.assertIn("composition_implies_form", system)
        self.assertIn("visual row order", system)
        self.assertIn("=== PAGE 1 ===", user)
        self.assertIn("Flash point: 12 °C", user)

    def test_cost_and_cache_key(self):
        self.assertAlmostEqual(llm.estimate_cost("claude-opus-5", 1_000_000, 100_000), 5.0 + 2.5)
        self.assertEqual(llm.estimate_cost("unknown-model", 10, 10), 0.0)
        self.assertEqual(llm.estimate_cost("qwen2.5:7b", 10, 10), 0.0)
        self.assertNotEqual(llm.cache_key("a", "s", "u"), llm.cache_key("b", "s", "u"))

    def test_local_model_uses_ollama_not_anthropic(self):
        self.assertFalse(llm.is_anthropic_model("qwen2.5:7b"))
        self.assertTrue(llm.is_anthropic_model("claude-opus-5"))
        captured = {}

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                body = {
                    "model": "qwen2.5:7b",
                    "done_reason": "stop",
                    "message": {"content": '{"product_name": "ok"}'},
                    "prompt_eval_count": 12,
                    "eval_count": 4,
                }
                return json.dumps(body).encode("utf-8")

        def fake_urlopen(request, timeout=0):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _Response()

        with patch("sdsbench.extractors.llm.urllib.request.urlopen", side_effect=fake_urlopen):
            payload = llm.call_model("qwen2.5:7b", "system", "user", max_tokens=32, num_ctx=2048)
        self.assertEqual(captured["url"], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(captured["body"]["options"]["num_ctx"], 2048)
        self.assertEqual(captured["body"]["options"]["temperature"], 0)
        self.assertIn("properties", captured["body"]["format"])
        self.assertEqual(payload["backend"], "ollama")
        self.assertEqual(payload["output"], {"product_name": "ok"})
        self.assertEqual(payload["usage"]["input_tokens"], 12)


if __name__ == "__main__":
    unittest.main()
