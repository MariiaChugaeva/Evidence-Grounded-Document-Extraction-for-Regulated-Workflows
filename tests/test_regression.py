"""Regression tests for the annotation and evaluation bugs already found."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from sdsbench.annotations import (
    Derivation,
    DocumentAnnotation,
    EvidenceRef,
    FieldAnnotation,
    IngredientAnnotation,
    ResolvedEvidence,
    resolve_document,
)
from sdsbench.evaluator import (
    NOT_APPLICABLE,
    SATISFIED,
    VIOLATED,
    evaluate_instance,
)
from sdsbench.matching import values_equal
from sdsbench.ontology import FieldState, ProductForm, ValueKind, field_is_defined_for
from sdsbench.schema import EvidenceSpan, FieldPrediction

SOURCE_DIR = Path("data/annotations/source")


def _resolved(
    page: int = 1,
    text: str = "Flash Point >=290 F",
    bbox: list[float] | None = None,
) -> ResolvedEvidence:
    return ResolvedEvidence(
        page=page,
        text=text,
        char_start=0,
        char_end=len(text),
        bbox=bbox or [0.10, 0.40, 0.55, 0.44],
        occurrences_on_page=1,
    )


def _document(field_name: str, gold: FieldAnnotation) -> DocumentAnnotation:
    return DocumentAnnotation(
        document="synthetic.pdf",
        split="dev",
        product_form=ProductForm.MIXTURE,
        fields={field_name: gold},
        ingredients=IngredientAnnotation(state=FieldState.NOT_STATED),
    )


def _present_gold() -> FieldAnnotation:
    return FieldAnnotation(
        state=FieldState.PRESENT,
        derivation=Derivation.EXPLICIT,
        value=">=290 ºF",
        accepted_values=[">=143.3 C"],
        resolved_evidence=[_resolved()],
    )


class FakeLayer:
    def page(self, _number: int):
        return object()


def _score(field_name, gold, prediction):
    with (
        patch("sdsbench.evaluator.textlayer.load", return_value=FakeLayer()),
        patch("sdsbench.evaluator.textlayer.occurs_on_page", return_value=True),
    ):
        return evaluate_instance(_document(field_name, gold), field_name, gold, prediction)


class TestAbsentNoLongerCountsAsEvidence(unittest.TestCase):
    """Old 'evidence accuracy' treated ABSENT/ABSENT as a localization success."""

    def test_not_stated_agreement_has_no_localization_axes(self):
        gold = FieldAnnotation(state=FieldState.NOT_STATED, derivation=Derivation.ABSENT)
        pred = FieldPrediction(state=FieldState.NOT_STATED)
        result = _score("flash_point", gold, pred)
        self.assertEqual(result.axes["state"], SATISFIED)
        self.assertEqual(result.axes["value"], SATISFIED)
        self.assertEqual(result.axes["page"], NOT_APPLICABLE)
        self.assertEqual(result.axes["span"], NOT_APPLICABLE)
        self.assertEqual(result.axes["bbox"], NOT_APPLICABLE)
        self.assertFalse(result.gold_requires_evidence)
        self.assertTrue(result.joint_correct)

    def test_present_field_missing_citation_fails_page_span_bbox(self):
        gold = _present_gold()
        pred = FieldPrediction(state=FieldState.PRESENT, value=">=290 ºF")
        result = _score("flash_point", gold, pred)
        self.assertEqual(result.axes["value"], SATISFIED)
        self.assertEqual(result.axes["page"], VIOLATED)
        self.assertEqual(result.axes["span"], VIOLATED)
        self.assertEqual(result.axes["bbox"], VIOLATED)
        self.assertFalse(result.joint_correct)


class TestPageAndBoxAreChecked(unittest.TestCase):
    def test_wrong_page_fails_even_if_value_matches(self):
        gold = _present_gold()
        pred = FieldPrediction(
            state=FieldState.PRESENT,
            value=">=290 ºF",
            evidence=EvidenceSpan(
                page=9,
                text="Flash Point >=290 F",
                bbox=[0.10, 0.40, 0.55, 0.44],
                bbox_provenance="system",
            ),
        )
        result = _score("flash_point", gold, pred)
        self.assertEqual(result.axes["value"], SATISFIED)
        self.assertEqual(result.axes["page"], VIOLATED)
        self.assertEqual(result.axes["span"], VIOLATED)
        self.assertFalse(result.joint_correct)

    def test_right_page_and_overlapping_box_can_pass_localization(self):
        gold = _present_gold()
        pred = FieldPrediction(
            state=FieldState.PRESENT,
            value=">=290 ºF",
            evidence=EvidenceSpan(
                page=1,
                text="Flash Point >=290 F",
                bbox=[0.10, 0.40, 0.55, 0.44],
                bbox_provenance="system",
            ),
        )
        result = _score("flash_point", gold, pred)
        self.assertEqual(result.axes["page"], SATISFIED)
        self.assertEqual(result.axes["span"], SATISFIED)
        self.assertEqual(result.axes["bbox"], SATISFIED)


class TestValueMatching(unittest.TestCase):
    def test_flash_point_fahrenheit_matches_celsius(self):
        self.assertTrue(
            values_equal(">=290 ºF", ">=143.3 C", ValueKind.TEMPERATURE, tolerance=0.6)
        )

    def test_formula_ignores_pdf_word_breaks(self):
        self.assertTrue(values_equal("F3CCH2 F", "F3CCH2F", ValueKind.FORMULA))

    def test_non_present_prediction_cannot_carry_a_value(self):
        with self.assertRaises(ValueError):
            FieldPrediction(state=FieldState.NO_DATA, value="61 C")

    def test_invented_present_value_on_no_data_fails(self):
        gold = FieldAnnotation(
            state=FieldState.NO_DATA,
            derivation=Derivation.EXPLICIT,
            resolved_evidence=[_resolved(text="Flash point: no data available")],
        )
        pred = FieldPrediction(
            state=FieldState.PRESENT,
            value="61 C",
            evidence=EvidenceSpan(page=1, text="Flash point: no data available"),
        )
        result = _score("flash_point", gold, pred)
        self.assertEqual(result.axes["state"], VIOLATED)
        self.assertEqual(result.axes["value"], VIOLATED)


class TestMixtureOntology(unittest.TestCase):
    def test_product_cas_is_undefined_for_mixtures(self):
        self.assertFalse(field_is_defined_for("product_cas_number", ProductForm.MIXTURE))
        self.assertFalse(field_is_defined_for("chemical_formula", ProductForm.MIXTURE))
        self.assertFalse(field_is_defined_for("molecular_weight", ProductForm.MIXTURE))
        self.assertTrue(field_is_defined_for("flash_point", ProductForm.MIXTURE))
        self.assertTrue(field_is_defined_for("product_cas_number", ProductForm.SUBSTANCE))


class TestCorrectedGold(unittest.TestCase):
    def test_sds_18_flash_point_is_present(self):
        with open(SOURCE_DIR / "sds_18.json", encoding="utf-8") as handle:
            payload = json.load(handle)
        flash = payload["fields"]["flash_point"]
        self.assertEqual(flash["state"], "PRESENT")
        self.assertIn("290", flash["value"])

    def test_sds_18_product_cas_is_not_an_ingredient_cas(self):
        with open(SOURCE_DIR / "sds_18.json", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["fields"]["product_cas_number"]["state"], "NOT_APPLICABLE")
        ingredient_cas = {item["cas_number"] for item in payload["ingredients"]["items"]}
        self.assertIn("101-68-8", ingredient_cas)

    def test_sds_02_does_not_promote_ingredient_cas_to_product(self):
        with open(SOURCE_DIR / "sds_02.json", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["fields"]["product_cas_number"]["state"], "NOT_APPLICABLE")
        self.assertEqual(payload["fields"]["flash_point"]["state"], "NOT_STATED")

    def test_fabricated_span_cannot_enter_gold(self):
        with open(SOURCE_DIR / "sds_17.json", encoding="utf-8") as handle:
            payload = json.load(handle)
        annotation = DocumentAnnotation.from_dict(payload)
        annotation.fields["flash_point"].evidence.append(
            EvidenceRef(page=1, text="THIS_SPAN_IS_NOT_IN_THE_PDF_XYZ987")
        )
        problems = resolve_document(annotation)
        self.assertTrue(any("span not found" in item for item in problems))


if __name__ == "__main__":
    unittest.main()
