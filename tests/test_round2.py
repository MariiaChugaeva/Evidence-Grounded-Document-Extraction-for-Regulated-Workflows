"""Regression tests for the round-2 review findings (P1-P8 in docs/known_problems.md).

Synthetic pages are built from tokens so that verbatim, span and box checks
run against real code paths without a PDF.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from sdsbench import derivation, oracle, textlayer
from sdsbench.annotations import (
    Derivation,
    DocumentAnnotation,
    FieldAnnotation,
    Ingredient,
    IngredientAnnotation,
    ResolvedEvidence,
    resolve_document,
)
from sdsbench.derivation import RULE_COMPOSITION_FORM, RULE_IDENTIFIER_UNDEFINED
from sdsbench.evaluator import (
    NOT_APPLICABLE,
    SATISFIED,
    UNDECIDABLE,
    VIOLATED,
    Summary,
    evaluate_ingredients,
    evaluate_instance,
)
from sdsbench.manifest import check_manifest, load_manifest
from sdsbench.matching import concentrations_equal, parse_concentration, span_overlap
from sdsbench.ontology import FieldState, ProductForm, ValueKind, classify_form_cue
from sdsbench.schema import (
    DocumentPrediction,
    EvidenceSpan,
    FieldPrediction,
    IngredientPrediction,
    IngredientsPrediction,
    PredictionSet,
)
from sdsbench.textlayer import DocumentText, Token

SOURCE_DIR = Path("data/annotations/source")


# --------------------------------------------------------------------------
# Synthetic page helpers
# --------------------------------------------------------------------------


def _tokens(words: list[str], y: float, x0: float = 0.10, step: float = 0.07) -> list[Token]:
    return [
        Token(text=word, bbox=(x0 + index * step, y, x0 + index * step + 0.06, y + 0.012))
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


def _span(page: textlayer.Page, text: str, bbox: list[float] | None = None) -> EvidenceSpan:
    if bbox is None:
        anchors = textlayer.find_all_in_page(page, text)
        bbox = list(anchors[0].bbox) if anchors else None
    return EvidenceSpan(page=1, text=text, bbox=bbox, bbox_provenance="system")


def _document(field_name: str, gold: FieldAnnotation) -> DocumentAnnotation:
    return DocumentAnnotation(
        document="synthetic.pdf",
        split="dev",
        product_form=ProductForm.MIXTURE,
        fields={field_name: gold},
        ingredients=IngredientAnnotation(state=FieldState.NOT_STATED),
    )


def _score(doc: DocumentText, field_name: str, gold: FieldAnnotation, prediction):
    with patch("sdsbench.evaluator.textlayer.load", return_value=doc):
        return evaluate_instance(_document(field_name, gold), field_name, gold, prediction)


FLASH_LINE = ["Flash", "Point:", "61", "°C"]


def _flash_page() -> textlayer.Page:
    return _page([(0.40, FLASH_LINE), (0.42, ["Boiling", "point:", "100", "°C"])])


def _flash_gold(page: textlayer.Page) -> FieldAnnotation:
    return FieldAnnotation(
        state=FieldState.PRESENT,
        derivation=Derivation.EXPLICIT,
        value="61 °C",
        resolved_evidence=[_resolved(page, "Flash Point: 61 °C")],
    )


# --------------------------------------------------------------------------
# P2: verbatim is exact
# --------------------------------------------------------------------------


class TestVerbatimIsExact(unittest.TestCase):
    def test_one_character_difference_is_not_verbatim(self):
        page = _flash_page()
        pred = FieldPrediction(
            state=FieldState.PRESENT, value="62 °C", evidence=_span(page, "Flash Point: 62 °C")
        )
        result = _score(_doc(page), "flash_point", _flash_gold(page), pred)
        self.assertFalse(result.verbatim)
        self.assertEqual(result.axes["support"], VIOLATED)
        self.assertFalse(result.joint_correct)
        self.assertGreater(result.verbatim_fuzzy_score, 90.0)

    def test_documented_folding_is_still_verbatim(self):
        page = _page([(0.40, ["Flash", "Point:", "61", "ºC"])])
        self.assertTrue(textlayer.occurs_on_page(page, "flash  point: 61 °C"))
        self.assertFalse(textlayer.occurs_on_page(page, "Flash Point: 61 °F"))

    def test_fuzzy_score_never_decides(self):
        page = _flash_page()
        self.assertFalse(textlayer.occurs_on_page(page, "Flash Point: 62 °C"))
        self.assertGreater(textlayer.fuzzy_occurrence_score(page, "Flash Point: 62 °C"), 90.0)
        # a quote cut mid-word is still an exact substring, hence verbatim
        self.assertTrue(textlayer.occurs_on_page(page, "Flash Point: 61 °C Boiling poin"))

    def test_section_headings_declare_the_form(self):
        self.assertIs(classify_form_cue("Mixtures Composition Comments: The components are not hazardous"), ProductForm.MIXTURE)
        self.assertIs(classify_form_cue("3.1. Substances"), ProductForm.SUBSTANCE)
        self.assertIsNone(classify_form_cue("Substances or 3.2 Mixtures"))
        self.assertIsNone(classify_form_cue("Substance name: sodium chloride CAS No.: 7647-14-5"))


# --------------------------------------------------------------------------
# Span axis: exact token overlap, no partial-ratio
# --------------------------------------------------------------------------


class TestSpanOverlapIsExact(unittest.TestCase):
    def test_short_fragment_of_gold_fails_coverage(self):
        page = _flash_page()
        pred = FieldPrediction(state=FieldState.PRESENT, value="61 °C", evidence=_span(page, "61 °C"))
        result = _score(_doc(page), "flash_point", _flash_gold(page), pred)
        self.assertTrue(result.verbatim)
        self.assertEqual(result.axes["span"], VIOLATED)
        self.assertLess(result.span_coverage, 0.8)

    def test_paragraph_citation_fails_iou(self):
        page = _page([(0.40, FLASH_LINE + ["Boiling", "point:", "100", "°C", "Density:", "1.0", "g/cm3", "pH:", "7"])])
        gold = _flash_gold(page)
        text = "Flash Point: 61 °C Boiling point: 100 °C Density: 1.0 g/cm3 pH: 7"
        pred = FieldPrediction(state=FieldState.PRESENT, value="61 °C", evidence=_span(page, text))
        result = _score(_doc(page), "flash_point", gold, pred)
        self.assertEqual(result.span_coverage, 1.0)
        self.assertEqual(result.axes["span"], VIOLATED)

    def test_overlap_counts_only_exact_tokens(self):
        overlap = span_overlap("Flash Point: 610 °C", "Flash Point: 61 °C")
        self.assertEqual(overlap.matched, 2)
        self.assertLess(overlap.coverage, 0.8)
        self.assertEqual(span_overlap("flash", "Flash Point: 61 °C").coverage, 0.25)


# --------------------------------------------------------------------------
# P3: boxes are checked in two dimensions
# --------------------------------------------------------------------------


class TestBoxDilutionIsTwoDimensional(unittest.TestCase):
    def test_full_width_box_fails(self):
        page = _flash_page()
        gold = _flash_gold(page)
        tight = list(gold.resolved_evidence[0].bbox)
        wide = [0.02, tight[1], 0.98, tight[3]]
        pred = FieldPrediction(
            state=FieldState.PRESENT, value="61 °C", evidence=_span(page, "Flash Point: 61 °C", wide)
        )
        result = _score(_doc(page), "flash_point", gold, pred)
        self.assertEqual(result.bbox_coverage, 1.0)
        self.assertLess(result.bbox_iou, 0.5)
        self.assertEqual(result.axes["bbox"], VIOLATED)

    def test_tight_box_passes(self):
        page = _flash_page()
        gold = _flash_gold(page)
        pred = FieldPrediction(
            state=FieldState.PRESENT, value="61 °C", evidence=_span(page, "Flash Point: 61 °C")
        )
        result = _score(_doc(page), "flash_point", gold, pred)
        self.assertEqual(result.axes["bbox"], SATISFIED)
        self.assertTrue(result.joint_correct)

    def test_containment_reports_area_dilution(self):
        coverage, dilution = textlayer.bbox_containment((0.0, 0.0, 1.0, 0.1), (0.2, 0.0, 0.4, 0.1))
        self.assertEqual(coverage, 1.0)
        self.assertAlmostEqual(dilution, 5.0)


# --------------------------------------------------------------------------
# P4: support is decisive; undecidable is not success
# --------------------------------------------------------------------------


class TestSupportIsDecisive(unittest.TestCase):
    def test_not_applicable_without_cue_is_undecidable_and_not_joint(self):
        page = _page([(0.40, ["Flash", "Point:", "see", "Section", "5"])])
        gold = FieldAnnotation(
            state=FieldState.NOT_APPLICABLE,
            derivation=Derivation.EXPLICIT,
            resolved_evidence=[_resolved(page, "Flash Point: see Section 5")],
        )
        pred = FieldPrediction(
            state=FieldState.NOT_APPLICABLE, evidence=_span(page, "Flash Point: see Section 5")
        )
        result = _score(_doc(page), "flash_point", gold, pred)
        self.assertEqual(result.axes["state"], SATISFIED)
        self.assertEqual(result.axes["span"], SATISFIED)
        self.assertEqual(result.axes["support"], UNDECIDABLE)
        self.assertFalse(result.joint_correct)

    def test_categorical_claim_is_checked_against_the_declaration(self):
        page = _page([(0.40, ["Substance/mixture", ":", "Mixture"]), (0.42, ["Section", "3:", "Composition"])])
        gold = FieldAnnotation(
            state=FieldState.PRESENT,
            derivation=Derivation.EXPLICIT,
            value="MIXTURE",
            resolved_evidence=[_resolved(page, "Substance/mixture : Mixture")],
        )
        right = FieldPrediction(
            state=FieldState.PRESENT, value="MIXTURE", evidence=_span(page, "Substance/mixture : Mixture")
        )
        wrong = FieldPrediction(
            state=FieldState.PRESENT, value="SUBSTANCE", evidence=_span(page, "Substance/mixture : Mixture")
        )
        silent = FieldPrediction(
            state=FieldState.PRESENT, value="MIXTURE", evidence=_span(page, "Section 3: Composition")
        )
        self.assertEqual(_score(_doc(page), "substance_or_mixture", gold, right).axes["support"], SATISFIED)
        self.assertEqual(_score(_doc(page), "substance_or_mixture", gold, wrong).axes["support"], VIOLATED)
        self.assertEqual(_score(_doc(page), "substance_or_mixture", gold, silent).axes["support"], UNDECIDABLE)

    def test_summary_counts_undecidable_as_failure(self):
        page = _page([(0.40, ["Flash", "Point:", "see", "Section", "5"])])
        gold = FieldAnnotation(
            state=FieldState.NOT_APPLICABLE,
            derivation=Derivation.EXPLICIT,
            resolved_evidence=[_resolved(page, "Flash Point: see Section 5")],
        )
        pred = FieldPrediction(
            state=FieldState.NOT_APPLICABLE, evidence=_span(page, "Flash Point: see Section 5")
        )
        summary = Summary(system="t", instances=[_score(_doc(page), "flash_point", gold, pred)])
        self.assertEqual(summary.joint_rate(), 0.0)
        self.assertEqual(summary.undecidable_count(), 1)
        self.assertEqual(summary.axis("support").undecidable, 1)

    def test_form_cue_lexicon(self):
        self.assertIs(classify_form_cue("Substance/mixture : Substance"), ProductForm.SUBSTANCE)
        self.assertIs(classify_form_cue("Substance or preparation: Preparation"), ProductForm.MIXTURE)
        self.assertIs(classify_form_cue("3.1. Substances Not applicable"), ProductForm.MIXTURE)
        self.assertIs(classify_form_cue("Substance: Non-applicable"), ProductForm.MIXTURE)
        self.assertIs(classify_form_cue('Mel-Rol is defined by OSHA as an "article."'), ProductForm.ARTICLE)
        self.assertIsNone(classify_form_cue("not a hazardous substance or mixture"))
        self.assertIsNone(classify_form_cue("Ingredient C.A.S. No. % by Wt"))


# --------------------------------------------------------------------------
# P5: derived claims cite premises and a rule
# --------------------------------------------------------------------------


class TestDerivedClaims(unittest.TestCase):
    PREMISE = "Substance/mixture : Mixture"

    def _setup(self):
        page = _page([(0.40, ["Substance/mixture", ":", "Mixture"])])
        gold = FieldAnnotation(
            state=FieldState.NOT_APPLICABLE,
            derivation=Derivation.ONTOLOGY,
            rule=RULE_IDENTIFIER_UNDEFINED,
            resolved_premises=[_resolved(page, self.PREMISE)],
        )
        return page, gold

    def test_premise_cited_as_direct_evidence_is_not_joint_correct(self):
        page, gold = self._setup()
        pred = FieldPrediction(state=FieldState.NOT_APPLICABLE, evidence=_span(page, self.PREMISE))
        result = _score(_doc(page), "product_cas_number", gold, pred)
        self.assertEqual(result.axes["derivation"], VIOLATED)
        self.assertEqual(result.axes["support"], UNDECIDABLE)
        self.assertFalse(result.joint_correct)

    def test_derived_claim_with_rule_and_premise_is_joint_correct(self):
        page, gold = self._setup()
        pred = FieldPrediction(
            state=FieldState.NOT_APPLICABLE,
            derivation="DERIVED",
            rule=RULE_IDENTIFIER_UNDEFINED,
            premises=[_span(page, self.PREMISE)],
        )
        result = _score(_doc(page), "product_cas_number", gold, pred)
        self.assertEqual(result.axes["page"], SATISFIED)
        self.assertEqual(result.axes["span"], SATISFIED)
        self.assertEqual(result.axes["bbox"], SATISFIED)
        self.assertEqual(result.axes["derivation"], SATISFIED)
        self.assertEqual(result.axes["support"], SATISFIED)
        self.assertTrue(result.joint_correct)
        self.assertEqual(result.evidence_role, "premise")

    def test_wrong_rule_fails_derivation_and_support(self):
        page, gold = self._setup()
        pred = FieldPrediction(
            state=FieldState.NOT_APPLICABLE,
            derivation="DERIVED",
            rule=RULE_COMPOSITION_FORM,
            premises=[_span(page, self.PREMISE)],
        )
        result = _score(_doc(page), "product_cas_number", gold, pred)
        self.assertEqual(result.axes["derivation"], VIOLATED)
        self.assertEqual(result.axes["support"], VIOLATED)

    def test_premise_that_establishes_a_substance_does_not_license_not_applicable(self):
        page = _page([(0.40, ["Substance/mixture", ":", "Substance"])])
        gold = FieldAnnotation(
            state=FieldState.NOT_APPLICABLE,
            derivation=Derivation.ONTOLOGY,
            rule=RULE_IDENTIFIER_UNDEFINED,
            resolved_premises=[_resolved(page, "Substance/mixture : Substance")],
        )
        pred = FieldPrediction(
            state=FieldState.NOT_APPLICABLE,
            derivation="DERIVED",
            rule=RULE_IDENTIFIER_UNDEFINED,
            premises=[_span(page, "Substance/mixture : Substance")],
        )
        result = _score(_doc(page), "product_cas_number", gold, pred)
        self.assertEqual(result.axes["support"], VIOLATED)

    def test_rule_registry(self):
        check = derivation.check_rule(
            RULE_IDENTIFIER_UNDEFINED, "chemical_formula", FieldState.NOT_APPLICABLE, None,
            ["Chemical description: Mixture of substances"],
        )
        self.assertEqual(check.status, SATISFIED)
        self.assertEqual(
            derivation.check_rule(
                RULE_IDENTIFIER_UNDEFINED, "flash_point", FieldState.NOT_APPLICABLE, None, ["Mixture"]
            ).status,
            VIOLATED,
        )
        self.assertEqual(
            derivation.check_rule(
                RULE_IDENTIFIER_UNDEFINED, "chemical_formula", FieldState.NOT_APPLICABLE, None,
                ["Section 3"],
            ).status,
            UNDECIDABLE,
        )
        self.assertEqual(
            derivation.check_rule(
                RULE_COMPOSITION_FORM, "substance_or_mixture", FieldState.PRESENT, "SUBSTANCE",
                ["Argon 100 7440-37-1"],
            ).status,
            SATISFIED,
        )
        self.assertEqual(
            derivation.check_rule(
                RULE_COMPOSITION_FORM, "substance_or_mixture", FieldState.PRESENT, "MIXTURE",
                ["Argon 100 7440-37-1"],
            ).status,
            VIOLATED,
        )
        self.assertIs(
            derivation.premise_form(["Aluminum 7429-90-5 1~10", "Copper 7440-50-8 1~15"]),
            ProductForm.MIXTURE,
        )

    def test_schema_keeps_premises_and_direct_evidence_apart(self):
        span = EvidenceSpan(page=1, text="Substance/mixture : Mixture")
        with self.assertRaises(ValueError):
            FieldPrediction(state=FieldState.NOT_APPLICABLE, derivation="DERIVED", premises=[span])
        with self.assertRaises(ValueError):
            FieldPrediction(
                state=FieldState.NOT_APPLICABLE,
                derivation="DERIVED",
                rule=RULE_IDENTIFIER_UNDEFINED,
                premises=[span],
                evidence=span,
            )
        with self.assertRaises(ValueError):
            FieldPrediction(state=FieldState.NOT_APPLICABLE, evidence=span, rule=RULE_IDENTIFIER_UNDEFINED)

    def test_gold_no_longer_inherits_form_spans_as_direct_evidence(self):
        with open(SOURCE_DIR / "sds_02.json", encoding="utf-8") as handle:
            annotation = DocumentAnnotation.from_dict(json.load(handle))
        problems = resolve_document(annotation)
        self.assertEqual(problems, [])
        cas = annotation.fields["product_cas_number"]
        self.assertEqual(cas.resolved_evidence, [])
        self.assertTrue(cas.resolved_premises)
        self.assertEqual(cas.rule, RULE_IDENTIFIER_UNDEFINED)
        self.assertTrue(annotation.fields["substance_or_mixture"].resolved_evidence)


# --------------------------------------------------------------------------
# P1: the oracle localizes by value only
# --------------------------------------------------------------------------


class TestOracleUsesNoGoldHints(unittest.TestCase):
    def test_value_absent_from_text_is_not_localized_even_if_the_gold_span_is(self):
        page = _page([(0.40, ["Substance", "or", "preparation:", "Preparation"])])
        gold = FieldAnnotation(
            state=FieldState.PRESENT,
            derivation=Derivation.EXPLICIT,
            value="MIXTURE",
            resolved_evidence=[_resolved(page, "Substance or preparation: Preparation")],
        )
        row = oracle.analyze_field(
            "synthetic.pdf", "substance_or_mixture", gold, _doc(page), None, ValueKind.CATEGORICAL
        )
        self.assertEqual(row.value_located_native, "no")
        self.assertEqual(row.span_exact_native, "yes")
        self.assertEqual(oracle.locate_value_exact(_doc(page), ["MIXTURE"]), [])

    def test_value_present_elsewhere_is_localized_without_page_hint(self):
        page = _page([(0.40, ["Flash", "Point:", "61", "°C"]), (0.42, ["Autoignition:", "300", "°C"])])
        gold = FieldAnnotation(
            state=FieldState.PRESENT,
            derivation=Derivation.EXPLICIT,
            value="61 °C",
            resolved_evidence=[_resolved(page, "Flash Point: 61 °C")],
        )
        row = oracle.analyze_field(
            "synthetic.pdf", "flash_point", gold, _doc(page), None, ValueKind.TEMPERATURE
        )
        self.assertEqual(row.value_located_native, "yes")
        self.assertEqual(row.value_on_gold_page_native, "yes")


# --------------------------------------------------------------------------
# P6: ingredient records
# --------------------------------------------------------------------------


class TestIngredientRecords(unittest.TestCase):
    def test_concentration_equivalences(self):
        self.assertTrue(concentrations_equal("20~60", "20 - 60"))
        self.assertTrue(concentrations_equal("1 - <10 %", "1-<10"))
        self.assertTrue(concentrations_equal("NMT 10", "≤ 10"))
        self.assertTrue(concentrations_equal("100", "100 %"))
        self.assertTrue(concentrations_equal("60% ~ 70%", "60 - 70"))
        self.assertFalse(concentrations_equal("< 20", "≤ 20"))
        self.assertFalse(concentrations_equal("1~10", "1~15"))
        self.assertIsNone(parse_concentration("none"))

    def _gold(self) -> DocumentAnnotation:
        return DocumentAnnotation(
            document="synthetic.pdf",
            split="dev",
            product_form=ProductForm.MIXTURE,
            fields={},
            ingredients=IngredientAnnotation(
                state=FieldState.PRESENT,
                items=[
                    Ingredient(name="Aluminum", cas_number="7429-90-5", cas_state=FieldState.PRESENT, concentration="1~10"),
                    Ingredient(name="Wood dust", cas_number=None, cas_state=FieldState.NOT_APPLICABLE, concentration="85-100"),
                ],
            ),
        )

    def _predictions(self, concentration: str) -> PredictionSet:
        return PredictionSet(
            system="t",
            documents=[
                DocumentPrediction(
                    document="synthetic.pdf",
                    fields={},
                    ingredients=IngredientsPrediction(
                        state=FieldState.PRESENT,
                        items=[
                            IngredientPrediction(
                                name="Aluminum", cas_number="7429-90-5", concentration=concentration
                            )
                        ],
                    ),
                )
            ],
        )

    def test_record_requires_every_attribute(self):
        good = evaluate_ingredients([self._gold()], self._predictions("1 - 10"))
        self.assertEqual(good.matched, 1)
        self.assertEqual(good.records_exact, 1)
        self.assertEqual(good.records_with_evidence, 0)
        bad = evaluate_ingredients([self._gold()], self._predictions("1 - 15"))
        self.assertEqual(bad.matched, 1)
        self.assertEqual(bad.records_exact, 0)
        self.assertFalse(bad.matches[0].concentration_ok)
        self.assertTrue(bad.matches[0].name_ok and bad.matches[0].cas_ok)

    def test_rows_without_cas_count_towards_recall(self):
        summary = evaluate_ingredients([self._gold()], self._predictions("1 - 10"))
        self.assertEqual(summary.gold_rows, 2)
        self.assertEqual(summary.recall, 0.5)
        self.assertEqual(summary.precision, 1.0)
        self.assertEqual(summary.state_agreement, (1, 1))


# --------------------------------------------------------------------------
# P7: splits and manifest
# --------------------------------------------------------------------------


class TestSplitsAndManifest(unittest.TestCase):
    def test_locked_split_is_refused_without_flag(self):
        from scripts.evaluate import LockedSplitError, guard_split

        locked = DocumentAnnotation(
            document="x.pdf",
            split="locked",
            product_form=ProductForm.UNDETERMINED,
            fields={},
            ingredients=IngredientAnnotation(state=FieldState.NOT_STATED),
        )
        with self.assertRaises(LockedSplitError):
            guard_split([locked], allow_locked=False)
        guard_split([locked], allow_locked=True)

    def test_manifest_covers_every_pdf(self):
        entries = load_manifest()
        self.assertEqual(check_manifest(entries), [])
        self.assertEqual(sum(entry.split == "locked" for entry in entries.values()), 10)

    def test_dev_documents_are_never_locked(self):
        entries = load_manifest()
        for number in range(1, 21):
            self.assertEqual(entries[f"sds_{number:02d}.pdf"].split, "dev")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


class TestJointOverEvidenceBearing(unittest.TestCase):
    def test_silent_agreement_is_excluded_from_evidence_bearing_joint(self):
        page = _flash_page()
        silent_gold = FieldAnnotation(state=FieldState.NOT_STATED, derivation=Derivation.ABSENT)
        silent = _score(_doc(page), "flash_point", silent_gold, FieldPrediction(state=FieldState.NOT_STATED))
        wrong = _score(
            _doc(page), "flash_point", _flash_gold(page), FieldPrediction(state=FieldState.NOT_STATED)
        )
        summary = Summary(system="t", instances=[silent, wrong])
        self.assertTrue(silent.joint_correct)
        self.assertEqual(silent.axes["support"], NOT_APPLICABLE)
        self.assertEqual(summary.joint_rate(), 0.5)
        self.assertEqual(summary.joint_rate(summary.evidence_bearing), 0.0)


if __name__ == "__main__":
    unittest.main()
