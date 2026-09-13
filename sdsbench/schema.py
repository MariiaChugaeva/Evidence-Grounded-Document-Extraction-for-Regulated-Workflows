"""Prediction output contract: state, value, and either direct evidence or premises + rule.

Boxes are page fractions in [0, 1], origin at the top-left of the displayed page.

An observed (DIRECT) claim cites one evidence span that states it. A DERIVED
claim cites one or more premise spans and names the rule that licenses the
conclusion (see ``sdsbench.derivation``); it must not pretend the premise is
direct evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .ontology import FieldState

BBoxProvenance = Literal["system", "derived", "none"]
DerivationKind = Literal["DIRECT", "DERIVED"]


class EvidenceSpan(BaseModel):
    page: int = Field(..., ge=1)
    text: str
    bbox: list[float] | None = None
    bbox_provenance: BBoxProvenance = "none"

    @field_validator("bbox")
    @classmethod
    def _check_bbox(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("bbox must have four elements")
        x0, y0, x1, y1 = value
        if not all(-0.01 <= item <= 1.01 for item in value):
            raise ValueError(f"bbox must be page fractions in [0, 1], got {value}")
        if x1 < x0 or y1 < y0:
            raise ValueError(f"bbox must be ordered [x0, y0, x1, y1], got {value}")
        return value


class FieldPrediction(BaseModel):
    state: FieldState
    value: str | None = None
    evidence: EvidenceSpan | None = None
    derivation: DerivationKind = "DIRECT"
    rule: str | None = None
    premises: list[EvidenceSpan] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    rationale: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "FieldPrediction":
        if self.state is not FieldState.PRESENT and self.value:
            raise ValueError(
                f"state {self.state.value} must not carry a value ({self.value!r})"
            )
        if self.state is FieldState.NOT_STATED and (self.evidence is not None or self.premises):
            raise ValueError("NOT_STATED must not carry evidence or premises")
        if self.derivation == "DERIVED":
            if not self.rule:
                raise ValueError("a DERIVED claim must name its rule")
            if not self.premises:
                raise ValueError("a DERIVED claim must cite at least one premise span")
            if self.evidence is not None:
                raise ValueError("a DERIVED claim cites premises, not direct evidence")
        elif self.rule or self.premises:
            raise ValueError("a DIRECT claim must not carry a rule or premises")
        return self

    @property
    def cited_spans(self) -> list[EvidenceSpan]:
        """The spans the claim rests on: premises for a derived claim, else the evidence."""
        if self.derivation == "DERIVED":
            return list(self.premises)
        return [self.evidence] if self.evidence is not None else []


class IngredientPrediction(BaseModel):
    name: str | None = None
    cas_number: str | None = None
    cas_state: FieldState = FieldState.PRESENT
    concentration: str | None = None
    evidence: EvidenceSpan | None = None


class IngredientsPrediction(BaseModel):
    state: FieldState
    items: list[IngredientPrediction] = Field(default_factory=list)
    evidence: EvidenceSpan | None = None


class DocumentPrediction(BaseModel):
    document: str
    fields: dict[str, FieldPrediction]
    ingredients: IngredientsPrediction | None = None
    error: str | None = None


class PredictionSet(BaseModel):
    system: str
    configuration: dict[str, str] = Field(default_factory=dict)
    documents: list[DocumentPrediction]

    def by_document(self) -> dict[str, DocumentPrediction]:
        return {item.document: item for item in self.documents}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.model_dump(mode="json"), handle, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "PredictionSet":
        with open(path, encoding="utf-8") as handle:
            return cls.model_validate(json.load(handle))
