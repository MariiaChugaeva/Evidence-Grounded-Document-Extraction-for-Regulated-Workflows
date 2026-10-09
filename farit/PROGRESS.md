# Research Progress — Complete-Evidence Retrieval under Reading Budgets

**Farit Sharafutdinov · status report · October 9, 2026**

Scope: first phase of topic scoping — verify that the problem (a) exists in our data, (b) is not solved by standard retrieval, (c) is not covered by recent literature. All results below are independently re-verified; all code and numbers are reproducible from this repo (see Reproducibility).

## 1. Research question

Given a limited reading budget (pages or tokens), retrieve the **complete** evidence set for each SDS field — all complementary premises — and know when the set is incomplete (abstain). This complements Maria's source-binding work; with the project now single-author, her framework is reused as infrastructure only.

## 2. Infrastructure audit (prerequisite)

- Environment rebuilt; all **68 regression tests pass**.
- Gold integrity verified independently: all 252 evidence/premise spans resolve to exact `textlayer` offsets; corpus control sums match (29 documents, 203 field instances).
- One latent dataset fact surfaced: `sds_30.pdf` exists in the manifest/OCR but is unannotated — excluded from all metrics, kept as a future distractor document.

## 3. Corpus built for the retrieval formulation

`farit/data/corpus/` (regenerable via `farit/src/retrieval/build_corpus.py`):

- Page-level units for both text layers: **339 pages × 2 layers** (native / OCR), median 292 tokens/page.
- Gold evidence sets per field instance, merging direct evidence and derivation premises: **252 fragments** (166 evidence + 86 premises).
- Completeness is frequent, not exotic: **70/203 instances (34%) need ≥2 fragments**; **26 are cross-page**; 48 fields are derived (premise-bearing).

## 4. Literature check (threat analysis)

Full analysis of all five recommended papers in `farit/results/threat_analysis.md`, numbers spot-checked against the PDFs:

- **ExtractBench** (Aug 2026): grounding is scored with **OR-acceptance** — one of N evidence readings suffices; completeness of the evidence set is not measured; cost is measured but never constrains reading; no abstention calibration.
- **SDS LLM Benchmark** (Apr 2026): value accuracy only, **no evidence annotations**, 10 documents, sections processed independently; calibration named only as future work.
- **LMDX** (ACL Findings 2024): extraction + localization, but one value per field, chunks processed independently — no cross-page premise fusion, no completeness, no budgets.
- **Learn then Test** (Angelopoulos et al.): not a threat but a **tool** — a distribution-free framework for calibrating an acceptance/abstention threshold with finite-sample risk control; we plan to reuse it for the stopping/abstention rule.
- **LayoutLMv3** (MM 2022): strong Document AI baseline methodology; token classification per page, no notion of evidence sets, completeness or abstention.
- Conclusion: evidence-set completeness under reading budgets, cross-page premise fusion, alternative derivations, and completeness-aware abstention are **open**.

## 5. Baseline experiments

Protocol: schema-driven queries (field name + ontology description only — no access to gold values), page-level retrieval, **dev split only** (locked split untouched, reserved for final audit), deterministic implementations, 125 evidence-bearing dev instances (140 dev field instances minus 15 with empty evidence sets; ingredients excluded).

**Headline (native layer, macro strict completeness — ALL gold evidence pages retrieved):**

| Method | complete@1 | complete@3 | complete@10 | piece-recall@3 (lenient) |
|---|---|---|---|---|
| Naive front-to-back | 0.464 | **0.696** | 0.992 | 0.736 |
| BM25 | 0.352 | 0.600 | 0.896 | 0.612 |
| Dense (MiniLM-L6) | 0.328 | 0.632 | 0.952 | 0.668 |

**Hard slices, complete@3 (BM25 → Dense):** multi-evidence 0.40 → 0.52; cross-page 0.316 → 0.316 (not a typo: 6 of 19 instances for both methods — neither retriever fixes cross-page dispersion); derived fields 0.22 → 0.56.

Verification notes: an initial piece-recall bug (pages counted against a fragment denominator) was caught by an invariant check (`piece_recall ≥ complete`) and fixed; `complete@k` values were unaffected. Determinism confirmed by bitwise-identical reruns; BM25/naive figures in the dense report are read from the frozen BM25 output, not recomputed.

## 6. Findings

1. **The phenomenon is real and frequent** — a third of fields need multi-fragment evidence (data property, not a modeling artifact).
2. **Standard single-pass retrieval does not solve completeness** — lenient recall is consistently higher than strict completeness; finding *one* span is easy, finding *all* is not.
3. **The failure is structural, not lexical** — BM25 ≈ dense on the hard slices, and native ≈ OCR; per-page scoring has no notion of *coverage*, which is exactly what multi-premise and cross-page fields require.
4. **A trivial document prior beats both retrievers** — SDS templates concentrate evidence early, so naive reading sets the bar any proposed method must clear.
5. **Completeness is only interesting under budget** — naive@10 ≈ 0.99, i.e. "read everything" trivially solves completeness. The non-trivial problem is achieving completeness *cheaply* and *knowing when to stop / abstain* — which no existing benchmark measures. This sharpens the proposed contribution: budget-aware evidence gathering with a calibrated stopping rule.

## 7. Open questions for the topic decision

- **Does the budget constraint actually bite?** Naive reading reaches ~93% completeness by page 5 and ~99% by page 10, so the problem may be hard only in a narrow budget band. Next step: plot completeness-vs-budget curves to find where (and whether) a genuinely hard regime exists in our data.
- **Is 29 annotated sheets enough?** Sufficient for this pilot; whether to extend annotation is an open decision — it depends on which method direction we choose, so it should follow the topic decision, not precede it.
- **What is the contribution's shape?** Options: (i) measurement framework alone (the corpus + completeness/budget metrics, with baselines' failure analysis); (ii) a method (budget-aware evidence gathering that **abstains** — declines a field and routes it to human review — when its evidence set is predicted incomplete, threshold calibrated via Learn then Test); (iii) both as one story.

## Reproducibility

- Corpus: `python farit/src/retrieval/build_corpus.py` (writes `farit/data/corpus/`, validates offsets/counts)
- Baselines: `python farit/src/retrieval/bm25_baseline.py`, `python farit/src/retrieval/dense_baseline.py`
- Results: `farit/results/bm25_baseline.{md,json}`, `farit/results/dense_baseline.{md,json}`
- Python env: `~/ds_env` (3.11); pinned `pymupdf==1.28.2` preserved (gold boxes depend on its segmentation)
