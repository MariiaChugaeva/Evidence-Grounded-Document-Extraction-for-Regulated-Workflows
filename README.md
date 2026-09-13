# Evidence-Grounded Document Extraction for Regulated Workflows

Bachelor thesis. The task is to extract a fixed set of fields from safety data sheets (SDS) and, whenever the document asserts a state other than silence, to cite the page, the verbatim span, and the bounding box that support that assertion. A claim that follows from the document through a rule (for example "the product is a mixture, therefore it has no product-level CAS number") is a *derived* claim: it cites its premise spans and the rule, never the premise as if it were direct evidence.

## 1. Corpus

| Resource | Coverage |
|---|---|
| Source PDFs (`documents/`) | 30 sheets |
| Tesseract OCR (`data/ocr/`) | 30 sheets |
| Manifest (`data/manifest.csv`) | 30 sheets: split, supplier, template family, regime |
| Gold annotations, split `dev` | sheets 1–20 |
| Gold annotations, split `locked` | sheets 21–29 (sheet 30 is image-only and excluded until an OCR-layer annotation path exists) |

The `locked` split is refused by `scripts.evaluate` unless `--allow-locked` is given. The rules (`rules-v3`) were frozen before sheets 21–29 were annotated and have been run on the locked split exactly once.

## 2. Field ontology

Product-level fields:

- product name
- supplier name
- substance or mixture
- product CAS number
- chemical formula
- molecular weight
- flash point

Section 3 composition is annotated separately as an ingredient list (name, CAS or CAS state, concentration, row evidence).

Product CAS, formula and molecular weight are defined only for a substance. On a mixture or an article they are `NOT_APPLICABLE` *by derivation*: the gold records the spans that establish the product form as premises and names the rule `identifier_undefined_for_form`. Sheets that never declare the product form are left `UNDETERMINED`; the conditional identifier fields then remain `NOT_STATED`.

Each field instance has one of four states:

| State | Meaning | Evidence required |
|---|---|---|
| `PRESENT` | The sheet asserts a usable value. | Yes |
| `NOT_APPLICABLE` | The sheet, or a rule applied to the sheet, rules the field out. | Yes (direct evidence or premises + rule) |
| `NO_DATA` | The sheet states that no value is available. | Yes |
| `NOT_STATED` | The sheet is silent. | No |

Derivation kinds in the gold (29 sheets): `EXPLICIT` (134 instances), `ONTOLOGY` (44, all `NOT_APPLICABLE`, rule `identifier_undefined_for_form`), `COMPOSITION_IMPLIED` (4, product form read off the composition table, rule `composition_implies_form`), `ABSENT` (21, `NOT_STATED`). The rules are registered in `sdsbench/derivation.py` with their regulatory sources.

## 3. Annotations

Hand-written gold is stored in `data/annotations/source/sds_01.json`–`sds_29.json`. Each citation is a page number and a verbatim text span. Bounding boxes are not drawn by hand: `python -m scripts.build_annotations` resolves every span against the native PDF text layer and writes character offsets and boxes to `data/annotations/gold.json`. A span that does not occur in the document is rejected. Direct evidence resolves into `resolved_evidence`; the premises of a derived field resolve into `resolved_premises` (for the identifier rule, every span that establishes the product form is an accepted premise).

Resolved gold contains 203 product-field instances (140 dev + 63 locked):

| State | dev | locked |
|---|---|---|
| `PRESENT` | 72 | 39 |
| `NOT_APPLICABLE` | 46 (14 explicit, 32 derived) | 16 (4 explicit, 12 derived) |
| `NO_DATA` | 7 | 2 |
| `NOT_STATED` | 15 | 6 |

252 resolved spans, of which 86 are premises. The gold premises of three derived fields (all in `sds_20`) cannot be verified by the rule registry; that sheet is annotated as ambiguous. Sheets 21–29 were annotated on 2026-09-13 from the native text layer, without running the extractor on them first, and still need a second blind reading.

## 4. Evaluation

`sdsbench/evaluator.py` scores seven axes independently and reports their conjunction:

| Axis | Definition |
|---|---|
| state | predicted state equals the gold state |
| value | typed comparison against the gold value or an accepted alternative |
| page | a cited span is on a gold page |
| span | exact token overlap with a gold span: coverage ≥ 0.8 and IoU ≥ 0.5 |
| bbox | the cited box covers ≥ 0.8 of the gold box and has IoU ≥ 0.5 with it |
| derivation | the claim is direct or derived as the gold says, with the same rule |
| support | gold-free: the cited spans are verbatim on their pages and back the claim (value in span; state cue in the value slot; form declaration for a categorical claim; rule check on the premises for a derived claim) |

Verbatim means an exact substring after a documented normalization (case, whitespace, dash and quote folding, ligatures, ordinal º); the normalization is listed in `sdsbench/textlayer.py`. Nothing in the evaluator is fuzzy; fuzzy scores are written to the results as diagnostics only. An axis that cannot be decided (`undecidable`) counts as a failure. Localization axes are undefined for gold `NOT_STATED`; derivation and support are undefined when the prediction claims nothing. The headline is reported twice: over all instances and over the instances that have gold evidence or premises to locate.

Ingredients are scored as records: rows are matched one-to-one by CAS number, then by normalized name; a matched record is complete when name, CAS (or CAS state) and concentration all agree, concentrations being compared as parsed intervals; row evidence counts when the cited row is verbatim, covers the gold row box and is at most 2.5× its height. Gold rows without a CAS number count towards recall. The ingredient block state (`PRESENT` / `NOT_APPLICABLE`) is scored per document.

Regression tests: `python -m unittest discover -s tests` (51 tests). They cover verbatim and box matching, derived claims, ingredient records, the locked-split guard, the diagnostics probes, and the LLM plumbing (including the Ollama request shape).

## 5. Oracle

`python -m scripts.run_oracle` measures whether a gold value can be recovered and localized from the value alone: no gold page, no gold span is used for the value columns. Gold-span recoverability is reported separately. Results are in `data/results/oracle.csv` (dev) and `data/results/locked/oracle-locked.csv`.

| Check (search by the value alone), dev | Native | OCR |
|---|---|---|
| `PRESENT` product values occurring exactly | 70/72 | 67/72 |
| same values located to a box, any page | 70/72 | 67/72 |
| of which at least one hit on a gold page | 67/72 | 65/72 |
| values recoverable by typed containment only | 63/72 | 60/72 |
| ingredient CAS numbers located | 53/53 | 47/53 |

| Gold spans occurring exactly, dev | Native | OCR |
|---|---|---|
| direct evidence (142 spans) | 142 | 119 (129 with fuzzy ≥ 90) |
| premises of derived fields (36 fields) | 36 | 31 |
| ingredient rows (53) | 53 | 42 |

The two native misses are `sds_02` / `substance_or_mixture` (gold `MIXTURE`, the sheet says "Preparation") and `sds_04` / `chemical_formula` (`F3CCH2F` is printed as `F3CCH2 F`; typed containment still finds it). On the locked split every value and every CAS number is locatable in the native layer (39/39, 19/19); in OCR the multi-line ingredient rows of the locked sheets are reproduced exactly in only 2 of 19 cases.

## 6. Rules baseline

The extractor is rule-based (`sdsbench/extractors/rules.py`, version `rules-v3`). It is run in a 2 × 2 grid: text layer (native PDF vs Tesseract) and reading order (visual rows vs linear). Predictions are in `data/predictions/`; per-instance and per-record scores are in `data/results/`. The label lists and form phrases were written on the development sheets.

Cumulative columns add constraints from left to right. The last two columns are the strict joint over all instances and over the evidence-bearing instances.

Development split (140 instances, 125 evidence-bearing):

| Configuration | State | + value | + page | + span | + bbox | + derivation | Joint (all) | Joint (evidence-bearing) |
|---|---|---|---|---|---|---|---|---|
| Native, rows | 87.9% | 84.3% | 79.3% | 65.7% | 62.9% | 60.7% | 59.3% | 55.2% |
| Native, linear | 79.3% | 60.7% | 55.7% | 36.4% | 35.7% | 33.6% | 33.6% | 26.4% |
| OCR, rows | 84.3% | 80.0% | 75.7% | 62.1% | 37.1% | 36.4% | 36.4% | 29.6% |
| OCR, linear | 74.3% | 57.1% | 52.1% | 35.7% | 23.6% | 22.9% | 22.9% | 14.4% |

Locked split (63 instances, 57 evidence-bearing; single run of `rules-v3`, 2026-09-13):

| Configuration | State | + value | + page | + span | + bbox | + derivation | Joint (all) | Joint (evidence-bearing) |
|---|---|---|---|---|---|---|---|---|
| Native, rows | 84.1% | 73.0% | 73.0% | 68.3% | 66.7% | 66.7% | 57.1% | 54.4% |
| Native, linear | 76.2% | 49.2% | 49.2% | 34.9% | 34.9% | 34.9% | 33.3% | 28.1% |
| OCR, rows | 84.1% | 74.6% | 74.6% | 68.3% | 31.7% | 31.7% | 27.0% | 19.3% |
| OCR, linear | 68.3% | 49.2% | 49.2% | 36.5% | 9.5% | 9.5% | 9.5% | 0.0% |

The headline barely moves between splits, but the failures differ: on the locked sheets the supplier field is 0/9 (the DR-software templates print the label as one token, "Manufacturer/Supplier:", which the label list does not contain), 7 of 51 citations are not verbatim (`rules-v3` drops a standalone ":" token between label and value, which the dev templates rarely have), and on `sds_28` the composition rule reads a stabilised substance as a mixture. By template family (native + rows): dr-software 71.4%, subsystems / wercs-two-column / schuelke / petrofer 57.1% each, sigma-aldrich 0%.

Native + rows on dev, per axis: support 101/111 satisfied with 5 undecidable; derivation 102/110. Per field, joint (all instances): product name 60%, supplier 70%, product form 60%, product CAS 35%, formula 65%, molecular weight 55%, flash point 70%. Ingredient records: 63 gold rows, 39 predicted, 39 matched by CAS; CAS correct in 100% of matches, concentration in 59%, name in 5%; 2 complete records. On the locked split: 20 gold rows, 19 matched, name 16%, concentration 58%, row evidence located 21%, 2 complete records.

Under the previous evaluator the dev configurations scored 65.7 / 38.6 / 40.0 / 25.7% joint. The difference is the corrected scoring: exact verbatim and span matching, two-dimensional box dilution, decisive support, and the derivation axis.

## 7. Diagnostics

`python -m scripts.run_diagnostics --layer native|ocr [--split locked --allow-locked]` runs three probes on the rules system and writes `data/results/diagnostics-<layer>.csv`. Each probe gives the system one part of the answer and scores the rest with the standard evaluator.

| Probe | Given | Measured | dev native | dev OCR | locked native |
|---|---|---|---|---|---|
| gold-value | the value / state / form | evidence located (page + span + bbox) | 45.6% | 22.4% | 49.1% |
| gold-evidence | the gold span | state and value read off it | 92.8% | 93.0% (of 115 spans present in OCR) | 84.2% |
| gold-verify | the gold pair | accepted by the gold-free verifier | 95.2% | 96.5% | 100% |
| full pipeline | nothing | joint | 55.2% | 29.6% | 54.4% |

Reading a value off the right span and verifying a pair both work; the pipeline loses its points finding the right span (candidate extraction and localization), and the OCR layer loses more on text representation.

## 8. LLM baseline

`sdsbench/extractors/llm.py` and `python -m scripts.run_llm` implement a no-fine-tuning diagnostic baseline: the model receives the page-tagged text of one sheet (native or OCR layer, rows or linear order), the ontology and the derivation rules, and must return state, value, page and a verbatim quote per field, premises and a rule id for derived claims, and the ingredient rows. Quotes are resolved with the evaluator's exact verbatim function; a quote that does not occur is kept without a box so the evaluator fails it.

Default backend: local Ollama, model `qwen2.5:7b` (Qwen2.5-7B-Instruct, Ollama's Q4_K_M tag), JSON-schema output, `num_ctx` 24576, `temperature` 0, `seed` 0. Eight-gigabyte laptop GPUs fit this quantization plus the 24k context; Q8 does not. Names starting with `claude-` still go through the official Anthropic SDK and `ANTHROPIC_API_KEY`. Responses are cached under `data/llm_cache/` keyed by model and prompt; a second run with the cache present does not call the server and reproduces the same predictions. Tokens, latency and (for API models) estimated cost are written next to the predictions in `*.meta.json`.

First run (2026-09-13, `native` + `rows`, development split, 20 sheets, 0 errors, no API cost). Values are often right; localization is not. Joint is 6.4% against 59.3% for `rules-v3` on the same configuration. Product-field quotes are verbatim in 22 of 39 citations. Ingredient *values* are stronger than the rules (32 complete records vs 2; among 49 matched rows, name 85.7%, CAS 83.7%, concentration 93.9%), but row evidence is located in only 20.4% of matches and the ingredient-block state agrees on 3 of 20 documents.

| Configuration | State | + value | + page | + span | + bbox | + derivation | Joint (all) | Joint (evidence-bearing) |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B, native, rows | 62.1% | 57.1% | 15.0% | 10.7% | 8.6% | 6.4% | 6.4% | 2.4% |

Per field, joint (all instances): product name 0%, supplier 0%, product form 0%, product CAS 10%, formula 15%, molecular weight 10%, flash point 10%. The model reads names and suppliers (state 100%) and then fails the citation axes.

`python -m scripts.run_llm --dry-run DIR` writes the prompts and the output schema without calling the model. A vision (page-image) variant is not implemented. OCR, linear order and the locked split have not been run.

## 9. Related work

The project sits next to two 2026 selective-risk-control papers (SCRC, arXiv 2512.12844; SCoRE, arXiv 2603.24704): both are general frameworks for selective prediction with finite-sample risk control, not document-extraction work. The calibration layer is taken from them; the room here is the risk definition, the evidence and derivation structure, the score design and the template-shift setting.

## 10. Repository layout

```
documents/                  source SDS PDFs (30)
data/manifest.csv           split, supplier, template family, regime per PDF
data/annotations/source/    hand-written gold, sheets 1–29 (with rule ids on derived fields)
data/annotations/gold.json  resolved gold (direct spans, premises, boxes)
data/ocr/                   Tesseract output, all 30 sheets
data/predictions/           rules-v3 and llm-qwen2.5-7b output for dev; locked rules in *-locked.json
data/llm_cache/             raw local-model responses (re-runs skip the server)
data/results/               evaluator, ingredient-record, oracle and diagnostics tables (dev)
data/results/locked/        the same for the locked split
sdsbench/                   ontology, derivation rules, text layer, matching, gold, evaluator, oracle, manifest, diagnostics, rules, llm
scripts/                    build_annotations, run_rules, evaluate, run_oracle, run_diagnostics, run_llm
ocr.py                      regenerate Tesseract JSON
tests/                      regression tests
```

## 11. Reproduction

The numbers in this file come from the committed prediction and result files. Re-scoring those files does not need a GPU, Ollama, Tesseract or an API key. Regenerating them does.

**Python.** 3.12 and the pins in `requirements.txt`:

```
python -m venv .venv
# Windows: .venv\Scripts\activate    Unix: source .venv/bin/activate
pip install -r requirements.txt
```

Gold bounding boxes in `data/annotations/gold.json` are derived from PyMuPDF's word segmentation; do not bump `pymupdf` without rebuilding and diffing gold.

**Tesseract** is needed only to regenerate `data/ocr/`. The binary path is set at the top of `ocr.py`.

**Ollama** is needed only to regenerate the LLM baseline (or to run it on a machine that does not have `data/llm_cache/`). Install the app, start the server on `http://127.0.0.1:11434`, and pull the tag used for the committed run:

```
# Windows: winget install --id Ollama.Ollama -e
# otherwise: https://ollama.com/download
ollama pull qwen2.5:7b
```

The table in §8 was produced on an RTX 3070 Ti Laptop (8 GB) with 32 GB RAM. `qwen2.5:7b` is Q4_K_M (~4.7 GB weights); with `num_ctx` 24576 the resident set is about 6 GB. A 16 GB card can use a Q8 tag (`--model qwen2.5:7b-instruct-q8_0`) but that is a different run and must not overwrite the committed `qwen2.5:7b` files. `claude-*` models additionally need `ANTHROPIC_API_KEY`.

**Re-score committed outputs** (no extractors):

```
python -m scripts.evaluate data/predictions/rules-native-rows.json --by-field --ingredients --by-template
python -m scripts.evaluate data/predictions/llm-qwen2.5-7b-native-rows.json --by-field --ingredients --by-template
python -m unittest discover -s tests
```

**Regenerate** (writes the same paths the tables were scored from):

```
python -m scripts.build_annotations --report
python -m scripts.run_rules
python -m scripts.run_oracle
python -m scripts.run_diagnostics --layer native
python -m scripts.run_llm --layer native --layout rows
python -m scripts.evaluate data/predictions/rules-native-rows.json --by-field --ingredients --by-template
python -m scripts.evaluate data/predictions/llm-qwen2.5-7b-native-rows.json --by-field --ingredients --by-template
python -m unittest discover -s tests
```

`run_rules` without flags writes all four native/OCR × rows/linear prediction files for the `dev` split. `run_llm` without flags is one configuration (`qwen2.5:7b`, native, rows, `num_ctx` 24576). If `data/llm_cache/` is present the LLM command is a cache replay and bit-identical to the committed predictions; `--no-cache` calls Ollama again. Inspect prompts without a server: `python -m scripts.run_llm --dry-run prompts/`.

Runs against the locked split need `--split locked --allow-locked`. `rules-v3` has used its one run; the LLM baseline has not.

## 12. Remaining work

The local LLM baseline has one development run (`native` + `rows`). It has not been run on the locked split, on OCR, or as a page-image VLM. Sheets 21–29 need a second blind reading; sheet 30 needs an OCR-layer annotation path. Calibration and a rejection threshold are not being retuned; earlier calibration figures are not valid under the present evaluator.
