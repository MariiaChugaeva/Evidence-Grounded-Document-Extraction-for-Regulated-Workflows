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

The `locked` split is refused by `scripts.evaluate` unless `--allow-locked` is given, and every run against it is appended to `docs/locked-runs.csv`. `rules-v3` was frozen before sheets 21–29 were annotated and ran on them exactly once (2026-09-13).

That run has since been read in detail, and one of its failures — `sds_28`, a stabilised substance that the composition rule called a mixture — is what `rules-v4` corrects. Sheets 21–29 are therefore **no longer a test set**: they are an audit split, usable for error analysis and regression but not for choosing a method. `rules-v4` has not been run on them. The next corpus round and the new holdout are specified in `docs/sampling-plan.md` and `docs/split-protocol.md`.

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

A rule in the registry does not accept a plausible-looking span; it names the **complete premise set** it needs, as a multiset of premise roles, and a derived claim is licensed only when every role is filled:

| Rule | Conclusion | Required premise roles |
|---|---|---|
| `identifier_undefined_for_form` | product CAS / formula / molecular weight are `NOT_APPLICABLE` | 1 × `form_declaration` |
| `composition_implies_form` → `SUBSTANCE` | product form | 1 × `sole_constituent` (one constituent at exactly 100 %) + 1 × `completeness_statement` (no other constituents, impurities or stabilisers) |
| `composition_implies_form` → `MIXTURE` | product form | 1 × `composition_header` + 2 × `partial_constituent` (distinct CAS numbers, each with an upper bound strictly below 100 %) |

This replaces two heuristics that `rules-v3` used and that are not sound under REACH Art. 3(1): "one ingredient ≥ 90 % implies a substance" and "more than one composition row implies a mixture". A substance may legally contain stabilisers and impurities, so a large share and a row count carry no conclusion; `sds_28` (diethyl ether at `≥ 90 – ≤ 100` with 200 ppm BHT as stabiliser) is exactly that case. A `composition_implies_form` claim whose premises contain an explicit form declaration is also rejected: a declared form is direct evidence and must be cited as such, not laundered through a derivation.

## 3. Annotations

Hand-written gold is stored in `data/annotations/source/sds_01.json`–`sds_29.json`. Each citation is a page number and a verbatim text span. Bounding boxes are not drawn by hand: `python -m scripts.build_annotations` resolves every span against the native PDF text layer and writes character offsets and boxes to `data/annotations/gold.json`. A span that does not occur in the document is rejected. Direct evidence resolves into `resolved_evidence`; the premises of a derived field resolve into `resolved_premises` (for the identifier rule, every span that establishes the product form is an accepted premise).

Resolved gold contains 203 product-field instances (140 dev + 63 locked):

| State | dev | locked |
|---|---|---|
| `PRESENT` | 72 | 39 |
| `NOT_APPLICABLE` | 46 (14 explicit, 32 derived) | 16 (4 explicit, 12 derived) |
| `NO_DATA` | 7 | 2 |
| `NOT_STATED` | 15 | 6 |

252 resolved spans, of which 86 are premises. `python -m scripts.build_annotations --report` reads the premise roles off the gold: 42 derived fields have a single-role premise set (a form declaration), one (`sds_04`) has the two-role substance set, and 5 have premises the registry cannot group into a complete set — the `composition_implies_form` fields of `sds_09`, `sds_18` and `sds_20`, and the two identifier fields of `sds_20`. Those five are scored leniently (one group, locate any premise) and are listed on every report run as due a second reading: either the annotation is missing a premise or the rule does not in fact cover the sheet. The gold-verify probe (§7) says which, per field:

| Field | What the registry says about the gold premises | Reading to make |
|---|---|---|
| `sds_09` / form | the premises declare the form ("...for the mixtures", section 9) | if that counts as a declaration, the derivation is `EXPLICIT`, not `COMPOSITION_IMPLIED`; if it is only an incidental use of the word, the premises should be the composition rows |
| `sds_18` / form | same, "35% of the mixture consists of..." on the composition page | same choice |
| `sds_20` / form | incomplete `MIXTURE` set: no composition header row | the sheet may simply not license the claim; it is already marked `ambiguous` |
| `sds_20` / CAS, molecular weight | the premises establish no form at all | depends on the form decision above |

These are not scoring bugs to be silenced. Each is a place where the gold asserts a derivation that the written rule does not support, which is the thing a rule registry is for. Sheets 21–29 were annotated on 2026-09-13 from the native text layer, without running the extractor on them first, and still need a second blind reading.

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

The three localization axes are scored over the **whole** target set, not over its best member. The gold targets of one instance are partitioned into role groups: alternative spans for the same role form one group with a minimum of one hit, while a rule that needs several roles produces several groups, each of which must be hit. A prediction that cites the 100 % constituent row of `sds_04` but not the "no other constituents" statement fails the span axis even though one of its citations is a perfect match; `--by-field` output and the results CSV carry `target_groups` and `groups_located` per instance. Group membership comes from the rule registry, so the evaluator and the extractor answer to the same definition of a complete premise set.

`scripts.evaluate` also refuses to score a prediction file against the wrong gold. Each `PredictionSet` derives a `prediction_id` (16 hex characters over the system name, the configuration and the document list; computed, not stored, so committed files stay byte-identical). Before any scoring the script checks that the file declares the split being requested, that its document set is exactly the documents of that split, and, when `--expect-id` is given, that the id matches. The split and the id are printed in the header of every report.

Ingredients are scored as records: rows are matched one-to-one by CAS number, then by normalized name; a matched record is complete when name, CAS (or CAS state) and concentration all agree, concentrations being compared as parsed intervals; row evidence counts when the cited row is verbatim, covers the gold row box and is at most 2.5× its height. Gold rows without a CAS number count towards recall. The ingredient block state (`PRESENT` / `NOT_APPLICABLE`) is scored per document.

Regression tests: `python -m unittest discover -s tests` (68 tests). They cover verbatim and box matching, derived claims and complete premise sets (including the stabilised-substance case), premise-group localization, ingredient records, the split and prediction-id guard, the diagnostics probes, and the LLM plumbing (including the Ollama request shape and the raw-contract report).

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

The extractor is rule-based (`sdsbench/extractors/rules.py`, version `rules-v4`). It is run in a 2 × 2 grid: text layer (native PDF vs Tesseract) and reading order (visual rows vs linear). Predictions are in `data/predictions/`; per-instance and per-record scores are in `data/results/`. The label lists and form phrases were written on the development sheets.

The product form is read in two steps. A declared form wins: a row that both contains a form phrase and classifies unambiguously as one form is cited as direct evidence. `rules-v4` confirms every phrase hit with `ontology.classify_form_cue`, so the boilerplate heading "3.1 Substances or 3.2 Mixtures" no longer produces a citation. The Annex II section headings `3.1. Substances` / `3.1 Substances` count as a declaration only when the composition section actually lists constituents, which is what separates `sds_16` from the empty heading in `sds_19`. Failing that, the form is derived, and only when exactly one candidate exists and the extractor's own premises pass `derivation.check_rule` — the same check the support axis applies. The identifier fields work the same way: `NOT_APPLICABLE` by `identifier_undefined_for_form` is emitted only if the form premises satisfy the rule. If neither path licenses a claim the form stays `UNDETERMINED` and the identifiers stay `NOT_STATED`, which is a refusal, not a guess.

Cumulative columns add constraints from left to right. The last two columns are the strict joint over all instances and over the evidence-bearing instances.

Development split (140 instances, 125 evidence-bearing), `rules-v4`:

| Configuration | State | + value | + page | + span | + bbox | + derivation | Joint (all) | Joint (evidence-bearing) |
|---|---|---|---|---|---|---|---|---|
| Native, rows | 85.0% | 81.4% | 79.3% | 66.4% | 63.6% | 62.1% | 60.7% | 56.8% |
| Native, linear | 76.4% | 57.9% | 55.7% | 37.1% | 36.4% | 35.0% | 35.0% | 28.0% |
| OCR, rows | 78.6% | 74.3% | 72.9% | 62.1% | 37.1% | 36.4% | 36.4% | 29.6% |
| OCR, linear | 68.6% | 51.4% | 49.3% | 35.7% | 23.6% | 22.9% | 22.9% | 14.4% |

Against `rules-v3` on the same split, state agreement falls (87.9 → 85.0% native + rows) and the joint rises (59.3 → 60.7% over all instances, 55.2 → 56.8% over the evidence-bearing ones). Six instances change, and they are the whole story:

- `sds_09`, four instances lost. The sheet's composition table has no header row that the layout exposes, so `rules-v4` cannot license `MIXTURE` and leaves the form `UNDETERMINED`; the three identifier fields lose their derived `NOT_APPLICABLE` — two of them read a value off the sheet instead, the third falls silent. `rules-v3` got the state right here by counting rows, but its derivation axis was violated and its support undecidable, so the instance never passed joint either way. The loss is four state points, not four correct answers.
- `sds_16`, one instance gained. `rules-v3` reached `SUBSTANCE` through the ≥ 90 % share and emitted it as a derivation, which the gold (an explicit Annex II 3.1 heading) scores as the wrong derivation. `rules-v4` cites the heading directly and the instance passes.
- `sds_18`, one instance gained. The form is now derived from the composition header plus two partial constituent rows, which is where the gold premises are, so span and bbox match.

No instance is `undecidable` on any axis any more (`rules-v3` had five). The form field's own joint rate goes from 60 to 70%. The sheets `rules-v4` refuses are honest gaps: their phrasings have no entry in the dev-written lexicon, and adding one on the strength of a held-out sheet would be tuning on the wrong split.

Locked split (63 instances, 57 evidence-bearing; single run of `rules-v3`, 2026-09-13; **not** re-run for `rules-v4`, see §1):

| Configuration | State | + value | + page | + span | + bbox | + derivation | Joint (all) | Joint (evidence-bearing) |
|---|---|---|---|---|---|---|---|---|
| Native, rows | 84.1% | 73.0% | 73.0% | 68.3% | 66.7% | 66.7% | 57.1% | 54.4% |
| Native, linear | 76.2% | 49.2% | 49.2% | 34.9% | 34.9% | 34.9% | 33.3% | 28.1% |
| OCR, rows | 84.1% | 74.6% | 74.6% | 68.3% | 31.7% | 31.7% | 27.0% | 19.3% |
| OCR, linear | 68.3% | 49.2% | 49.2% | 36.5% | 9.5% | 9.5% | 9.5% | 0.0% |

The headline barely moved between splits, but the failures differed: on the locked sheets the supplier field is 0/9 (the DR-software templates print the label as one token, "Manufacturer/Supplier:", which the label list does not contain), 7 of 51 citations are not verbatim (`rules-v3` drops a standalone ":" token between label and value, which the dev templates rarely have), and on `sds_28` the composition rule read a stabilised substance as a mixture. By template family (native + rows): dr-software 71.4%, subsystems / wercs-two-column / schuelke / petrofer 57.1% each, sigma-aldrich 0%. The `sds_28` error is the one `rules-v4` fixes, and it is why this table cannot be refreshed: see §1.

`rules-v4`, native + rows on dev, per axis: support 104/109 satisfied, derivation 100/108, nothing undecidable. Per field, joint (all instances): product name 60%, supplier 70%, product form 70%, product CAS 35%, formula 65%, molecular weight 55%, flash point 70%. Ingredient records are unchanged by `rules-v4` (nothing in the composition extractor moved): 63 gold rows, 39 predicted, 39 matched by CAS; CAS correct in 100% of matches, concentration in 59%, name in 5%; 2 complete records; the ingredient block state agrees on 14 of 20 documents. On the locked split under `rules-v3`: 20 gold rows, 19 matched, name 16%, concentration 58%, row evidence located 21%, 2 complete records.

Under the previous evaluator the dev configurations scored 65.7 / 38.6 / 40.0 / 25.7% joint. The difference is the corrected scoring: exact verbatim and span matching, two-dimensional box dilution, decisive support, and the derivation axis.

## 7. Diagnostics

`python -m scripts.run_diagnostics --layer native|ocr [--split locked --allow-locked]` runs three probes on the rules system and writes `data/results/diagnostics-<layer>.csv`. Each probe gives the system one part of the answer and scores the rest with the standard evaluator.

| Probe | Given | Measured | dev native | dev OCR | locked native |
|---|---|---|---|---|---|
| gold-value | the value / state / form | evidence located (page + span + bbox) | 45.6% | 22.4% | 49.1% |
| gold-evidence | the gold span | state and value read off it | 92.8% | 93.0% (of 115 spans present in OCR) | 84.2% |
| gold-verify | the gold pair | accepted by the gold-free verifier | 93.6% | 94.7% | 100% |
| full pipeline | nothing | joint | 56.8% | 29.6% | 54.4% |

The dev columns are `rules-v4`; the locked column is the single `rules-v3` run. Reading a value off the right span and verifying a pair both work; the pipeline loses its points finding the right span (candidate extraction and localization), and the OCR layer loses more on text representation. The verifier itself is now stricter: on the native layer it accepts 117 of 125 gold pairs against 119 before, because a derived pair whose premise set is incomplete is no longer accepted on one good premise. All eight failures are in the gold, and each comes with a reason in `verify_note` — three violated (`sds_09` and `sds_18` cite a form declaration under a composition rule, `sds_20` / supplier is a known loose span) and five undecidable (`sds_12` cites a span with no form cue, the four `sds_20` fields have premises that establish nothing). Running the verifier over the gold is now an annotation review tool, which is the use §3 puts it to.

## 8. LLM baseline

`sdsbench/extractors/llm.py` and `python -m scripts.run_llm` implement a no-fine-tuning diagnostic baseline: the model receives the page-tagged text of one sheet (native or OCR layer, rows or linear order), the ontology and the derivation rules, and must return state, value, page and a verbatim quote per field, premises and a rule id for derived claims, and the ingredient rows. Quotes are resolved with the evaluator's exact verbatim function; a quote that does not occur is kept without a box so the evaluator fails it.

Default backend: local Ollama, model `qwen2.5:7b` (Qwen2.5-7B-Instruct, Ollama's Q4_K_M tag), JSON-schema output, `num_ctx` 24576, `temperature` 0, `seed` 0. Eight-gigabyte laptop GPUs fit this quantization plus the 24k context; Q8 does not. Names starting with `claude-` still go through the official Anthropic SDK and `ANTHROPIC_API_KEY`. Responses are cached under `data/llm_cache/` keyed by model and prompt; a second run with the cache present does not call the server and reproduces the same predictions. Tokens, latency and (for API models) estimated cost are written next to the predictions in `*.meta.json`.

First run (2026-09-13, `native` + `rows`, development split, 20 sheets, 0 errors, no API cost). Values are often right; localization is not. Joint is 6.4% against 60.7% for `rules-v4` on the same configuration. Product-field quotes are verbatim in 22 of 39 citations. Ingredient *values* are stronger than the rules (32 complete records vs 2; among 49 matched rows, name 85.7%, CAS 83.7%, concentration 93.9%), but row evidence is located in only 20.4% of matches and the ingredient-block state agrees on 3 of 20 documents.

| Configuration | State | + value | + page | + span | + bbox | + derivation | Joint (all) | Joint (evidence-bearing) |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B, native, rows | 62.1% | 57.1% | 15.0% | 10.7% | 8.6% | 6.4% | 6.4% | 2.4% |

Per field, joint (all instances): product name 0%, supplier 0%, product form 0%, product CAS 10%, formula 15%, molecular weight 10%, flash point 10%. The model reads names and suppliers (state 100%) and then fails the citation axes.

### Raw output versus the repaired prediction

The scores above are computed on the *repaired* prediction: the converter drops a value carried by a non-`PRESENT` state, strips evidence from `NOT_STATED`, demotes a `DERIVED` claim that names no rule, and keeps an unresolvable quote without a box. Scoring only that output hides how often the model breaks the contract, so the breaches are now counted before the repair and reported next to it. `scripts.run_llm` writes `data/results/<system>_raw_contract.csv` (one row per breach: document, field, code, description, detail) and prints the totals; `*.meta.json` carries the counts and the prediction id. The repair itself is unchanged, so the committed prediction file is bit-identical to the 2026-09-13 run.

Qwen2.5-7B, native + rows, dev: **165 breaches over 20 of 20 documents**, against 140 field instances.

| Code | Claims | Meaning |
|---|---|---|
| `quote_not_verbatim` | 62 | the quote does not occur on the page it was cited from |
| `claim_without_citation` | 42 | a claim other than `NOT_STATED` cites nothing |
| `derived_premise_set_incomplete` | 24 | a `DERIVED` claim's premises do not complete the rule's premise set |
| `direct_with_rule_or_premises` | 20 | a `DIRECT` claim names a rule or cites premises |
| `not_stated_with_citation` | 9 | `NOT_STATED` carries evidence or premises |
| `state_carries_value` | 8 | a state other than `PRESENT` carries a value |

The distribution matters more than the total. Roughly half the breaches are localization (`quote_not_verbatim`, `claim_without_citation`) and are already visible in the page/span/bbox axes. The other half are *structural*: the model does not keep direct and derived claims apart (20 + 24 + 9 + 8 = 61 breaches), and the 24 incomplete premise sets are the same failure the rules extractor is now forbidden to make. A constrained decoder or a schema cannot fix that, because the schema is already enforced by Ollama's JSON mode — every one of these 165 breaches is in output that validated against the schema.

Fourteen codes are defined in `sdsbench/extractors/llm.py` (`CONTRACT_CODES`); the eight that did not fire on this run include the document-level ones (`no_structured_output`, `schema_invalid`) and `derived_unknown_rule`.

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
data/predictions/           rules-v4 and llm-qwen2.5-7b output for dev; the rules-v3 locked run in *-locked.json
data/llm_cache/             raw local-model responses (re-runs skip the server)
data/results/               evaluator, ingredient-record, oracle, diagnostics and raw-contract tables (dev)
data/results/locked/        the same for the locked split
docs/locked-runs.csv        every evaluation run against a non-development split
docs/sampling-plan.md       strata and quotas for the next corpus round
docs/split-protocol.md      split roles, holdout rules, freeze order
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

Runs against a non-development split need `--split locked --allow-locked`, are checked against the prediction file's declared split, document set and (with `--expect-id`) prediction id, and are appended to `docs/locked-runs.csv`. `rules-v3` has used its one run on sheets 21–29; `rules-v4` will not use one, because that split has been retired to an audit role (§1, `docs/split-protocol.md`).

## 12. Remaining work

In order:

1. **Corpus.** Collect the 40 sheets of `docs/sampling-plan.md`, strata first. The binding constraint on every claim in this file is that one stabilised substance (`sds_28`) and four composition-derived forms are all the evidence the derivation rules have.
2. **Second reading of the gold.** Start with the five fields §3 lists, then sheets 21–29, then sheet 30 once an OCR-layer annotation path exists.
3. **New holdout.** Freeze a method version, then annotate and run once, per `docs/split-protocol.md`.
4. **LLM baseline.** One development run exists (`native` + `rows`); OCR, linear order and a page-image VLM variant do not. The structural half of the contract report (§8) is the interesting target: a prompt or a decoder that keeps direct and derived claims apart.
5. **Calibration.** Not being retuned; earlier calibration figures are not valid under the present evaluator.
