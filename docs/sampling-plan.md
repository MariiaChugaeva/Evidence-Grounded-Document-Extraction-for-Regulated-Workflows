# Sampling plan for the next corpus round

Written before collection, so that the corpus is not assembled out of whatever is easy to
find and then described afterwards. Nothing may be added to `documents/` until the sheet is
assigned to a stratum below and recorded in `data/manifest.csv`.

## 1. What the present 30 sheets do and do not cover

| Dimension | Present coverage | Gap |
|---|---|---|
| Template family | 19 families, 6 of them once only | families beyond the common generators (ChemGes, Chemeter, Sub Systems, CHEMDOX, WERCS) |
| Supplier | 30 suppliers, no repeats | — |
| Regime | REACH-EU 10, OSHA-HCS 9, REACH-UK 5, AU-WHS 2, WHMIS / GOST / JP-transport 1 each | WHMIS, GOST, and every Asian regime are single sheets or absent |
| Text layer | 28 native, 2 scanned (1 usable OCR, 1 image-only) | scans and image-only sheets |
| Product form | 11 mixtures, 3 substances, 1 article, 4 undetermined (dev) | substances, in particular stabilised ones |
| Composition shape | one substance with a completeness statement (`sds_04`), one stabilised substance (`sds_28`) | the whole class the form rule is most likely to misread |
| Withheld composition | none | trade-secret / proprietary claims |

The last three rows are the reason for this round. `sds_28` is a single sheet and it is the
one that broke the old form heuristic; a rule cannot be trusted on a class represented by
one example.

## 2. Strata and quotas

Target: 40 new sheets, taking the corpus to 70. A sheet may satisfy several strata, but each
quota must be met by sheets counted independently, so at least 40 sheets are needed and some
strata will need dedicated searching.

| # | Stratum | Definition | Quota |
|---|---|---|---|
| S1 | New template family | the generator or in-house layout does not appear in sheets 1–30 | 20 sheets over ≥ 14 new families |
| S2 | New supplier | the legal entity does not appear in sheets 1–30 | all 40 |
| S3 | Stabilised / impure substance | declared a substance while section 3 lists stabilisers, inhibitors, additives or impurities | 6 |
| S4 | Clean single-constituent substance | one constituent at 100 % plus a completeness statement | 4 |
| S5 | Open-ended composition | section 3 rows whose top bound touches 100 % (`≥ 90 – ≤ 100`, `> 95`, `balance`) | 6 |
| S6 | Withheld composition | trade secret, "proprietary", CAS withheld under a confidentiality claim | 4 |
| S7 | Difficult text layer | scanned, image-only, rotated, or a native layer under 4 kB | 6 |
| S8 | Hard layout | multi-sheet bundle, two-column label/value, non-GHS section numbering, transport-only, or over 25 pages | 8 |
| S9 | Other regulatory regime | WHMIS, GOST, AU-WHS, JP, KR, CN, IN, BR, or a non-English original with an English sheet | 8 |
| S10 | Article or non-SDS documentation | product documentation supplied in lieu of an SDS | 3 |

S3, S5 and S6 are the strata that decide whether the derivation rules hold; if a stratum
cannot be filled, the shortfall is recorded in this file rather than silently absorbed.

## 3. Exclusion criteria

- No sheet that was found by searching for a phrase the extractor already handles.
- No second sheet from the same supplier *and* template family unless it is deliberately a
  within-family pair (like `sds_16` / `sds_21`), and then it is marked as such.
- No sheet whose original language is not represented by an English or German text layer, as
  the field parsers are only defined for those.
- No sheet without a retrievable public source.

## 4. Provenance record

Every new sheet adds a row to `data/manifest.csv` and a provenance entry with: source URL,
retrieval date, SHA-256 of the PDF, the strata it fills, and the reason it was selected. A
sheet whose provenance is incomplete is not annotated.

## 5. Order of work

1. Write the search queries per stratum, then collect. Collection is blind to extractor
   behaviour: the extractor is not run on a candidate before the decision to include it.
2. Record provenance and strata; assign the split according to `docs/split-protocol.md`.
3. Annotate, two independent readings, disagreements logged and resolved against the
   regulation text rather than against extractor output.
4. Only then run anything on the new sheets.
