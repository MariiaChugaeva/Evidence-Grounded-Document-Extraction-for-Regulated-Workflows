# BM25 Baseline Results (dev split only)

## Queries used

- **product_name**: `product name Trade name / product identifier.`
- **supplier_name**: `supplier name Supplier, manufacturer or distributor named on the SDS.`
- **substance_or_mixture**: `substance or mixture Whether the product is a substance, mixture or article.`
- **product_cas_number**: `product cas number CAS of the product itself. Defined only for a substance.`
- **chemical_formula**: `chemical formula Molecular formula of the product. Defined only for a substance.`
- **molecular_weight**: `molecular weight Molecular weight in g/mol. Defined only for a substance.`
- **flash_point**: `flash point Flash point, compared in degrees Celsius.`

## Inclusion / exclusion statistics

- Total dev instances: 160
- Excluded (ingredients): 20
- Excluded (empty evidence_set): 15
- Excluded (NOT_STATED / no evidence required): 0
- **Included in metrics**: 125

## Sanity checks

- A1: `sds_01.pdf` / `product_name` page 1 rank in native: **2**

## Native layer

### Native

| Method | Metric | @1 | @2 | @3 | @5 | @10 | @500tok | @1000tok | @2000tok |
|---|---|---|---|---|---|---|---|---|---|
| BM25 | macro complete | 0.352 | 0.544 | 0.600 | 0.720 | 0.896 | 0.376 | 0.576 | 0.776 |
|  | macro piece_recall | 0.372 | 0.552 | 0.612 | 0.740 | 0.912 | 0.392 | 0.576 | 0.796 |
|  | micro piece_recall | 0.329 | 0.497 | 0.561 | 0.699 | 0.902 | 0.347 | 0.520 | 0.757 |
| Naive order | macro complete | 0.464 | 0.600 | 0.696 | 0.928 | 0.992 | 0.496 | 0.728 | 0.944 |
|  | macro piece_recall | 0.492 | 0.620 | 0.736 | 0.952 | 0.996 | 0.520 | 0.756 | 0.948 |
|  | micro piece_recall | 0.474 | 0.624 | 0.740 | 0.942 | 0.994 | 0.520 | 0.763 | 0.948 |

### Slice breakdown (BM25 macro complete)

| Slice | n | @1 | @2 | @3 | @5 | @10 | @500tok | @1000tok | @2000tok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_evidence | 77 | 0.468 | 0.675 | 0.727 | 0.831 | 0.935 | 0.494 | 0.701 | 0.883 |
| multi_evidence | 48 | 0.167 | 0.333 | 0.396 | 0.542 | 0.833 | 0.188 | 0.375 | 0.604 |
| cross_page_true | 19 | 0.000 | 0.263 | 0.316 | 0.421 | 0.737 | 0.053 | 0.316 | 0.421 |
| cross_page_false | 106 | 0.415 | 0.594 | 0.651 | 0.774 | 0.925 | 0.434 | 0.623 | 0.840 |
| explicit | 89 | 0.449 | 0.685 | 0.753 | 0.843 | 0.944 | 0.483 | 0.730 | 0.865 |
| derived | 36 | 0.111 | 0.194 | 0.222 | 0.417 | 0.778 | 0.111 | 0.194 | 0.556 |

## OCR layer

### OCR

| Method | Metric | @1 | @2 | @3 | @5 | @10 | @500tok | @1000tok | @2000tok |
|---|---|---|---|---|---|---|---|---|---|
| BM25 | macro complete | 0.360 | 0.520 | 0.592 | 0.712 | 0.904 | 0.368 | 0.576 | 0.768 |
|  | macro piece_recall | 0.380 | 0.528 | 0.604 | 0.732 | 0.916 | 0.384 | 0.588 | 0.788 |
|  | micro piece_recall | 0.335 | 0.480 | 0.572 | 0.699 | 0.908 | 0.341 | 0.549 | 0.757 |
| Naive order | macro complete | 0.464 | 0.600 | 0.696 | 0.928 | 0.992 | 0.496 | 0.728 | 0.920 |
|  | macro piece_recall | 0.492 | 0.620 | 0.736 | 0.952 | 0.996 | 0.520 | 0.740 | 0.944 |
|  | micro piece_recall | 0.474 | 0.624 | 0.740 | 0.942 | 0.994 | 0.520 | 0.740 | 0.931 |

### Slice breakdown (BM25 macro complete)

| Slice | n | @1 | @2 | @3 | @5 | @10 | @500tok | @1000tok | @2000tok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_evidence | 77 | 0.481 | 0.636 | 0.675 | 0.805 | 0.935 | 0.481 | 0.675 | 0.857 |
| multi_evidence | 48 | 0.167 | 0.333 | 0.458 | 0.562 | 0.854 | 0.188 | 0.417 | 0.625 |
| cross_page_true | 19 | 0.000 | 0.263 | 0.316 | 0.421 | 0.789 | 0.053 | 0.316 | 0.421 |
| cross_page_false | 106 | 0.425 | 0.566 | 0.642 | 0.764 | 0.925 | 0.425 | 0.623 | 0.830 |
| explicit | 89 | 0.461 | 0.663 | 0.730 | 0.831 | 0.955 | 0.472 | 0.730 | 0.854 |
| derived | 36 | 0.111 | 0.167 | 0.250 | 0.417 | 0.778 | 0.111 | 0.194 | 0.556 |

## Interpretation

BM25 achieves moderate piece-level recall but struggles with strict complete-evidence retrieval. Complete@3 is low because multi-evidence instances require all pages to be retrieved, and BM25 ranks pages independently without awareness of coverage. Cross-page instances are harder than single-page ones. The naive page-order baseline is surprisingly competitive on some metrics, suggesting that evidence often appears early in SDS documents. Native and OCR layers perform similarly, with OCR slightly lagging on fine-grained matches.
