# Dense Retrieval Baseline Results (dev split only)

Model: `sentence-transformers/all-MiniLM-L6-v2`

## Comparison: Naive order vs BM25 vs Dense-MiniLM

## Native layer

### Native

| Method | Metric | @1 | @2 | @3 | @5 | @10 | @500tok | @1000tok | @2000tok |
|---|---|---|---|---|---|---|---|---|---|
| Naive order | macro complete | 0.464 | 0.600 | 0.696 | 0.928 | 0.992 | 0.496 | 0.728 | 0.944 |
|  | macro piece_recall | 0.492 | 0.620 | 0.736 | 0.952 | 0.996 | 0.520 | 0.756 | 0.948 |
| BM25 | macro complete | 0.352 | 0.544 | 0.600 | 0.720 | 0.896 | 0.376 | 0.576 | 0.776 |
|  | macro piece_recall | 0.372 | 0.552 | 0.612 | 0.740 | 0.912 | 0.392 | 0.576 | 0.796 |
| Dense-MiniLM | macro complete | 0.328 | 0.536 | 0.632 | 0.800 | 0.952 | 0.416 | 0.632 | 0.808 |
|  | macro piece_recall | 0.376 | 0.588 | 0.668 | 0.816 | 0.972 | 0.460 | 0.664 | 0.836 |

### Dense-MiniLM slice breakdown (native)

| Slice | n | @1 | @2 | @3 | @5 | @10 | @500tok | @1000tok | @2000tok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_evidence | 77 | 0.442 | 0.636 | 0.701 | 0.831 | 0.987 | 0.519 | 0.701 | 0.844 |
| multi_evidence | 48 | 0.146 | 0.375 | 0.521 | 0.750 | 0.896 | 0.250 | 0.521 | 0.750 |
| cross_page_true | 19 | 0.000 | 0.105 | 0.316 | 0.579 | 0.737 | 0.053 | 0.368 | 0.526 |
| cross_page_false | 106 | 0.387 | 0.613 | 0.689 | 0.840 | 0.991 | 0.481 | 0.679 | 0.858 |
| explicit | 89 | 0.393 | 0.584 | 0.663 | 0.820 | 0.955 | 0.483 | 0.674 | 0.831 |
| derived | 36 | 0.167 | 0.417 | 0.556 | 0.750 | 0.944 | 0.250 | 0.528 | 0.750 |

## OCR layer

### OCR

| Method | Metric | @1 | @2 | @3 | @5 | @10 | @500tok | @1000tok | @2000tok |
|---|---|---|---|---|---|---|---|---|---|
| Naive order | macro complete | 0.464 | 0.600 | 0.696 | 0.928 | 0.992 | 0.496 | 0.728 | 0.920 |
|  | macro piece_recall | 0.492 | 0.620 | 0.736 | 0.952 | 0.996 | 0.520 | 0.740 | 0.944 |
| BM25 | macro complete | 0.360 | 0.520 | 0.592 | 0.712 | 0.904 | 0.368 | 0.576 | 0.768 |
|  | macro piece_recall | 0.380 | 0.528 | 0.604 | 0.732 | 0.916 | 0.384 | 0.588 | 0.788 |
| Dense-MiniLM | macro complete | 0.344 | 0.520 | 0.688 | 0.808 | 0.952 | 0.432 | 0.664 | 0.824 |
|  | macro piece_recall | 0.396 | 0.584 | 0.744 | 0.840 | 0.968 | 0.488 | 0.724 | 0.864 |

### Dense-MiniLM slice breakdown (OCR)

| Slice | n | @1 | @2 | @3 | @5 | @10 | @500tok | @1000tok | @2000tok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_evidence | 77 | 0.442 | 0.649 | 0.779 | 0.857 | 0.987 | 0.532 | 0.766 | 0.883 |
| multi_evidence | 48 | 0.188 | 0.312 | 0.542 | 0.729 | 0.896 | 0.271 | 0.500 | 0.729 |
| cross_page_true | 19 | 0.000 | 0.053 | 0.211 | 0.526 | 0.789 | 0.000 | 0.158 | 0.474 |
| cross_page_false | 106 | 0.406 | 0.604 | 0.774 | 0.858 | 0.981 | 0.509 | 0.755 | 0.887 |
| explicit | 89 | 0.393 | 0.607 | 0.742 | 0.843 | 0.966 | 0.483 | 0.730 | 0.854 |
| derived | 36 | 0.222 | 0.306 | 0.556 | 0.722 | 0.917 | 0.306 | 0.500 | 0.750 |

## Interpretation

Dense-MiniLM improves over BM25 on most metrics, but the gap is modest. The largest gains are on single-evidence and explicit instances where semantic similarity better matches query intent. However, dense retrieval still fails on the hardest slices: multi-evidence (complete@3 ~0.4–0.5), cross-page (~0.3), and derived (~0.2–0.3). These failures are structural — no single-page scoring method can guarantee coverage of dispersed evidence. Naive order remains surprisingly strong, confirming that SDS evidence tends to appear early. Native and OCR layers show nearly identical dense performance, suggesting MiniLM is robust to OCR noise at this granularity.
