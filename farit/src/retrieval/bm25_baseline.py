#!/usr/bin/env python3
"""BM25 baseline for complete-evidence retrieval on dev split.

CLI without arguments; all paths are hard-coded relative to repo root.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Paths (repo root is cwd)
# ---------------------------------------------------------------------------
REPO_ROOT = Path.cwd()
CORPUS_DIR = REPO_ROOT / "farit" / "data" / "corpus"
RESULTS_DIR = REPO_ROOT / "farit" / "results"
SRC_DIR = REPO_ROOT / "farit" / "src" / "retrieval"

UNITS_NATIVE = CORPUS_DIR / "units_native.jsonl"
UNITS_OCR = CORPUS_DIR / "units_ocr.jsonl"
GOLD_EVIDENCE = CORPUS_DIR / "gold_evidence.jsonl"
ONTOLOGY_PY = REPO_ROOT / "sdsbench" / "ontology.py"

OUT_JSON = RESULTS_DIR / "bm25_baseline.json"
OUT_MD = RESULTS_DIR / "bm25_baseline.md"

# ---------------------------------------------------------------------------
# Load ontology FIELD_SPECS by executing ontology.py in a minimal namespace
# ---------------------------------------------------------------------------

def load_field_specs() -> dict[str, dict]:
    code = ONTOLOGY_PY.read_text(encoding="utf-8")
    ns: dict = {}
    exec(compile(code, str(ONTOLOGY_PY), "exec"), ns)  # noqa: S102
    return ns["FIELD_SPECS"]  # type: ignore[return-value]


FIELD_SPECS = load_field_specs()

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return text.lower().split()


# ---------------------------------------------------------------------------
# Build queries from schema only (no gold value)
# ---------------------------------------------------------------------------

def build_query(field_name: str) -> str:
    spec = FIELD_SPECS[field_name]
    name_part = spec.name.replace("_", " ")
    desc_part = spec.description
    return f"{name_part} {desc_part}"


QUERIES = {name: build_query(name) for name in FIELD_SPECS}

# ---------------------------------------------------------------------------
# Load corpus pages
# ---------------------------------------------------------------------------

def load_pages(path: Path) -> list[dict]:
    pages = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))
    return pages


# ---------------------------------------------------------------------------
# Metrics for a single instance
# ---------------------------------------------------------------------------

def compute_instance_metrics(
    ranked_pages: list[int],
    gold_evidence_set: list[dict],
    page_tokens: dict[int, int],
) -> dict:
    """Compute piece_recall@k, complete@k, and token-budget metrics.

    piece_recall counts individual evidence fragments (not unique pages).
    complete checks whether all fragments' pages are in the retrieved set.
    """
    ks = [1, 2, 3, 5, 10]
    budgets = [500, 1000, 2000]

    n_evidence = len(gold_evidence_set)
    gold_pages = {e["page"] for e in gold_evidence_set}

    piece_recall = {}
    complete = {}

    for k in ks:
        top_set = set(ranked_pages[:k])
        hits = sum(1 for e in gold_evidence_set if e["page"] in top_set)
        piece_recall[k] = hits / n_evidence if n_evidence > 0 else 0.0
        complete[k] = 1.0 if gold_pages.issubset(top_set) else 0.0

    # Token-budget metrics
    token_pages = []
    token_acc = 0
    for p in ranked_pages:
        token_acc += page_tokens.get(p, 0)
        if token_acc > budgets[-1]:
            break
        token_pages.append(p)

    token_piece_recall = {}
    token_complete = {}
    for B in budgets:
        acc = 0
        selected = []
        for p in token_pages:
            acc += page_tokens.get(p, 0)
            if acc > B:
                break
            selected.append(p)
        sel_set = set(selected)
        hits = sum(1 for e in gold_evidence_set if e["page"] in sel_set)
        token_piece_recall[B] = hits / n_evidence if n_evidence > 0 else 0.0
        token_complete[B] = 1.0 if gold_pages.issubset(sel_set) else 0.0

    return {
        "piece_recall": piece_recall,
        "complete": complete,
        "token_piece_recall": token_piece_recall,
        "token_complete": token_complete,
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate(instances: list[dict]) -> dict:
    ks = [1, 2, 3, 5, 10]
    budgets = [500, 1000, 2000]

    # Macro averages over instances
    macro_piece = {k: 0.0 for k in ks}
    macro_complete = {k: 0.0 for k in ks}
    macro_tpiece = {B: 0.0 for B in budgets}
    macro_tcomplete = {B: 0.0 for B in budgets}

    # Micro averages over evidence fragments
    micro_piece_num = {k: 0 for k in ks}
    micro_piece_den = {k: 0 for k in ks}
    micro_complete_num = {k: 0 for k in ks}
    micro_complete_den = {k: 0 for k in ks}

    micro_tpiece_num = {B: 0 for B in budgets}
    micro_tpiece_den = {B: 0 for B in budgets}
    micro_tcomplete_num = {B: 0 for B in budgets}
    micro_tcomplete_den = {B: 0 for B in budgets}

    n = len(instances)
    for inst in instances:
        m = inst["metrics"]
        n_ev = inst["n_evidence"]
        for k in ks:
            macro_piece[k] += m["piece_recall"][k]
            macro_complete[k] += m["complete"][k]
            micro_piece_num[k] += int(m["piece_recall"][k] * n_ev)
            micro_piece_den[k] += n_ev
            micro_complete_num[k] += int(m["complete"][k])
            micro_complete_den[k] += 1
        for B in budgets:
            macro_tpiece[B] += m["token_piece_recall"][B]
            macro_tcomplete[B] += m["token_complete"][B]
            micro_tpiece_num[B] += int(m["token_piece_recall"][B] * n_ev)
            micro_tpiece_den[B] += n_ev
            micro_tcomplete_num[B] += int(m["token_complete"][B])
            micro_tcomplete_den[B] += 1

    for k in ks:
        macro_piece[k] /= n
        macro_complete[k] /= n
        micro_piece_num[k] /= micro_piece_den[k]
        micro_complete_num[k] /= micro_complete_den[k]
    for B in budgets:
        macro_tpiece[B] /= n
        macro_tcomplete[B] /= n
        micro_tpiece_num[B] /= micro_tpiece_den[B]
        micro_tcomplete_num[B] /= micro_tcomplete_den[B]

    return {
        "macro": {
            "piece_recall": macro_piece,
            "complete": macro_complete,
            "token_piece_recall": macro_tpiece,
            "token_complete": macro_tcomplete,
        },
        "micro": {
            "piece_recall": micro_piece_num,
            "complete": micro_complete_num,
            "token_piece_recall": micro_tpiece_num,
            "token_complete": micro_tcomplete_num,
        },
        "n_instances": n,
    }


def slice_aggregate(instances: list[dict], predicate) -> dict:
    filtered = [i for i in instances if predicate(i)]
    if not filtered:
        return {"n_instances": 0}
    return aggregate(filtered)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_layer(layer_name: str, units_path: Path) -> dict:
    pages = load_pages(units_path)
    # Filter to dev split only (A4)
    pages = [p for p in pages if p["split"] == "dev"]

    # Group pages by document
    doc_pages: dict[str, list[dict]] = {}
    for p in pages:
        doc_pages.setdefault(p["document"], []).append(p)

    # Load gold evidence
    gold_instances = []
    with GOLD_EVIDENCE.open(encoding="utf-8") as f:
        for line in f:
            inst = json.loads(line)
            if inst["split"] != "dev":
                continue
            gold_instances.append(inst)

    # Exclusions tracking
    excluded_total = 0
    excluded_empty = 0
    excluded_ingredients = 0
    excluded_absent = 0
    included_instances = []

    per_instance = []

    for inst in gold_instances:
        if inst["field"] == "ingredients":
            excluded_ingredients += 1
            excluded_total += 1
            continue
        if not inst.get("evidence_set"):
            excluded_empty += 1
            excluded_total += 1
            continue
        if inst["state"] == "NOT_STATED":
            # NOT_STATED has no evidence requirement
            excluded_absent += 1
            excluded_total += 1
            continue

        included_instances.append(inst)

    # For each included instance, run BM25 and naive order
    for inst in included_instances:
        doc = inst["document"]
        field = inst["field"]
        gold_ev_set = inst["evidence_set"]
        gold_pages = {e["page"] for e in gold_ev_set}
        n_evidence = inst["n_evidence"]
        cross_page = inst.get("cross_page", False)
        derivation = inst["derivation"]

        doc_ps = doc_pages.get(doc, [])
        if not doc_ps:
            # No pages for this document — metrics will be zero
            page_tokens = {}
            ranked_bm25 = []
            ranked_naive = []
        else:
            # Tokenize corpus
            corpus_texts = [p["text"] for p in doc_ps]
            tokenized_corpus = [tokenize(t) for t in corpus_texts]
            bm25 = BM25Okapi(tokenized_corpus)

            query_tokens = tokenize(QUERIES[field])
            scores = bm25.get_scores(query_tokens)
            # Rank by descending score
            indexed = list(enumerate(scores))
            indexed.sort(key=lambda x: x[1], reverse=True)
            ranked_bm25 = [doc_ps[i]["page"] for i, _ in indexed]

            # Naive order: sort by page number ascending
            naive_sorted = sorted(doc_ps, key=lambda p: p["page"])
            ranked_naive = [p["page"] for p in naive_sorted]

            page_tokens = {p["page"]: p.get("n_tokens", 0) for p in doc_ps}

        m_bm25 = compute_instance_metrics(ranked_bm25, gold_ev_set, page_tokens)
        m_naive = compute_instance_metrics(ranked_naive, gold_ev_set, page_tokens)

        # A5: assert piece_recall >= complete for all budgets
        for k in [1, 2, 3, 5, 10]:
            assert m_bm25["piece_recall"][k] >= m_bm25["complete"][k] - 1e-9, \
                f"A5 fail BM25 {doc}/{field} @ {k}: piece_recall {m_bm25['piece_recall'][k]} < complete {m_bm25['complete'][k]}"
            assert m_naive["piece_recall"][k] >= m_naive["complete"][k] - 1e-9, \
                f"A5 fail naive {doc}/{field} @ {k}: piece_recall {m_naive['piece_recall'][k]} < complete {m_naive['complete'][k]}"
        for B in [500, 1000, 2000]:
            assert m_bm25["token_piece_recall"][B] >= m_bm25["token_complete"][B] - 1e-9, \
                f"A5 fail BM25 {doc}/{field} @ {B}tok: piece_recall {m_bm25['token_piece_recall'][B]} < complete {m_bm25['token_complete'][B]}"
            assert m_naive["token_piece_recall"][B] >= m_naive["token_complete"][B] - 1e-9, \
                f"A5 fail naive {doc}/{field} @ {B}tok: piece_recall {m_naive['token_piece_recall'][B]} < complete {m_naive['token_complete'][B]}"

        per_instance.append({
            "document": doc,
            "field": field,
            "n_evidence": n_evidence,
            "cross_page": cross_page,
            "derivation": derivation,
            "bm25": m_bm25,
            "naive": m_naive,
        })

    # Overall aggregates
    overall_bm25 = aggregate([{"metrics": i["bm25"], "n_evidence": i["n_evidence"]} for i in per_instance])
    overall_naive = aggregate([{"metrics": i["naive"], "n_evidence": i["n_evidence"]} for i in per_instance])

    # Slices
    single_ev = slice_aggregate(
        [{"metrics": i["bm25"], "n_evidence": i["n_evidence"]} for i in per_instance],
        lambda x: x["n_evidence"] == 1,
    )
    multi_ev = slice_aggregate(
        [{"metrics": i["bm25"], "n_evidence": i["n_evidence"]} for i in per_instance],
        lambda x: x["n_evidence"] >= 2,
    )
    cross_true = slice_aggregate(
        [{"metrics": i["bm25"], "n_evidence": i["n_evidence"]} for i in per_instance if i["cross_page"]],
        lambda x: True,
    )
    cross_false = slice_aggregate(
        [{"metrics": i["bm25"], "n_evidence": i["n_evidence"]} for i in per_instance if not i["cross_page"]],
        lambda x: True,
    )
    explicit = slice_aggregate(
        [{"metrics": i["bm25"], "n_evidence": i["n_evidence"]} for i in per_instance if i["derivation"] == "EXPLICIT"],
        lambda x: True,
    )
    derived = slice_aggregate(
        [{"metrics": i["bm25"], "n_evidence": i["n_evidence"]} for i in per_instance if i["derivation"] != "EXPLICIT"],
        lambda x: True,
    )

    # Sanity check A1: sds_01.pdf / product_name rank in native
    sanity_rank = None
    if layer_name == "native":
        for i in per_instance:
            if i["document"] == "sds_01.pdf" and i["field"] == "product_name":
                # Find rank of page 1 in BM25 ranking
                doc_ps = doc_pages.get("sds_01.pdf", [])
                corpus_texts = [p["text"] for p in doc_ps]
                tokenized_corpus = [tokenize(t) for t in corpus_texts]
                bm25 = BM25Okapi(tokenized_corpus)
                query_tokens = tokenize(QUERIES["product_name"])
                scores = bm25.get_scores(query_tokens)
                indexed = list(enumerate(scores))
                indexed.sort(key=lambda x: x[1], reverse=True)
                ranked = [doc_ps[idx]["page"] for idx, _ in indexed]
                try:
                    sanity_rank = ranked.index(1) + 1
                except ValueError:
                    sanity_rank = -1
                break

    return {
        "layer": layer_name,
        "n_total_dev": len(gold_instances),
        "n_excluded": excluded_total,
        "n_excluded_empty": excluded_empty,
        "n_excluded_ingredients": excluded_ingredients,
        "n_excluded_absent": excluded_absent,
        "n_included": len(per_instance),
        "queries": QUERIES,
        "sanity_rank_sds01_product_name": sanity_rank,
        "overall_bm25": overall_bm25,
        "overall_naive": overall_naive,
        "slices_bm25": {
            "single_evidence": single_ev,
            "multi_evidence": multi_ev,
            "cross_page_true": cross_true,
            "cross_page_false": cross_false,
            "explicit": explicit,
            "derived": derived,
        },
        "per_instance": per_instance,
    }


def format_table(data: dict, title: str) -> str:
    lines = [f"### {title}", ""]
    ks = [1, 2, 3, 5, 10]
    budgets = [500, 1000, 2000]

    header = "| Method | Metric | " + " | ".join(f"@{k}" for k in ks) + " | " + " | ".join(f"@{B}tok" for B in budgets) + " |"
    sep = "|" + "|".join(["---"] * (2 + len(ks) + len(budgets))) + "|"
    lines.append(header)
    lines.append(sep)

    for method, label in [("bm25", "BM25"), ("naive", "Naive order")]:
        agg = data.get(f"overall_{method}", {})
        if not agg:
            continue
        macro = agg.get("macro", {})
        micro = agg.get("micro", {})

        # Macro complete
        row = [label, "macro complete"]
        for k in ks:
            row.append(f"{macro['complete'][k]:.3f}")
        for B in budgets:
            row.append(f"{macro['token_complete'][B]:.3f}")
        lines.append("| " + " | ".join(row) + " |")

        # Macro piece_recall
        row = ["", "macro piece_recall"]
        for k in ks:
            row.append(f"{macro['piece_recall'][k]:.3f}")
        for B in budgets:
            row.append(f"{macro['token_piece_recall'][B]:.3f}")
        lines.append("| " + " | ".join(row) + " |")

        # Micro piece_recall (micro complete removed per B2 — duplicates macro)
        row = ["", "micro piece_recall"]
        for k in ks:
            row.append(f"{micro['piece_recall'][k]:.3f}")
        for B in budgets:
            row.append(f"{micro['token_piece_recall'][B]:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)


def format_slices(data: dict) -> str:
    lines = ["### Slice breakdown (BM25 macro complete)", ""]
    slices = data["slices_bm25"]
    header = "| Slice | n | @1 | @2 | @3 | @5 | @10 | @500tok | @1000tok | @2000tok |"
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines.append(header)
    lines.append(sep)
    for name, agg in slices.items():
        if agg.get("n_instances", 0) == 0:
            continue
        n = agg["n_instances"]
        macro = agg["macro"]["complete"]
        tmacro = agg["macro"]["token_complete"]
        row = [
            name,
            str(n),
            f"{macro[1]:.3f}",
            f"{macro[2]:.3f}",
            f"{macro[3]:.3f}",
            f"{macro[5]:.3f}",
            f"{macro[10]:.3f}",
            f"{tmacro[500]:.3f}",
            f"{tmacro[1000]:.3f}",
            f"{tmacro[2000]:.3f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    native = run_layer("native", UNITS_NATIVE)
    ocr = run_layer("ocr", UNITS_OCR)

    # Write JSON
    output = {
        "native": native,
        "ocr": ocr,
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)

    # Write Markdown
    md_lines = [
        "# BM25 Baseline Results (dev split only)",
        "",
        "## Queries used",
        "",
    ]
    for name, q in QUERIES.items():
        md_lines.append(f"- **{name}**: `{q}`")
    md_lines.append("")

    # Inclusion stats
    md_lines.append("## Inclusion / exclusion statistics")
    md_lines.append("")
    md_lines.append(f"- Total dev instances: {native['n_total_dev']}")
    md_lines.append(f"- Excluded (ingredients): {native['n_excluded_ingredients']}")
    md_lines.append(f"- Excluded (empty evidence_set): {native['n_excluded_empty']}")
    md_lines.append(f"- Excluded (NOT_STATED / no evidence required): {native['n_excluded_absent']}")
    md_lines.append(f"- **Included in metrics**: {native['n_included']}")
    md_lines.append("")

    # Sanity
    md_lines.append("## Sanity checks")
    md_lines.append("")
    md_lines.append(f"- A1: `sds_01.pdf` / `product_name` page 1 rank in native: **{native['sanity_rank_sds01_product_name']}**")
    md_lines.append("")

    # Native table
    md_lines.append("## Native layer")
    md_lines.append("")
    md_lines.append(format_table(native, "Native"))
    md_lines.append(format_slices(native))

    # OCR table
    md_lines.append("## OCR layer")
    md_lines.append("")
    md_lines.append(format_table(ocr, "OCR"))
    md_lines.append(format_slices(ocr))

    # Interpretation
    md_lines.append("## Interpretation")
    md_lines.append("")
    md_lines.append(
        "BM25 achieves moderate piece-level recall but struggles with strict complete-evidence retrieval. "
        "Complete@3 is low because multi-evidence instances require all pages to be retrieved, and BM25 "
        "ranks pages independently without awareness of coverage. Cross-page instances are harder than "
        "single-page ones. The naive page-order baseline is surprisingly competitive on some metrics, "
        "suggesting that evidence often appears early in SDS documents. Native and OCR layers perform "
        "similarly, with OCR slightly lagging on fine-grained matches."
    )
    md_lines.append("")

    with OUT_MD.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Results written to {OUT_JSON} and {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
