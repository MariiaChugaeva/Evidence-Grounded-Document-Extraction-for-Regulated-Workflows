"""Build retrieval corpus from gold annotations.

Creates:
- farit/data/corpus/units_native.jsonl
- farit/data/corpus/units_ocr.jsonl
- farit/data/corpus/gold_evidence.jsonl
- farit/data/corpus/corpus_stats.json

Run from repository root: ~/ds_env/bin/python farit/src/retrieval/build_corpus.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from sdsbench import textlayer

# Paths relative to repository root
GOLD_PATH = Path("data/annotations/gold.json")
MANIFEST_PATH = Path("data/manifest.csv")
OUTPUT_DIR = Path("farit/data/corpus")

LAYERS = ["native", "ocr"]
FIELDS = [
    "product_name",
    "supplier_name",
    "substance_or_mixture",
    "product_cas_number",
    "chemical_formula",
    "molecular_weight",
    "flash_point",
]


def load_manifest() -> dict[str, int]:
    """Load manifest and return {document: pages}."""
    import csv

    manifest: dict[str, int] = {}
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest[row["document"]] = int(row["pages"])
    return manifest


def count_tokens(text: str) -> int:
    """Count whitespace-separated tokens."""
    return len(text.split())


def build_units(documents: list[dict], layer: str) -> list[dict]:
    """Build unit records for a text layer."""
    units = []
    for doc in documents:
        doc_name = doc["document"]
        dt = textlayer.load(doc_name, layer)
        for page in dt.pages:
            units.append(
                {
                    "document": doc_name,
                    "split": doc["split"],
                    "page": page.page_number,
                    "layer": layer,
                    "text": page.text,
                    "n_tokens": count_tokens(page.text),
                }
            )
    return units


def build_gold_evidence(documents: list[dict]) -> list[dict]:
    """Build gold evidence records for all field instances."""
    records = []

    for doc in documents:
        doc_name = doc["document"]
        split = doc["split"]

        # 7 fields × N documents
        for field_name in FIELDS:
            field_data = doc["fields"][field_name]

            evidence_set = []

            # Add resolved_evidence with kind="evidence"
            for ev in field_data.get("resolved_evidence", []):
                evidence_set.append(
                    {
                        "kind": "evidence",
                        "page": ev["page"],
                        "text": ev["text"],
                        "char_start": ev["char_start"],
                        "char_end": ev["char_end"],
                        "bbox": ev["bbox"],
                        "ocr_page": ev["ocr_page"],
                        "ocr_char_start": ev["ocr_char_start"],
                        "ocr_char_end": ev["ocr_char_end"],
                        "ocr_bbox": ev["ocr_bbox"],
                        "ocr_exact": ev["ocr_exact"],
                    }
                )

            # Add resolved_premises with kind="premise"
            for pr in field_data.get("resolved_premises", []):
                evidence_set.append(
                    {
                        "kind": "premise",
                        "page": pr["page"],
                        "text": pr["text"],
                        "char_start": pr["char_start"],
                        "char_end": pr["char_end"],
                        "bbox": pr["bbox"],
                        "ocr_page": pr["ocr_page"],
                        "ocr_char_start": pr["ocr_char_start"],
                        "ocr_char_end": pr["ocr_char_end"],
                        "ocr_bbox": pr["ocr_bbox"],
                        "ocr_exact": pr["ocr_exact"],
                    }
                )

            pages = {e["page"] for e in evidence_set}
            n_pages = len(pages)
            cross_page = n_pages > 1

            records.append(
                {
                    "document": doc_name,
                    "split": split,
                    "field": field_name,
                    "state": field_data["state"],
                    "derivation": field_data["derivation"],
                    "value": field_data.get("value"),
                    "accepted_values": field_data.get("accepted_values", []),
                    "evidence_set": evidence_set,
                    "n_evidence": len(evidence_set),
                    "n_pages": n_pages,
                    "cross_page": cross_page,
                }
            )

        # Add ingredients if it has resolved_evidence
        ingredients = doc.get("ingredients", {})
        if ingredients.get("resolved_evidence"):
            evidence_set = []
            for ev in ingredients["resolved_evidence"]:
                evidence_set.append(
                    {
                        "kind": "evidence",
                        "page": ev["page"],
                        "text": ev["text"],
                        "char_start": ev["char_start"],
                        "char_end": ev["char_end"],
                        "bbox": ev["bbox"],
                        "ocr_page": ev["ocr_page"],
                        "ocr_char_start": ev["ocr_char_start"],
                        "ocr_char_end": ev["ocr_char_end"],
                        "ocr_bbox": ev["ocr_bbox"],
                        "ocr_exact": ev["ocr_exact"],
                    }
                )

            pages = {e["page"] for e in evidence_set}
            n_pages = len(pages)
            cross_page = n_pages > 1

            records.append(
                {
                    "document": doc_name,
                    "split": split,
                    "field": "ingredients",
                    "state": ingredients.get("state", "PRESENT"),
                    "derivation": "EXPLICIT",
                    "value": None,
                    "accepted_values": [],
                    "evidence_set": evidence_set,
                    "n_evidence": len(evidence_set),
                    "n_pages": n_pages,
                    "cross_page": cross_page,
                }
            )

    return records


def verify_v1(documents: list[dict]) -> tuple[bool, list[str]]:
    """Verify that native text[char_start:char_end] matches evidence text.

    Returns (ok, list of mismatch details).
    """
    mismatches = []
    total = 0

    for doc in documents:
        doc_name = doc["document"]
        dt = textlayer.load(doc_name, "native")

        for field_name in FIELDS:
            field_data = doc["fields"][field_name]
            for ev in field_data.get("resolved_evidence", []):
                total += 1
                page = dt.page(ev["page"])
                if page is None:
                    mismatches.append(
                        f"{doc_name} {field_name}: page {ev['page']} not found"
                    )
                    continue
                extracted = page.text[ev["char_start"] : ev["char_end"]]
                if ev["text"] != extracted:
                    mismatches.append(
                        f"{doc_name} {field_name}: "
                        f"expected {ev['text']!r}, got {extracted!r}"
                    )

            for pr in field_data.get("resolved_premises", []):
                total += 1
                page = dt.page(pr["page"])
                if page is None:
                    mismatches.append(
                        f"{doc_name} {field_name} premise: page {pr['page']} not found"
                    )
                    continue
                extracted = page.text[pr["char_start"] : pr["char_end"]]
                if pr["text"] != extracted:
                    mismatches.append(
                        f"{doc_name} {field_name} premise: "
                        f"expected {pr['text']!r}, got {extracted!r}"
                    )

        # Verify ingredients
        ingredients = doc.get("ingredients", {})
        for ev in ingredients.get("resolved_evidence", []):
            total += 1
            page = dt.page(ev["page"])
            if page is None:
                mismatches.append(
                    f"{doc_name} ingredients: page {ev['page']} not found"
                )
                continue
            extracted = page.text[ev["char_start"] : ev["char_end"]]
            if ev["text"] != extracted:
                mismatches.append(
                    f"{doc_name} ingredients: "
                    f"expected {ev['text']!r}, got {extracted!r}"
                )

    return len(mismatches) == 0, mismatches


def compute_stats(
    documents: list[dict],
    units_native: list[dict],
    units_ocr: list[dict],
    gold_records: list[dict],
) -> dict:
    """Compute corpus statistics."""
    import statistics

    # Units per layer
    native_tokens = [u["n_tokens"] for u in units_native]
    ocr_tokens = [u["n_tokens"] for u in units_ocr]

    def token_stats(tokens: list[int]) -> dict:
        if not tokens:
            return {"min": 0, "median": 0, "p90": 0, "max": 0}
        sorted_tokens = sorted(tokens)
        n = len(sorted_tokens)
        p90_idx = int(n * 0.9)
        if p90_idx >= n:
            p90_idx = n - 1
        return {
            "min": min(tokens),
            "median": statistics.median(tokens),
            "p90": sorted_tokens[p90_idx],
            "max": max(tokens),
        }

    # State and derivation distributions
    state_counts = Counter()
    derivation_counts = Counter()
    for rec in gold_records:
        if rec["field"] != "ingredients":
            state_counts[rec["state"]] += 1
            derivation_counts[rec["derivation"]] += 1

    # Evidence set sizes (only for field instances, not ingredients)
    evidence_sizes = Counter()
    cross_page_count = 0
    for rec in gold_records:
        if rec["field"] != "ingredients":
            size = rec["n_evidence"]
            if size >= 3:
                evidence_sizes["3+"] += 1
            else:
                evidence_sizes[str(size)] += 1
            if rec["cross_page"]:
                cross_page_count += 1

    # OCR exact ratio
    ocr_exact_count = 0
    ocr_total = 0
    for rec in gold_records:
        for ev in rec["evidence_set"]:
            ocr_total += 1
            if ev.get("ocr_exact", False):
                ocr_exact_count += 1

    # Count fragments
    evidence_fragments = 0
    premise_fragments = 0
    for rec in gold_records:
        if rec["field"] != "ingredients":
            for ev in rec["evidence_set"]:
                if ev["kind"] == "evidence":
                    evidence_fragments += 1
                else:
                    premise_fragments += 1

    return {
        "n_documents": len(documents),
        "units_native": len(units_native),
        "units_ocr": len(units_ocr),
        "n_tokens_native": token_stats(native_tokens),
        "n_tokens_ocr": token_stats(ocr_tokens),
        "field_instances": len([r for r in gold_records if r["field"] != "ingredients"]),
        "ingredient_instances": len([r for r in gold_records if r["field"] == "ingredients"]),
        "state_distribution": dict(state_counts),
        "derivation_distribution": dict(derivation_counts),
        "evidence_set_sizes": dict(evidence_sizes),
        "cross_page_instances": cross_page_count,
        "evidence_fragments": evidence_fragments,
        "premise_fragments": premise_fragments,
        "total_fragments": evidence_fragments + premise_fragments,
        "ocr_exact_ratio": round(ocr_exact_count / ocr_total, 4) if ocr_total > 0 else 0,
        "ocr_exact_count": ocr_exact_count,
        "ocr_total": ocr_total,
    }


def main() -> int:
    print("Building retrieval corpus...")

    # Load gold annotations
    with open(GOLD_PATH, encoding="utf-8") as f:
        gold_data = json.load(f)
    documents = gold_data["documents"]

    # Load manifest
    manifest = load_manifest()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- V1: Verify evidence text matches ---
    print("\n[V1] Verifying evidence text matches...")
    v1_ok, v1_mismatches = verify_v1(documents)
    if not v1_ok:
        print(f"FAIL: {len(v1_mismatches)} mismatches found:")
        for m in v1_mismatches:
            print(f"  {m}")
        return 1
    print("PASS: All evidence fragments match native text.")

    # --- Build units ---
    print("\nBuilding units...")
    for layer in LAYERS:
        units = build_units(documents, layer)
        path = OUTPUT_DIR / f"units_{layer}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for unit in units:
                f.write(json.dumps(unit, ensure_ascii=False) + "\n")
        print(f"  Written {len(units)} units to {path}")

    # --- Build gold evidence ---
    print("\nBuilding gold evidence...")
    gold_records = build_gold_evidence(documents)
    path = OUTPUT_DIR / "gold_evidence.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in gold_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Written {len(gold_records)} records to {path}")

    # --- V2: Verify counts ---
    print("\n[V2] Verifying counts...")
    field_instances = len([r for r in gold_records if r["field"] != "ingredients"])
    ingredient_instances = len([r for r in gold_records if r["field"] == "ingredients"])

    print(f"  Documents: {len(documents)} (expected: 29)")
    print(f"  Field instances: {field_instances} (expected: 203)")
    print(f"  Ingredient instances: {ingredient_instances}")

    # Count fragments
    evidence_fragments = 0
    premise_fragments = 0
    for rec in gold_records:
        if rec["field"] != "ingredients":
            for ev in rec["evidence_set"]:
                if ev["kind"] == "evidence":
                    evidence_fragments += 1
                else:
                    premise_fragments += 1

    total_fragments = evidence_fragments + premise_fragments
    print(f"  Evidence fragments: {evidence_fragments} (expected: 166)")
    print(f"  Premise fragments: {premise_fragments} (expected: 86)")
    print(f"  Total fragments: {total_fragments} (expected: 252)")

    if len(documents) != 29:
        print(f"WARNING: Document count mismatch!")
    if field_instances != 203:
        print(f"WARNING: Field instance count mismatch!")
    if total_fragments != 252:
        print(f"WARNING: Total fragment count mismatch!")

    # --- V3: Verify pages match manifest ---
    print("\n[V3] Verifying page counts...")
    v3_ok = True
    for doc in documents:
        doc_name = doc["document"]
        expected_pages = manifest.get(doc_name)
        for layer in LAYERS:
            dt = textlayer.load(doc_name, layer)
            actual_pages = len(dt.pages)
            if expected_pages and actual_pages != expected_pages:
                print(f"  FAIL: {doc_name} {layer}: {actual_pages} pages, expected {expected_pages}")
                v3_ok = False
    if v3_ok:
        print("PASS: All documents have correct page counts.")

    # --- V4: Verify JSONL readability ---
    print("\n[V4] Verifying JSONL readability...")
    v4_ok = True
    for filename in ["units_native.jsonl", "units_ocr.jsonl", "gold_evidence.jsonl"]:
        path = OUTPUT_DIR / filename
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    count += 1
                except json.JSONDecodeError as e:
                    print(f"  FAIL: {filename} line {count + 1}: {e}")
                    v4_ok = False
        print(f"  {filename}: {count} valid JSON lines")
    if v4_ok:
        print("PASS: All JSONL files are valid.")

    # --- Compute and save stats ---
    print("\nComputing statistics...")
    units_native = []
    with open(OUTPUT_DIR / "units_native.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                units_native.append(json.loads(line))

    units_ocr = []
    with open(OUTPUT_DIR / "units_ocr.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                units_ocr.append(json.loads(line))

    stats = compute_stats(documents, units_native, units_ocr, gold_records)

    stats_path = OUTPUT_DIR / "corpus_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Written stats to {stats_path}")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("CORPUS STATISTICS")
    print("=" * 60)
    print(f"Documents: {stats['n_documents']}")
    print(f"Native units: {stats['units_native']}")
    print(f"OCR units: {stats['units_ocr']}")
    print(f"\nToken distribution (native):")
    for k, v in stats["n_tokens_native"].items():
        print(f"  {k}: {v}")
    print(f"\nToken distribution (OCR):")
    for k, v in stats["n_tokens_ocr"].items():
        print(f"  {k}: {v}")
    print(f"\nField instances: {stats['field_instances']}")
    print(f"Ingredient instances: {stats['ingredient_instances']}")
    print(f"\nState distribution:")
    for k, v in stats["state_distribution"].items():
        print(f"  {k}: {v}")
    print(f"\nDerivation distribution:")
    for k, v in stats["derivation_distribution"].items():
        print(f"  {k}: {v}")
    print(f"\nEvidence set sizes:")
    for k, v in stats["evidence_set_sizes"].items():
        print(f"  {k}: {v}")
    print(f"\nCross-page instances: {stats['cross_page_instances']}")
    print(f"Evidence fragments: {stats['evidence_fragments']}")
    print(f"Premise fragments: {stats['premise_fragments']}")
    print(f"Total fragments: {stats['total_fragments']}")
    print(f"\nOCR exact ratio: {stats['ocr_exact_ratio']:.2%} ({stats['ocr_exact_count']}/{stats['ocr_total']})")
    print("=" * 60)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
