#!/usr/bin/env python3
"""Explore SDS annotations to understand evidence distribution."""

import json
from pathlib import Path
from collections import Counter

# Путь к данным (внутри репо)
DATA_DIR = Path("data/annotations/source")


def load_annotations():
    """Load all annotation files."""
    docs = []
    for f in sorted(DATA_DIR.glob("sds_*.json")):
        with open(f) as fh:
            docs.append(json.load(fh))
    return docs


def analyze_evidence(docs):
    """Analyze evidence distribution."""
    stats = {
        "total_docs": len(docs),
        "product_forms": Counter(),
        "evidence_per_field": [],
        "derived_vs_direct": Counter(),
    }

    for doc in docs:
        form = doc.get("product_form", "UNKNOWN")
        stats["product_forms"][form] += 1

        fields = doc.get("fields", {})
        for field_name, field_data in fields.items():
            evidence_list = field_data.get("evidence", [])
            if not isinstance(evidence_list, list):
                evidence_list = [evidence_list] if evidence_list else []

            stats["evidence_per_field"].append({
                "doc": doc["document"],
                "field": field_name,
                "state": field_data.get("state", "UNKNOWN"),
                "evidence_count": len(evidence_list),
                "pages": [e.get("page", 0) for e in evidence_list],
                "derivation": field_data.get("derivation", "EXPLICIT"),
            })

            deriv = field_data.get("derivation", "EXPLICIT")
            stats["derived_vs_direct"][deriv] += 1

    return stats


def print_stats(stats):
    """Print summary statistics."""
    print("=== SDS Dataset Overview ===\n")
    print(f"Total documents: {stats['total_docs']}")
    print(f"Product forms: {dict(stats['product_forms'])}")

    print(f"\n=== Evidence Distribution ===")
    evidence_counts = Counter(e["evidence_count"] for e in stats["evidence_per_field"])
    print(f"Evidence pieces per field: {dict(sorted(evidence_counts.items()))}")

    print(f"\n=== Derivation Types ===")
    for deriv_type, count in stats["derived_vs_direct"].items():
        print(f"  {deriv_type}: {count}")

    print(f"\n=== Multi-evidence Fields ===")
    multi = [e for e in stats["evidence_per_field"] if e["evidence_count"] > 1]
    print(f"Fields with >1 evidence: {len(multi)}")
    for e in multi[:10]:
        print(f"  {e['doc']}/{e['field']}: {e['evidence_count']} pieces, pages {e['pages']}, {e['derivation']}")


if __name__ == "__main__":
    docs = load_annotations()
    stats = analyze_evidence(docs)
    print_stats(stats)
