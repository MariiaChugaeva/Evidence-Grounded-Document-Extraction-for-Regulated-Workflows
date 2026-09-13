"""Corpus manifest: one row per PDF with split, supplier, template family and regime.

The manifest is the authority on which documents belong to which split. The
``locked`` split may only be scored with an explicit flag, and every such run
should be recorded; the rules and cue lexicons must never be tuned on it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .textlayer import DOCUMENTS_DIR

MANIFEST_PATH = Path("data/manifest.csv")

SPLIT_DEV = "dev"
SPLIT_LOCKED = "locked"

FIELDS = (
    "document",
    "split",
    "supplier",
    "template_family",
    "regime",
    "language",
    "scan",
    "pages",
    "product_form",
    "note",
)


@dataclass(frozen=True)
class ManifestEntry:
    document: str
    split: str
    supplier: str
    template_family: str
    regime: str
    language: str
    scan: bool
    pages: int
    product_form: str
    note: str


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, ManifestEntry]:
    entries: dict[str, ManifestEntry] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            entries[row["document"]] = ManifestEntry(
                document=row["document"],
                split=row["split"],
                supplier=row.get("supplier", ""),
                template_family=row.get("template_family", ""),
                regime=row.get("regime", ""),
                language=row.get("language", ""),
                scan=row.get("scan", "").strip().lower() in ("yes", "true", "1"),
                pages=int(row["pages"]) if row.get("pages") else 0,
                product_form=row.get("product_form", ""),
                note=row.get("note", ""),
            )
    return entries


def check_manifest(
    entries: dict[str, ManifestEntry], documents_dir: Path = DOCUMENTS_DIR
) -> list[str]:
    """Every PDF has a row, every row has a PDF, every row has a split and a template family."""
    problems: list[str] = []
    on_disk = {path.name for path in documents_dir.glob("*.pdf")}
    for document in sorted(on_disk - set(entries)):
        problems.append(f"{document}: PDF has no manifest row")
    for document in sorted(set(entries) - on_disk):
        problems.append(f"{document}: manifest row has no PDF")
    for document, entry in sorted(entries.items()):
        if entry.split not in (SPLIT_DEV, SPLIT_LOCKED):
            problems.append(f"{document}: unknown split {entry.split!r}")
        if not entry.template_family:
            problems.append(f"{document}: no template family")
    return problems


def template_families(entries: dict[str, ManifestEntry]) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    for document, entry in sorted(entries.items()):
        families.setdefault(entry.template_family, []).append(document)
    return families
