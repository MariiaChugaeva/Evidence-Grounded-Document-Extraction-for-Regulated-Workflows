"""Native PDF and OCR text, with boxes as page fractions in [0, 1].

Verbatim matching
-----------------
A quote is *verbatim* when, after the normalization below, it is a contiguous
substring of the page's token text in reading order or in visual-row order.
Nothing else counts: no similarity threshold, no partial matches. The
normalization is limited to lookalike folding and whitespace:

- case is folded;
- runs of whitespace (including no-break spaces) collapse to one space;
- ordinal º and degree ° both become °;
- en dash, em dash and minus become '-';
- curly quotes become straight quotes;
- the ligatures ff / fi / fl are expanded;
- the ™ and ® marks are dropped.

Hyphenation at line breaks is *not* undone: a quote must reproduce the text
as printed. Fuzzy scores exist only as diagnostics (``fuzzy_occurrence_score``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dataclass_field
from functools import lru_cache
from pathlib import Path

import pymupdf
from rapidfuzz import fuzz

BBox = tuple[float, float, float, float]

DOCUMENTS_DIR = Path("documents")
OCR_DIR = Path("data/ocr")

NATIVE = "native"
OCR = "ocr"


@dataclass(frozen=True)
class Token:
    text: str
    bbox: BBox
    confidence: float | None = None


@dataclass
class Page:
    page_number: int
    tokens: list[Token]
    text: str
    token_spans: list[tuple[int, int]]
    width: float
    height: float
    row_cache: dict[float, list["Row"]] = dataclass_field(
        default_factory=dict, repr=False, compare=False
    )

    def tokens_in_char_range(self, start: int, end: int) -> list[Token]:
        return [
            token
            for token, (t_start, t_end) in zip(self.tokens, self.token_spans)
            if t_start < end and t_end > start
        ]

    @property
    def row_text(self) -> str:
        return "\n".join(row.text for row in rows(self))


@dataclass
class DocumentText:
    document: str
    layer: str
    pages: list[Page]

    def page(self, page_number: int) -> Page | None:
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    @property
    def full_text(self) -> str:
        return "\n".join(page.text for page in self.pages)


@dataclass
class Anchor:
    page_number: int
    text: str
    char_start: int
    char_end: int
    bbox: BBox | None
    score: float
    exact: bool


def union_bbox(boxes: list[BBox]) -> BBox | None:
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def bbox_iou(a: BBox | None, b: BBox | None) -> float:
    if a is None or b is None:
        return 0.0
    intersection = _intersection(a, b)
    union = _area(a) + _area(b) - intersection
    return intersection / union if union > 0 else 0.0


def bbox_containment(predicted: BBox | None, gold: BBox | None) -> tuple[float, float]:
    """Return (area coverage of gold, predicted area / gold area).

    The second number is a two-dimensional dilution: a box that is the gold's
    height but the page's width is diluted, and so is a box that is the gold's
    width but three lines tall.
    """
    if predicted is None or gold is None:
        return 0.0, 0.0
    gold_area = _area(gold)
    if gold_area <= 0:
        return 0.0, 0.0
    return _intersection(predicted, gold) / gold_area, _area(predicted) / gold_area


def bbox_height_ratio(predicted: BBox | None, gold: BBox | None) -> float:
    if predicted is None or gold is None:
        return 0.0
    gold_height = max(1e-9, gold[3] - gold[1])
    return max(0.0, predicted[3] - predicted[1]) / gold_height


def _build_page(
    page_number: int,
    lines: list[list[Token]],
    width: float,
    height: float,
) -> Page:
    tokens: list[Token] = []
    token_spans: list[tuple[int, int]] = []
    pieces: list[str] = []
    cursor = 0
    for line_index, line in enumerate(lines):
        if line_index:
            pieces.append("\n")
            cursor += 1
        for token_index, token in enumerate(line):
            if token_index:
                pieces.append(" ")
                cursor += 1
            pieces.append(token.text)
            tokens.append(token)
            token_spans.append((cursor, cursor + len(token.text)))
            cursor += len(token.text)
    return Page(
        page_number=page_number,
        tokens=tokens,
        text="".join(pieces),
        token_spans=token_spans,
        width=width,
        height=height,
    )


def load_native(document: str, documents_dir: Path = DOCUMENTS_DIR) -> DocumentText:
    path = documents_dir / document
    pdf = pymupdf.open(path)
    pages: list[Page] = []
    try:
        for index, pdf_page in enumerate(pdf, start=1):
            rect = pdf_page.rect
            width, height = rect.width, rect.height
            grouped: dict[tuple[int, int], list[tuple[int, Token]]] = {}
            for x0, y0, x1, y1, word, block_no, line_no, word_no in pdf_page.get_text(
                "words"
            ):
                if not word.strip():
                    continue
                token = Token(
                    text=word,
                    bbox=(x0 / width, y0 / height, x1 / width, y1 / height),
                )
                grouped.setdefault((block_no, line_no), []).append((word_no, token))
            lines = [
                [token for _, token in sorted(entries, key=lambda item: item[0])]
                for _, entries in sorted(grouped.items())
            ]
            pages.append(_build_page(index, lines, width, height))
    finally:
        pdf.close()
    return DocumentText(document=document, layer=NATIVE, pages=pages)


def _group_ocr_lines(words: list[dict], height: float) -> list[list[Token]]:
    lines: list[list[Token]] = []
    current: list[Token] = []
    current_top = current_bottom = 0.0
    for word in words:
        x0, y0, x1, y1 = word["bbox"]
        centre = (y0 + y1) / 2
        if current and not (current_top <= centre <= current_bottom):
            lines.append(current)
            current = []
        if not current:
            current_top, current_bottom = y0, y1
        else:
            current_top = min(current_top, y0)
            current_bottom = max(current_bottom, y1)
        current.append(
            Token(
                text=word["text"],
                bbox=(x0, y0, x1, y1),
                confidence=word.get("confidence"),
            )
        )
    if current:
        lines.append(current)
    return lines


def load_ocr(document: str, ocr_dir: Path = OCR_DIR) -> DocumentText:
    path = ocr_dir / (Path(document).stem + ".json")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    pages: list[Page] = []
    for raw_page in data["pages"]:
        image_width = float(raw_page["image_width"])
        image_height = float(raw_page["image_height"])
        lines = _group_ocr_lines(raw_page["words"], image_height)
        normalized_lines = [
            [
                Token(
                    text=token.text,
                    bbox=(
                        token.bbox[0] / image_width,
                        token.bbox[1] / image_height,
                        token.bbox[2] / image_width,
                        token.bbox[3] / image_height,
                    ),
                    confidence=token.confidence,
                )
                for token in line
            ]
            for line in lines
        ]
        pages.append(
            _build_page(
                raw_page["page_number"], normalized_lines, image_width, image_height
            )
        )
    return DocumentText(document=data["document"], layer=OCR, pages=pages)


@lru_cache(maxsize=None)
def load(document: str, layer: str = NATIVE) -> DocumentText:
    if layer == NATIVE:
        return load_native(document)
    if layer == OCR:
        return load_ocr(document)
    raise ValueError(f"unknown text layer: {layer!r}")


_CHAR_FOLDING = {
    "º": "°",
    "°": "°",
    "–": "-",
    "—": "-",
    "−": "-",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    " ": " ",
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "™": "",
    "®": "",
}


def fold(text: str) -> str:
    return "".join(_CHAR_FOLDING.get(char, char) for char in text)


def normalize_for_search(text: str, casefold: bool = True) -> tuple[str, list[int]]:
    """Apply the module's normalization. offsets[i] maps back to the original index."""
    normalized: list[str] = []
    offsets: list[int] = []
    previous_was_space = True
    for index, char in enumerate(text):
        folded = _CHAR_FOLDING.get(char, char)
        if not folded:
            continue
        if folded.isspace():
            if previous_was_space:
                continue
            normalized.append(" ")
            offsets.append(index)
            previous_was_space = True
            continue
        for piece in folded.lower() if casefold else folded:
            normalized.append(piece)
            offsets.append(index)
        previous_was_space = False
    while normalized and normalized[-1] == " ":
        normalized.pop()
        offsets.pop()
    return "".join(normalized), offsets


def _anchor_from_norm_range(
    page: Page,
    offsets: list[int],
    start_norm: int,
    end_norm: int,
    score: float,
    exact: bool,
) -> Anchor:
    char_start = offsets[start_norm]
    char_end = offsets[min(end_norm, len(offsets)) - 1] + 1
    matched_tokens = page.tokens_in_char_range(char_start, char_end)
    return Anchor(
        page_number=page.page_number,
        text=page.text[char_start:char_end],
        char_start=char_start,
        char_end=char_end,
        bbox=union_bbox([token.bbox for token in matched_tokens]),
        score=score,
        exact=exact,
    )


def find_all_in_page(page: Page, needle: str, casefold: bool = True) -> list[Anchor]:
    """Every exact occurrence of the needle in the page's reading-order text."""
    haystack, offsets = normalize_for_search(page.text, casefold)
    query, _ = normalize_for_search(needle, casefold)
    if not query or not haystack:
        return []
    anchors: list[Anchor] = []
    position = haystack.find(query)
    while position != -1:
        anchors.append(
            _anchor_from_norm_range(
                page, offsets, position, position + len(query), 100.0, True
            )
        )
        position = haystack.find(query, position + 1)
    return anchors


def find_in_page(page: Page, needle: str, min_score: float = 90.0) -> Anchor | None:
    """Exact occurrence if there is one, else the best fuzzy alignment above min_score.

    Diagnostic use only: nothing in the evaluator accepts a fuzzy anchor.
    """
    haystack, offsets = normalize_for_search(page.text)
    query, _ = normalize_for_search(needle)
    if not query or not haystack:
        return None

    position = haystack.find(query)
    exact = position != -1
    if exact:
        start_norm, end_norm, score = position, position + len(query), 100.0
    else:
        alignment = fuzz.partial_ratio_alignment(query, haystack)
        if alignment is None or alignment.score < min_score:
            return None
        start_norm, end_norm = alignment.dest_start, alignment.dest_end
        score = alignment.score
        if end_norm <= start_norm:
            return None

    return _anchor_from_norm_range(page, offsets, start_norm, end_norm, score, exact)


def find_in_document(
    document_text: DocumentText,
    needle: str,
    page_number: int | None = None,
    min_score: float = 90.0,
) -> Anchor | None:
    if page_number is not None:
        page = document_text.page(page_number)
        return find_in_page(page, needle, min_score) if page else None

    best: Anchor | None = None
    for page in document_text.pages:
        anchor = find_in_page(page, needle, min_score)
        if anchor is None:
            continue
        if best is None or (anchor.exact, anchor.score) > (best.exact, best.score):
            best = anchor
        if best.exact:
            break
    return best


def occurs_on_page(page: Page, quote: str) -> bool:
    """Verbatim test, exact after normalization, in reading order or visual-row order."""
    needle, _ = normalize_for_search(quote)
    if not needle:
        return False
    for haystack_source in (page.text, page.row_text):
        haystack, _ = normalize_for_search(haystack_source)
        if needle in haystack:
            return True
    return False


def fuzzy_occurrence_score(page: Page, quote: str) -> float:
    """Best partial-ratio of the quote against the page (0-100). Diagnostic only."""
    needle, _ = normalize_for_search(quote)
    if not needle:
        return 0.0
    best = 0.0
    for haystack_source in (page.text, page.row_text):
        haystack, _ = normalize_for_search(haystack_source)
        if haystack:
            best = max(best, float(fuzz.partial_ratio(needle, haystack)))
    return best


def available_documents(documents_dir: Path = DOCUMENTS_DIR) -> list[str]:
    return sorted(path.name for path in documents_dir.glob("*.pdf"))


@dataclass
class Row:
    page_number: int
    tokens: list[Token]

    @property
    def text(self) -> str:
        return " ".join(token.text for token in self.tokens)


def rows(page: Page, overlap: float = 0.4) -> list[Row]:
    cached = page.row_cache.get(overlap)
    if cached is not None:
        return cached

    ordered = sorted(page.tokens, key=lambda token: (token.bbox[1], token.bbox[0]))
    grouped: list[list[Token]] = []
    for token in ordered:
        placed = False
        if grouped:
            current = grouped[-1]
            top = min(item.bbox[1] for item in current)
            bottom = max(item.bbox[3] for item in current)
            height = min(bottom - top, token.bbox[3] - token.bbox[1])
            shared = min(bottom, token.bbox[3]) - max(top, token.bbox[1])
            if height > 0 and shared / height >= overlap:
                current.append(token)
                placed = True
        if not placed:
            grouped.append([token])
    result = [
        Row(page_number=page.page_number, tokens=sorted(group, key=lambda t: t.bbox[0]))
        for group in grouped
    ]
    page.row_cache[overlap] = result
    return result
