"""Turn the medical PDFs into clean per-page text and token-bounded chunks.

Pipeline (contract C2 of the project README):

    pdf_inventory.csv --> pages.jsonl   (one record per PDF page, cleaned text)
                      --> chunks.jsonl  (512-token windows, 102 overlap, one page each)
                      --> ingest_report.md (per-document statistics, GARBLED flags)

Chunking stays inside a single page so that every citation points to one exact
page. Token counts and window boundaries come from the BGE-M3 tokenizer, and
windows are cut from the ORIGINAL string through the tokenizer's offset mapping
(no decode artefacts).

Usage (run from 03_CODE/):
    python src/ingest.py                    # all documents
    python src/ingest.py --only DOC_ID      # re-ingest one document, keep the others
    python src/ingest.py --pages-only       # step A only (no tokenizer needed)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import config

LOGGER = logging.getLogger("ingest")

INVENTORY_PATH = config.DATASETS / "pdf_inventory.csv"
PAGES_PATH = config.PROCESSED_DIR / "pages.jsonl"
CHUNKS_PATH = config.PROCESSED_DIR / "chunks.jsonl"
REPORT_PATH = config.PROCESSED_DIR / "ingest_report.md"

GARBLED_CHARS_PER_PAGE = 300  # average below this => scanned or broken text
REPEAT_RATIO = 0.5  # a line on more than this share of pages is a header/footer
MIN_PAGES_FOR_REPEAT_DETECTION = 4
MAX_HEADING_CHARS = 80

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_PAGE_NUMBER_LINE = re.compile(
    r"^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.IGNORECASE
)
_NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z]")
_MULTI_SPACE = re.compile(r"[ \t ]+")


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
def load_inventory(path: Path = INVENTORY_PATH) -> list[dict[str, str]]:
    """Read the PDF inventory (contract C1) and return its rows in file order."""
    if not path.exists():
        raise FileNotFoundError(f"Inventory not found: {path}")
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"doc_id", "title", "source", "url", "year", "pages", "file_name"}
    missing = required - set(rows[0].keys()) if rows else required
    if missing:
        raise ValueError(f"Inventory is missing columns: {sorted(missing)}")
    return rows


# --------------------------------------------------------------------------- #
# Step A: PDF -> cleaned pages
# --------------------------------------------------------------------------- #
def extract_raw_pages(pdf_path: Path) -> list[str]:
    """Return the raw text of every page of a PDF (empty string on page failure).

    Raises an exception when the PDF cannot be opened at all or stays encrypted;
    the caller logs it and skips the document.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    if reader.is_encrypted:
        # An empty user password opens many "encrypted" public documents.
        if reader.decrypt("") == 0:
            raise ValueError("PDF is encrypted and needs a password")
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            pages.append(page.extract_text() or "")
        except Exception as error:  # noqa: BLE001 - one bad page must not stop the run
            LOGGER.warning("%s page %d: extraction failed (%s)", pdf_path.name, index, error)
            pages.append("")
    return pages


def _normalise_lines(raw: str) -> list[str]:
    """NFKC-normalise a page, strip control characters and return non-empty lines."""
    text = unicodedata.normalize("NFKC", raw)
    text = _CONTROL_CHARS.sub("", text)
    lines = []
    for line in text.splitlines():
        line = _MULTI_SPACE.sub(" ", line).strip()
        if line:
            lines.append(line)
    return lines


def _repeat_key(line: str) -> str:
    """Key used to detect running headers/footers (digits are masked)."""
    return re.sub(r"\d+", "#", line.lower())


def find_repeated_lines(pages_lines: list[list[str]]) -> set[str]:
    """Return the keys of lines present on more than half of a document's pages."""
    total = len(pages_lines)
    if total < MIN_PAGES_FOR_REPEAT_DETECTION:
        return set()
    counter: Counter[str] = Counter()
    for lines in pages_lines:
        counter.update({_repeat_key(line) for line in lines if len(line) <= 120})
    return {key for key, count in counter.items() if count / total > REPEAT_RATIO}


def join_hyphenated(lines: list[str]) -> list[str]:
    """Re-join words split by a hyphen at the end of a line ("treat-" + "ment")."""
    merged: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        while (
            line.endswith("-")
            and len(line) > 1
            and line[-2].isalpha()
            and index + 1 < len(lines)
            and lines[index + 1][:1].islower()
        ):
            index += 1
            line = line[:-1] + lines[index]
        merged.append(line)
        index += 1
    return merged


def clean_page(raw: str, repeated: set[str]) -> list[str]:
    """Clean one page and return its lines (headers, footers, page numbers removed)."""
    lines = _normalise_lines(raw)
    lines = [
        line
        for line in lines
        if not _PAGE_NUMBER_LINE.match(line) and _repeat_key(line) not in repeated
    ]
    return join_hyphenated(lines)


def is_heading(line: str) -> bool:
    """Heuristic: does this line look like a section heading?"""
    if not 3 <= len(line) <= MAX_HEADING_CHARS or line.endswith((".", ",", ";", ":")):
        return False
    if _NUMBERED_HEADING.match(line):
        return True
    letters = [char for char in line if char.isalpha()]
    if len(letters) >= 3 and all(char.isupper() for char in letters):
        return True
    words = line.split()
    if 2 <= len(words) <= 10 and line[0].isupper():
        small = {"of", "and", "the", "in", "for", "to", "a", "on", "with", "or", "by"}
        return all(word[0].isupper() or word.lower() in small for word in words if word[0].isalpha())
    return False


def build_pages(row: dict[str, str]) -> list[dict[str, Any]]:
    """Step A for one document: PDF -> list of page records (all pages kept)."""
    pdf_path = config.PDF_DIR / row["file_name"]
    raw_pages = extract_raw_pages(pdf_path)
    pages_lines = [_normalise_lines(raw) for raw in raw_pages]
    repeated = find_repeated_lines(pages_lines)
    records: list[dict[str, Any]] = []
    section: str | None = None
    for number, raw in enumerate(raw_pages, start=1):
        lines = clean_page(raw, repeated)
        for line in lines:
            if is_heading(line):
                section = line
        records.append(
            {
                "doc_id": row["doc_id"],
                "source": row["file_name"],
                "page": number,
                "section": section,
                "text": " ".join(lines),
            }
        )
    return records


# --------------------------------------------------------------------------- #
# Step B: pages -> chunks
# --------------------------------------------------------------------------- #
def load_tokenizer() -> Any:
    """Load the BGE-M3 fast tokenizer (tokenizer files only, no model weights)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.EMBED_MODEL, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("A fast tokenizer is required for offset mapping")
    return tokenizer


def token_offsets(text: str, tokenizer: Any) -> list[tuple[int, int]]:
    """Return the (start, end) character offsets of every token of ``text``."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [tuple(pair) for pair in encoded["offset_mapping"]]


def chunk_page(
    text: str,
    tokenizer: Any,
    size: int = config.CHUNK_TOKENS,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[tuple[str, int]]:
    """Cut ``text`` into windows of ``size`` tokens with ``overlap`` tokens shared.

    Returns a list of ``(chunk_text, n_tokens)``. The stride is ``size - overlap``
    (410 with the project parameters). Chunk text is a slice of the original string.
    """
    offsets = token_offsets(text, tokenizer)
    total = len(offsets)
    stride = size - overlap
    chunks: list[tuple[str, int]] = []
    start = 0
    while start < total:
        end = min(start + size, total)
        piece = text[offsets[start][0] : offsets[end - 1][1]].strip()
        if piece:
            chunks.append((piece, end - start))
        if end == total:
            break
        start += stride
    return chunks


def build_chunks(
    pages: list[dict[str, Any]], row: dict[str, str], tokenizer: Any
) -> list[dict[str, Any]]:
    """Step B for one document: page records -> chunk records (contract C2)."""
    chunks: list[dict[str, Any]] = []
    for page in pages:
        text = page["text"]
        if not text or len(token_offsets(text, tokenizer)) < config.MIN_PAGE_TOKENS:
            continue
        for index, (piece, n_tokens) in enumerate(chunk_page(text, tokenizer), start=1):
            chunks.append(
                {
                    "chunk_id": f"{row['doc_id']}_p{page['page']:04d}_c{index:02d}",
                    "doc_id": row["doc_id"],
                    "source": row["file_name"],
                    "page": page["page"],
                    "section": page["section"],
                    "text": piece,
                    "n_tokens": n_tokens,
                    "year": row["year"],
                }
            )
    return chunks


# --------------------------------------------------------------------------- #
# Files and report
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file; return an empty list when it does not exist."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as UTF-8 JSON lines (one object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_report(
    rows: list[dict[str, str]],
    pages: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    failed: list[str],
) -> int:
    """Write ingest_report.md and return the total number of chunks."""
    pages_by_doc: dict[str, list[dict[str, Any]]] = {}
    for page in pages:
        pages_by_doc.setdefault(page["doc_id"], []).append(page)
    chunk_count: Counter[str] = Counter(chunk["doc_id"] for chunk in chunks)

    lines = [
        "# Ingestion report",
        "",
        "| doc_id | pages | avg chars/page | empty/short pages | chunks | flag |",
        "|---|---:|---:|---:|---:|---|",
    ]
    total_pages = total_empty = 0
    for row in rows:
        doc_pages = pages_by_doc.get(row["doc_id"], [])
        if not doc_pages:
            lines.append(f"| {row['doc_id']} | 0 | 0 | 0 | 0 | FAILED |")
            continue
        chars = sum(len(page["text"]) for page in doc_pages)
        average = chars / len(doc_pages)
        empty = sum(1 for page in doc_pages if len(page["text"]) < 50)
        flag = "GARBLED" if average < GARBLED_CHARS_PER_PAGE else ""
        total_pages += len(doc_pages)
        total_empty += empty
        lines.append(
            f"| {row['doc_id']} | {len(doc_pages)} | {average:.0f} | {empty} "
            f"| {chunk_count[row['doc_id']]} | {flag} |"
        )
    total_chunks = len(chunks)
    lines += [
        "",
        f"- Documents: {len(pages_by_doc)} / {len(rows)}",
        f"- Total pages: {total_pages}",
        f"- Empty or very short pages: {total_empty}",
        f"- **Total chunks: {total_chunks}** (limit {config.MAX_CHUNKS})",
    ]
    if failed:
        lines.append(f"- Failed documents (skipped): {', '.join(failed)}")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return total_chunks


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(only: str | None, pages_only: bool) -> int:
    """Run the ingestion and return a process exit code (0 = success)."""
    rows = load_inventory()
    order = {row["doc_id"]: position for position, row in enumerate(rows)}
    if only is not None and only not in order:
        LOGGER.error("Unknown doc_id %r (not in the inventory)", only)
        return 1
    targets = [row for row in rows if only is None or row["doc_id"] == only]

    pages = [p for p in read_jsonl(PAGES_PATH) if only is not None and p["doc_id"] != only]
    chunks = [c for c in read_jsonl(CHUNKS_PATH) if only is not None and c["doc_id"] != only]
    tokenizer = None if pages_only else load_tokenizer()
    failed: list[str] = []

    for number, row in enumerate(targets, start=1):
        LOGGER.info("[%d/%d] %s", number, len(targets), row["doc_id"])
        try:
            doc_pages = build_pages(row)
        except Exception as error:  # noqa: BLE001 - log and continue with the next PDF
            LOGGER.error("%s: skipped (%s)", row["doc_id"], error)
            failed.append(row["doc_id"])
            continue
        pages.extend(doc_pages)
        if tokenizer is not None:
            chunks.extend(build_chunks(doc_pages, row, tokenizer))

    pages.sort(key=lambda p: (order.get(p["doc_id"], 10**6), p["page"]))
    chunks.sort(key=lambda c: (order.get(c["doc_id"], 10**6), c["page"], c["chunk_id"]))

    write_jsonl(PAGES_PATH, [{k: p[k] for k in ("doc_id", "source", "page", "section", "text")} for p in pages])
    if tokenizer is not None:
        write_jsonl(CHUNKS_PATH, chunks)
    total = write_report(rows, pages, chunks if tokenizer is not None else read_jsonl(CHUNKS_PATH), failed)
    LOGGER.info("Wrote %s", REPORT_PATH)

    if tokenizer is not None and total > config.MAX_CHUNKS:
        LOGGER.error(
            "Too many chunks: %d > %d (over by %d). Drop or cut documents.",
            total, config.MAX_CHUNKS, total - config.MAX_CHUNKS,
        )
        return 2
    return 1 if failed else 0


def main() -> None:
    """Parse the command line and run the ingestion."""
    parser = argparse.ArgumentParser(description="Ingest the medical PDFs into pages and chunks.")
    parser.add_argument("--only", metavar="DOC_ID", help="re-ingest a single document")
    parser.add_argument("--pages-only", action="store_true", help="run step A only (no tokenizer)")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
    sys.exit(run(arguments.only, arguments.pages_only))


if __name__ == "__main__":
    main()
