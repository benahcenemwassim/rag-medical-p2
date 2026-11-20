"""Download the PDFs listed in pdf_inventory.csv into 05_DATASETS/pdfs/.

A file that already exists is never overwritten. A download is accepted only if it
starts with the bytes ``%PDF`` and is larger than 20 KB (websites sometimes answer
with an HTML page). Failures are written to 07_EXTRA/results/download_failures.txt
so they can be downloaded by hand in a browser.

Usage (run from 03_CODE/):
    python src/download_pdfs.py
    python src/download_pdfs.py --only DOC_ID
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

import config

INVENTORY_PATH = config.DATASETS / "pdf_inventory.csv"
FAILURES_PATH = config.RESULTS_DIR / "download_failures.txt"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*;q=0.8",
}
RETRIES = 3
TIMEOUT_SECONDS = 60
PAUSE_SECONDS = 1.0
MIN_BYTES = 20 * 1024


def load_inventory(path: Path = INVENTORY_PATH) -> list[dict[str, str]]:
    """Read the inventory CSV (contract C1) and return its rows."""
    if not path.exists():
        raise FileNotFoundError(f"Inventory not found: {path}")
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_pdf(content: bytes) -> str | None:
    """Return None when ``content`` looks like a real PDF, else the reason it does not."""
    if content.lstrip()[:1] == b"<":
        return "got an HTML page instead of a PDF"
    if not content.startswith(b"%PDF"):
        return "content does not start with %PDF"
    if len(content) <= MIN_BYTES:
        return f"file too small ({len(content)} bytes)"
    return None


def fetch(url: str) -> tuple[bytes | None, str]:
    """GET ``url`` with up to RETRIES attempts; return (content, reason_if_failed)."""
    reason = "unknown error"
    for attempt in range(1, RETRIES + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS, allow_redirects=True)
            response.raise_for_status()
            problem = validate_pdf(response.content)
            if problem is None:
                return response.content, ""
            reason = problem
        except requests.RequestException as error:
            reason = f"{type(error).__name__}: {error}"
        if attempt < RETRIES:
            time.sleep(2 * attempt)
    return None, reason


def download_all(rows: list[dict[str, str]]) -> tuple[list[str], list[str], list[tuple[str, str, str]]]:
    """Download every missing PDF. Return (downloaded, skipped, failures)."""
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    skipped: list[str] = []
    failures: list[tuple[str, str, str]] = []
    for number, row in enumerate(rows, start=1):
        doc_id, url = row["doc_id"], row["url"]
        destination = config.PDF_DIR / row["file_name"]
        if destination.exists():
            skipped.append(doc_id)
            continue
        print(f"[{number}/{len(rows)}] {doc_id}")
        content, reason = fetch(url)
        if content is None:
            failures.append((doc_id, url, reason))
            print(f"  FAILED: {reason}")
        else:
            partial = destination.with_suffix(".part")
            partial.write_bytes(content)
            partial.replace(destination)  # a half-written file never keeps the final name
            downloaded.append(doc_id)
        time.sleep(PAUSE_SECONDS)
    return downloaded, skipped, failures


def write_failures(failures: list[tuple[str, str, str]]) -> None:
    """Write the failure list (doc_id, url, reason) to download_failures.txt."""
    FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILURES_PATH, "w", encoding="utf-8") as handle:
        for doc_id, url, reason in failures:
            handle.write(f"{doc_id}\t{url}\t{reason}\n")


def main() -> None:
    """Parse the command line, download the PDFs and print a summary."""
    parser = argparse.ArgumentParser(description="Download the PDFs listed in the inventory.")
    parser.add_argument("--only", metavar="DOC_ID", help="download a single document")
    arguments = parser.parse_args()

    rows = load_inventory()
    if arguments.only:
        rows = [row for row in rows if row["doc_id"] == arguments.only]
        if not rows:
            sys.exit(f"Unknown doc_id: {arguments.only}")

    downloaded, skipped, failures = download_all(rows)
    write_failures(failures)
    attempted = len(downloaded) + len(failures)
    print(f"downloaded {len(downloaded)}/{attempted}, skipped (already present): {len(skipped)}, "
          f"failed: {[doc_id for doc_id, _, _ in failures]}")
    if failures:
        print(f"Details in {FAILURES_PATH}. Download those files by hand into {config.PDF_DIR}.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
