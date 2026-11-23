"""Embed every chunk with BGE-M3 and load it into the "medical" Chroma collection.

Reads  05_DATASETS/processed/chunks.jsonl  (contract C2).
Writes the Chroma database and  <CHROMA_DIR>/index_meta.json.

We always pass our OWN embeddings to Chroma: its default embedding model is a
different one, and mixing models silently ruins retrieval.

Usage (run from 03_CODE/):
    python src/index.py               # stops if the collection already exists
    python src/index.py --rebuild     # delete and rebuild the collection

Never run this while the app or any process using the same Chroma directory is
open: Chroma locks its files on Windows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from importlib import metadata
from pathlib import Path
from typing import Any

import config
from retrieve import collection_exists, get_client, get_embedder

CHUNKS_PATH = config.PROCESSED_DIR / "chunks.jsonl"
META_PATH = config.CHROMA_DIR / "index_meta.json"
ADD_BATCH = 500
ENCODE_BATCH = 32
SANITY_QUERY = "What are the recommended treatments?"


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict[str, Any]]:
    """Read chunks.jsonl (one JSON object per line)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python src/ingest.py")
    with open(path, encoding="utf-8") as handle:
        chunks = [json.loads(line) for line in handle if line.strip()]
    if not chunks:
        raise ValueError(f"{path} is empty")
    return chunks


def check_unique_ids(chunks: list[dict[str, Any]]) -> None:
    """Fail early with a readable message if two chunks share a chunk_id."""
    seen: set[str] = set()
    for chunk in chunks:
        if chunk["chunk_id"] in seen:
            raise ValueError(f"Duplicate chunk_id in chunks.jsonl: {chunk['chunk_id']}")
        seen.add(chunk["chunk_id"])


def build_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata for one chunk (Chroma rejects None, so null section -> "")."""
    year = str(chunk.get("year", "")).strip()
    return {
        "doc_id": chunk["doc_id"],
        "source": chunk["source"],
        "page": int(chunk["page"]),
        "section": chunk.get("section") or "",
        "year": int(year) if year.isdigit() else 0,
        "n_tokens": int(chunk["n_tokens"]),
    }


def write_meta(count: int) -> None:
    """Write index_meta.json (model, date, chunk parameters, count, package versions)."""
    info = {
        "embedding_model": config.EMBED_MODEL,
        "date": date.today().isoformat(),
        "chunk_tokens": config.CHUNK_TOKENS,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "total_chunks": count,
        "chromadb_version": metadata.version("chromadb"),
        "sentence_transformers_version": metadata.version("sentence-transformers"),
    }
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)


def build_index(rebuild: bool) -> int:
    """Embed all chunks and fill the collection. Return a process exit code."""
    chunks = load_chunks()
    check_unique_ids(chunks)
    client = get_client()
    exists = collection_exists(client, config.COLLECTION)

    if exists and not rebuild:
        print(f"Collection '{config.COLLECTION}' already exists. "
              "Use --rebuild to replace it. Nothing was changed.")
        return 0

    # Encode BEFORE touching the database: if this step crashes (e.g. GPU out of
    # memory) a previously working index is still intact.
    start = time.perf_counter()
    model = get_embedder()
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=ENCODE_BATCH,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    if exists:
        client.delete_collection(config.COLLECTION)
    collection = client.create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})

    try:
        for begin in range(0, len(chunks), ADD_BATCH):
            part = chunks[begin : begin + ADD_BATCH]
            collection.add(
                ids=[chunk["chunk_id"] for chunk in part],
                documents=[chunk["text"] for chunk in part],
                metadatas=[build_metadata(chunk) for chunk in part],
                embeddings=embeddings[begin : begin + ADD_BATCH].tolist(),
            )
        count = collection.count()
        if count != len(chunks):
            raise RuntimeError(f"Chroma holds {count} chunks, expected {len(chunks)}")
    except Exception:
        # Never leave a half-built collection behind: the next run would see it
        # as "already exists" and refuse to build.
        client.delete_collection(config.COLLECTION)
        raise

    query_vector = model.encode([SANITY_QUERY], normalize_embeddings=True)[0].tolist()
    top = collection.query(query_embeddings=[query_vector], n_results=1,
                           include=["documents", "metadatas", "distances"])
    meta = top["metadatas"][0][0]
    print(f"Sanity query: {SANITY_QUERY!r}")
    print(f"  top: {meta['source']} p.{meta['page']}  similarity={1 - top['distances'][0][0]:.3f}")
    print(f"  {top['documents'][0][0][:200]!r}")

    write_meta(count)
    print(f"Indexed {count} chunks in {time.perf_counter() - start:.1f}s -> {config.CHROMA_DIR}")
    return 0


def main() -> None:
    """Parse the command line and build the index."""
    parser = argparse.ArgumentParser(description="Embed the chunks and build the Chroma index.")
    parser.add_argument("--rebuild", action="store_true",
                        help="delete the existing collection and rebuild it")
    arguments = parser.parse_args()
    sys.exit(build_index(arguments.rebuild))


if __name__ == "__main__":
    main()