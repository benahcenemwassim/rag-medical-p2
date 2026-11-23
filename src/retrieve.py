"""Retrieval over the medical Chroma collection (contract C4).

Only dense retrieval exists for now; "hybrid" and "hybrid_rerank" are added later.
The embedding model and the Chroma collection are loaded once (first call) and
cached, so repeated calls to ``retrieve`` stay fast.

Scores are cosine similarities: Chroma returns a distance, similarity = 1 - distance.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import chromadb
from chromadb.config import Settings

import config


@lru_cache(maxsize=1)
def get_embedder() -> Any:
    """Load BGE-M3 once: fp16 on CUDA when available, float32 on CPU."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(config.EMBED_MODEL, device=device)
    if device == "cuda":
        model = model.half()
    return model


def get_client() -> Any:
    """Open the persistent Chroma client on the configured directory (telemetry off)."""
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def collection_exists(client: Any, name: str) -> bool:
    """Return True if a collection called ``name`` exists in this Chroma database."""
    names = [getattr(item, "name", item) for item in client.list_collections()]
    return name in names


@lru_cache(maxsize=1)
def get_collection() -> Any:
    """Return the cached "medical" collection; fail clearly if it was never built.

    Only the "missing collection" case gets the friendly message: any other
    Chroma error (locked or corrupt database) is allowed to surface unchanged.
    """
    client = get_client()
    if not collection_exists(client, config.COLLECTION):
        raise RuntimeError(
            f"Chroma collection '{config.COLLECTION}' not found in {config.CHROMA_DIR}. "
            "Run: python src/index.py --rebuild"
        )
    return client.get_collection(config.COLLECTION)


def embed_query(query: str) -> list[float]:
    """Embed one query with the same model and normalisation used for the chunks."""
    vector = get_embedder().encode([query], normalize_embeddings=True)[0]
    return vector.astype("float32").tolist()


def dense_search(query: str, n: int) -> list[dict[str, Any]]:
    """Return the ``n`` most similar chunks, each with its cosine similarity as ``score``."""
    collection = get_collection()
    n = min(n, collection.count())
    if n <= 0:
        return []
    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict[str, Any]] = []
    for chunk_id, text, meta, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {
                "chunk_id": chunk_id,
                "source": meta["source"],
                "page": int(meta["page"]),
                "section": meta.get("section") or None,
                "text": text,
                "score": 1.0 - float(distance),
            }
        )
    return hits


def retrieve(query: str, k: int = config.TOP_K, mode: str = "dense") -> list[dict[str, Any]]:
    """Retrieve the top-``k`` chunks for ``query`` (contract C4).

    Each dict has exactly: chunk_id, source, page, section, text, score.
    """
    if not query or not query.strip():
        raise ValueError("The query is empty.")
    if mode == "dense":
        return dense_search(query, k)
    if mode in ("hybrid", "hybrid_rerank"):
        raise NotImplementedError(f"mode '{mode}' is not implemented yet")
    raise ValueError(f"Unknown retrieval mode: {mode!r}")