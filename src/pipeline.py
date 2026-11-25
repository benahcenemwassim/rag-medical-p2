"""End-to-end question answering pipeline (contract C6).

Wires together retrieval (C4, ``retrieve.py``) and generation (C5, ``generate.py``):
retrieve the top-k chunks, apply the refusal gate on the dense top-1 cosine
similarity, and either refuse directly or call the LLM. Neither the retrieval
algorithms (dense/hybrid/hybrid_rerank) nor the generation/citation logic are
reimplemented here; this module only orchestrates them.

Command line (test helper, from 03_CODE/):
    python src/pipeline.py "question" --model llama3 --retrieval dense --k 5 --tau 0.6
    python src/pipeline.py "question" --no-gate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

# The public contract requires the parameter name `config` on answer(question, config),
# which would shadow the project's config module if imported as `import config`.
# Aliased here only; every other file keeps using `import config`.
import config as cfg
import generate as generate_module
import retrieve as retrieve_module


def _dense_top1_score(question: str, chunks: list[dict[str, Any]], mode: str) -> float | None:
    """Return the dense top-1 cosine similarity to use for the refusal gate.

    C4 documents ``score`` as a cosine similarity, which is only true for
    ``mode="dense"``. For "dense" we reuse the chunk already retrieved (no
    extra call). For any other mode (hybrid / hybrid_rerank, once implemented)
    ``score`` there is a fusion or rerank score, not cosine, so we fetch the
    dense top-1 separately instead of gating on the wrong number. Returns
    ``None`` when nothing can be retrieved (empty collection).
    """
    if mode == "dense":
        return chunks[0]["score"] if chunks else None
    dense_hits = retrieve_module.retrieve(question, k=1, mode="dense")
    return dense_hits[0]["score"] if dense_hits else None


def _chunks_to_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format retrieved chunks as the C6 ``sources`` list."""
    return [
        {
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "page": chunk["page"],
            "section": chunk["section"],
            "score": chunk["score"],
            "text": chunk["text"],
        }
        for chunk in chunks
    ]


def _refusal_answer() -> str:
    """The exact refusal sentence plus the disclaimer, reusing generate.py's helper
    so the disclaimer logic is not duplicated here."""
    return generate_module.add_disclaimer(cfg.REFUSAL_SENTENCE)


def answer(question: str, config: dict[str, Any]) -> dict[str, Any]:
    """Answer ``question`` end to end and return the C6 result dict (contract C6).

    ``config`` keys (all optional, defaulting from ``cfg``/project conventions):
        model            -- passed to generate() (default: cfg.DEFAULT_LLM)
        retrieval         -- "dense", "hybrid" or "hybrid_rerank" (default: "dense")
        k                 -- number of chunks to retrieve (default: cfg.TOP_K)
        tau               -- refusal threshold (default: cfg.TAU)
        use_refusal_gate  -- whether to gate before calling the LLM (default: True)

    Returns exactly:
        {question, answer, refused, top_score, sources, config, latency_s}
    """
    start = time.perf_counter()

    model = config.get("model", cfg.DEFAULT_LLM)
    retrieval_mode = config.get("retrieval", "dense")
    k = config.get("k", cfg.TOP_K)
    tau = config.get("tau", cfg.TAU)
    use_refusal_gate = config.get("use_refusal_gate", True)

    chunks = retrieve_module.retrieve(question, k=k, mode=retrieval_mode)
    top_score = _dense_top1_score(question, chunks, retrieval_mode)
    # Sources reflect what was retrieved, independent of whether the final
    # answer ends up being a refusal (needed to analyze retrieval failures).
    sources = _chunks_to_sources(chunks)

    # Empty retrieval result only forces a direct refusal when the gate is
    # enabled (nothing to gate against). With the gate disabled (e.g. baseline
    # B1), generate() is still called with the empty chunks list, same as any
    # other retrieval result. top_score can be None even with non-empty chunks
    # (hybrid mode retrieved something, but the separate dense lookup used for
    # the gate found nothing); that case also refuses rather than crashing on
    # `None < tau`.
    empty_retrieval = not chunks
    if use_refusal_gate and (
        empty_retrieval or top_score is None or top_score < tau
    ):
        refused = True
    else:
        refused = False

    if refused:
        final_answer = _refusal_answer()
    else:
        gen_result = generate_module.generate(question, chunks, model=model)
        final_answer = gen_result["answer"]
        # The gate may have passed (or been disabled), but the LLM can still
        # refuse on its own (e.g. retrieved chunks that don't actually answer
        # the question, or an empty chunks list under a disabled gate).
        refused = generate_module._is_refusal(final_answer)

    latency_s = time.perf_counter() - start

    return {
        "question": question,
        "answer": final_answer,
        "refused": refused,
        "top_score": top_score if top_score is not None else 0.0,
        "sources": sources,
        "config": {
            "model": model,
            "retrieval": retrieval_mode,
            "k": k,
            "tau": tau,
            "use_refusal_gate": use_refusal_gate,
        },
        "latency_s": latency_s,
    }


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def main() -> None:
    """Parse the command line, run one full answer() call and print the result as JSON."""
    parser = argparse.ArgumentParser(
        description="Run the full retrieve + gate + generate pipeline on one question."
    )
    parser.add_argument("question", help="the medical question to answer")
    parser.add_argument("--model", default=cfg.DEFAULT_LLM, help="llama3, llama3.2:3b, gpt-4o-mini...")
    parser.add_argument("--retrieval", default="dense", choices=["dense", "hybrid", "hybrid_rerank"])
    parser.add_argument("--k", type=int, default=cfg.TOP_K)
    parser.add_argument("--tau", type=float, default=cfg.TAU)
    parser.add_argument("--no-gate", action="store_true", help="disable the refusal gate")
    arguments = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    run_config = {
        "model": arguments.model,
        "retrieval": arguments.retrieval,
        "k": arguments.k,
        "tau": arguments.tau,
        "use_refusal_gate": not arguments.no_gate,
    }

    try:
        result = answer(arguments.question, run_config)
    except (RuntimeError, NotImplementedError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()