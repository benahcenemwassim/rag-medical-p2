"""Answer a medical question from retrieved chunks (contract C5).

Builds the Listing-23 prompt from the chunks, calls the language model (Ollama
for local models, the OpenAI API for ``gpt-*`` models), verifies in code that
every citation points to a chunk that was really given to the model, and appends
the medical disclaimer. The model's own output is never trusted for any of these.

Also provides ``generate_plain`` for baseline B0 (no context, no citations).

Command line (test helper, from 03_CODE/):
    python src/generate.py "question" --chunks some_chunks.jsonl [--model llama3]
    python src/generate.py "question" --plain
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import config

PROMPT_TEMPLATE = """You are a strict medical question-answering assistant.
Answer the user's question using ONLY the retrieved context below.

RULES:
1. Cite every factual claim using EXACTLY this format: [DOC001.pdf p.12]
2. NEVER use academic reference numbers found in the text (such as [1] or [12]).
3. If the context does not contain the answer, output EXACTLY: "I cannot answer this question from the provided documents."
4. If only part of the question can be answered, explain what is known and what is missing.

EXAMPLES OF CORRECT BEHAVIOR:
- Answer: "The target blood glucose is <7 mmol/L [DOC001.pdf p.10]."
- Refusal: "I cannot answer this question from the provided documents."

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

Answer:"""

PLAIN_PROMPT = "Answer the following medical question concisely"

# A citation is a PDF file name followed, within a few characters, by a page
# marker and a number. This accepts the variants models actually write:
#   [DOC002.pdf p.17]   (DOC002.pdf, p. 17, section 3.2)   Source: DOC002.pdf, page 17
_CITATION = re.compile(
    r"([\w\-.]+\.pdf)\W{0,3}(?:pp?|pages?)\.?\s*(\d+)", re.IGNORECASE
)


# --------------------------------------------------------------------------- #
# Prompt construction and post-processing
# --------------------------------------------------------------------------- #
def build_context(chunks: list[dict[str, Any]]) -> str:
    """Format chunks as ``[<source> p.<page>] (section: <section>) <text>``.

    One block per chunk, separated by a blank line, in the retrieved order.
    A missing section is written as ``unknown``.
    """
    blocks = []
    for chunk in chunks:
        section = chunk.get("section") or "unknown"
        blocks.append(
            f"[{chunk['source']} p.{chunk['page']}] (section: {section}) {chunk['text']}"
        )
    return "\n\n".join(blocks)


def add_disclaimer(text: str) -> str:
    """Append the medical disclaimer to ``text`` (idempotent)."""
    if config.DISCLAIMER in text:
        return text
    return f"{text.rstrip()}\n\n{config.DISCLAIMER}"


def _strip_disclaimer(text: str) -> str:
    """Remove the disclaimer (if present) so it never counts as answer content."""
    return text.replace(config.DISCLAIMER, "").strip()


def _is_refusal(answer: str) -> bool:
    """True when the answer contains the refusal sentence and cites nothing."""
    text = _strip_disclaimer(answer)
    return config.REFUSAL_SENTENCE in text and not _CITATION.search(text)


def check_citations(answer: str, chunks: list[dict[str, Any]]) -> bool:
    """Verify every citation of ``answer`` against the chunks given to the model.

    A refusal (refusal sentence, no citation) is valid. Any other answer is valid
    only if it contains at least one citation and every cited (file, page) pair
    exists in ``chunks``. File names are compared case-insensitively.

    This checks that the cited pages were retrieved, not that the claim is
    actually supported by them (that is what the RAGAS faithfulness metric is for).
    """
    if _is_refusal(answer):
        return True
    found = [(m.group(1).lower(), int(m.group(2))) for m in _CITATION.finditer(answer)]
    if not found:
        return False
    allowed = {(chunk["source"].lower(), int(chunk["page"])) for chunk in chunks}
    return all(pair in allowed for pair in found)


# --------------------------------------------------------------------------- #
# LLM dispatch
# --------------------------------------------------------------------------- #
def _call_ollama(prompt: str, model: str) -> str:
    """Send ``prompt`` to a local Ollama model and return the reply text."""
    import httpx
    import ollama

    start_hint = (
        "Cannot reach Ollama. Start it (open the Ollama app or run 'ollama serve') "
        f"and make sure the model is available: 'ollama pull {model}'."
    )
    try:
        client = ollama.Client(timeout=config.LLM_TIMEOUT_S)
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": config.LLM_TEMPERATURE,
                "seed": config.LLM_SEED,
                "num_ctx": config.LLM_NUM_CTX,
                "num_predict": config.LLM_NUM_PREDICT,
            },
        )
    except ollama.ResponseError as error:
        if getattr(error, "status_code", None) == 404:
            raise RuntimeError(
                f"Model '{model}' is not installed. Run: ollama pull {model}"
            ) from None
        raise RuntimeError(f"Ollama returned an error: {error}") from None
    except httpx.TimeoutException:
        raise RuntimeError(
            f"Ollama did not answer within {config.LLM_TIMEOUT_S}s (model '{model}')."
        ) from None
    except (ConnectionError, httpx.TransportError):
        raise RuntimeError(start_hint) from None

    # Ollama silently drops the START of a prompt that does not fit the context.
    used = getattr(response, "prompt_eval_count", None)
    if used is not None and used >= config.LLM_NUM_CTX - 256:
        print(
            f"WARNING: prompt used {used} of {config.LLM_NUM_CTX} tokens; "
            "the context may have been truncated.",
            file=sys.stderr,
        )
    return response["message"]["content"]


def _call_openai(prompt: str, model: str) -> str:
    """Send ``prompt`` to the OpenAI API and return the reply text."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set the environment variable "
            "(PowerShell: $env:OPENAI_API_KEY = 'sk-...') and retry."
        )
    import openai

    try:
        client = openai.OpenAI(
            api_key=api_key,
            timeout=config.LLM_TIMEOUT_S,
            max_retries=config.OPENAI_MAX_RETRIES,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.LLM_TEMPERATURE,
            seed=config.LLM_SEED,
            max_tokens=config.LLM_NUM_PREDICT,
        )
        return response.choices[0].message.content or ""
    except openai.OpenAIError as error:
        raise RuntimeError(f"OpenAI call failed after retries: {error}") from None


def _call_llm(prompt: str, model: str) -> str:
    """Route the prompt to OpenAI (gpt-* models) or to Ollama (everything else)."""
    if model.startswith("gpt-"):
        return _call_openai(prompt, model)
    return _call_ollama(prompt, model)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate(
    question: str, chunks: list[dict[str, Any]], model: str = config.DEFAULT_LLM
) -> dict[str, Any]:
    """Answer ``question`` from ``chunks`` and return the C5 result dict.

    Returns exactly ``{"answer": str, "model": str, "citations_valid": bool}``.
    The answer always ends with the disclaimer; citations are verified in code.
    """
    prompt = PROMPT_TEMPLATE.format(context=build_context(chunks), question=question)
    raw_answer = _call_llm(prompt, model).strip()
    return {
        "answer": add_disclaimer(raw_answer),
        "model": model,
        "citations_valid": check_citations(raw_answer, chunks),
    }


def generate_plain(question: str, model: str = config.DEFAULT_LLM) -> str:
    """Baseline B0: no context, no citations, no disclaimer. Returns the raw reply."""
    return _call_llm(f"{PLAIN_PROMPT}: {question}", model).strip()


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def load_chunks(path: Path) -> list[dict[str, Any]]:
    """Read a chunks .jsonl file and drop the retrieval ``score`` key."""
    chunks = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunk = json.loads(line)
                chunk.pop("score", None)
                chunks.append(chunk)
    return chunks


def main() -> None:
    """Parse the command line, run one generation and print the result as JSON."""
    parser = argparse.ArgumentParser(
        description="Generate a cited, disclaimer-stamped answer from retrieved chunks."
    )
    parser.add_argument("question", help="the medical question to answer")
    parser.add_argument("--chunks", type=Path, help="JSON-lines file of retrieved chunks")
    parser.add_argument(
        "--model", default=config.DEFAULT_LLM, help="llama3, llama3.2:3b, gpt-4o-mini..."
    )
    parser.add_argument("--plain", action="store_true", help="baseline B0: no context, no citations")
    arguments = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    try:
        if arguments.plain:
            result: Any = {
                "answer": generate_plain(arguments.question, arguments.model),
                "model": arguments.model,
            }
        else:
            if arguments.chunks is None:
                parser.error("--chunks is required unless --plain is used")
            result = generate(arguments.question, load_chunks(arguments.chunks), arguments.model)
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
