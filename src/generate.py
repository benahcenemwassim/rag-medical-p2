"""Build the RAG prompt, call the LLM, verify citations and enforce the disclaimer.

Contract C5 of the project README:

    question + retrieved chunks --> prompt --> LLM (Ollama or OpenAI)
                                --> {"answer", "model", "citations_valid"}

Zero trust in model output: citations are always checked in code
(``check_citations``) and the medical disclaimer is always appended in code
(``add_disclaimer``). Ollama is called with an explicit ``num_ctx`` because its
default context window silently truncates long prompts, which would make a
model that lost the retrieved chunks look like it hallucinated.

Usage (run from 03_CODE/):
    python src/generate.py "what is metformin used for" --chunks path/to/chunks.jsonl
    python src/generate.py "what is metformin used for" --plain          # baseline B0
    python src/generate.py "..." --chunks c.jsonl --model gpt-4o-mini    # needs OPENAI_API_KEY
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

PROMPT_TEMPLATE = """You are a medical assistant. Answer ONLY using the context below.
For every claim, cite the source (document, page, section).
If the context does not contain the answer, say:
"I cannot answer this question from the provided documents."

Context:
{context}

Question: {question}

Answer (with citations):"""

PLAIN_PROMPT = "Answer the following medical question concisely"

# Matches [file.pdf p.N], [file.pdf, p. N], (file.pdf p.N) and (file.pdf, p. N).
_CITATION = re.compile(
    r"[\[(]\s*([\w\-.]+\.pdf)\s*,?\s*p\.?\s*(\d+)\s*[\])]", re.IGNORECASE
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
    """True when the answer is exactly the refusal sentence (quotes tolerated)."""
    return _strip_disclaimer(answer).strip(" \t\r\n\"'") == config.REFUSAL_SENTENCE


def check_citations(answer: str, chunks: list[dict[str, Any]]) -> bool:
    """Verify every citation of ``answer`` against the chunks given to the model.

    Returns True only if (1) at least one citation is found, unless the answer
    is the exact refusal sentence (then there must be zero citations), and
    (2) every cited (file, page) pair exists in ``chunks``.
    """
    found = [(m.group(1), int(m.group(2))) for m in _CITATION.finditer(answer)]
    if _is_refusal(answer):
        return not found
    if not found:
        return False
    allowed = {(chunk["source"], int(chunk["page"])) for chunk in chunks}
    return all(pair in allowed for pair in found)


# --------------------------------------------------------------------------- #
# LLM dispatch
# --------------------------------------------------------------------------- #
def _call_ollama(prompt: str, model: str) -> str:
    """Send ``prompt`` to a local Ollama model and return the reply text."""
    import httpx
    import ollama

    start_hint = (
        f"Cannot reach Ollama. Start it (open the Ollama app or run 'ollama serve') "
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
            },
        )
        return response["message"]["content"]
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
    parser.add_argument("--model", default=config.DEFAULT_LLM, help="llama3, llama3.2:3b, gpt-4o-mini...")
    parser.add_argument("--plain", action="store_true", help="baseline B0: no context, no citations")
    arguments = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    try:
        if arguments.plain:
            result: Any = {"answer": generate_plain(arguments.question, arguments.model),
                           "model": arguments.model}
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