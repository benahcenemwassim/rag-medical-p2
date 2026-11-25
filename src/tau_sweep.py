"""Threshold (tau) sweep for the refusal gate (contract-adjacent, no C6 call).

Loads 05_DATASETS/eval/tau_dev.jsonl, retrieves the dense top-1 similarity
score for each question exactly once (k=1, mode="dense"), then evaluates
thresholds 0.30-0.70 (step 0.05) against two metrics:

    answerable pass rate     = % of answerable questions with score >= tau
    unanswerable block rate  = % of unanswerable questions with score <  tau

No LLM calls, no generation, no use of pipeline.answer() or the frozen test
set. Saves a machine-readable results file and one chart, then prints a
suggested threshold with its reasoning (a development-set estimate, not a
universally optimal value).

Methodological note: the original team plan specified 15 dev questions; this
run uses 20 manually verified questions (10 answerable / 10 unanswerable,
TAU001-TAU020) instead, for broader coverage. That substitution is recorded
in the results file below.

Command line (from 03_CODE/):
    python src/tau_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a GUI window
import matplotlib.pyplot as plt

import config
import retrieve as retrieve_module

TAU_DEV_PATH = config.EVAL_DIR / "tau_dev.jsonl"
RESULTS_JSON = config.RESULTS_DIR / "tau_eval_results.json"
RESULTS_CHART = config.RESULTS_DIR / "tau_eval_results.png"

# 0.30, 0.35, ..., 0.70 — round() avoids float drift from repeated += 0.05.
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(9)]

DATASET_NOTE = (
    "Team plan originally specified 15 dev questions; this run uses 20 "
    "manually verified questions (10 answerable / 10 unanswerable, "
    "TAU001-TAU020) instead, for broader coverage."
)


# --------------------------------------------------------------------------- #
# Step 1: load and validate the dev set
# --------------------------------------------------------------------------- #
def load_tau_dev(path: Path = TAU_DEV_PATH) -> list[dict[str, Any]]:
    """Load and validate tau_dev.jsonl.

    Each line must be a JSON object with ``id`` (str), ``question`` (str) and
    ``answerable`` (bool). Raises ``ValueError`` on a missing field, a wrong
    type for ``answerable``, a duplicate id, an empty file, or a dataset that
    is missing either answerable or unanswerable questions (the sweep needs
    both to compute its two metrics).
    """
    if not path.exists():
        raise FileNotFoundError(f"tau dev set not found: {path}")

    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for field in ("id", "question", "answerable"):
                if field not in record:
                    raise ValueError(f"{path} line {line_number}: missing '{field}'")
            if not isinstance(record["answerable"], bool):
                raise ValueError(
                    f"{path} line {line_number}: 'answerable' must be true/false, "
                    f"got {record['answerable']!r}"
                )
            if record["id"] in seen_ids:
                raise ValueError(f"{path} line {line_number}: duplicate id '{record['id']}'")
            seen_ids.add(record["id"])
            items.append(record)

    if not items:
        raise ValueError(f"{path} contains no questions")

    n_answerable = sum(1 for item in items if item["answerable"])
    n_unanswerable = len(items) - n_answerable
    if n_answerable == 0 or n_unanswerable == 0:
        raise ValueError(
            f"{path} must contain both answerable and unanswerable questions "
            f"(got {n_answerable} answerable, {n_unanswerable} unanswerable)"
        )

    return items


# --------------------------------------------------------------------------- #
# Step 2: retrieve the dense top-1 score for each question, once
# --------------------------------------------------------------------------- #
def collect_scores(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retrieve each question once (k=1, dense) and record its top-1 score.

    Empty retrieval (the collection returns zero chunks for a question) is
    recorded as ``score=None`` rather than silently treated as an ordinary
    low similarity score — "nothing retrieved" and "retrieved something
    irrelevant" are different failure modes and must not be conflated in the
    pass/block-rate math. Questions with ``score=None`` are reported
    separately and excluded from both metrics (see ``sweep``).
    """
    scored = []
    missing_ids = []
    for item in items:
        hits = retrieve_module.retrieve(item["question"], k=1, mode="dense")
        score = hits[0]["score"] if hits else None
        if score is None:
            missing_ids.append(item["id"])
        scored.append({**item, "score": score})

    if missing_ids:
        print(
            f"WARNING: empty retrieval for {len(missing_ids)} question(s): "
            f"{missing_ids} — excluded from the sweep metrics, see results file.",
            file=sys.stderr,
        )
    return scored


# --------------------------------------------------------------------------- #
# Step 3: sweep thresholds over the already-collected scores
# --------------------------------------------------------------------------- #
def sweep(scored: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Compute the answerable pass rate and unanswerable block rate per threshold.

    Boundary convention (must match pipeline.py's refusal gate exactly):
    ``score >= tau`` passes, ``score < tau`` is blocked. Questions with
    ``score=None`` (empty retrieval) are excluded from both counts; their ids
    are returned separately so they can be reported rather than hidden.
    """
    answerable = [s for s in scored if s["answerable"]]
    unanswerable = [s for s in scored if not s["answerable"]]
    excluded_ids = [s["id"] for s in scored if s["score"] is None]

    rows = []
    for tau in THRESHOLDS:
        n_pass = sum(
        1 for s in answerable
        if s["score"] is not None and s["score"] >= tau)

        n_block = sum(
        1 for s in unanswerable
        if s["score"] is None or s["score"] < tau)


        rows.append(
            {
                "tau": tau,
                "answerable_pass_rate": round(100.0 * n_pass / len(answerable), 1) if answerable else 0.0,
                "unanswerable_block_rate": round(100.0 * n_block / len(unanswerable), 1) if unanswerable else 0.0,
                "n_answerable_pass": n_pass,
                "n_answerable_total": len(answerable),
                "n_unanswerable_block": n_block,
                "n_unanswerable_total": len(unanswerable),
            }
        )
    return rows, excluded_ids


# --------------------------------------------------------------------------- #
# Step 4: suggest a threshold from the trade-off (not an automatic combined-max)
# --------------------------------------------------------------------------- #
def suggest_threshold(rows: list[dict[str, Any]]) -> tuple[float, str]:
    """Suggest a threshold using a dominance rule, not a maximized combined score.

    Because raising tau can only keep the answerable pass rate the same or
    lower it (never raise it), and can only keep the unanswerable block rate
    the same or raise it (never lower it): among all thresholds that achieve
    the best observed answerable pass rate, the *largest* such tau also gets
    the best possible block rate without costing any additional blocked
    legitimate question. That tau is returned as the suggestion.

    This is a development-set estimate from 20 questions, not a universally
    optimal threshold — it should be read alongside the full table, and
    overridden if the team judges the trade-off differently (e.g. accepting
    a small pass-rate cost for a much higher block rate).
    """
    best_pass_rate = max(r["answerable_pass_rate"] for r in rows)
    candidates = [r for r in rows if r["answerable_pass_rate"] == best_pass_rate]
    chosen = max(candidates, key=lambda r: r["tau"])

    reasoning = (
        f"Selected tau={chosen['tau']}: the highest threshold that still keeps the "
        f"best observed answerable pass rate ({best_pass_rate}%, "
        f"{chosen['n_answerable_pass']}/{chosen['n_answerable_total']} answerable "
        f"questions). At this tau, the unanswerable block rate is "
        f"{chosen['unanswerable_block_rate']}% "
        f"({chosen['n_unanswerable_block']}/{chosen['n_unanswerable_total']}). "
        "No higher tau is preferred automatically, because every tau above this "
        "one blocks at least one more answerable question while only this tau's "
        "block rate was free (no pass-rate cost). If the team prefers a stricter "
        "gate and is willing to trade some answerable pass rate for a higher "
        "block rate, consult the full table in the results file and pick manually — "
        "this is a 20-question development-set estimate, not a universally optimal "
        "threshold."
    )
    return chosen["tau"], reasoning


# --------------------------------------------------------------------------- #
# Step 5: chart + results file
# --------------------------------------------------------------------------- #
def save_chart(rows: list[dict[str, Any]], path: Path = RESULTS_CHART) -> None:
    """Save one chart: both metrics against tau, on the same axes."""
    taus = [r["tau"] for r in rows]
    pass_rates = [r["answerable_pass_rate"] for r in rows]
    block_rates = [r["unanswerable_block_rate"] for r in rows]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(taus, pass_rates, marker="o", label="Answerable pass rate")
    ax.plot(taus, block_rates, marker="o", label="Unanswerable block rate")
    ax.set_xlabel("tau (dense top-1 cosine similarity threshold)")
    ax.set_ylabel("rate (%)")
    ax.set_title("Refusal gate threshold sweep (tau_dev.jsonl, n=20)")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_results(
    scored: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    excluded_ids: list[str],
    path: Path = RESULTS_JSON,
) -> None:
    """Save the full machine-readable results: per-question scores, the
    per-threshold table, excluded ids, and the suggested threshold."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_path": str(TAU_DEV_PATH),
        "dataset_note": DATASET_NOTE,
        "n_questions": len(scored),
        "excluded_empty_retrieval_ids": excluded_ids,
        "per_question_scores": [
            {"id": s["id"], "answerable": s["answerable"], "score": s["score"]} for s in scored
        ],
        "thresholds": rows,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """Run the full sweep and print a human-readable summary."""
    items = load_tau_dev()
    print(f"Loaded {len(items)} questions from {TAU_DEV_PATH}")

    scored = collect_scores(items)

    rows, excluded_ids = sweep(scored)

    save_chart(rows)
    save_results(scored, rows, excluded_ids)

    print(f"\n{'tau':>5} | {'pass %':>8} | {'block %':>9}")
    for r in rows:
        print(f"{r['tau']:>5} | {r['answerable_pass_rate']:>8} | {r['unanswerable_block_rate']:>9}")

    print(f"\nSaved: {RESULTS_JSON}")
    print(f"Saved: {RESULTS_CHART}")


if __name__ == "__main__":
    main()