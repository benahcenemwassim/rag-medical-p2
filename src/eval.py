import json
import re
from pathlib import Path
from datetime import datetime

from retrieve import retrieve


# ============================================================
# 1. CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = BASE_DIR.parent.parent / "05_DATASETS" / "processed" / "chunks.jsonl"
DEV_PATH    = BASE_DIR.parent.parent / "05_DATASETS" / "eval" / "dev.jsonl"
REPORT_TXT  = BASE_DIR.parent.parent / "05_DATASETS" / "eval" / "retrieval_comparison.txt"
REPORT_JSON = BASE_DIR.parent.parent / "05_DATASETS" / "eval" / "retrieval_eval.json"

TOP_K = 5


# ============================================================
# 2. NORMALIZATION + FACT MATCHING
# ============================================================

_OP = {
    "\u2265": ">=", "\u2264": "<=", "\u2A7E": ">=", "\u2A7D": "<=",
    "at least": ">=", "no less than": ">=", "at most": "<=",
    "greater than or equal to": ">=", "less than or equal to": "<=",
}
_NUMWORD = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def norm(s: str) -> str:
    s = s.lower()
    for k, v in _OP.items():
        s = s.replace(k, v)
    for w, d in _NUMWORD.items():
        s = re.sub(rf"\b{w}\b", d, s)
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    s = re.sub(r"(?<=[a-z])-(?=[a-z])", "", s)         # isoniazid-resistant -> isoniazidresistant
    s = re.sub(r"(\d)\s*(mmhg|mg|ml|%)", r"\1\2", s)   # 130 mmHg -> 130mmhg
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def fact_in_text(fact_text: str, chunk_text: str) -> bool:
    return norm(fact_text) in norm(chunk_text)


def score_chunk(chunk: dict, required_facts: list) -> tuple[list, list]:
    """Return (supported_ids, missing_ids) for one chunk."""
    if not required_facts:
        return [], []
    text = chunk.get("text", "") or ""
    supported, missing = [], []
    for f in required_facts:
        (supported if fact_in_text(f["text"], text) else missing).append(f["id"])
    return supported, missing


# ============================================================
# 3. PER-QUESTION EVAL
# ============================================================
# status:  "pass"  | "fail" | "error" | "skip"
#   pass  : a retrieved chunk supports every required_fact
#   fail  : retrieval ran, no chunk supports every required_fact
#   error : retrieve() raised — not a quality signal, tracked separately
#   skip  : question has no required_facts (schema issue, not a retrieval miss)

def eval_question(q: dict, retrieved: list, error: Exception | None = None) -> dict:
    base = {
        "id": q["id"],
        "status": None,
        "first_support_rank": None,
        "missing": [],
        "per_chunk": [],
    }

    if error is not None:
        base["status"] = "error"
        base["error"] = f"{type(error).__name__}: {error}"
        return base

    req = q.get("required_facts") or []
    if not req:
        base["status"] = "skip"
        base["note"] = "no required_facts"
        return base

    first_rank = None
    best_missing = [f["id"] for f in req]

    for rank, c in enumerate(retrieved, 1):
        sup, miss = score_chunk(c, req)
        base["per_chunk"].append({
            "rank": rank,
            "source": c.get("source"),
            "page": c.get("page"),
            "supported": sup,
            "missing": miss,
        })
        if not miss and first_rank is None:
            first_rank = rank
            best_missing = []
        elif first_rank is None and len(miss) < len(best_missing):
            best_missing = miss

    base["status"] = "pass" if first_rank is not None else "fail"
    base["first_support_rank"] = first_rank
    base["missing"] = best_missing
    return base


# ============================================================
# 4. WRITERS
# ============================================================

def write_chunk(file, chunk):
    if chunk is None:
        file.write("No matching chunk found.\n")
        return
    file.write(f"Source: {chunk.get('source', 'Unknown')}\n")
    file.write(f"Page:   {chunk.get('page', 'Unknown')}\n")
    file.write("-" * 80 + "\n")
    file.write((chunk.get("text", "No text available") or "") + "\n")
    file.write("-" * 80 + "\n")


def write_retrieved(file, rank, chunk, required_facts):
    file.write(f"\n--- RANK {rank} ---\n")
    write_chunk(file, chunk)
    if required_facts:
        sup, miss = score_chunk(chunk, required_facts)
        file.write(f"Facts OK     : {sup or '—'}\n")
        if miss:
            file.write(f"Facts MISSING: {miss}\n")


# ============================================================
# 5. MAIN
# ============================================================

def main():

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    with open(DEV_PATH, encoding="utf-8-sig") as f:
        dev = [json.loads(line) for line in f if line.strip()]

    summary = []

    with open(REPORT_TXT, "w", encoding="utf-8") as report:

        report.write("MEDICAL RAG - FACT-PRESENCE EVALUATION\n")
        report.write("=" * 100 + "\n")
        report.write(f"Run:          {datetime.now().isoformat(timespec='seconds')}\n")
        report.write(f"Total chunks: {len(chunks)}\n")
        report.write(f"Total dev Qs: {len(dev)}\n")

        evaluated = 0

        for q in dev:
            if not q.get("answerable", True):
                continue
            evaluated += 1

            req = q.get("required_facts") or []

            report.write("\n\n" + "=" * 100 + "\n")
            report.write(f"QUESTION [{q['id']}]: {q['question']}\n")
            report.write(f"\nGOLD ANSWER: {q['gold_answer']}\n")

            if req:
                report.write("\nREQUIRED FACTS:\n")
                for f in req:
                    report.write(f"  {f['id']} [{f.get('type','?')}] {f['text']}\n")

            report.write("\n\n### RETRIEVED CHUNKS (fact-gated) ###\n")

            retrieved, error = [], None
            try:
                retrieved = retrieve(q["question"], k=TOP_K, mode="dense")
            except Exception as e:
                error = e
                report.write(f"ERROR in retrieve(): {type(e).__name__}: {e}\n")

            if error is None:
                if not retrieved:
                    report.write("No chunks retrieved.\n")
                else:
                    for rank, c in enumerate(retrieved, 1):
                        write_retrieved(report, rank, c, req)

            result = eval_question(q, retrieved, error)
            summary.append(result)

            if result["status"] == "pass":
                report.write(f"\n>>> VERDICT: PASS  (first fully-supporting chunk: rank {result['first_support_rank']})\n")
            elif result["status"] == "fail":
                report.write(f"\n>>> VERDICT: FAIL  (missing facts: {result['missing']})\n")
            elif result["status"] == "error":
                report.write(f"\n>>> VERDICT: ERROR  ({result['error']})\n")
            else:  # skip
                report.write(f"\n>>> VERDICT: SKIP  ({result.get('note','no facts')})\n")

        # ---------------- summary ----------------
        n_pass  = sum(1 for s in summary if s["status"] == "pass")
        n_fail  = sum(1 for s in summary if s["status"] == "fail")
        n_error = sum(1 for s in summary if s["status"] == "error")
        n_skip  = sum(1 for s in summary if s["status"] == "skip")
        n_scored = n_pass + n_fail            # denominator for answer_support@k

        report.write("\n\n" + "=" * 100 + "\n")
        report.write("SUMMARY\n")
        report.write("=" * 100 + "\n")
        report.write(f"{'ID':<10}{'status':<8}{'rank':<7}{'missing / error'}\n")
        report.write("-" * 80 + "\n")
        for s in summary:
            detail = ""
            if s["status"] == "fail":
                detail = ",".join(s["missing"])
            elif s["status"] == "error":
                detail = s["error"]
            elif s["status"] == "skip":
                detail = s.get("note", "")
            report.write(
                f"{s['id']:<10}"
                f"{s['status']:<8}"
                f"{str(s['first_support_rank'] or '-'):<7}"
                f"{detail}\n"
            )
        report.write("-" * 80 + "\n")
        report.write(f"pass  : {n_pass}\n")
        report.write(f"fail  : {n_fail}\n")
        report.write(f"error : {n_error}\n")
        report.write(f"skip  : {n_skip}\n")
        if n_scored:
            report.write(f"answer_support@k : {n_pass}/{n_scored}  ({100*n_pass/n_scored:.0f}%)"
                         f"  [excludes {n_error} error, {n_skip} skip]\n")
        else:
            report.write("answer_support@k : n/a (no scorable questions)\n")
        report.write(f"\nEvaluated questions: {evaluated}\n")

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run": datetime.now().isoformat(timespec="seconds"),
                "metrics": {
                    "pass": n_pass,
                    "fail": n_fail,
                    "error": n_error,
                    "skip": n_skip,
                    "scored": n_scored,
                    "answer_support_at_k": n_pass / n_scored if n_scored else None,
                },
                "questions": summary,
            },
            f, indent=2, ensure_ascii=False,
        )

    print(f"Done. {n_pass}/{n_scored} scored questions passed "
          f"({n_error} error, {n_skip} skip).")
    print(f"Text report: {REPORT_TXT}")
    print(f"JSON report: {REPORT_JSON}")


if __name__ == "__main__":
    main()