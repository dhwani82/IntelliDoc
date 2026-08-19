"""
Compare RAG prompt strategies on the existing eval set.

Templates:
  1. plain            — instruction-only grounded RAG prompt
  2. few_shot         — same rules plus 3 example Q&A pairs
  3. chain_of_thought — step-by-step reasoning, then a final answer,
                        still grounded only in retrieved context

Usage (from pdf-chat-rag/backend):
    python prompt_experiments.py <doc_id>
    python prompt_experiments.py          # uses doc_id from eval/eval_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

load_dotenv(BACKEND_ROOT / ".env")

EVAL_DIR = BACKEND_ROOT / "eval"
EVAL_SET_PATH = EVAL_DIR / "eval_set.json"
EVAL_RESULTS_PATH = EVAL_DIR / "eval_results.json"
CSV_PATH = EVAL_DIR / "prompt_comparison.csv"
MD_PATH = EVAL_DIR / "prompt_comparison.md"
JSON_PATH = EVAL_DIR / "prompt_comparison.json"
WINNER_PATH = EVAL_DIR / "prompt_strategy_winner.json"

# Updated after a comparison run. Overridden by PROMPT_STRATEGY in the environment.
DEFAULT_PROMPT_STRATEGY = os.getenv("PROMPT_STRATEGY", "few_shot").strip() or "few_shot"

STRATEGY_ORDER = ("plain", "few_shot", "chain_of_thought")

PromptBuilder = Callable[[str, str], Tuple[str, str]]

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "to", "of", "in", "on", "for", "with", "as", "at", "by",
    "from", "is", "are", "was", "were", "be", "been", "being", "it", "its",
    "into", "about", "over", "after", "before", "also", "can", "could",
    "should", "would", "may", "might", "will", "just", "not", "no", "yes",
    "using", "used", "use", "only", "context", "document", "question",
    "answer", "final", "step", "first", "second", "third", "because",
    "therefore", "thus", "based", "provided", "information", "cannot",
    "find", "does", "did", "do", "they", "them", "their", "there",
}

_REFUSAL_PHRASES = (
    "cannot find this information in the document",
    "i cannot find this information",
    "can't find this information",
    "not in the document",
    "not found in the document",
)

PLAIN_SYSTEM = """You are a helpful assistant that answers questions based solely on the provided context from a document.
If the answer cannot be found in the context, say: "I cannot find this information in the document."
Do not make up information or use knowledge outside the provided context."""

FEW_SHOT_SYSTEM = """You are a helpful assistant that answers questions based solely on the provided context from a document.
Follow the style of the examples: quote only facts that appear in the context, keep the answer concise, and if the context does not contain the answer say exactly: "I cannot find this information in the document."
Do not use knowledge outside the provided context."""

COT_SYSTEM = """You are a careful assistant that answers from retrieved document context only.
Think step by step, but every step must cite or paraphrase the context. Do not use outside knowledge.
If the context does not contain the answer, the final answer must be: "I cannot find this information in the document." """

FEW_SHOT_EXAMPLES = """
Here are examples of how to answer using only the given context.

Example 1:
Context: Atlas is a billing microservice deployed on Cloud Run in project pay-prod.
Question: Where is Atlas deployed?
Answer: Atlas is deployed on Cloud Run in project pay-prod.

Example 2:
Context: API keys for the payments service are stored in Secret Manager and mounted as environment variables.
Question: How are API keys stored?
Answer: API keys are stored in Secret Manager and mounted as environment variables.

Example 3:
Context: The runbook covers deploy steps for the billing service. It does not mention a database vendor.
Question: Which database does the billing service use?
Answer: I cannot find this information in the document.
""".strip()


def _valid_strategy(name: str) -> str:
    key = (name or "").strip().lower().replace("-", "_")
    aliases = {"cot": "chain_of_thought", "fewshot": "few_shot", "rag": "plain"}
    key = aliases.get(key, key)
    if key not in STRATEGY_ORDER:
        raise ValueError(
            f"Unknown prompt strategy {name!r}. Expected one of: {', '.join(STRATEGY_ORDER)}"
        )
    return key


def get_default_strategy() -> str:
    env = os.getenv("PROMPT_STRATEGY")
    if env and env.strip():
        return _valid_strategy(env)
    if WINNER_PATH.exists():
        try:
            with open(WINNER_PATH, encoding="utf-8") as f:
                winner = json.load(f).get("winner")
            if winner:
                return _valid_strategy(str(winner))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return _valid_strategy(DEFAULT_PROMPT_STRATEGY)


def build_plain_prompts(context: str, question: str) -> Tuple[str, str]:
    user = f"""Context from document:
{context}

Question: {question}

Answer the question using only the information from the context above. If the answer is not in the context, say so."""
    return PLAIN_SYSTEM, user


def build_few_shot_prompts(context: str, question: str) -> Tuple[str, str]:
    user = f"""{FEW_SHOT_EXAMPLES}

Now answer the real question using only the context below.

Context from document:
{context}

Question: {question}

Answer:"""
    return FEW_SHOT_SYSTEM, user


def build_chain_of_thought_prompts(context: str, question: str) -> Tuple[str, str]:
    user = f"""Context from document:
{context}

Question: {question}

Reason step by step using only the context:
1. Select the sentences (if any) that could answer the question.
2. Check whether they actually contain the requested facts.
3. Write a concise final answer after the heading "Final answer:".

If the context does not contain the answer, the final answer must be: I cannot find this information in the document.
Do not use knowledge that is not in the context."""
    return COT_SYSTEM, user


PROMPT_BUILDERS: Dict[str, PromptBuilder] = {
    "plain": build_plain_prompts,
    "few_shot": build_few_shot_prompts,
    "chain_of_thought": build_chain_of_thought_prompts,
}


def build_prompts(
    context: str,
    question: str,
    strategy: Optional[str] = None,
) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt) for the selected strategy."""
    name = _valid_strategy(strategy or get_default_strategy())
    return PROMPT_BUILDERS[name](context, question)


def extract_final_answer(text: str) -> str:
    """Prefer the 'Final answer:' / last 'Answer:' span produced by CoT/few-shot."""
    raw = (text or "").strip()
    if not raw:
        return ""
    for marker in ("Final answer:", "Final Answer:", "\nAnswer:"):
        if marker in raw:
            return raw.split(marker)[-1].strip()
    return raw


def _keyword_in_text(keyword: str, text: str) -> bool:
    return keyword.lower() in (text or "").lower()


def _any_keyword_in_text(keywords: Sequence[str], text: str) -> bool:
    return any(_keyword_in_text(k, text) for k in keywords)


def _is_refusal(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(phrase in lowered for phrase in _REFUSAL_PHRASES)


def _content_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9+#/-]*", (text or "").lower())
    cleaned = [t.strip(".-") for t in tokens if t.strip(".-")]
    return [t for t in cleaned if len(t) >= 3 and t not in _STOPWORDS]


def accuracy_hit(answer: str, expected_keywords: Sequence[str]) -> bool:
    """True when the final answer contains an expected keyword and is not a refusal."""
    final = extract_final_answer(answer)
    if _is_refusal(final):
        return False
    return _any_keyword_in_text(list(expected_keywords), final)


def groundedness_hit(
    answer: str,
    context: str,
    threshold: float = 0.7,
) -> bool:
    """
    True when the final answer's content words mostly appear in retrieved context,
    or when the model refused instead of inventing an answer.
    """
    final = extract_final_answer(answer)
    if _is_refusal(final):
        return True
    tokens = _content_tokens(final)
    if not tokens:
        return False
    blob = (context or "").lower()
    supported = sum(1 for t in tokens if t in blob)
    return (supported / len(tokens)) >= threshold


def groundedness_ratio(answer: str, context: str) -> float:
    final = extract_final_answer(answer)
    if _is_refusal(final):
        return 1.0
    tokens = _content_tokens(final)
    if not tokens:
        return 0.0
    blob = (context or "").lower()
    return round(sum(1 for t in tokens if t in blob) / len(tokens), 4)


def _is_failed_generation(answer: str) -> bool:
    lowered = (answer or "").lower()
    return (
        "couldn't generate an answer" in lowered
        or "gemini api error" in lowered
        or ("429" in lowered and "quota" in lowered)
    )


def _row_failed(row: Dict[str, Any]) -> bool:
    if row.get("failed_generation"):
        return True
    return _is_failed_generation(row.get("answer_preview") or "")


def _combined_score(accuracy: float, groundedness: float) -> float:
    return round(0.5 * accuracy + 0.5 * groundedness, 4)
    return round(0.5 * accuracy + 0.5 * groundedness, 4)


def _choose_winner(summaries: Sequence[Dict[str, Any]]) -> str:
    # Highest combined score, then groundedness, then accuracy; break remaining
    # ties with lower latency so the cheaper equally-good prompt wins.
    ranked = sorted(
        summaries,
        key=lambda row: (
            row["combined_score"],
            row["groundedness"],
            row["accuracy"],
            -float(row.get("avg_latency_seconds") or 0.0),
        ),
        reverse=True,
    )
    return ranked[0]["strategy"]


def _write_csv(summaries: Sequence[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "strategy",
        "accuracy",
        "groundedness",
        "combined_score",
        "accuracy_hits",
        "groundedness_hits",
        "num_questions",
        "avg_latency_seconds",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def _write_markdown(
    summaries: Sequence[Dict[str, Any]],
    winner: str,
    path: Path,
    note: Optional[str] = None,
) -> None:
    lines = [
        "# Prompt strategy comparison",
        "",
        f"Winner (default): **{winner}**",
        "",
        "| Strategy | Accuracy | Groundedness | Combined | Acc hits | Ground hits | Avg latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['strategy']} "
            f"| {row['accuracy']:.2%} "
            f"| {row['groundedness']:.2%} "
            f"| {row['combined_score']:.2%} "
            f"| {row['accuracy_hits']}/{row['num_questions']} "
            f"| {row['groundedness_hits']}/{row['num_questions']} "
            f"| {row['avg_latency_seconds']:.3f} |"
        )
    lines.append("")
    lines.append(
        "Accuracy: expected keyword appears in the final answer (refusals count as misses).  "
        "Groundedness: final-answer content words are supported by retrieved context, "
        "or the model refused instead of hallucinating."
    )
    if note:
        lines.extend(["", f"_{note}_"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _default_doc_id() -> Optional[str]:
    if not EVAL_RESULTS_PATH.exists():
        return None
    try:
        with open(EVAL_RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f).get("doc_id")
    except (OSError, json.JSONDecodeError):
        return None


async def run_prompt_experiments(
    doc_id: str,
    sleep_s: float = 8.0,
    generation_backend: Optional[str] = None,
) -> Dict[str, Any]:
    from chat_service import ChatService
    from local_llm import resolve_generation_backend
    from vector_store import VectorStore

    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set: List[Dict[str, Any]] = json.load(f)
    if not eval_set:
        raise SystemExit(f"Eval set at {EVAL_SET_PATH} is empty.")

    vector_store = VectorStore()
    vector_store.load_existing()
    if doc_id not in vector_store.doc_indices:
        raise SystemExit(
            f"doc_id={doc_id!r} not found. Known: {list(vector_store.doc_indices.keys())[-8:]}"
        )

    chat = ChatService()
    backend = resolve_generation_backend(generation_backend)
    if backend == "gemini" and not chat.client_ready:
        raise SystemExit("GEMINI_API_KEY is not configured.")

    per_strategy: Dict[str, List[Dict[str, Any]]] = {name: [] for name in STRATEGY_ORDER}

    for item in eval_set:
        question = item["question"]
        expected = item.get("expected_keywords") or []
        retrieved = await chat._retrieve_chunks(doc_id, question, vector_store)
        context = chat._build_context(retrieved) if retrieved else ""
        print(f"\n=== {question} ===")

        for strategy in STRATEGY_ORDER:
            system_prompt, user_prompt = build_prompts(context, question, strategy)
            t0 = time.perf_counter()
            if not retrieved:
                answer = "No relevant information found in the document."
            else:
                answer, _ = await chat._generate_answer(system_prompt, user_prompt, backend)
            latency = time.perf_counter() - t0

            acc = accuracy_hit(answer, expected)
            grounded = groundedness_hit(answer, context)
            failed = _is_failed_generation(answer)
            row = {
                "question": question,
                "expected_keywords": expected,
                "strategy": strategy,
                "accuracy_hit": acc and not failed,
                "groundedness_hit": grounded and not failed,
                "groundedness_ratio": groundedness_ratio(answer, context),
                "latency_seconds": round(latency, 3),
                "num_retrieved_chunks": len(retrieved),
                "failed_generation": failed,
                "answer_preview": (answer[:240] + "...") if len(answer) > 240 else answer,
            }
            per_strategy[strategy].append(row)
            print(
                f"  [{strategy:16}] "
                f"{'A' if row['accuracy_hit'] else '-'}{'G' if row['groundedness_hit'] else '-'} "
                f"{latency:.2f}s"
                f"{' (failed)' if failed else ''}"
            )
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

    summaries, comparable_n, total_n = build_summaries(per_strategy)
    winner = _choose_winner(summaries)
    return {
        "doc_id": doc_id,
        "generation_backend": backend,
        "num_questions": total_n,
        "comparable_questions": comparable_n,
        "winner": winner,
        "summaries": summaries,
        "per_strategy": per_strategy,
        "note": (
            None
            if comparable_n == total_n
            else (
                f"Scored {comparable_n}/{total_n} questions where all three "
                "strategies returned a real model answer (API errors excluded)."
            )
        ),
    }


def build_summaries(
    per_strategy: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Score strategies on questions where every template returned a real answer.
    That keeps a quota-truncated run from treating API errors as model misses.
    """
    questions = [row["question"] for row in per_strategy[STRATEGY_ORDER[0]]]
    comparable: List[str] = []
    for question in questions:
        rows = [
            next(r for r in per_strategy[name] if r["question"] == question)
            for name in STRATEGY_ORDER
        ]
        if not any(_row_failed(r) for r in rows):
            comparable.append(question)

    scored_questions = comparable or questions
    n = len(scored_questions) or 1
    summaries: List[Dict[str, Any]] = []
    for strategy in STRATEGY_ORDER:
        rows = [
            r for r in per_strategy[strategy]
            if r["question"] in scored_questions and not _row_failed(r)
        ]
        denom = len(rows) or 1
        acc_hits = sum(1 for r in rows if r["accuracy_hit"])
        g_hits = sum(1 for r in rows if r["groundedness_hit"])
        acc = acc_hits / denom
        grounded = g_hits / denom
        summaries.append({
            "strategy": strategy,
            "accuracy": round(acc, 4),
            "groundedness": round(grounded, 4),
            "combined_score": _combined_score(acc, grounded),
            "accuracy_hits": acc_hits,
            "groundedness_hits": g_hits,
            "num_questions": len(rows),
            "avg_latency_seconds": round(
                sum(r["latency_seconds"] for r in rows) / denom, 3
            ) if rows else 0.0,
        })
    return summaries, len(comparable), len(questions)


def persist_report(report: Dict[str, Any]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _write_csv(report["summaries"], CSV_PATH)
    _write_markdown(
        report["summaries"],
        report["winner"],
        MD_PATH,
        note=report.get("note"),
    )
    with open(WINNER_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "winner": report["winner"],
                "combined_score": next(
                    s["combined_score"]
                    for s in report["summaries"]
                    if s["strategy"] == report["winner"]
                ),
                "doc_id": report["doc_id"],
            },
            f,
            indent=2,
        )
    print(f"\nSaved {CSV_PATH}")
    print(f"Saved {MD_PATH}")
    print(f"Saved {JSON_PATH}")
    print(f"Winner ({report['winner']}) written to {WINNER_PATH}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare plain / few-shot / CoT prompts on the RAG eval set."
    )
    parser.add_argument(
        "doc_id",
        nargs="?",
        default=None,
        help="Indexed document id (defaults to eval/eval_results.json)",
    )
    parser.add_argument("--sleep", type=float, default=8.0, help="Seconds between generations")
    parser.add_argument(
        "--backend",
        default=None,
        help="gemini or local (defaults to GENERATION_BACKEND)",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Recompute winner from eval/prompt_comparison.json without calling the API",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def rescore_existing() -> Dict[str, Any]:
    if not JSON_PATH.exists():
        raise SystemExit(f"No existing report at {JSON_PATH}")
    with open(JSON_PATH, encoding="utf-8") as f:
        report = json.load(f)
    summaries, comparable_n, total_n = build_summaries(report["per_strategy"])
    report["summaries"] = summaries
    report["comparable_questions"] = comparable_n
    report["num_questions"] = total_n
    report["winner"] = _choose_winner(summaries)
    report["note"] = (
        None
        if comparable_n == total_n
        else (
            f"Scored {comparable_n}/{total_n} questions where all three "
            "strategies returned a real model answer (API errors excluded)."
        )
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.rescore:
        report = rescore_existing()
    else:
        doc_id = args.doc_id or _default_doc_id()
        if not doc_id:
            raise SystemExit("Pass a doc_id or run eval once so eval/eval_results.json exists.")
        report = asyncio.run(
            run_prompt_experiments(doc_id, sleep_s=args.sleep, generation_backend=args.backend)
        )
    persist_report(report)
    print("\n=== Prompt comparison ===")
    for row in report["summaries"]:
        print(
            f"{row['strategy']:18}  acc={row['accuracy']:.2%}  "
            f"ground={row['groundedness']:.2%}  combined={row['combined_score']:.2%}"
        )
    print(f"\nWinning strategy: {report['winner']}")


if __name__ == "__main__":
    main()
