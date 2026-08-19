"""
RAG evaluation harness.

Usage (from pdf-chat-rag/backend):
    python -m eval.run_eval <doc_id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Ensure backend package root is on sys.path when run as a script
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

load_dotenv(BACKEND_ROOT / ".env")

from chat_service import ChatService  # noqa: E402
from vector_store import VectorStore  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
EVAL_SET_PATH = EVAL_DIR / "eval_set.json"
RESULTS_PATH = EVAL_DIR / "eval_results.json"


def _keyword_in_text(keyword: str, text: str) -> bool:
    return keyword.lower() in (text or "").lower()


def _any_keyword_in_text(keywords: List[str], text: str) -> bool:
    return any(_keyword_in_text(k, text) for k in keywords)


def _answer_says_cannot_find(answer: str) -> bool:
    lowered = (answer or "").lower()
    return "cannot find" in lowered or "can't find" in lowered


async def run_eval(doc_id: str) -> Dict[str, Any]:
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set: List[Dict[str, Any]] = json.load(f)

    vector_store = VectorStore()
    vector_store.load_existing()
    chat_service = ChatService()

    if doc_id not in vector_store.doc_indices:
        raise SystemExit(
            f"doc_id={doc_id!r} not found in vector store. "
            f"Known docs: {list(vector_store.doc_indices.keys())}"
        )

    per_question: List[Dict[str, Any]] = []
    retrieval_hits = 0
    grounded_hits = 0
    latencies: List[float] = []

    for item in eval_set:
        question = item["question"]
        expected_keywords = item.get("expected_keywords") or []

        retrieved = await chat_service._retrieve_chunks(doc_id, question, vector_store)
        retrieved_text = "\n".join(
            r.get("metadata", {}).get("text", "") for r in retrieved
        )
        retrieval_ok = _any_keyword_in_text(expected_keywords, retrieved_text)

        t0 = time.perf_counter()
        result = await chat_service.get_answer(doc_id, question, vector_store)
        latency_s = time.perf_counter() - t0
        latencies.append(latency_s)
        await asyncio.sleep(13)

        answer = result.get("answer", "")
        grounded_ok = (
            _any_keyword_in_text(expected_keywords, answer)
            and not _answer_says_cannot_find(answer)
        )

        if retrieval_ok:
            retrieval_hits += 1
        if grounded_ok:
            grounded_hits += 1

        row = {
            "question": question,
            "expected_keywords": expected_keywords,
            "latency_seconds": round(latency_s, 3),
            "retrieval_hit": retrieval_ok,
            "groundedness_hit": grounded_ok,
            "num_retrieved_chunks": len(retrieved),
            "answer_preview": (answer[:200] + "...") if len(answer) > 200 else answer,
        }
        per_question.append(row)
        print(
            f"[{'R' if retrieval_ok else '-'}{'G' if grounded_ok else '-'}] "
            f"{latency_s:.2f}s | {question}"
        )

    n = len(eval_set) or 1
    summary = {
        "doc_id": doc_id,
        "num_questions": len(eval_set),
        "retrieval_accuracy": round(retrieval_hits / n, 4),
        "groundedness_accuracy": round(grounded_hits / n, 4),
        "avg_latency_seconds": round(sum(latencies) / n, 3) if latencies else 0.0,
        "retrieval_hits": retrieval_hits,
        "groundedness_hits": grounded_hits,
        "results": per_question,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG eval against a document.")
    parser.add_argument("doc_id", help="Document ID previously indexed via /upload")
    args = parser.parse_args()

    summary = asyncio.run(run_eval(args.doc_id))

    print("\n=== Eval Summary ===")
    print(f"doc_id:                 {summary['doc_id']}")
    print(f"questions:              {summary['num_questions']}")
    print(f"retrieval_accuracy:     {summary['retrieval_accuracy']:.2%}")
    print(f"groundedness_accuracy:  {summary['groundedness_accuracy']:.2%}")
    print(f"avg_latency_seconds:    {summary['avg_latency_seconds']:.3f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
