"""
Compare base MiniLM vs fine-tuned embeddings on the RAG eval set.

Metric: top-k retrieval hit rate. A query is a hit if any expected keyword
appears in the concatenated text of the top-k retrieved chunks.

This script embeds the existing corpus with each model directly (it does not
reuse the Gemini FAISS index), so the comparison is isolated to embedding quality.

Usage (from pdf-chat-rag/backend):
    python -m eval.compare_embeddings
    python -m eval.compare_embeddings --top-k 5 --doc_id <id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from finetune_embeddings import (  # noqa: E402
    BASE_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    load_corpus_chunks,
)

EVAL_DIR = Path(__file__).resolve().parent
EVAL_SET_PATH = EVAL_DIR / "eval_set.json"
RESULTS_PATH = EVAL_DIR / "embedding_compare_results.json"


def _keyword_in_text(keyword: str, text: str) -> bool:
    return keyword.lower() in (text or "").lower()


def _any_keyword_in_text(keywords: Sequence[str], text: str) -> bool:
    return any(_keyword_in_text(k, text) for k in keywords)


def _load_eval_set(path: Path = EVAL_SET_PATH) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        eval_set = json.load(f)
    if not isinstance(eval_set, list) or not eval_set:
        raise ValueError(f"Eval set at {path} is empty.")
    return eval_set


def _filter_chunks(chunks: List[Dict], doc_id: Optional[str]) -> List[Dict]:
    if not doc_id:
        return chunks
    filtered = [c for c in chunks if str(c.get("doc_id")) == str(doc_id)]
    if not filtered:
        known = sorted({str(c.get("doc_id")) for c in chunks if c.get("doc_id")})
        raise SystemExit(
            f"doc_id={doc_id!r} not found in corpus metadata. Known docs: {known}"
        )
    return filtered


def _load_model(model_path: str):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading {model_path!r} on {device}")
    return SentenceTransformer(model_path, device=device)


def evaluate_model(
    model,
    model_label: str,
    eval_set: Sequence[Dict[str, Any]],
    corpus_texts: Sequence[str],
    top_k: int,
) -> Dict[str, Any]:
    """Encode corpus + queries and compute top-k keyword hit rate."""
    print(f"[INFO] Encoding {len(corpus_texts)} chunks with {model_label}")
    corpus_emb = model.encode(
        list(corpus_texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    queries = [item["question"] for item in eval_set]
    print(f"[INFO] Encoding {len(queries)} eval queries with {model_label}")
    query_emb = model.encode(
        queries,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    k = min(top_k, len(corpus_texts))
    # Cosine similarity because both sides are L2-normalized.
    scores = query_emb @ corpus_emb.T
    ranked = np.argsort(-scores, axis=1)[:, :k]

    hits = 0
    per_question: List[Dict[str, Any]] = []
    for i, item in enumerate(eval_set):
        top_idx = ranked[i]
        retrieved_text = "\n".join(corpus_texts[int(j)] for j in top_idx)
        expected = item.get("expected_keywords") or []
        hit = _any_keyword_in_text(expected, retrieved_text)
        if hit:
            hits += 1
        per_question.append({
            "question": item["question"],
            "expected_keywords": expected,
            "hit": hit,
            "top_k_preview": retrieved_text[:240].replace("\n", " "),
        })
        print(f"[{'H' if hit else '-'}] {model_label} | {item['question']}")

    n = len(eval_set) or 1
    hit_rate = hits / n
    return {
        "model": model_label,
        "top_k": k,
        "num_questions": len(eval_set),
        "num_chunks": len(corpus_texts),
        "hits": hits,
        "hit_rate": round(hit_rate, 4),
        "results": per_question,
    }


def compare(
    top_k: int = 5,
    doc_id: Optional[str] = None,
    base_model: str = BASE_MODEL_NAME,
    finetuned_path: Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Any]:
    eval_set = _load_eval_set()
    chunks = _filter_chunks(load_corpus_chunks(), doc_id)
    corpus_texts = [c["text"] for c in chunks]

    base = _load_model(base_model)
    base_summary = evaluate_model(base, "base", eval_set, corpus_texts, top_k)

    finetuned_path = Path(finetuned_path)
    if not finetuned_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found at {finetuned_path}. "
            "Run `python finetune_embeddings.py` first."
        )
    finetuned = _load_model(str(finetuned_path))
    ft_summary = evaluate_model(finetuned, "finetuned", eval_set, corpus_texts, top_k)

    delta = round(ft_summary["hit_rate"] - base_summary["hit_rate"], 4)
    comparison = {
        "doc_id": doc_id,
        "top_k": top_k,
        "num_questions": len(eval_set),
        "num_chunks": len(corpus_texts),
        "base": {
            "model": base_model,
            "hits": base_summary["hits"],
            "hit_rate": base_summary["hit_rate"],
        },
        "finetuned": {
            "model": str(finetuned_path),
            "hits": ft_summary["hits"],
            "hit_rate": ft_summary["hit_rate"],
        },
        "hit_rate_delta": delta,
        "per_question": {
            "base": base_summary["results"],
            "finetuned": ft_summary["results"],
        },
    }
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare base vs fine-tuned embedding top-k hit rate."
    )
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval cutoff (default 5)")
    parser.add_argument(
        "--doc_id",
        default=None,
        help="Limit the corpus to one indexed document (default: all chunks)",
    )
    parser.add_argument("--base-model", default=BASE_MODEL_NAME)
    parser.add_argument("--finetuned-path", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    summary = compare(
        top_k=args.top_k,
        doc_id=args.doc_id,
        base_model=args.base_model,
        finetuned_path=Path(args.finetuned_path),
    )

    print("\n=== Embedding comparison (top-k hit rate) ===")
    print(f"questions:           {summary['num_questions']}")
    print(f"chunks:              {summary['num_chunks']}")
    print(f"top_k:               {summary['top_k']}")
    print(f"base hit rate:       {summary['base']['hit_rate']:.2%} ({summary['base']['hits']}/{summary['num_questions']})")
    print(
        f"finetuned hit rate:  {summary['finetuned']['hit_rate']:.2%} "
        f"({summary['finetuned']['hits']}/{summary['num_questions']})"
    )
    print(f"delta (ft - base):   {summary['hit_rate_delta']:+.2%}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
