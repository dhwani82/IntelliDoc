"""
Fine-tune a sentence-transformers embedding model on this project's document corpus.

Loads all-MiniLM-L6-v2, builds (query, document) pairs from indexed chunks
(synthetic queries if no labeled pairs file is present), trains with
MultipleNegativesRankingLoss, and writes weights to ./models/finetuned-embeddings.

Usage (from pdf-chat-rag/backend):
    python finetune_embeddings.py
    python finetune_embeddings.py --epochs 2 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

load_dotenv(BACKEND_ROOT / ".env")

BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "models" / "finetuned-embeddings"
METADATA_PATH = BACKEND_ROOT / "data" / "metadata.json"
LABELED_PAIRS_CANDIDATES = (
    BACKEND_ROOT / "eval" / "labeled_pairs.json",
    BACKEND_ROOT / "data" / "labeled_pairs.json",
)

MIN_CHUNK_CHARS = 40
MIN_QUERY_CHARS = 12
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "to", "of", "in", "on", "for", "with", "as", "at", "by",
    "from", "is", "are", "was", "were", "be", "been", "being", "it", "its",
    "into", "about", "over", "after", "before", "between", "through", "during",
    "without", "within", "also", "can", "could", "should", "would", "may",
    "might", "will", "just", "not", "no", "yes",
}


def load_corpus_chunks(metadata_path: Path = METADATA_PATH) -> List[Dict]:
    """Load chunk dicts (must include `text`) from the existing FAISS metadata dump."""
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No indexed corpus at {metadata_path}. Upload and process a PDF first "
            "so data/metadata.json exists."
        )
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, list) or not metadata:
        raise ValueError(f"Corpus at {metadata_path} is empty. Index a document first.")
    chunks = []
    for row in metadata:
        text = " ".join(str((row or {}).get("text") or "").split()).strip()
        if len(text) < MIN_CHUNK_CHARS:
            continue
        item = dict(row)
        item["text"] = text
        chunks.append(item)
    if not chunks:
        raise ValueError("No usable chunks (all were empty or too short).")
    return chunks


def _load_labeled_pairs_file(path: Path) -> List[Tuple[str, str]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list of query-document objects.")
    pairs: List[Tuple[str, str]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        query = row.get("query") or row.get("anchor") or row.get("question")
        document = row.get("document") or row.get("positive") or row.get("text")
        query = " ".join(str(query or "").split()).strip()
        document = " ".join(str(document or "").split()).strip()
        if query and document:
            pairs.append((query, document))
    return pairs


def load_labeled_pairs() -> Optional[List[Tuple[str, str]]]:
    """Return labeled (query, document) pairs if a sidecar file exists."""
    for path in LABELED_PAIRS_CANDIDATES:
        if path.exists():
            pairs = _load_labeled_pairs_file(path)
            if pairs:
                print(f"[INFO] Loaded {len(pairs)} labeled pairs from {path}")
                return pairs
            print(f"[INFO] {path} exists but contained no valid pairs; falling back to synthetic queries")
    return None


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _extract_key_phrase(text: str, max_words: int = 8) -> str:
    """Pick a short content phrase for a synthetic interrogative query."""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9+#/_-]*", text)
    content = [t for t in tokens if t.lower() not in STOPWORDS]
    if not content:
        content = tokens
    phrase = " ".join(content[:max_words]).strip()
    return phrase or text[:80].strip()


def generate_synthetic_queries(chunk_text: str) -> List[str]:
    """
    Turn one corpus chunk into one or two retrieval queries.

    Uses the first sentence (inverse-cloze style) plus a What/How question
    built from a key phrase so the model sees both statement and interrogative
    queries against the same passage.
    """
    sentences = _split_sentences(chunk_text)
    queries: List[str] = []

    first = sentences[0] if sentences else chunk_text
    first = first.strip()
    if len(first) >= MIN_QUERY_CHARS and first.lower() != chunk_text.lower():
        queries.append(first.rstrip(".!?"))

    phrase = _extract_key_phrase(chunk_text)
    if phrase:
        if re.search(r"\b(how|deploy|implement|handle|debug|work)\b", chunk_text, re.I):
            question = f"How does {phrase} work?"
        else:
            question = f"What does the document say about {phrase}?"
        if len(question) >= MIN_QUERY_CHARS:
            queries.append(question)

    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            unique.append(q)
    if not unique:
        unique.append(chunk_text[:120].rstrip(".!?") or chunk_text)
    return unique


def build_training_pairs(
    chunks: Sequence[Dict],
    labeled_pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> List[Tuple[str, str]]:
    """Prefer labeled pairs; otherwise synthesize a query per chunk from the corpus."""
    if labeled_pairs:
        return [(q, d) for q, d in labeled_pairs if q.strip() and d.strip()]

    pairs: List[Tuple[str, str]] = []
    for chunk in chunks:
        text = chunk["text"]
        for query in generate_synthetic_queries(text):
            if query.lower() == text.lower():
                continue
            pairs.append((query, text))
    if len(pairs) < 2:
        raise ValueError(
            "Need at least 2 query-document pairs for MultipleNegativesRankingLoss "
            "(in-batch negatives). Index more document text and retry."
        )
    return pairs


def _train_with_fit(
    model,
    pairs: Sequence[Tuple[str, str]],
    epochs: int,
    batch_size: int,
    output_dir: Path,
) -> None:
    """Legacy fit() path for sentence-transformers versions that still expose it."""
    from torch.utils.data import DataLoader

    try:
        from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    except ImportError:
        from sentence_transformers.losses import MultipleNegativesRankingLoss

    from sentence_transformers import InputExample

    examples = [InputExample(texts=[query, document]) for query, document in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = MultipleNegativesRankingLoss(model)
    steps_per_epoch = max(1, (len(examples) + batch_size - 1) // batch_size)
    warmup_steps = max(1, int(0.1 * steps_per_epoch * epochs))
    print(
        f"[INFO] Training with model.fit / MultipleNegativesRankingLoss "
        f"(epochs={epochs}, batch_size={batch_size}, warmup_steps={warmup_steps})"
    )
    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        show_progress_bar=True,
    )


def _train_with_trainer(
    model,
    pairs: Sequence[Tuple[str, str]],
    epochs: int,
    batch_size: int,
    output_dir: Path,
) -> None:
    from datasets import Dataset

    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

    try:
        from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    except ImportError:
        from sentence_transformers.losses import MultipleNegativesRankingLoss

    train_dataset = Dataset.from_dict({
        "anchor": [q for q, _ in pairs],
        "positive": [d for _, d in pairs],
    })
    loss = MultipleNegativesRankingLoss(model)
    checkpoint_dir = output_dir.parent / "finetuned-embeddings-checkpoints"
    steps_per_epoch = max(1, (len(pairs) + batch_size - 1) // batch_size)
    warmup_steps = max(1, int(0.1 * steps_per_epoch * epochs))
    args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        warmup_steps=warmup_steps,
        fp16=False,
        bf16=False,
        logging_steps=max(1, steps_per_epoch // 5),
        save_strategy="no",
        dataloader_pin_memory=False,
        report_to="none",
    )
    print(
        f"[INFO] Training with SentenceTransformerTrainer / MultipleNegativesRankingLoss "
        f"(epochs={epochs}, batch_size={batch_size}, warmup_steps={warmup_steps})"
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()
    model.save(str(output_dir))


def finetune(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    base_model: str = BASE_MODEL_NAME,
    epochs: int = 2,
    batch_size: int = 16,
    seed: int = 42,
) -> Path:
    """Run MNRL fine-tuning and return the directory the model was saved to."""
    import torch
    from sentence_transformers import SentenceTransformer

    random.seed(seed)
    torch.manual_seed(seed)

    chunks = load_corpus_chunks()
    labeled = load_labeled_pairs()
    pairs = build_training_pairs(chunks, labeled)
    random.shuffle(pairs)

    batch_size = max(2, min(batch_size, len(pairs)))
    print(
        f"[INFO] Fine-tuning dataset: {len(pairs)} query-document pairs "
        f"from {len(chunks)} chunks "
        f"({'labeled' if labeled else 'synthetic queries'})"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading base model {base_model} on {device}")
    model = SentenceTransformer(base_model, device=device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _train_with_trainer(model, pairs, epochs, batch_size, output_dir)
    except ImportError as exc:
        print(f"[INFO] Trainer API unavailable ({exc}); falling back to model.fit")
        _train_with_fit(model, pairs, epochs, batch_size, output_dir)
    print(f"[INFO] Saved fine-tuned embeddings to {output_dir}")
    return output_dir


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune MiniLM embeddings on the indexed document corpus."
    )
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL_NAME,
        help="sentence-transformers model id to start from",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write the fine-tuned model",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    finetune(
        output_dir=Path(args.output_dir),
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
