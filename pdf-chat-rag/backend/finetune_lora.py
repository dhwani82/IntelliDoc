"""
LoRA fine-tune the local instruct model on question-answer pairs.

Pairs are taken from eval/qa_pairs.json (eval-set format plus an `answer`
field) when present. Otherwise they are built from eval/eval_set.json plus
synthetic Q/A derived from the indexed document corpus.

Saves PEFT adapters to ./models/lora-adapter, which local_llm.py loads when
USE_LORA_ADAPTER is true or the directory exists.

Usage (from pdf-chat-rag/backend):
    python finetune_lora.py
    python finetune_lora.py --epochs 1 --max-pairs 128
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

load_dotenv(BACKEND_ROOT / ".env")

from finetune_embeddings import (  # noqa: E402
    _split_sentences,
    generate_synthetic_queries,
    load_corpus_chunks,
)
from local_llm import DEFAULT_LOCAL_MODEL, LORA_ADAPTER_PATH  # noqa: E402

EVAL_SET_PATH = BACKEND_ROOT / "eval" / "eval_set.json"
LABELED_QA_CANDIDATES = (
    BACKEND_ROOT / "eval" / "qa_pairs.json",
    BACKEND_ROOT / "data" / "qa_pairs.json",
)
RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based solely on the "
    "provided context from a document. If the answer cannot be found in the "
    "context, say: \"I cannot find this information in the document.\""
)
LORA_TARGET_CANDIDATES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "qkv_proj",
    "gate_up_proj",
)


def _keyword_in_text(keyword: str, text: str) -> bool:
    return keyword.lower() in (text or "").lower()


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return [row for row in raw if isinstance(row, dict)]


def load_labeled_qa() -> Optional[List[Dict[str, str]]]:
    """Load {question, answer} rows if a sidecar file exists (eval-set shape)."""
    for path in LABELED_QA_CANDIDATES:
        if not path.exists():
            continue
        rows = _load_json_list(path)
        pairs = []
        for row in rows:
            question = " ".join(str(row.get("question") or "").split()).strip()
            answer = " ".join(str(row.get("answer") or "").split()).strip()
            if question and answer:
                pairs.append({"question": question, "answer": answer})
        if pairs:
            print(f"[INFO] Loaded {len(pairs)} labeled QA pairs from {path}")
            return pairs
        print(f"[INFO] {path} exists but had no question/answer fields")
    return None


def _answer_from_text(text: str, keywords: Optional[Sequence[str]] = None) -> str:
    sentences = _split_sentences(text)
    if keywords:
        scored: List[Tuple[int, str]] = []
        for sentence in sentences:
            hits = sum(1 for k in keywords if _keyword_in_text(k, sentence))
            if hits:
                scored.append((hits, sentence))
        if scored:
            scored.sort(key=lambda item: -item[0])
            return " ".join(sentence for _, sentence in scored[:2]).strip()
    if sentences:
        return " ".join(sentences[:2]).strip()
    return text[:400].strip()


def qa_from_eval_set(chunks: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Turn eval/eval_set.json into QA pairs in the same schema, filling `answer`
    from the first corpus chunk that contains an expected keyword.
    """
    if not EVAL_SET_PATH.exists():
        return []
    eval_set = _load_json_list(EVAL_SET_PATH)
    pairs: List[Dict[str, str]] = []
    for item in eval_set:
        question = " ".join(str(item.get("question") or "").split()).strip()
        if not question:
            continue
        existing_answer = " ".join(str(item.get("answer") or "").split()).strip()
        if existing_answer:
            pairs.append({"question": question, "answer": existing_answer})
            continue
        keywords = item.get("expected_keywords") or []
        match_text = ""
        best_hits = 0
        for chunk in chunks:
            text = chunk.get("text") or ""
            if not keywords:
                continue
            hits = sum(1 for k in keywords if _keyword_in_text(k, text))
            if hits > best_hits:
                best_hits = hits
                match_text = text
        if not match_text and chunks:
            match_text = chunks[0].get("text") or ""
        answer = _answer_from_text(match_text, keywords)
        if question and answer:
            pairs.append({"question": question, "answer": answer})
    print(f"[INFO] Built {len(pairs)} QA pairs from {EVAL_SET_PATH.name}")
    return pairs


def qa_from_corpus(chunks: Sequence[Dict[str, Any]], max_pairs: int) -> List[Dict[str, str]]:
    """Synthetic extractive QA: question from the chunk, answer = leading sentences."""
    pairs: List[Dict[str, str]] = []
    for chunk in chunks:
        text = chunk.get("text") or ""
        answer = _answer_from_text(text)
        if not answer:
            continue
        for question in generate_synthetic_queries(text):
            if question.lower() == answer.lower():
                continue
            pairs.append({"question": question, "answer": answer})
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def build_qa_dataset(max_pairs: int = 200) -> List[Dict[str, str]]:
    labeled = load_labeled_qa()
    if labeled:
        return labeled[:max_pairs]

    chunks = load_corpus_chunks()
    pairs = qa_from_eval_set(chunks)
    remaining = max(0, max_pairs - len(pairs))
    if remaining:
        synthetic = qa_from_corpus(chunks, remaining)
        seen = {p["question"].lower() for p in pairs}
        for row in synthetic:
            if row["question"].lower() in seen:
                continue
            pairs.append(row)
            seen.add(row["question"].lower())
            if len(pairs) >= max_pairs:
                break
    if len(pairs) < 2:
        raise ValueError(
            "Need at least 2 QA pairs to LoRA-tune. Index a document and/or "
            "add eval/qa_pairs.json in eval-set format with an `answer` field."
        )
    return pairs[:max_pairs]


def infer_lora_target_modules(model) -> List[str]:
    names = {name.split(".")[-1] for name, _ in model.named_modules()}
    targets = [mod for mod in LORA_TARGET_CANDIDATES if mod in names]
    if not targets:
        raise ValueError(
            "Could not infer LoRA target modules for this architecture. "
            f"Known module suffixes include: {sorted(names)[:30]}"
        )
    return targets


def _tokenize_qa(tokenizer, question: str, answer: str, max_seq_len: int) -> Dict[str, List[int]]:
    user_content = (
        "Answer the question using only information from the document context "
        "implied by the question.\n\n"
        f"Question: {question}"
    )
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer},
    ]
    prompt_messages = messages[:-1]
    if getattr(tokenizer, "chat_template", None):
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    else:
        prompt_text = f"{RAG_SYSTEM_PROMPT}\n\n{user_content}\n\nAnswer:"
        full_text = f"{prompt_text} {answer}"
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    full_ids = full_ids[:max_seq_len]
    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


class _PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention_mask = [], [], []
        for feat in features:
            pad = max_len - len(feat["input_ids"])
            input_ids.append(feat["input_ids"] + [self.pad_token_id] * pad)
            labels.append(feat["labels"] + [-100] * pad)
            attention_mask.append(feat["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def finetune_lora(
    output_dir: Path = LORA_ADAPTER_PATH,
    base_model: str = DEFAULT_LOCAL_MODEL,
    epochs: int = 1,
    batch_size: int = 1,
    lora_r: int = 8,
    lora_alpha: int = 16,
    max_seq_len: int = 512,
    max_pairs: int = 200,
    seed: int = 42,
    learning_rate: float = 2e-4,
) -> Path:
    """Train LoRA adapters and write them to output_dir."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    random.seed(seed)
    torch.manual_seed(seed)

    pairs = build_qa_dataset(max_pairs=max_pairs)
    random.shuffle(pairs)
    print(f"[INFO] LoRA dataset: {len(pairs)} question-answer pairs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"[INFO] Loading base model {base_model} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    if not getattr(model, "hf_device_map", None):
        model = model.to(device)

    target_modules = infer_lora_target_modules(model)
    print(f"[INFO] LoRA target modules: {target_modules}")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False

    tokenized = [
        _tokenize_qa(tokenizer, row["question"], row["answer"], max_seq_len)
        for row in pairs
    ]
    tokenized = [ex for ex in tokenized if any(label != -100 for label in ex["labels"])]
    if len(tokenized) < 2:
        raise ValueError("Tokenization produced too few usable examples.")
    train_dataset = Dataset.from_list(tokenized)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir.parent / "lora-adapter-checkpoints"

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 8 // max(1, batch_size)),
        learning_rate=learning_rate,
        logging_steps=max(1, len(tokenized) // 10),
        save_strategy="no",
        report_to="none",
        fp16=bool(torch.cuda.is_available()),
        bf16=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=bool(torch.cuda.is_available()),
        remove_unused_columns=False,
        warmup_steps=max(1, int(0.05 * ((len(tokenized) + batch_size - 1) // batch_size) * epochs)),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=_PadCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[INFO] Saved LoRA adapter to {output_dir}")
    return output_dir


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune the local instruct model on corpus QA pairs."
    )
    parser.add_argument("--base-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--output-dir", default=str(LORA_ADAPTER_PATH))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    finetune_lora(
        output_dir=Path(args.output_dir),
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_len=args.max_seq_len,
        max_pairs=args.max_pairs,
        seed=args.seed,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
