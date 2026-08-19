"""
Lightweight Keras query router for the RAG chat pipeline.

Classifies a user question into:
  - needs_retrieval  → run the existing RAG path
  - general_chitchat → skip retrieval, return a short greeting
  - out_of_scope     → skip retrieval, explain the assistant only answers
                       from the uploaded document

Usage (from pdf-chat-rag/backend):
    python query_router.py            # train and save ./models/router
    python query_router.py --evaluate
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

BACKEND_ROOT = Path(__file__).resolve().parent
TRAIN_SET_PATH = BACKEND_ROOT / "eval" / "router_train.json"
MODEL_DIR = Path(os.getenv("ROUTER_MODEL_DIR", str(BACKEND_ROOT / "models" / "router")))
LOG_PATH = BACKEND_ROOT / "data" / "router_decisions.jsonl"

LABELS = ("needs_retrieval", "general_chitchat", "out_of_scope")
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}

MAX_TOKENS = 2000
SEQUENCE_LENGTH = 24
EMBEDDING_DIM = 48
CONFIDENCE_FLOOR = 0.40

CHITCHAT_RESPONSE = (
    "Hi! I'm here to help you explore the uploaded document. "
    "Ask me about the projects, experience, tools, or technical details in the PDF."
)
OUT_OF_SCOPE_RESPONSE = (
    "That question is outside this document. I can only answer from the uploaded PDF "
    "— for example the Kubernetes/EKS work, DocuMind, Event-Connect, or the Unity interview."
)

_load_lock = threading.Lock()
_log_lock = threading.Lock()
_model = None
_model_path: Optional[Path] = None


def router_enabled() -> bool:
    raw = os.getenv("USE_QUERY_ROUTER", "true").strip().lower()
    return raw not in {"0", "false", "no"}


def canned_response_for(label: str) -> Optional[str]:
    if label == "general_chitchat":
        return CHITCHAT_RESPONSE
    if label == "out_of_scope":
        return OUT_OF_SCOPE_RESPONSE
    return None


def build_training_examples() -> List[Dict[str, str]]:
    """
    ~180 labeled queries for this project's document domain (resume/portfolio
    covering Kubernetes, DocuMind, Event-Connect, and a Unity interview).
    """
    retrieval = [
        "What Kubernetes project is described?",
        "Describe the Kubernetes project in this document",
        "Tell me about the Spring Boot e-commerce app",
        "What cloud platform was the app deployed to?",
        "Was the app deployed to EKS?",
        "Which AWS Kubernetes service was used?",
        "What tool was used to templatize the Kubernetes manifests?",
        "Did they use Helm charts?",
        "How were the Kubernetes YAML files templated?",
        "What CI/CD tools were used?",
        "Was Jenkins part of the pipeline?",
        "How did GitHub fit into CI/CD?",
        "Was the Kubernetes work professional experience or coursework?",
        "Is the K8s project from SWE 645?",
        "What happens if a pod crashes?",
        "How does the Deployment controller keep replicas running?",
        "How were secrets like database credentials handled?",
        "Did they use Kubernetes Secrets?",
        "What Kubernetes object type was used for the three services, and why not StatefulSet?",
        "Why were the services Deployments instead of StatefulSets?",
        "How did the services discover each other in the cluster?",
        "Was DNS used for Kubernetes service discovery?",
        "How would you debug a service that is not responding in the cluster?",
        "Which kubectl commands would you use to debug a pod?",
        "Why EKS instead of self-managing Kubernetes?",
        "What does the document say about the EKS control plane?",
        "What is DocuMind?",
        "Tell me about the DocuMind project",
        "What five tools does the DocuMind agent use?",
        "Does DocuMind use grep_code and vector_search?",
        "How was DocuMind evaluated?",
        "Was RAGAS used to evaluate DocuMind?",
        "What limitation did vector-only retrieval have in DocuMind?",
        "Why did DocuMind need a call graph besides vector search?",
        "What is Event-Connect and how was it deployed?",
        "Describe the Event-Connect microservices",
        "Was Event-Connect deployed with ECR and EC2?",
        "Who is the interview with and what company?",
        "Who is Wei Chen?",
        "What company was the interview with?",
        "What does Unity Gaming Services provide?",
        "Does Unity offer multiplayer and analytics services?",
        "Summarize this document",
        "Summarize the uploaded PDF",
        "What projects are listed in this resume?",
        "Walk me through the candidate's experience",
        "What skills are listed?",
        "What programming languages does the candidate know?",
        "Tell me about the education section",
        "What is on page 1 of the document?",
        "Quote the experience section",
        "List the backend technologies mentioned",
        "What did they build with LangGraph?",
        "Explain the React agent in DocuMind",
        "How many services were in the e-commerce app?",
        "What is the product catalog / cart / checkout split?",
        "Any certifications mentioned?",
        "What cloud services appear in this PDF?",
        "Compare the Kubernetes project and Event-Connect",
        "What structural retrieval means in DocuMind",
        "find_symbol and who_calls tools",
        "How is impact_of used in DocuMind?",
        "What was the SWE 645 assignment about?",
        "Does the resume mention FastAPI or Spring Boot?",
        "Give me a short bio based on this document",
        "What should I ask this candidate in an interview?",
        "Highlight the strongest project in the PDF",
        "Where was the e-commerce app hosted?",
        "How were Helm values used?",
        "What happens to replicas when a pod dies?",
        "Is there anything about Jenkins pipelines?",
        "Explain the Unity interview takeaways",
        "What analytics does Unity Gaming Services include?",
        "What does this person know about Kubernetes?",
        "Which project used ECR?",
    ]
    chitchat = [
        "hi",
        "hello",
        "hey",
        "hey there",
        "hiya",
        "yo",
        "good morning",
        "good afternoon",
        "good evening",
        "howdy",
        "how's it going",
        "how are you",
        "how are you doing",
        "how are you today",
        "what's up",
        "sup",
        "thanks",
        "thank you",
        "thanks a lot",
        "thank you so much",
        "thx",
        "cheers",
        "great job",
        "awesome",
        "nice",
        "cool",
        "you are helpful",
        "you're the best",
        "lol",
        "haha",
        "who are you",
        "what's your name",
        "what can you do",
        "can you help me",
        "I have a question",
        "ok",
        "okay",
        "got it",
        "sounds good",
        "never mind",
        "that's all",
        "nothing else",
        "bye",
        "goodbye",
        "see you",
        "take care",
        "have a nice day",
        "good night",
        "please",
        "sorry",
        "my bad",
        "wait",
        "hold on",
        "tell me a joke",
        "make me laugh",
        "are you an AI",
        "are you ChatGPT",
        "how old are you",
        "nice to meet you",
        "let's chat",
        "talk to me",
        "I'm bored",
        "how is your day",
        "what's your favorite color",
        "do you like me",
        "good to know",
        "interesting",
        "wow",
        "oh",
        "hmm",
        "I see",
        "copy that",
        "hi helper",
        "hello there",
        "thanks for the help",
        "just saying hi",
    ]
    out_of_scope = [
        "what's the weather today",
        "weather in Seattle",
        "will it rain tomorrow",
        "who won the Super Bowl",
        "NBA score last night",
        "stock price of Apple",
        "what's Bitcoin worth",
        "should I buy Tesla stock",
        "how do I cook pasta",
        "give me a pizza recipe",
        "calorie count of a banana",
        "workout plan for beginners",
        "how to train a dog",
        "I have a headache, what medicine should I take",
        "should I take ibuprofen",
        "legal advice for a speeding ticket",
        "write a Python bubble sort",
        "implement quicksort in Java",
        "debug my React app that's not in this PDF",
        "translate hello to Spanish",
        "what is the capital of France",
        "who is the president of the United States",
        "latest world news",
        "celebrity gossip today",
        "Netflix show recommendations",
        "best restaurants near me",
        "Uber to the airport",
        "flight status for DL123",
        "how to reset my WiFi router",
        "help me install Windows",
        "how to jailbreak an iPhone",
        "explain quantum physics",
        "solve this integral for my homework",
        "what is 234 times 891",
        "play rock paper scissors",
        "write a poem about cats",
        "lyrics to a popular song",
        "sports betting odds",
        "cryptocurrency trading tips",
        "how to get a mortgage",
        "plan a trip to Japan",
        "what's on sale at Amazon",
        "recipe for chocolate cake",
        "my printer won't connect",
        "fix a blue screen of death",
        "who invented the telephone",
        "explain the plot of Inception",
        "best GPU for gaming",
        "how do I learn guitar",
        "write an email to my landlord",
        "what's the meaning of life",
        "horoscope for Leo",
        "is it going to snow",
        "vaccine side effects",
        "divorce lawyer recommendations",
        "how to beat the stock market",
        "generate a logo for my startup",
        "make a fortnite strategy",
        "what's the score of the soccer match",
        "convert 80 fahrenheit to celsius",
        "when is the next solar eclipse",
        "book a hotel in New York",
        "how much is a Tesla Model 3",
        "teach me French grammar",
        "write SQL to drop a table",
        "what time is it in Tokyo",
        "find cheap flights to London",
        "my car won't start",
        "how to get rid of ants",
    ]

    examples: List[Dict[str, str]] = []
    seen = set()
    per_class_cap = 66
    for label, texts in (
        ("needs_retrieval", retrieval),
        ("general_chitchat", chitchat),
        ("out_of_scope", out_of_scope),
    ):
        kept = 0
        for text in texts:
            if kept >= per_class_cap:
                break
            key = " ".join(text.lower().split())
            if not key or key in seen:
                continue
            seen.add(key)
            examples.append({"text": text, "label": label})
            kept += 1
    return examples


def save_training_set(path: Path = TRAIN_SET_PATH) -> List[Dict[str, str]]:
    examples = build_training_examples()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    counts = {label: 0 for label in LABELS}
    for row in examples:
        counts[row["label"]] += 1
    print(f"[ROUTER] Wrote {len(examples)} examples to {path} ({counts})")
    return examples


def load_training_set(path: Path = TRAIN_SET_PATH) -> List[Dict[str, str]]:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        examples = [
            {"text": str(r["text"]).strip(), "label": str(r["label"]).strip()}
            for r in rows
            if isinstance(r, dict) and r.get("text") and r.get("label") in LABEL_TO_ID
        ]
        if examples:
            return examples
    return save_training_set(path)


def _build_model(tf_mod: Any):
    keras = tf_mod.keras
    layers = keras.layers

    vectorizer = layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
        name="query_vectorizer",
    )
    inputs = keras.Input(shape=(), dtype=tf_mod.string, name="query")
    tokens = vectorizer(inputs)
    x = layers.Embedding(
        MAX_TOKENS,
        EMBEDDING_DIM,
        mask_zero=True,
        name="query_embedding",
    )(tokens)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(len(LABELS), activation="softmax", name="intent")(x)
    model = keras.Model(inputs, outputs, name="query_router")
    return model, vectorizer


def train_router(
    output_dir: Path = MODEL_DIR,
    epochs: int = 30,
    batch_size: int = 16,
) -> Path:
    """Fit the Keras classifier and save it under ./models/router."""
    import tensorflow as tf

    examples = load_training_set()
    rng = random.Random(42)
    rng.shuffle(examples)
    texts = [row["text"] for row in examples]
    y = tf.keras.utils.to_categorical(
        [LABEL_TO_ID[row["label"]] for row in examples],
        num_classes=len(LABELS),
    )

    model, vectorizer = _build_model(tf)
    vectorizer.adapt(tf.constant(texts))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history = model.fit(
        tf.constant(texts),
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        verbose=2,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.keras"
    model.save(model_path)
    meta = {
        "labels": list(LABELS),
        "num_examples": len(examples),
        "history": {
            "accuracy": [float(x) for x in history.history.get("accuracy", [])],
            "val_accuracy": [float(x) for x in history.history.get("val_accuracy", [])],
        },
    }
    with open(output_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    save_training_set(TRAIN_SET_PATH)
    print(f"[ROUTER] Saved model to {model_path}")
    return output_dir


def _model_file(model_dir: Path = MODEL_DIR) -> Path:
    return Path(model_dir) / "model.keras"


def model_available(model_dir: Path = MODEL_DIR) -> bool:
    return _model_file(model_dir).exists()


def load_router(model_dir: Path = MODEL_DIR, force: bool = False):
    """Load and cache the trained Keras model."""
    global _model, _model_path
    path = _model_file(model_dir)
    with _load_lock:
        if not force and _model is not None and _model_path == path:
            return _model
        if not path.exists():
            raise FileNotFoundError(
                f"Query router model not found at {path}. Run `python query_router.py` first."
            )
        import tensorflow as tf

        print(f"[ROUTER] Loading Keras model from {path}")
        _model = tf.keras.models.load_model(path)
        _model_path = path
        return _model


def predict_label(text: str, apply_confidence_floor: bool = True) -> Tuple[str, Dict[str, float]]:
    """Return (label, class→probability)."""
    import tensorflow as tf

    model = load_router()
    cleaned = " ".join(str(text or "").split()) or " "
    probs_arr = model.predict(tf.constant([cleaned]), verbose=0)[0]
    probs = {label: float(probs_arr[i]) for i, label in enumerate(LABELS)}
    label = max(probs, key=probs.get)
    if apply_confidence_floor and probs[label] < CONFIDENCE_FLOOR:
        label = "needs_retrieval"
    return label, probs


def log_routing_decision(decision: Dict[str, Any]) -> None:
    """Append one JSON line for later analysis; also print a compact log line."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(decision, ensure_ascii=False)
    with _log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    probs = decision.get("probs") or {}
    top = max(probs.values()) if probs else 0.0
    print(
        f"[ROUTER] label={decision.get('label')} conf={top:.2f} "
        f"skip={decision.get('skipped_retrieval')} "
        f"source={decision.get('source')} "
        f"query={str(decision.get('query', ''))[:80]!r}"
    )


def route_query(text: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Classify a query and log the decision.

    If the router is disabled or the model is missing, fail open to
    needs_retrieval so the RAG pipeline still runs.
    """
    source = "model"
    label = "needs_retrieval"
    probs = {name: 0.0 for name in LABELS}
    probs["needs_retrieval"] = 1.0

    if not router_enabled():
        source = "disabled"
    elif not model_available():
        source = "missing_model"
        print("[ROUTER] Model missing; defaulting to needs_retrieval")
    else:
        try:
            label, probs = predict_label(text)
            source = "model"
        except Exception as exc:
            source = "error"
            label = "needs_retrieval"
            print(f"[ROUTER] Inference failed ({exc}); defaulting to needs_retrieval")

    decision = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": text,
        "doc_id": doc_id,
        "label": label,
        "probs": {k: round(v, 4) for k, v in probs.items()},
        "skipped_retrieval": label != "needs_retrieval",
        "response": canned_response_for(label),
        "source": source,
    }
    log_routing_decision(decision)
    return decision


def evaluate_examples(examples: Optional[Sequence[Dict[str, str]]] = None) -> Dict[str, Any]:
    rows = list(examples or load_training_set())
    load_router()
    hits = 0
    per_label = {label: {"correct": 0, "total": 0} for label in LABELS}
    mistakes: List[Dict[str, Any]] = []
    for row in rows:
        pred, probs = predict_label(row["text"], apply_confidence_floor=False)
        gold = row["label"]
        per_label[gold]["total"] += 1
        if pred == gold:
            hits += 1
            per_label[gold]["correct"] += 1
        else:
            mistakes.append({
                "text": row["text"],
                "gold": gold,
                "pred": pred,
                "probs": {k: round(v, 3) for k, v in probs.items()},
            })
    n = len(rows) or 1
    summary = {
        "accuracy": round(hits / n, 4),
        "hits": hits,
        "num_examples": len(rows),
        "per_label": per_label,
        "mistakes": mistakes[:12],
    }
    print(f"[ROUTER] Train-set accuracy {summary['accuracy']:.2%} ({hits}/{len(rows)})")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or inspect the Keras query router.")
    parser.add_argument("--evaluate", action="store_true", help="Load the saved model and score the training set")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.evaluate:
        evaluate_examples()
        return
    train_router(epochs=args.epochs, batch_size=args.batch_size)
    evaluate_examples()


if __name__ == "__main__":
    main()
