import os
import asyncio
from pathlib import Path
from typing import List, Optional

import google.generativeai as genai

BACKEND_ROOT = Path(__file__).resolve().parent

# Local sentence-transformers models. USE_FINETUNED_EMBEDDINGS switches between
# the off-the-shelf MiniLM checkpoint and the domain-adapted weights produced
# by finetune_embeddings.py.
BASE_ST_MODEL_NAME = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
FINETUNED_EMBEDDINGS_PATH = os.getenv(
    "FINETUNED_EMBEDDINGS_PATH",
    str(BACKEND_ROOT / "models" / "finetuned-embeddings"),
)
USE_FINETUNED_EMBEDDINGS = os.getenv("USE_FINETUNED_EMBEDDINGS", "false").lower() in {
    "1",
    "true",
    "yes",
}

# gemini | sentence_transformers. When unset, fine-tuned mode implies local ST.
_EMBEDDING_BACKEND_ENV = os.getenv("EMBEDDING_BACKEND", "").strip().lower().replace("-", "_")

_client_initialized = False
_st_model = None
_st_model_id: Optional[str] = None


def init_gemini():
    global _client_initialized
    if not _client_initialized:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_key_here":
            raise ValueError("GEMINI_API_KEY not set. Please set it in .env file")
        genai.configure(api_key=api_key)
        _client_initialized = True


def use_sentence_transformers() -> bool:
    """True when retrieval should use a local MiniLM (base or fine-tuned) model."""
    if _EMBEDDING_BACKEND_ENV in {"sentence_transformers", "st", "local"}:
        return True
    if _EMBEDDING_BACKEND_ENV in {"gemini"}:
        return False
    return USE_FINETUNED_EMBEDDINGS


def get_st_model_path() -> str:
    """Return the Hugging Face id or local directory for the active ST model."""
    if USE_FINETUNED_EMBEDDINGS:
        path = Path(FINETUNED_EMBEDDINGS_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"Fine-tuned embeddings not found at {path}. "
                "Run `python finetune_embeddings.py` first, or set "
                "USE_FINETUNED_EMBEDDINGS=false to use the base MiniLM model."
            )
        return str(path)
    return BASE_ST_MODEL_NAME


def get_embedding_model_id() -> str:
    """Stable id stored with the FAISS index so a model switch is detected."""
    if use_sentence_transformers():
        if USE_FINETUNED_EMBEDDINGS:
            return "finetuned-embeddings"
        return BASE_ST_MODEL_NAME
    return "gemini-embedding-001"


def _get_sentence_transformer():
    """Load and cache the active sentence-transformers model on CPU or CUDA."""
    global _st_model, _st_model_id
    model_id = get_embedding_model_id()
    if _st_model is None or _st_model_id != model_id:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        path = get_st_model_path()
        print(f"[INFO] Loading sentence-transformers model {path!r} on {device}")
        _st_model = SentenceTransformer(path, device=device)
        _st_model_id = model_id
    return _st_model


def _clean_texts(texts: List[str]) -> List[str]:
    cleaned = []
    for text in texts:
        if text is None:
            cleaned.append("")
        else:
            cleaned.append(" ".join(str(text).split()).strip())
    return cleaned


def _get_st_embeddings_sync(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    cleaned_texts = _clean_texts(texts)
    for i, text in enumerate(cleaned_texts, start=1):
        if not text:
            raise ValueError(f"Empty text found at embedding item {i}")

    model = _get_sentence_transformer()
    print(
        f"[INFO] Encoding {len(cleaned_texts)} texts with {get_embedding_model_id()} "
        f"(USE_FINETUNED_EMBEDDINGS={USE_FINETUNED_EMBEDDINGS})"
    )
    # Unit-normalized so FAISS L2 ranking matches cosine similarity.
    vectors = model.encode(
        cleaned_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    print(f"[INFO] Local embedding call successful, received {len(vectors)} embeddings")
    return [vec.tolist() for vec in vectors]


def _get_gemini_embeddings_sync(texts: List[str], task_type: str) -> List[List[float]]:
    try:
        init_gemini()

        if not texts:
            return []

        cleaned_texts = _clean_texts(texts)

        print(
            f"[INFO] Calling Gemini Embedding API for {len(cleaned_texts)} texts "
            f"with task_type={task_type}..."
        )

        embeddings = []
        for i, text in enumerate(cleaned_texts, start=1):
            if not text:
                raise ValueError(f"Empty text found at embedding item {i}")

            response = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type=task_type,
            )

            if "embedding" not in response or not response["embedding"]:
                raise ValueError(f"No embedding returned for item {i}")

            embeddings.append(response["embedding"])

        print(f"[INFO] Gemini embedding call successful, received {len(embeddings)} embeddings")
        return embeddings

    except ValueError as e:
        error_msg = f"Configuration/validation error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else "Unknown Gemini API error"
        full_error = f"Gemini API error ({error_type}): {error_msg}"
        print(f"[ERROR] {full_error}")
        raise Exception(full_error)


async def get_embeddings(texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
    """
    Embed a list of texts with the configured backend.

    Backends:
        - gemini (default): models/gemini-embedding-001. task_type is passed
          through so document vs query embeddings stay aligned.
        - sentence_transformers: all-MiniLM-L6-v2, or ./models/finetuned-embeddings
          when USE_FINETUNED_EMBEDDINGS=true. task_type is ignored; MiniLM uses
          one encoder for queries and documents.

    Set EMBEDDING_BACKEND=sentence_transformers to use MiniLM without fine-tuning.
    USE_FINETUNED_EMBEDDINGS=true also selects the local ST backend.

    Note:
        Sync model/API calls run in a thread executor so the asyncio event loop
        is not blocked.
    """
    loop = asyncio.get_running_loop()
    if use_sentence_transformers():
        return await loop.run_in_executor(None, _get_st_embeddings_sync, texts)
    return await loop.run_in_executor(None, _get_gemini_embeddings_sync, texts, task_type)
