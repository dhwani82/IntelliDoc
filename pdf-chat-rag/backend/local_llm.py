"""
Local Hugging Face generation backend for the RAG pipeline.

Loads a small instruct model (default: microsoft/Phi-3-mini-4k-instruct) with
transformers and optionally attaches a PEFT LoRA adapter from
./models/lora-adapter. Used when GENERATION_BACKEND=local or when /chat is
called with ?backend=local.

The model is loaded lazily on first generate() so Gemini-only deploys do not
pay the download/RAM cost at import time.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Optional, Tuple

BACKEND_ROOT = Path(__file__).resolve().parent

DEFAULT_LOCAL_MODEL = os.getenv(
    "LOCAL_LLM_MODEL", "microsoft/Phi-3-mini-4k-instruct"
)
LORA_ADAPTER_PATH = Path(
    os.getenv("LORA_ADAPTER_PATH", str(BACKEND_ROOT / "models" / "lora-adapter"))
)
VALID_GENERATION_BACKENDS = ("gemini", "local")

_load_lock = threading.Lock()
_model = None
_tokenizer = None
_loaded_key: Optional[Tuple[str, Optional[str]]] = None


def _env_flag(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return None
    return str(raw).strip().lower() in {"1", "true", "yes"}


def resolve_generation_backend(override: Optional[str] = None) -> str:
    """Return 'gemini' or 'local'. override (e.g. query param) wins over env."""
    raw = (override if override is not None else os.getenv("GENERATION_BACKEND", "gemini"))
    backend = str(raw or "gemini").strip().lower()
    if backend not in VALID_GENERATION_BACKENDS:
        raise ValueError(
            f"Invalid generation backend {backend!r}. "
            f"Expected one of: {', '.join(VALID_GENERATION_BACKENDS)}"
        )
    return backend


def should_load_lora() -> bool:
    """
    Load LoRA when USE_LORA_ADAPTER is true, or automatically when the adapter
    directory exists and the flag is unset. Explicit false disables it.
    """
    flag = _env_flag("USE_LORA_ADAPTER")
    adapter_exists = LORA_ADAPTER_PATH.exists() and any(LORA_ADAPTER_PATH.iterdir())
    if flag is True:
        return True
    if flag is False:
        return False
    return adapter_exists


def _pick_torch_dtype(torch_mod: Any):
    if torch_mod.cuda.is_available():
        return torch_mod.float16
    return torch_mod.float32


def _move_to_device(model, torch_mod: Any):
    if getattr(model, "hf_device_map", None):
        return model
    device = "cuda" if torch_mod.cuda.is_available() else "cpu"
    return model.to(device)


def load_local_model(force_reload: bool = False):
    """
    Load (and cache) the base instruct model, attaching LoRA adapters when
    should_load_lora() is true.
    """
    global _model, _tokenizer, _loaded_key
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = DEFAULT_LOCAL_MODEL
    use_lora = should_load_lora()
    adapter_key = str(LORA_ADAPTER_PATH.resolve()) if use_lora else None
    cache_key = (model_id, adapter_key)

    with _load_lock:
        if not force_reload and _model is not None and _loaded_key == cache_key:
            return _model, _tokenizer

        if use_lora and not LORA_ADAPTER_PATH.exists():
            raise FileNotFoundError(
                f"USE_LORA_ADAPTER is set but no adapter found at {LORA_ADAPTER_PATH}. "
                "Run `python finetune_lora.py` first, or set USE_LORA_ADAPTER=false."
            )

        dtype = _pick_torch_dtype(torch)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[LOCAL_LLM] Loading {model_id!r} on {device} dtype={dtype}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "torch_dtype": dtype,
        }
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        model = _move_to_device(model, torch)

        if use_lora:
            from peft import PeftModel

            print(f"[LOCAL_LLM] Attaching LoRA adapter from {LORA_ADAPTER_PATH}")
            model = PeftModel.from_pretrained(model, str(LORA_ADAPTER_PATH))
            model = _move_to_device(model, torch)

        model.eval()
        _model = model
        _tokenizer = tokenizer
        _loaded_key = cache_key
        print(
            f"[LOCAL_LLM] Ready (lora={'on' if use_lora else 'off'}, "
            f"model={model_id})"
        )
        return _model, _tokenizer


def _build_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{system_prompt.strip()}\n\n{user_prompt.strip()}\n\nAnswer:"


def generate_local(
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: Optional[int] = None,
) -> str:
    """Run a single non-streaming completion and return the assistant text."""
    import torch

    model, tokenizer = load_local_model()
    prompt = _build_prompt(tokenizer, system_prompt, user_prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    if max_new_tokens is None:
        max_new_tokens = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "256"))

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    print(f"[LOCAL_LLM] Generating (max_new_tokens={max_new_tokens})...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print(f"[LOCAL_LLM] Generated {len(text)} chars")
    return text
