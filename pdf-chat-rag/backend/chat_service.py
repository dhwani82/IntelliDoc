import os
import json
import asyncio
from typing import Dict, AsyncGenerator, List, Any, Optional, Tuple

import google.generativeai as genai

from embeddings import get_embeddings
from local_llm import resolve_generation_backend
from prompt_experiments import build_prompts, get_default_strategy
from vector_store import VectorStore

GEMINI_MODEL = "models/gemini-2.5-flash"
FRIENDLY_GENERATION_ERROR = (
    "I couldn't generate an answer from the document. Please try rephrasing your question."
)
CITATION_PREVIEW_LENGTH = 100
KNOWN_SECTIONS = (
    "Projects", "Experience", "Work Experience", "Education", "Skills",
    "Summary", "Certifications", "Publications", "Awards", "Contact",
    "Objective", "References", "Technical Skills", "Professional Experience",
)


def _infer_section(text: str, metadata: Dict[str, Any]) -> Optional[str]:
    section = metadata.get("section") or metadata.get("title")
    if section:
        return str(section).strip() or None

    normalized = " ".join(text.split()).strip()
    if not normalized:
        return None

    for name in KNOWN_SECTIONS:
        if normalized.lower().startswith(name.lower()):
            remainder = normalized[len(name):]
            if not remainder or remainder[0] in " :.-":
                return name

    colon_idx = normalized.find(":")
    if 0 < colon_idx <= 40:
        candidate = normalized[:colon_idx].strip()
        if candidate and candidate[0].isupper() and not candidate.endswith("."):
            return candidate

    return None


def _make_preview(text: str, section: Optional[str] = None, max_len: int = CITATION_PREVIEW_LENGTH) -> str:
    body = " ".join(text.split()).strip()
    if section and body.lower().startswith(section.lower()):
        body = body[len(section):].lstrip(" :.-")

    if not body:
        return ""

    if len(body) <= max_len:
        return body

    truncated = body[:max_len]
    last_space = truncated.rfind(" ")
    if last_space > int(max_len * 0.6):
        truncated = truncated[:last_space]
    return truncated.rstrip(".,;:- ") + "..."


def _build_citations(results: List[Dict]) -> List[Dict]:
    """Build deduplicated, merged citation objects for the UI."""
    merged: Dict[Tuple[int, str], Dict[str, Any]] = {}

    for result in results:
        metadata = result["metadata"]
        page = int(metadata["page"])
        text = metadata.get("text", "")
        section = _infer_section(text, metadata)
        preview = _make_preview(text, section)
        if not preview:
            continue

        key = (page, section or "")
        existing = merged.get(key)
        if existing:
            if existing["preview"] == preview:
                continue
            if len(preview) > len(existing["preview"]):
                existing["preview"] = preview
            existing.pop("chunk_id", None)
            continue

        citation: Dict[str, Any] = {"page": page, "preview": preview}
        if section:
            citation["section"] = section
        chunk_id = metadata.get("chunk_id")
        if chunk_id is not None:
            citation["chunk_id"] = chunk_id
        merged[key] = citation

    return sorted(
        merged.values(),
        key=lambda item: (item["page"], item.get("section") or "", item["preview"]),
    )


def _finish_reason_label(reason: Any) -> str:
    if reason is None:
        return "unknown"
    name = getattr(reason, "name", None)
    if name:
        return str(name)
    return str(reason)


def _log_gemini_response_debug(response: Any, context: str) -> None:
    print(f"[CHAT] Gemini response issue: {context}")
    if response is None:
        print("[CHAT] response is None")
        return

    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback is not None:
        print(f"[CHAT] prompt_feedback: {prompt_feedback}")

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        print("[CHAT] no candidates in response")
    for index, candidate in enumerate(candidates):
        finish_reason = getattr(candidate, "finish_reason", None)
        safety_ratings = getattr(candidate, "safety_ratings", None)
        print(
            f"[CHAT] candidate[{index}] finish_reason={_finish_reason_label(finish_reason)} "
            f"({finish_reason!r})"
        )
        if safety_ratings:
            print(f"[CHAT] candidate[{index}] safety_ratings={safety_ratings}")

    try:
        print(f"[CHAT] full Gemini response: {response}")
    except Exception as exc:
        print(f"[CHAT] could not stringify full response: {exc}")


def _text_from_parts(parts: Any) -> str:
    if not parts:
        return ""
    texts: List[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            texts.append(text)
    return "".join(texts)


def _extract_text_from_response(response: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Safely extract text from a Gemini GenerateContentResponse.
    Returns (text, error_message). error_message is set when no text is available.
    """
    if response is None:
        _log_gemini_response_debug(None, "response missing")
        return None, FRIENDLY_GENERATION_ERROR

    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback is not None:
        block_reason = getattr(prompt_feedback, "block_reason", None)
        if block_reason:
            _log_gemini_response_debug(
                response, f"prompt blocked (block_reason={block_reason})"
            )
            return None, FRIENDLY_GENERATION_ERROR

    candidates = getattr(response, "candidates", None)
    if not candidates:
        _log_gemini_response_debug(response, "candidates missing or empty")
        return None, FRIENDLY_GENERATION_ERROR

    collected: List[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        parts = getattr(content, "parts", None)
        if not parts:
            continue
        part_text = _text_from_parts(parts)
        if part_text:
            collected.append(part_text)

    if collected:
        return "".join(collected), None

    _log_gemini_response_debug(response, "no text parts in any candidate")
    return None, FRIENDLY_GENERATION_ERROR


class ChatService:
    def __init__(self):
        print("[INFO] chat_service.py loaded")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_key_here":
            self.client_ready = False
            print("[INFO] GEMINI_API_KEY not configured")
        else:
            genai.configure(api_key=api_key)
            self.client_ready = True
            print("[INFO] Gemini configured successfully")
        try:
            default_backend = resolve_generation_backend()
        except ValueError:
            default_backend = "gemini"
        print(f"[INFO] Default GENERATION_BACKEND={default_backend}")

    def _skip_retrieval_response(
        self, question: str, doc_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Run the Keras query router before retrieval.

        Returns a canned {answer, sources} dict when the query is chitchat or
        out of scope so the RAG path is skipped. Returns None to continue.
        """
        from query_router import route_query

        decision = route_query(question, doc_id=doc_id)
        if not decision.get("skipped_retrieval"):
            return None
        return {
            "answer": decision.get("response") or "",
            "sources": [],
            "route": decision.get("label"),
        }

    async def _retrieve_chunks(
        self, doc_id: str, question: str, vector_store: VectorStore, top_k: int = 10
    ) -> List[Dict]:
        """Embed the question and retrieve relevant chunks, with fallback for small documents."""
        question_embeddings = await get_embeddings([question], task_type="retrieval_query")
        question_embedding = question_embeddings[0]
        print(f"[CHAT] Question embedding dim={len(question_embedding)}")

        results = vector_store.search(question_embedding, doc_id, top_k=top_k)

        if not results:
            all_chunks = vector_store.get_all_chunks_for_doc(doc_id)
            total_chars = sum(len(r["metadata"].get("text", "")) for r in all_chunks)
            print(
                f"[CHAT] Semantic search returned 0 results. "
                f"Fallback check: {len(all_chunks)} chunks, {total_chars} chars"
            )
            if all_chunks and len(all_chunks) <= 20 and total_chars <= 20000:
                print(f"[CHAT] Using full-document fallback ({len(all_chunks)} chunks)")
                results = all_chunks
            else:
                print("[CHAT] Fallback not applicable; returning empty results")

        return results

    def _build_context(self, results: List[Dict]) -> str:
        context_chunks = [result["metadata"]["text"] for result in results]
        context = "\n\n".join(context_chunks)
        print(f"[CHAT] Context: {len(context_chunks)} chunks, {len(context)} chars")
        print(f"[CHAT] Context preview: {context[:500]!r}")
        return context

    def _build_prompts(
        self,
        context: str,
        question: str,
        strategy: Optional[str] = None,
    ) -> tuple[str, str]:
        return build_prompts(context, question, strategy or get_default_strategy())

    def _generate_gemini_response(self, system_prompt: str, user_prompt: str) -> Any:
        print("[CHAT] Starting Gemini generation...")
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system_prompt,
        )
        response = model.generate_content(user_prompt)
        print("[CHAT] Gemini generation completed")
        return response

    def _generate_local_response(self, system_prompt: str, user_prompt: str) -> str:
        from local_llm import generate_local

        print("[CHAT] Starting local Hugging Face generation...")
        text = generate_local(system_prompt, user_prompt)
        print("[CHAT] Local generation completed")
        return text

    async def _generate_answer(
        self,
        system_prompt: str,
        user_prompt: str,
        backend: str,
    ) -> Tuple[str, Optional[str]]:
        # Both Gemini and local HF generate() are synchronous. Run them in a
        # worker thread via run_in_executor so the async event loop stays
        # responsive while waiting on the API or on local tokens.
        loop = asyncio.get_running_loop()
        if backend == "local":
            try:
                text = await loop.run_in_executor(
                    None, self._generate_local_response, system_prompt, user_prompt
                )
            except Exception as exc:
                print(f"[CHAT] Local LLM error: {exc}")
                return FRIENDLY_GENERATION_ERROR, str(exc)
            return (text or FRIENDLY_GENERATION_ERROR), None

        try:
            response = await loop.run_in_executor(
                None, self._generate_gemini_response, system_prompt, user_prompt
            )
        except Exception as exc:
            print(f"[CHAT] Gemini API error: {exc}")
            return FRIENDLY_GENERATION_ERROR, str(exc)

        text, error = _extract_text_from_response(response)
        if error:
            return error, error
        return text or FRIENDLY_GENERATION_ERROR, None

    async def get_answer(
        self,
        doc_id: str,
        question: str,
        vector_store: VectorStore,
        top_k: int = 10,
        generation_backend: Optional[str] = None,
        prompt_strategy: Optional[str] = None,
        skip_router: bool = False,
    ) -> Dict:
        print(f"[CHAT] get_answer called for doc_id={doc_id}")
        backend = resolve_generation_backend(generation_backend)
        print(f"[CHAT] generation backend={backend}")
        strategy = prompt_strategy or get_default_strategy()
        print(f"[CHAT] prompt strategy={strategy}")

        if not skip_router:
            routed = self._skip_retrieval_response(question, doc_id)
            if routed is not None:
                return routed

        results = await self._retrieve_chunks(doc_id, question, vector_store, top_k=top_k)
        print(f"[CHAT] Retrieval returned {len(results)} chunks")

        if not results:
            return {
                "answer": "No relevant information found in the document.",
                "sources": []
            }

        context = self._build_context(results)
        sources = _build_citations(results)
        system_prompt, user_prompt = self._build_prompts(context, question, strategy)

        if backend == "gemini" and not self.client_ready:
            return {
                "answer": "Error: GEMINI_API_KEY not configured. Please set your API key in the .env file.",
                "sources": []
            }

        answer, _ = await self._generate_answer(system_prompt, user_prompt, backend)

        return {
            "answer": answer,
            "sources": sources
        }

    async def get_answer_stream(
        self,
        doc_id: str,
        question: str,
        vector_store: VectorStore,
        top_k: int = 10,
        generation_backend: Optional[str] = None,
        prompt_strategy: Optional[str] = None,
        skip_router: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        NDJSON chat response for the /chat StreamingResponse.

        Generation itself is non-streaming (one full answer), then yielded as
        NDJSON chunks so the existing frontend contract stays unchanged. The
        blocking Gemini/local call is offloaded inside _generate_answer so this
        async generator does not freeze the event loop.
        """
        print(f"[CHAT] Received question for doc_id={doc_id}: {question}")
        backend = resolve_generation_backend(generation_backend)
        print(f"[CHAT] generation backend={backend}")
        strategy = prompt_strategy or get_default_strategy()
        print(f"[CHAT] prompt strategy={strategy}")

        if not skip_router:
            routed = self._skip_retrieval_response(question, doc_id)
            if routed is not None:
                yield json.dumps({"chunk": routed["answer"]}) + "\n"
                yield json.dumps({
                    "done": True,
                    "sources": [],
                    "route": routed.get("route"),
                }) + "\n"
                return

        results = await self._retrieve_chunks(doc_id, question, vector_store, top_k=top_k)
        print(f"[CHAT] Retrieval returned {len(results)} chunks")

        if not results:
            yield json.dumps({
                "chunk": "No relevant information found in the document.",
                "done": True,
                "sources": []
            }) + "\n"
            return

        context = self._build_context(results)
        sources = _build_citations(results)
        system_prompt, user_prompt = self._build_prompts(context, question, strategy)

        if backend == "gemini" and not self.client_ready:
            yield json.dumps({
                "chunk": "Error: GEMINI_API_KEY not configured.",
                "done": True,
                "sources": []
            }) + "\n"
            return

        # Sync generation happens in a thread via _generate_answer; we only
        # yield once the full answer is ready (NDJSON still matches the client).
        answer, _ = await self._generate_answer(system_prompt, user_prompt, backend)
        yield json.dumps({"chunk": answer}) + "\n"
        yield json.dumps({"done": True, "sources": sources}) + "\n"
