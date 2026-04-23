import os
import json
import threading
from queue import Queue, Empty
from typing import Dict, AsyncGenerator
import asyncio

import google.generativeai as genai

from embeddings import get_embeddings
from vector_store import VectorStore


class ChatService:
    def __init__(self):
        print("[INFO] Gemini chat_service.py loaded")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_key_here":
            self.client_ready = False
            print("[INFO] GEMINI_API_KEY not configured")
        else:
            genai.configure(api_key=api_key)
            self.client_ready = True
            print("[INFO] Gemini configured successfully")

    async def get_answer(self, doc_id: str, question: str, vector_store: VectorStore, top_k: int = 5) -> Dict:
        print(f"[CHAT] get_answer called for doc_id={doc_id}")

        question_embeddings = await get_embeddings([question])
        question_embedding = question_embeddings[0]
        print("[CHAT] Question embedding created")

        results = vector_store.search(question_embedding, doc_id, top_k=top_k)
        print(f"[CHAT] Vector search returned {len(results)} results")

        if not results:
            return {
                "answer": "No relevant information found in the document.",
                "sources": []
            }

        context_chunks = []
        sources = []

        for result in results:
            metadata = result["metadata"]
            context_chunks.append(metadata["text"])
            sources.append({
                "page": metadata["page"],
                "chunk_id": metadata["chunk_id"],
                "text_preview": metadata["text"][:200] + "..." if len(metadata["text"]) > 200 else metadata["text"]
            })

        context = "\n\n".join(context_chunks)
        print(f"[CHAT] Built context with {len(context_chunks)} chunks")

        system_prompt = """You are a helpful assistant that answers questions based solely on the provided context from a document.
If the answer cannot be found in the context, say: "I cannot find this information in the document."
Do not make up information or use knowledge outside the provided context."""

        user_prompt = f"""Context from document:
{context}

Question: {question}

Answer the question using only the information from the context above. If the answer is not in the context, say so."""

        if not self.client_ready:
            return {
                "answer": "Error: GEMINI_API_KEY not configured. Please set your API key in the .env file.",
                "sources": []
            }

        def _get_gemini_response():
            print("[CHAT] Starting Gemini non-streamed generation...")
            model = genai.GenerativeModel(
                model_name="models/gemini-1.5-flash",
                system_instruction=system_prompt
            )
            response = model.generate_content(user_prompt)
            print("[CHAT] Gemini non-streamed generation completed")
            return response.text if hasattr(response, "text") and response.text else "I cannot find this information in the document."

        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(None, _get_gemini_response)

        return {
            "answer": answer,
            "sources": sources
        }

    async def get_answer_stream(
        self, doc_id: str, question: str, vector_store: VectorStore, top_k: int = 5
    ) -> AsyncGenerator[str, None]:
        print(f"[CHAT] Received question for doc_id={doc_id}: {question}")

        question_embeddings = await get_embeddings([question], task_type="retrieval_query")
        question_embedding = question_embeddings[0]
        print("[CHAT] Question embedding created")

        results = vector_store.search(question_embedding, doc_id, top_k=top_k)
        print(f"[CHAT] Vector search returned {len(results)} results")

        if not results:
            yield json.dumps({
                "chunk": "No relevant information found in the document.",
                "done": True,
                "sources": []
            }) + "\n"
            return

        context_chunks = []
        sources = []

        for result in results:
            metadata = result["metadata"]
            context_chunks.append(metadata["text"])
            sources.append({
                "page": metadata["page"],
                "chunk_id": metadata["chunk_id"],
                "text_preview": metadata["text"][:200] + "..." if len(metadata["text"]) > 200 else metadata["text"]
            })

        context = "\n\n".join(context_chunks)
        print(f"[CHAT] Built context with {len(context_chunks)} chunks")

        system_prompt = """You are a helpful assistant that answers questions based solely on the provided context from a document.
If the answer cannot be found in the context, say: "I cannot find this information in the document."
Do not make up information or use knowledge outside the provided context."""

        user_prompt = f"""Context from document:
{context}

Question: {question}

Answer the question using only the information from the context above. If the answer is not in the context, say so."""

        if not self.client_ready:
            yield json.dumps({
                "chunk": "Error: GEMINI_API_KEY not configured.",
                "done": True,
                "sources": []
            }) + "\n"
            return

        chunk_queue: Queue = Queue()
        done_sentinel = object()

        def _run_stream():
            try:
                print("[CHAT] Starting Gemini streamed generation...")
                model = genai.GenerativeModel(
                    model_name="models/gemini-2.5-flash",
                    system_instruction=system_prompt
                )

                response = model.generate_content(user_prompt, stream=True)

                got_any_text = False
                for chunk in response:
                    text = getattr(chunk, "text", None)
                    if text:
                        got_any_text = True
                        print(f"[CHAT] Stream chunk received: {text[:80]!r}")
                        chunk_queue.put(text)

                if not got_any_text:
                    print("[CHAT] Gemini returned no stream text")
                    chunk_queue.put("I cannot find this information in the document.")

            except Exception as e:
                print(f"[CHAT] Gemini streaming error: {e}")
                chunk_queue.put(f"\n[ERROR] Gemini streaming failed: {str(e)}")
            finally:
                print("[CHAT] Stream finished")
                chunk_queue.put(done_sentinel)

        thread = threading.Thread(target=_run_stream, daemon=True)
        thread.start()

        while True:
            try:
                item = chunk_queue.get(timeout=0.1)
            except Empty:
                await asyncio.sleep(0.02)
                continue

            if item is done_sentinel:
                break

            yield json.dumps({"chunk": item}) + "\n"

        yield json.dumps({"done": True, "sources": sources}) + "\n"