import os
import asyncio
from typing import List
import google.generativeai as genai

_client_initialized = False


def init_gemini():
    global _client_initialized
    if not _client_initialized:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_key_here":
            raise ValueError("GEMINI_API_KEY not set. Please set it in .env file")
        genai.configure(api_key=api_key)
        _client_initialized = True


async def get_embeddings(texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
    """
    Create Gemini embeddings for a list of texts.

    Args:
        texts: List of input strings
        task_type: Gemini embedding task type
                   - retrieval_document for PDF chunks
                   - retrieval_query for user questions

    Returns:
        List of embedding vectors
    """
    def _get_embeddings_sync(texts: List[str], task_type: str) -> List[List[float]]:
        try:
            init_gemini()

            if not texts:
                return []

            cleaned_texts = []
            for text in texts:
                if text is None:
                    cleaned_texts.append("")
                else:
                    cleaned_texts.append(" ".join(str(text).split()).strip())

            print(f"[INFO] Calling Gemini Embedding API for {len(cleaned_texts)} texts with task_type={task_type}...")

            embeddings = []
            for i, text in enumerate(cleaned_texts, start=1):
                if not text:
                    raise ValueError(f"Empty text found at embedding item {i}")

                response = genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=text,
                    task_type=task_type
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

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _get_embeddings_sync, texts, task_type)