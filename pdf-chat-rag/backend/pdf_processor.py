import fitz  # PyMuPDF
import os
import gc
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from chunking import chunk_text
from embeddings import get_embeddings
from vector_store import VectorStore

# Thread pool so PDF extraction doesn't block the event loop
_executor = ThreadPoolExecutor(max_workers=2)

# Reduce memory: smaller batches, sequential (no concurrent)
EMBED_BATCH_SIZE = 25
EMBED_MAX_CONCURRENT = 1
MAX_CHUNKS = 500  # Limit chunks to avoid MemoryError on large PDFs
MAX_TEXT_CHARS = 500_000  # Truncate very large documents


def _extract_all_pages(doc_path: str) -> List[Dict]:
    """Extract text from all pages (runs in thread, opens file once)."""
    doc = fitz.open(doc_path)
    try:
        return [
            {"page": i + 1, "text": doc[i].get_text()}
            for i in range(len(doc))
        ]
    finally:
        doc.close()


async def process_pdf(file_path: str, doc_id: str, vector_store: VectorStore):
    try:
        print(f"[INFO] Step 1: Extracting text from PDF: {file_path}")
        loop = asyncio.get_running_loop()
        pages_text = await loop.run_in_executor(_executor, _extract_all_pages, file_path)
        print(f"[INFO] Step 1 done: extracted {len(pages_text)} pages")

        full_text = "\n\n".join([page["text"] for page in pages_text])
        print("[DEBUG] First 1000 characters of extracted full_text:")
        print(full_text[:1000])

        print("[DEBUG] First page raw text preview:")
        print(pages_text[0]["text"][:1000] if pages_text else "No pages found")

        if not full_text.strip():
            raise ValueError("No text found in PDF")

        if len(full_text) > MAX_TEXT_CHARS:
            print(f"[INFO] Truncating from {len(full_text)} to {MAX_TEXT_CHARS} chars")
            full_text = full_text[:MAX_TEXT_CHARS] + "\n\n[Document truncated for memory limits.]"

        print(f"[INFO] Step 2: Chunking text...")
        chunks = chunk_text(full_text)

        if len(chunks) > MAX_CHUNKS:
            print(f"[INFO] Limiting chunks from {len(chunks)} to {MAX_CHUNKS}")
            chunks = chunks[:MAX_CHUNKS]

        print(f"[INFO] Step 2 done: created {len(chunks)} chunks")
        del full_text

        chunk_texts = [" ".join(chunk["text"].split()) for chunk in chunks]
        print(f"[INFO] Step 3: Creating embeddings for {len(chunk_texts)} chunks")

        all_embeddings = []
        batches = [
            chunk_texts[i:i + EMBED_BATCH_SIZE]
            for i in range(0, len(chunk_texts), EMBED_BATCH_SIZE)
        ]

        for idx, b in enumerate(batches):
            print(f"[INFO] Embedding batch {idx + 1}/{len(batches)} with {len(b)} chunks")
            result = await get_embeddings(b, task_type="retrieval_document")
            print(f"[INFO] Batch {idx + 1} returned {len(result)} embeddings")
            all_embeddings.extend(result)
            gc.collect()

        embeddings = all_embeddings
        print(f"[INFO] Step 3 done: total embeddings = {len(embeddings)}")

        if len(embeddings) != len(chunks):
            raise ValueError(f"Embeddings count mismatch: {len(embeddings)} embeddings for {len(chunks)} chunks")

        print(f"[INFO] Step 4: Preparing metadata...")
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            page_num = 1
            char_count = 0
            for page in pages_text:
                if char_count + len(page["text"]) >= chunk["start"]:
                    page_num = page["page"]
                    break
                char_count += len(page["text"]) + 2

            chunk_metadata.append({
                "doc_id": doc_id,
                "chunk_id": i,
                "text": chunk["text"],
                "start": chunk["start"],
                "end": chunk["end"],
                "page": page_num
            })

        print(f"[INFO] Step 5: Storing vectors in FAISS...")
        vector_store.add_document(doc_id, embeddings, chunk_metadata)
        print(f"[INFO] Step 5 done: vectors stored successfully")

        return {
            "doc_id": doc_id,
            "num_chunks": len(chunks),
            "num_pages": len(pages_text)
        }

    except Exception as e:
        print(f"[ERROR] Error in process_pdf: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise