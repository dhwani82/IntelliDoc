from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> List[Dict]:
    """
    Split document text into overlapping chunks for embedding and retrieval.

    Uses RecursiveCharacterTextSplitter with separators ordered from coarsest
    to finest: paragraph (\\n\\n) → line (\\n) → sentence (. ? !) → word → character.
    The splitter prefers earlier separators so chunks break on natural boundaries
    whenever possible. That preserves semantic units better than a naive
    fixed-length slice, which often cuts mid-sentence and hurts retrieval quality.

    Returns dicts with cleaned text plus start/end offsets in the original string
    so callers can map chunks back to pages.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        # Prefer paragraph/line/sentence breaks before falling back to words/chars.
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)

    chunks: List[Dict] = []
    search_from = 0

    for raw_chunk in raw_chunks:
        cleaned_chunk = " ".join(raw_chunk.split()).strip()
        if not cleaned_chunk:
            continue
        start = text.find(raw_chunk.strip(), search_from)
        if start == -1:
            start = search_from
        end = start + len(raw_chunk.strip())
        search_from = max(start + 1, end - overlap)
        chunks.append({"text": cleaned_chunk, "start": start, "end": end})

    return chunks
