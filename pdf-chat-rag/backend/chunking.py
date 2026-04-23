from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> List[Dict]:
    """
    Split text into semantically meaningful chunks with overlap.
    Optimized for RAG pipelines.

    Args:
        text: Full document text
        chunk_size: Maximum characters per chunk
        overlap: Overlap between chunks

    Returns:
        List of chunk dictionaries with text/start/end
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    text_length = len(text)

    chunks: List[Dict] = []
    start = 0

    while start < text_length:

        end = min(start + chunk_size, text_length)
        chunk = text[start:end]

        # Try to split on natural boundaries
        if end < text_length:

            break_points = [
                chunk.rfind("\n\n"),   # paragraph break
                chunk.rfind("\n"),     # newline
                chunk.rfind(". "),     # sentence end
                chunk.rfind("? "),
                chunk.rfind("! "),
            ]

            best_break = max(break_points)

            # Only adjust if break point is near end
            if best_break > chunk_size * 0.6:
                end = start + best_break + 1
                chunk = text[start:end]

        cleaned_chunk = " ".join(chunk.split()).strip()

        if cleaned_chunk:
            chunks.append({
                "text": cleaned_chunk,
                "start": start,
                "end": end
            })

        # Stop if we reached end
        if end >= text_length:
            break

        next_start = end - overlap

        # Ensure forward progress
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks