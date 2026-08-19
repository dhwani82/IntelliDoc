import faiss
import numpy as np
import json
import os
from typing import List, Dict

from embeddings import get_embedding_model_id


class VectorStore:
    def __init__(self):
        self.index = None
        self.metadata = []  # index-aligned with FAISS
        self.dimension = None  # set dynamically from first embedding batch
        self.doc_indices = {}  # Maps doc_id to list of chunk indices
        self.embedding_model = None  # id of the encoder that produced stored vectors

    def _initialize_index(self, dimension: int):
        """Initialize FAISS index if it doesn't exist."""
        if self.index is None:
            self.dimension = dimension
            # Exact L2 search: correct and simple for small/medium corpora, but
            # O(n) per query so it won't scale to millions of vectors. At that
            # size, swap to an approximate index (e.g. IndexIVFFlat / HNSW).
            self.index = faiss.IndexFlatL2(dimension)

    def add_document(self, doc_id: str, embeddings: List[List[float]], chunk_metadata: List[Dict]):
        """
        Add document embeddings and metadata to the vector store.
        """
        if not embeddings:
            raise ValueError("No embeddings provided")

        embeddings_array = np.array(embeddings, dtype="float32")

        if embeddings_array.ndim != 2:
            raise ValueError(f"Embeddings must be a 2D array, got shape {embeddings_array.shape}")

        num_vectors, embedding_dim = embeddings_array.shape

        if num_vectors != len(chunk_metadata):
            raise ValueError(
                f"Embeddings count mismatch: {num_vectors} embeddings for {len(chunk_metadata)} metadata rows"
            )

        # Initialize FAISS dynamically from actual embedding dimension
        self._initialize_index(embedding_dim)
        self._ensure_embedding_model()

        # Safety check if index already exists
        if self.index.d != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: FAISS index expects {self.index.d}, got {embedding_dim}. "
                "Re-index documents after switching EMBEDDING_BACKEND / USE_FINETUNED_EMBEDDINGS."
            )

        start_idx = self.index.ntotal
        self.index.add(embeddings_array)

        chunk_indices = []
        for i, metadata in enumerate(chunk_metadata):
            idx = start_idx + i
            self.metadata.append(metadata)
            chunk_indices.append(idx)

        self.doc_indices[doc_id] = chunk_indices
        print(
            f"[STORE] Added doc_id={doc_id}: {len(chunk_indices)} vectors "
            f"(indices {chunk_indices[0]}-{chunk_indices[-1]}), index.ntotal={self.index.ntotal}"
        )
        self._save()

    def get_all_chunks_for_doc(self, doc_id: str) -> List[Dict]:
        """Return every chunk for a document (used as fallback for small PDFs)."""
        if doc_id not in self.doc_indices:
            return []
        results = []
        for idx in self.doc_indices[doc_id]:
            if 0 <= idx < len(self.metadata):
                results.append({
                    "metadata": self.metadata[idx],
                    "distance": 0.0,
                })
        return results

    def search(self, query_embedding: List[float], doc_id: str, top_k: int = 10) -> List[Dict]:
        """
        Search for similar chunks within a specific document only.

        A global FAISS top-k over a shared index can return neighbors from other
        uploads and leave the current doc with zero hits after filtering. Earlier
        approaches overfetched (e.g. top_k * 2) then filtered by doc_id; that
        still failed when other docs dominated the neighbor list. Building a
        per-document sub-index searches only that doc's vectors, so filtering
        is unnecessary and retrieval stays scoped to the active PDF.
        """
        if self.index is None:
            print("[SEARCH] FAISS index is None")
            return []

        if doc_id not in self.doc_indices:
            known = list(self.doc_indices.keys())
            print(f"[SEARCH] doc_id not found: {doc_id!r}. Known doc_ids ({len(known)}): {known[-3:]}")
            return []

        chunk_indices = self.doc_indices[doc_id]
        print(
            f"[SEARCH] index.ntotal={self.index.ntotal}, doc_id={doc_id}, "
            f"doc_chunks={len(chunk_indices)}, top_k={top_k}"
        )

        if not chunk_indices:
            return []

        query_array = np.array([query_embedding], dtype="float32")

        if query_array.ndim != 2:
            raise ValueError(f"Query embedding must be 2D after wrapping, got shape {query_array.shape}")

        if query_array.shape[1] != self.index.d:
            raise ValueError(
                f"Query embedding dimension mismatch: FAISS index expects {self.index.d}, got {query_array.shape[1]}"
            )

        # Restrict search to this document's vectors (avoids cross-doc crowding).
        doc_vectors = np.vstack(
            [self.index.reconstruct(int(i)) for i in chunk_indices]
        ).astype("float32")
        sub_index = faiss.IndexFlatL2(self.index.d)
        sub_index.add(doc_vectors)

        k = min(top_k, len(chunk_indices))
        distances, local_indices = sub_index.search(query_array, k)

        results = []
        for dist, local_idx in zip(distances[0], local_indices[0]):
            if local_idx < 0:
                continue
            global_idx = chunk_indices[int(local_idx)]
            if global_idx >= len(self.metadata):
                continue
            meta = self.metadata[global_idx]
            results.append({
                "metadata": meta,
                "distance": float(dist),
            })
            preview = meta.get("text", "")[:80].replace("\n", " ")
            print(
                f"[SEARCH] hit global_idx={global_idx} chunk_id={meta.get('chunk_id')} "
                f"page={meta.get('page')} distance={dist:.4f} text={preview!r}"
            )

        print(f"[SEARCH] returning {len(results)} results for doc_id={doc_id}")
        return results

    def _save(self):
        """Save FAISS index and metadata to disk."""
        os.makedirs("data", exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, "data/faiss.index")

        with open("data/metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        with open("data/doc_indices.json", "w", encoding="utf-8") as f:
            json.dump(self.doc_indices, f, indent=2)

        with open("data/config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dimension": self.dimension,
                    "embedding_model": self.embedding_model or get_embedding_model_id(),
                },
                f,
                indent=2,
            )

    def _load(self):
        """Load FAISS index and metadata from disk."""
        index_path = "data/faiss.index"
        metadata_path = "data/metadata.json"
        doc_indices_path = "data/doc_indices.json"
        config_path = "data/config.json"

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.dimension = self.index.d

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        if os.path.exists(doc_indices_path):
            with open(doc_indices_path, "r", encoding="utf-8") as f:
                self.doc_indices = json.load(f)

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                self.dimension = config.get("dimension", self.dimension)
                self.embedding_model = config.get("embedding_model", self.embedding_model)

        current_model = get_embedding_model_id()
        if self.embedding_model and self.embedding_model != current_model:
            print(
                f"[STORE] Warning: index was built with {self.embedding_model!r} but "
                f"the active encoder is {current_model!r}. Re-index after changing "
                "USE_FINETUNED_EMBEDDINGS / EMBEDDING_BACKEND or search quality will drop."
            )

    def load_existing(self):
        """Load existing data if available."""
        self._load()

    def _ensure_embedding_model(self):
        """Record the active encoder; refuse mixing two embedding spaces in one index."""
        current = get_embedding_model_id()
        if self.embedding_model is None:
            self.embedding_model = current
            return
        if self.embedding_model != current:
            raise ValueError(
                f"Embedding model mismatch: store has {self.embedding_model!r}, "
                f"active encoder is {current!r}. Delete data/ or re-index after "
                "switching USE_FINETUNED_EMBEDDINGS / EMBEDDING_BACKEND."
            )