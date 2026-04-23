import faiss
import numpy as np
import json
import os
from typing import List, Dict


class VectorStore:
    def __init__(self):
        self.index = None
        self.metadata = []  # index-aligned with FAISS
        self.dimension = None  # set dynamically from first embedding batch
        self.doc_indices = {}  # Maps doc_id to list of chunk indices

    def _initialize_index(self, dimension: int):
        """Initialize FAISS index if it doesn't exist."""
        if self.index is None:
            self.dimension = dimension
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

        # Safety check if index already exists
        if self.index.d != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: FAISS index expects {self.index.d}, got {embedding_dim}"
            )

        start_idx = self.index.ntotal
        self.index.add(embeddings_array)

        chunk_indices = []
        for i, metadata in enumerate(chunk_metadata):
            idx = start_idx + i
            self.metadata.append(metadata)
            chunk_indices.append(idx)

        self.doc_indices[doc_id] = chunk_indices
        self._save()

    def search(self, query_embedding: List[float], doc_id: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks in a specific document.
        """
        if self.index is None or doc_id not in self.doc_indices:
            return []

        query_array = np.array([query_embedding], dtype="float32")

        if query_array.ndim != 2:
            raise ValueError(f"Query embedding must be 2D after wrapping, got shape {query_array.shape}")

        if query_array.shape[1] != self.index.d:
            raise ValueError(
                f"Query embedding dimension mismatch: FAISS index expects {self.index.d}, got {query_array.shape[1]}"
            )

        k = min(top_k * 2, self.index.ntotal)
        distances, indices = self.index.search(query_array, k)

        doc_chunk_indices = set(self.doc_indices[doc_id])
        results = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx in doc_chunk_indices and idx < len(self.metadata):
                results.append({
                    "metadata": self.metadata[idx],
                    "distance": float(dist)
                })
                if len(results) >= top_k:
                    break

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
            json.dump({"dimension": self.dimension}, f, indent=2)

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

    def load_existing(self):
        """Load existing data if available."""
        self._load()