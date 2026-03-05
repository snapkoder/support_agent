"""
In-Memory Vector Store Adapter - Simple cosine similarity search in memory
Used when ChromaDB/SQLite are not needed (default for production pipeline)
"""

import math
import logging
from typing import List, Dict, Any, Optional

from ..ports import VectorStorePort, DocumentChunk, RetrievedChunk, VectorStoreError

logger = logging.getLogger(__name__)


class InMemoryVectorStoreAdapter(VectorStorePort):
    """
    In-memory vector store using cosine similarity.
    Stores document chunks and their embeddings for fast retrieval.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.InMemoryVectorStoreAdapter")
        self._chunks: List[DocumentChunk] = []
        self._embeddings: List[List[float]] = []
        self.logger.info("InMemoryVectorStoreAdapter initialized")

    async def upsert(self, chunks: List[DocumentChunk], embeddings: Optional[List[List[float]]] = None) -> None:
        """Insert or update document chunks with their embeddings."""
        if not chunks:
            return
        if embeddings and len(embeddings) != len(chunks):
            raise VectorStoreError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

        for i, chunk in enumerate(chunks):
            # Check if chunk already exists (update)
            existing_idx = next(
                (idx for idx, c in enumerate(self._chunks) if c.chunk_id == chunk.chunk_id),
                None
            )
            if existing_idx is not None:
                self._chunks[existing_idx] = chunk
                if embeddings:
                    self._embeddings[existing_idx] = embeddings[i]
            else:
                self._chunks.append(chunk)
                if embeddings:
                    self._embeddings.append(embeddings[i])
                else:
                    self._embeddings.append([])

        self.logger.info(f"Upserted {len(chunks)} chunks (total: {len(self._chunks)})")

    async def similarity_search(
        self,
        query_vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """Search for similar chunks using cosine similarity."""
        if not self._chunks or not query_vector:
            return []

        scored = []
        for i, (chunk, embedding) in enumerate(zip(self._chunks, self._embeddings)):
            if not embedding:
                continue
            # Apply metadata filters
            if filters and not self._matches_filters(chunk, filters):
                continue
            score = self._cosine_similarity(query_vector, embedding)
            scored.append((chunk, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for chunk, score in scored[:top_k]:
            results.append(RetrievedChunk(chunk=chunk, score=score, retrieval_method="cosine"))

        return results

    async def persist(self) -> None:
        """No-op for in-memory store."""
        pass

    async def load(self) -> None:
        """No-op for in-memory store."""
        pass

    async def health_check(self) -> bool:
        return True

    async def get_stats(self) -> Dict[str, Any]:
        return {
            "total_chunks": len(self._chunks),
            "indexed_chunks": sum(1 for e in self._embeddings if e),
        }

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)

    @staticmethod
    def _matches_filters(chunk: DocumentChunk, filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            chunk_val = getattr(chunk, key, None)
            if chunk_val is None:
                return False
            if isinstance(value, list):
                if chunk_val not in value:
                    return False
            elif chunk_val != value:
                return False
        return True
