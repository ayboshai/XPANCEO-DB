"""
Qdrant vector index implementation (placeholder).
To be implemented when switching from FAISS.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class QdrantIndex:
    """
    Qdrant-based vector index.
    Placeholder implementation - use FAISS for now.
    """
    
    def __init__(
        self,
        collection_name: str = "xpanceo_chunks",
        host: str = "localhost",
        port: int = 6333,
        dimension: int = 1536,
    ):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.dimension = dimension
        
        raise NotImplementedError(
            "Qdrant index not yet implemented. Use FAISS by setting index_backend: faiss in config."
        )
    
    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict],
    ) -> None:
        """Insert or update vectors with metadata."""
        raise NotImplementedError()
    
    def search(
        self,
        vector: list[float],
        top_k: int = 5,
    ) -> list[tuple[str, float, dict]]:
        """Search for similar vectors."""
        raise NotImplementedError()
    
    def delete(self, ids: list[str]) -> None:
        """Delete vectors by ID."""
        raise NotImplementedError()
    
    def clear(self) -> None:
        """Clear all data from index."""
        raise NotImplementedError()
    
    @property
    def count(self) -> int:
        """Number of vectors in index."""
        raise NotImplementedError()
