"""
FAISS vector index implementation.
Stores embeddings with metadata for retrieval.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    FAISS-based vector index with metadata storage.
    Uses IndexFlatIP (inner product) for cosine similarity with normalized vectors.
    """
    
    def __init__(
        self,
        index_dir: str,
        dimension: int = 1536,  # text-embedding-3-small dimension
    ):
        self.index_dir = index_dir
        self.dimension = dimension
        
        self._index = None
        self._metadata: dict[str, dict] = {}  # chunk_id -> metadata
        self._id_map: list[str] = []  # position -> chunk_id
        
        os.makedirs(index_dir, exist_ok=True)
        
        # Load existing index if available
        self._load()
    
    @property
    def index(self):
        """Lazy initialization of FAISS index."""
        if self._index is None:
            try:
                import faiss
            except ImportError:
                raise ImportError("faiss not installed. Run: pip install faiss-cpu")
            
            # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
            self._index = faiss.IndexFlatIP(self.dimension)
        
        return self._index
    
    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict],
    ) -> None:
        """
        Insert or update vectors with metadata.
        
        Args:
            ids: List of chunk IDs
            vectors: List of embedding vectors
            metadata: List of metadata dicts
        """
        if not ids:
            return
        
        import faiss
        
        # Normalize vectors for cosine similarity
        vectors_np = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors_np)
        
        # Remove existing entries (for update)
        existing_ids = set(ids) & set(self._id_map)
        if existing_ids:
            # For simplicity, rebuild index without existing entries
            # In production, would use IDMap for efficient updates
            self._remove_ids(existing_ids)
        
        # Add new vectors
        self.index.add(vectors_np)
        
        # Update mappings
        for chunk_id, meta in zip(ids, metadata):
            self._id_map.append(chunk_id)
            self._metadata[chunk_id] = meta
        
        # Persist
        self._save()
        
        logger.info(f"Indexed {len(ids)} vectors. Total: {self.index.ntotal}")
    
    def search(
        self,
        vector: list[float],
        top_k: int = 5,
    ) -> list[tuple[str, float, dict]]:
        """
        Search for similar vectors.
        
        Args:
            vector: Query vector
            top_k: Number of results
            
        Returns:
            List of (chunk_id, score, metadata) tuples
        """
        import faiss
        
        if self.index.ntotal == 0:
            return []
        
        # Normalize query vector
        query = np.array([vector], dtype=np.float32)
        faiss.normalize_L2(query)
        
        # Search
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._id_map):
                chunk_id = self._id_map[idx]
                meta = self._metadata.get(chunk_id, {})
                results.append((chunk_id, float(score), meta))
        
        return results
    
    def delete(self, ids: list[str]) -> None:
        """Delete vectors by ID."""
        self._remove_ids(set(ids))
        self._save()
    
    def _remove_ids(self, ids_to_remove: set[str]) -> None:
        """Remove IDs and rebuild index."""
        import faiss
        
        if not ids_to_remove:
            return
        
        # Get indices to keep
        keep_indices = [
            i for i, chunk_id in enumerate(self._id_map)
            if chunk_id not in ids_to_remove
        ]
        
        if not keep_indices:
            # Clear everything
            self._index = faiss.IndexFlatIP(self.dimension)
            self._id_map = []
            for chunk_id in ids_to_remove:
                self._metadata.pop(chunk_id, None)
            return
        
        # Extract vectors to keep
        vectors_to_keep = np.array([
            self.index.reconstruct(i) for i in keep_indices
        ], dtype=np.float32)
        
        # Create new index
        new_index = faiss.IndexFlatIP(self.dimension)
        new_index.add(vectors_to_keep)
        
        # Update mappings
        new_id_map = [self._id_map[i] for i in keep_indices]
        for chunk_id in ids_to_remove:
            self._metadata.pop(chunk_id, None)
        
        self._index = new_index
        self._id_map = new_id_map
    
    def _save(self) -> None:
        """Persist index and metadata to disk."""
        import faiss
        
        index_path = Path(self.index_dir) / "index.faiss"
        meta_path = Path(self.index_dir) / "metadata.pkl"
        
        try:
            faiss.write_index(self.index, str(index_path))
            with open(meta_path, "wb") as f:
                pickle.dump({
                    "metadata": self._metadata,
                    "id_map": self._id_map,
                }, f)
            logger.debug(f"Saved index to {self.index_dir}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def _load(self) -> None:
        """Load index and metadata from disk."""
        import faiss
        
        index_path = Path(self.index_dir) / "index.faiss"
        meta_path = Path(self.index_dir) / "metadata.pkl"
        
        if index_path.exists() and meta_path.exists():
            try:
                self._index = faiss.read_index(str(index_path))
                with open(meta_path, "rb") as f:
                    data = pickle.load(f)
                    self._metadata = data.get("metadata", {})
                    self._id_map = data.get("id_map", [])
                logger.info(f"Loaded index: {self.index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}. Starting fresh.")
    
    def clear(self) -> None:
        """Clear all data from index."""
        import faiss
        
        self._index = faiss.IndexFlatIP(self.dimension)
        self._metadata = {}
        self._id_map = []
        self._save()
    
    @property
    def count(self) -> int:
        """Number of vectors in index."""
        return self.index.ntotal if self._index else 0


def create_index(config: dict) -> FAISSIndex:
    """Factory function to create index from config."""
    return FAISSIndex(
        index_dir=config.get("index_dir", "data/index"),
        dimension=1536,  # text-embedding-3-small
    )
