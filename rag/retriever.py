"""
Retriever module - vector search with optional BM25 hybrid.
"""

from __future__ import annotations

import logging
from typing import Optional

from ingestion.embedder import OpenAIEmbedder, create_embedder
from ingestion.index_faiss import FAISSIndex, create_index
from ingestion.models import Chunk, ScoredChunk

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves relevant chunks for a query.
    Supports dense (vector) search and optional BM25 hybrid.
    """
    
    def __init__(
        self,
        embedder: OpenAIEmbedder,
        index: FAISSIndex,
        chunks_lookup: Dict[str, Chunk],  # chunk_id -> Chunk
        top_k: int = 5,
        hybrid_enabled: bool = False,
    ):
        self.embedder = embedder
        self.index = index
        self.chunks_lookup = chunks_lookup
        self.top_k = top_k
        self.hybrid_enabled = hybrid_enabled
        
        self._bm25 = None
        if hybrid_enabled:
            self._init_bm25()
    
    def _init_bm25(self) -> None:
        """Initialize BM25 index for hybrid search."""
        try:
            from rank_bm25 import BM25Okapi
            
            # Tokenize all chunk contents
            self._bm25_chunk_ids = list(self.chunks_lookup.keys())
            corpus = [
                self.chunks_lookup[cid].content.lower().split()
                for cid in self._bm25_chunk_ids
            ]
            self._bm25 = BM25Okapi(corpus)
            logger.info(f"BM25 initialized with {len(corpus)} documents")
            
        except ImportError:
            logger.warning("rank-bm25 not installed. Hybrid search disabled.")
            self._bm25 = None
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[ScoredChunk]:
        """
        Search for relevant chunks.
        
        Args:
            query: User question
            top_k: Number of results (uses default if None)
            
        Returns:
            List of ScoredChunk sorted by relevance
        """
        k = top_k or self.top_k
        
        # Dense search
        query_vector = self.embedder.embed_single(query)
        dense_results = self.index.search(query_vector, k * 2 if self.hybrid_enabled else k)
        
        # Convert to ScoredChunk
        scored_chunks: Dict[str, ScoredChunk] = {}
        
        for chunk_id, score, meta in dense_results:
            chunk = self.chunks_lookup.get(chunk_id)
            if chunk:
                scored_chunks[chunk_id] = ScoredChunk(chunk=chunk, score=score)
        
        # Optional BM25 hybrid
        if self.hybrid_enabled and self._bm25:
            bm25_results = self._search_bm25(query, k)
            for chunk_id, bm25_score in bm25_results:
                if chunk_id in scored_chunks:
                    # Combine scores (simple average)
                    existing = scored_chunks[chunk_id]
                    combined_score = (existing.score + bm25_score) / 2
                    scored_chunks[chunk_id] = ScoredChunk(
                        chunk=existing.chunk,
                        score=combined_score,
                    )
                else:
                    chunk = self.chunks_lookup.get(chunk_id)
                    if chunk:
                        scored_chunks[chunk_id] = ScoredChunk(chunk=chunk, score=bm25_score * 0.5)
        
        # Sort by score descending
        results = sorted(scored_chunks.values(), key=lambda x: x.score, reverse=True)
        
        return results[:k]
    
    def _search_bm25(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """BM25 search."""
        if not self._bm25:
            return []
        
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk_id = self._bm25_chunk_ids[idx]
                # Normalize BM25 score to 0-1 range (approximate)
                normalized_score = min(scores[idx] / 10.0, 1.0)
                results.append((chunk_id, normalized_score))
        
        return results


def create_retriever(config: dict, chunks: List[Chunk]) -> Retriever:
    """Factory function to create retriever from config."""
    embedder = create_embedder(config)
    index = create_index(config)
    
    # Build lookup
    chunks_lookup = {c.chunk_id: c for c in chunks}
    
    return Retriever(
        embedder=embedder,
        index=index,
        chunks_lookup=chunks_lookup,
        top_k=config.get("top_k", 5),
        hybrid_enabled=config.get("hybrid_enabled", False),
    )
