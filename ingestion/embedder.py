"""
Embedding client with OpenAI integration and caching.
Provider abstraction for easy model/provider switching.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict,  Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)

# OpenAI limit observed in practice: 300k tokens per request.
# Keep a safety margin to avoid request failures on large corpora.
EMBED_MAX_TOKENS_PER_REQUEST = 240_000
# Also cap number of texts per request to keep latency predictable.
EMBED_MAX_TEXTS_PER_REQUEST = 512


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed list of texts, return list of vectors."""
        ...
    
    def embed_single(self, text: str) -> List[float]:
        """Embed single text, return vector."""
        ...


class OpenAIEmbedder:
    """
    OpenAI embeddings client with caching and retry logic.
    Default model: text-embedding-3-small
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        cache_dir: Optional[str] = None,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        timeout: int = 30,
        rate_limit_rpm: int = 500,
    ):
        self.api_key = api_key
        self.model = model
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        self.rate_limit_rpm = rate_limit_rpm
        
        # Rate limiting: min delay between requests
        self._min_delay = 60.0 / rate_limit_rpm if rate_limit_rpm > 0 else 0
        self._last_request_time = 0.0
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        self._client = None
        self._tokenizer = None
        try:
            import tiktoken  # type: ignore
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback to a conservative heuristic if tiktoken is unavailable.
            self._tokenizer = None
    
    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai not installed. Run: pip install openai")
            self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed list of texts.
        Uses cache for previously embedded texts.
        """
        if not texts:
            return []
        
        results = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self._get_cached(text)
            if cached is not None:
                results[i] = cached
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Embed uncached texts
        if texts_to_embed:
            embeddings = self._embed_with_retry(texts_to_embed)
            
            for idx, (text, embedding) in zip(indices_to_embed, zip(texts_to_embed, embeddings)):
                results[idx] = embedding
                self._save_cache(text, embedding)
        
        return results
    
    def embed_single(self, text: str) -> List[float]:
        """Embed single text."""
        results = self.embed([text])
        return results[0] if results else []
    
    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts with retry logic, rate limiting, and concurrency control.
        Large requests are automatically split into safe token-bounded batches.
        """
        if not texts:
            return []

        embeddings: List[List[float]] = []
        for batch in self._batch_texts(texts):
            embeddings.extend(self._embed_batch_with_retry(batch))
        return embeddings

    def _embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Embed a single batch with retry logic and limiter protection."""
        from shared import get_sync_limiter
        limiter = get_sync_limiter()
        
        for attempt in range(self.max_retries):
            try:
                # Acquire semaphore + rate limit
                limiter.acquire_embedding()
                
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=texts,
                        timeout=self.timeout,
                    )
                    
                    # Sort by index to maintain order
                    embeddings = [None] * len(texts)
                    for item in response.data:
                        embeddings[item.index] = item.embedding
                    
                    return embeddings
                    
                finally:
                    limiter.release_embedding()
                
            except Exception as e:
                wait_time = self.backoff_base ** attempt
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}. Retry in {wait_time}s...")
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise
        
        return []

    def _batch_texts(self, texts: List[str]) -> List[List[str]]:
        """Split texts into batches bounded by token and count budgets."""
        batches: List[List[str]] = []
        current: List[str] = []
        current_tokens = 0

        for text in texts:
            est_tokens = self._estimate_tokens(text)

            would_exceed_tokens = current and (current_tokens + est_tokens > EMBED_MAX_TOKENS_PER_REQUEST)
            would_exceed_count = current and (len(current) >= EMBED_MAX_TEXTS_PER_REQUEST)
            if would_exceed_tokens or would_exceed_count:
                batches.append(current)
                current = []
                current_tokens = 0

            current.append(text)
            current_tokens += est_tokens

        if current:
            batches.append(current)

        if len(batches) > 1:
            logger.info(f"Embedding split into {len(batches)} batches due to token budget")

        return batches

    def _estimate_tokens(self, text: str) -> int:
        """Conservative token estimate to keep batch sizes below API limits."""
        if not text:
            return 1
        if self._tokenizer is not None:
            try:
                return max(1, len(self._tokenizer.encode(text)))
            except Exception:
                pass
        # Conservative fallback: assume ~3 characters per token.
        return max(1, len(text) // 3)

    def _compute_hash(self, text: str) -> str:
        """Compute hash for cache key."""
        content = f"{self.model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached(self, text: str) -> Optional[List[float]]:
        """Get cached embedding if exists."""
        if not self.cache_dir:
            return None
        
        cache_file = Path(self.cache_dir) / f"{self._compute_hash(text)}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                return None
        return None
    
    def _save_cache(self, text: str, embedding: List[float]) -> None:
        """Save embedding to cache."""
        if not self.cache_dir:
            return
        
        cache_file = Path(self.cache_dir) / f"{self._compute_hash(text)}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")


def create_embedder(config: dict) -> OpenAIEmbedder:
    """Factory function to create embedder from config."""
    return OpenAIEmbedder(
        api_key=config.get("openai_api_key", os.getenv("OPENAI_API_KEY", "")),
        model=config.get("model_embed", "text-embedding-3-small"),
        cache_dir=config.get("embedding_cache_dir"),
        max_retries=config.get("api_max_retries", 3),
        backoff_base=config.get("api_backoff_base", 2.0),
        timeout=config.get("api_timeout", 30),
        rate_limit_rpm=config.get("api_rate_limit_rpm", 500),
    )
