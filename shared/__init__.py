"""
Shared utilities for XPANCEO DB.
Configuration loading, OCR quality checking, retry decorator.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import yaml

logger = logging.getLogger(__name__)

# Type for retry decorator
T = TypeVar("T")


# =============================================================================
# Configuration Loading
# =============================================================================

_config_cache: Optional[dict] = None


def load_config(config_path: str = "config/master_config.yaml", force_reload: bool = False) -> dict:
    """
    Load configuration from YAML file.
    Resolves environment variables and caches result.
    
    Args:
        config_path: Path to config file
        force_reload: Force reload from disk
        
    Returns:
        Configuration dictionary
    """
    global _config_cache
    
    if _config_cache is not None and not force_reload:
        return _config_cache
    
    def resolve_env(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var, "")
        return value
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Resolve environment variables
    for key, value in config.items():
        config[key] = resolve_env(value)
    
    _config_cache = config
    return config


def get_config_value(key: str, default: Any = None) -> Any:
    """Get single config value."""
    config = load_config()
    return config.get(key, default)


# =============================================================================
# OCR Quality Checker (Single Source of Truth)
# =============================================================================

class OCRQualityChecker:
    """
    Centralized OCR quality checking using config thresholds.
    Use this everywhere instead of hardcoded values.
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        
        # Load thresholds from config
        self.confidence_threshold = self.config.get("ocr_confidence_threshold", 60)
        self.min_chars = self.config.get("ocr_min_chars", 50)
        self.min_tokens = self.config.get("ocr_min_tokens", 10)
        self.min_alpha_ratio = self.config.get("ocr_min_alpha_ratio", 0.4)
    
    def is_failed(
        self,
        confidence: float,
        char_count: int,
        word_count: int,
        alpha_ratio: float,
    ) -> bool:
        """
        Check if OCR result indicates failure.
        ANY condition being True triggers failure.
        
        Args:
            confidence: Tesseract mean confidence 0-100
            char_count: Total character count
            word_count: Total word count
            alpha_ratio: Ratio of alphanumeric characters
            
        Returns:
            True if OCR failed
        """
        return (
            confidence < self.confidence_threshold or
            char_count < self.min_chars or
            word_count < self.min_tokens or
            alpha_ratio < self.min_alpha_ratio
        )
    
    def get_failure_reason(
        self,
        confidence: float,
        char_count: int,
        word_count: int,
        alpha_ratio: float,
    ) -> Optional[str]:
        """Get human-readable failure reason."""
        reasons = []
        
        if confidence < self.confidence_threshold:
            reasons.append(f"confidence={confidence:.1f}<{self.confidence_threshold}")
        if char_count < self.min_chars:
            reasons.append(f"chars={char_count}<{self.min_chars}")
        if word_count < self.min_tokens:
            reasons.append(f"tokens={word_count}<{self.min_tokens}")
        if alpha_ratio < self.min_alpha_ratio:
            reasons.append(f"alpha_ratio={alpha_ratio:.2f}<{self.min_alpha_ratio}")
        
        return "; ".join(reasons) if reasons else None


# Global instance (lazy loaded)
_ocr_checker: Optional[OCRQualityChecker] = None


def get_ocr_checker(config: Optional[dict] = None) -> OCRQualityChecker:
    """Get or create OCR quality checker."""
    global _ocr_checker
    if _ocr_checker is None or config is not None:
        _ocr_checker = OCRQualityChecker(config)
    return _ocr_checker


# =============================================================================
# Retry Decorator with Exponential Backoff
# =============================================================================

def retry_with_backoff(
    max_retries: Optional[int] = None,
    backoff_base: Optional[float] = None,
    timeout: Optional[int] = None,
    exceptions: tuple = (Exception,),
    on_failure: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retry with exponential backoff.
    
    Args:
        max_retries: Max retry attempts (from config if None)
        backoff_base: Backoff base in seconds (from config if None)
        timeout: Request timeout (from config if None)
        exceptions: Exceptions to catch
        on_failure: Callback on each failure (exception, attempt)
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            config = load_config()
            retries = max_retries or config.get("api_max_retries", 3)
            base = backoff_base or config.get("api_backoff_base", 2.0)
            
            last_exception = None
            
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    wait_time = base ** attempt
                    
                    if on_failure:
                        on_failure(e, attempt)
                    else:
                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1}/{retries} failed: {e}. "
                            f"Retry in {wait_time:.1f}s..."
                        )
                    
                    if attempt < retries - 1:
                        time.sleep(wait_time)
            
            # All retries exhausted
            raise last_exception
        
        return wrapper
    return decorator


# =============================================================================
# Async Semaphore Pool (for parallelism limits)
# =============================================================================

import asyncio
from contextlib import asynccontextmanager


class ParallelismLimiter:
    """
    Limits concurrent operations based on config.
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self._embedding_semaphore: Optional[asyncio.Semaphore] = None
        self._vision_semaphore: Optional[asyncio.Semaphore] = None
    
    @property
    def embedding_semaphore(self) -> asyncio.Semaphore:
        if self._embedding_semaphore is None:
            limit = self.config.get("max_concurrent_embeddings", 5)
            self._embedding_semaphore = asyncio.Semaphore(limit)
        return self._embedding_semaphore
    
    @property
    def vision_semaphore(self) -> asyncio.Semaphore:
        if self._vision_semaphore is None:
            limit = self.config.get("max_concurrent_vision", 2)
            self._vision_semaphore = asyncio.Semaphore(limit)
        return self._vision_semaphore
    
    @asynccontextmanager
    async def limit_embeddings(self):
        """Context manager for embedding rate limiting."""
        async with self.embedding_semaphore:
            yield
    
    @asynccontextmanager
    async def limit_vision(self):
        """Context manager for vision rate limiting."""
        async with self.vision_semaphore:
            yield


# Global limiter
_limiter: Optional[ParallelismLimiter] = None
_limiter_sig: Optional[tuple] = None


def get_limiter(config: Optional[dict] = None) -> ParallelismLimiter:
    """Return a cached async limiter, reinitializing only on config change."""
    global _limiter
    global _limiter_sig

    def _sig(cfg: Optional[dict]) -> tuple:
        cfg = cfg or load_config()
        return (
            cfg.get("max_concurrent_embeddings", 5),
            cfg.get("max_concurrent_vision", 2),
        )

    if _limiter is None:
        _limiter = ParallelismLimiter(config)
        _limiter_sig = _sig(config)
        return _limiter

    if config is not None:
        new_sig = _sig(config)
        if new_sig != _limiter_sig:
            logger.info(
                "Reinitializing async limiter due to config change: "
                f"{_limiter_sig} -> {new_sig}"
            )
            _limiter = ParallelismLimiter(config)
            _limiter_sig = new_sig

    return _limiter


import threading


class SyncRateLimiter:
    """Thread-safe semaphore + RPM limiter for sync API calls."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()

        max_embed = self.config.get("max_concurrent_embeddings", 5)
        max_vision = self.config.get("max_concurrent_vision", 2)
        max_judge = self.config.get("max_concurrent_judge", 3)

        self._embed_sem = threading.Semaphore(max_embed)
        self._vision_sem = threading.Semaphore(max_vision)
        self._judge_sem = threading.Semaphore(max_judge)

        # Single global RPM gate across API types.
        self.rpm = self.config.get("api_rate_limit_rpm", 500)
        self._min_delay = 60.0 / self.rpm if self.rpm > 0 else 0
        self._last_api_call = 0.0
        self._lock = threading.Lock()

    def _wait_for_rate_limit(self) -> None:
        """Sleep to respect the shared RPM budget."""
        if self._min_delay <= 0:
            return

        with self._lock:
            now = time.time()
            elapsed = now - self._last_api_call

            if elapsed < self._min_delay:
                time.sleep(self._min_delay - elapsed)

            self._last_api_call = time.time()

    def acquire_embedding(self) -> None:
        """Acquire embedding slot."""
        self._embed_sem.acquire()
        self._wait_for_rate_limit()

    def release_embedding(self) -> None:
        """Release embedding slot."""
        self._embed_sem.release()

    def acquire_vision(self) -> None:
        """Acquire vision slot."""
        self._vision_sem.acquire()
        self._wait_for_rate_limit()

    def release_vision(self) -> None:
        """Release vision slot."""
        self._vision_sem.release()

    def acquire_judge(self) -> None:
        """Acquire judge slot."""
        self._judge_sem.acquire()
        self._wait_for_rate_limit()

    def release_judge(self) -> None:
        """Release judge slot."""
        self._judge_sem.release()


# Global sync limiter
_sync_limiter: Optional[SyncRateLimiter] = None
_sync_limiter_sig: Optional[tuple] = None
_sync_limiter_lock = threading.Lock()


def get_sync_limiter(config: Optional[dict] = None) -> SyncRateLimiter:
    """Return a cached sync limiter, reinitializing only on config change."""
    global _sync_limiter
    global _sync_limiter_sig

    def _sig(cfg: Optional[dict]) -> tuple:
        cfg = cfg or load_config()
        return (
            cfg.get("max_concurrent_embeddings", 5),
            cfg.get("max_concurrent_vision", 2),
            cfg.get("max_concurrent_judge", 3),
            cfg.get("api_rate_limit_rpm", 500),
        )

    with _sync_limiter_lock:
        if _sync_limiter is None:
            _sync_limiter = SyncRateLimiter(config)
            _sync_limiter_sig = _sig(config)
            logger.info(
                "Initialized sync limiter: "
                f"embed={_sync_limiter_sig[0]} vision={_sync_limiter_sig[1]} "
                f"judge={_sync_limiter_sig[2]} rpm={_sync_limiter_sig[3]}"
            )
            return _sync_limiter

        if config is not None:
            new_sig = _sig(config)
            if new_sig != _sync_limiter_sig:
                logger.info(
                    "Reinitializing sync limiter due to config change: "
                    f"{_sync_limiter_sig} -> {new_sig}"
                )
                _sync_limiter = SyncRateLimiter(config)
                _sync_limiter_sig = new_sig

        return _sync_limiter
