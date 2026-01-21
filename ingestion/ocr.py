"""
OCR module using Tesseract with caching and quality heuristics.
Always runs OCR on images; Vision fallback handled separately.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from PIL import Image

from .models import OCRResult

logger = logging.getLogger(__name__)


def compute_image_hash(image_path: str) -> str:
    """Compute MD5 hash of image file for caching."""
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def count_tokens(text: str) -> int:
    """Simple whitespace tokenization for token count."""
    return len(text.split())


def compute_alpha_ratio(text: str) -> float:
    """Compute ratio of alphanumeric characters to total characters."""
    if not text:
        return 0.0
    alnum = sum(1 for c in text if c.isalnum())
    return alnum / len(text)


class OCRProcessor:
    """
    Tesseract OCR wrapper with caching and quality heuristics.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        lang: str = "eng+rus",  # Multi-language support
        config: Optional[dict] = None,
    ):
        self.cache_dir = cache_dir
        self.lang = lang
        self.config = config or {}
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Load thresholds from config
        self.confidence_threshold = self.config.get("ocr_confidence_threshold", 60)
        self.min_chars = self.config.get("ocr_min_chars", 50)
        self.min_tokens = self.config.get("ocr_min_tokens", 10)
        self.min_alpha_ratio = self.config.get("ocr_min_alpha_ratio", 0.4)
    
    def run(self, image_path: str) -> OCRResult:
        """
        Run OCR on image. Returns OCRResult with text and quality metrics.
        Uses cache if available.
        """
        image_hash = compute_image_hash(image_path)
        
        # Check cache
        cached = self._get_cached(image_hash)
        if cached:
            logger.debug(f"OCR cache hit for {image_path}")
            return cached
        
        # Run Tesseract
        result = self._run_tesseract(image_path)
        
        # Cache result
        self._save_cache(image_hash, result)
        
        return result
    
    def _run_tesseract(self, image_path: str) -> OCRResult:
        """Execute Tesseract OCR on image."""
        try:
            import pytesseract
            from pytesseract import Output
        except ImportError:
            raise ImportError("pytesseract not installed. Run: pip install pytesseract")
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Get detailed OCR data with confidence scores
            ocr_data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                output_type=Output.DICT,
            )
            
            # Extract text and compute metrics
            words = []
            confidences = []
            
            for i, word in enumerate(ocr_data["text"]):
                conf = ocr_data["conf"][i]
                if word.strip() and conf > 0:  # Filter empty and invalid
                    words.append(word)
                    confidences.append(conf)
            
            text = " ".join(words)
            
            # Compute mean confidence (only valid scores)
            mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return OCRResult(
                text=text,
                confidence=mean_confidence,
                word_count=len(words),
                char_count=len(text),
                alpha_ratio=compute_alpha_ratio(text),
            )
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed for {image_path}: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                word_count=0,
                char_count=0,
                alpha_ratio=0.0,
            )
    
    def is_ocr_failed(self, result: OCRResult) -> bool:
        """
        Check if OCR result indicates failure based on quality heuristics.
        Any condition being True triggers failure.
        """
        return (
            result.confidence < self.confidence_threshold or
            result.char_count < self.min_chars or
            result.word_count < self.min_tokens or
            result.alpha_ratio < self.min_alpha_ratio
        )
    
    def get_failure_reason(self, result: OCRResult) -> Optional[str]:
        """Get human-readable failure reason."""
        reasons = []
        
        if result.confidence < self.confidence_threshold:
            reasons.append(f"confidence={result.confidence:.1f}<{self.confidence_threshold}")
        if result.char_count < self.min_chars:
            reasons.append(f"chars={result.char_count}<{self.min_chars}")
        if result.word_count < self.min_tokens:
            reasons.append(f"tokens={result.word_count}<{self.min_tokens}")
        if result.alpha_ratio < self.min_alpha_ratio:
            reasons.append(f"alpha_ratio={result.alpha_ratio:.2f}<{self.min_alpha_ratio}")
        
        return "; ".join(reasons) if reasons else None
    
    def _get_cached(self, image_hash: str) -> Optional[OCRResult]:
        """Load cached OCR result if exists."""
        if not self.cache_dir:
            return None
        
        cache_file = Path(self.cache_dir) / f"{image_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return OCRResult(**data)
            except Exception:
                return None
        return None
    
    def _save_cache(self, image_hash: str, result: OCRResult) -> None:
        """Save OCR result to cache."""
        if not self.cache_dir:
            return
        
        cache_file = Path(self.cache_dir) / f"{image_hash}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to cache OCR result: {e}")
