"""
Vision captioning module using GPT-4o-mini as fallback for failed OCR.
Implements two-step Extract→Summarize prompt per specification.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Vision prompt per specification
VISION_PROMPT = """Analyze this image from a technical PDF.

Step 1 (Extract): List all visible text, numbers, labels, axis titles, legend entries, table headers, and values.
If any text is unreadable or too small, explicitly say "unreadable" for those parts.

Step 2 (Summarize): Based ONLY on what you extracted in Step 1, briefly describe what the image shows.
- For graphs: describe the type, axes, trends (e.g., "line graph showing X increasing over Y")
- For tables: summarize the structure and key data points
- For diagrams: describe the components and relationships

RULES:
- Do NOT speculate or infer beyond what is visible
- Do NOT make up values or labels
- If the image is unclear, say so explicitly
- Keep the summary concise (2-3 sentences max)"""


class VisionCaptioner:
    """
    Generate image captions using GPT-4o-mini vision.
    Used as fallback when OCR fails.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        cache_dir: Optional[str] = None,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        timeout: int = 30,
    ):
        self.api_key = api_key
        self.model = model
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def run(self, image_path: str) -> tuple[str, bool]:
        """
        Generate caption for image.
        
        Returns:
            tuple: (caption_text, success_flag)
        """
        # Compute hash for caching
        image_hash = self._compute_hash(image_path)
        
        # Check cache
        cached = self._get_cached(image_hash)
        if cached is not None:
            logger.debug(f"Vision cache hit for {image_path}")
            return cached, True
        
        # Call Vision API with retry
        for attempt in range(self.max_retries):
            try:
                caption = self._call_vision_api(image_path)
                self._save_cache(image_hash, caption)
                return caption, True
            except Exception as e:
                wait_time = self.backoff_base ** attempt
                logger.warning(f"Vision API attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
        
        # All retries failed
        logger.error(f"Vision API failed after {self.max_retries} attempts for {image_path}")
        return "", False
    
    def _call_vision_api(self, image_path: str) -> str:
        """Call OpenAI Vision API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")
        
        client = OpenAI(api_key=self.api_key)
        
        # Encode image to base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Determine image type
        ext = Path(image_path).suffix.lower()
        media_type = "image/png" if ext == ".png" else "image/jpeg"
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}",
                                "detail": "high",  # High detail for technical content
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
            timeout=self.timeout,
        )
        
        return response.choices[0].message.content.strip()
    
    def _compute_hash(self, image_path: str) -> str:
        """Compute MD5 hash of image file."""
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _get_cached(self, image_hash: str) -> Optional[str]:
        """Load cached caption if exists."""
        if not self.cache_dir:
            return None
        
        cache_file = Path(self.cache_dir) / f"{image_hash}.txt"
        if cache_file.exists():
            try:
                return cache_file.read_text(encoding="utf-8")
            except Exception:
                return None
        return None
    
    def _save_cache(self, image_hash: str, caption: str) -> None:
        """Save caption to cache."""
        if not self.cache_dir:
            return
        
        cache_file = Path(self.cache_dir) / f"{image_hash}.txt"
        try:
            cache_file.write_text(caption, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to cache vision caption: {e}")
