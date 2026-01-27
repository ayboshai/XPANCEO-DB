"""
Vision captioning module using GPT-4o-mini as fallback for failed OCR.
Implements two-step Extract→Summarize prompt per specification.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

from .ocr import compute_image_hash

logger = logging.getLogger(__name__)

# Vision prompt focused on plain-text output (avoid prompt-echo artifacts)
VISION_PROMPT = """Analyze this image from a technical PDF.

Extract all visible text, numbers, labels, axis titles, legend entries, table headers, and values.
Then briefly describe what the image shows using only what is visible.

Rules:
- If text is unreadable, say "unreadable" for that part.
- Do not speculate or invent values.
- Output plain text only (no headings, bullet lists, or markdown)."""

MAX_VISION_IMAGE_DIM = 3000


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
        rate_limit_rpm: int = 60,  # Vision has stricter limits
        config: Optional[dict] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        self.rate_limit_rpm = rate_limit_rpm
        self.config = config

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def run(self, image_path: str) -> Tuple[str, bool]:
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
        
        # Call Vision API with retry, rate limiting, and concurrency control
        from shared import get_sync_limiter
        limiter = get_sync_limiter()
        
        for attempt in range(self.max_retries):
            try:
                # Acquire semaphore + rate limit
                limiter.acquire_vision()
                
                try:
                    caption = self._call_vision_api(image_path)
                    caption = self._clean_caption(caption)
                    self._save_cache(image_hash, caption)
                    return caption, True
                finally:
                    limiter.release_vision()
                    
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
        
        # Disable SDK-level retries to avoid hidden long backoffs.
        # We already implement explicit retry logic in run(...).
        client = OpenAI(api_key=self.api_key, timeout=self.timeout, max_retries=0)
        
        # Encode image to base64 (standardize format to avoid unsupported types)
        image_data, media_type = self._prepare_image_bytes(image_path)

        t0 = time.time()
        # Guard against pathological latency on complex figures.
        # This does not drop chunks: OCR fallback still runs.
        vision_timeout = max(self.timeout, 45)
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
                                # Let the model decide detail level to reduce
                                # pathological latency on some figures.
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
            timeout=vision_timeout,
        )
        dt = time.time() - t0
        if dt > 60:
            logger.warning(f"Slow Vision call ({dt:.1f}s): {image_path}")
        else:
            logger.debug(f"Vision call completed in {dt:.1f}s: {image_path}")

        return response.choices[0].message.content.strip()

    def _clean_caption(self, caption: str) -> str:
        """
        Remove prompt-echo artifacts like "Step 1/Step 2" headers.
        Keeps the original caption if cleaning would empty it.
        """
        if not caption:
            return caption

        cleaned_lines = []
        for line in caption.splitlines():
            stripped = line.strip()
            lower = stripped.lower()
            if not stripped:
                continue
            if "step 1" in lower or "step 2" in lower:
                continue
            if lower.startswith("rules") or lower.startswith("rules:"):
                continue
            cleaned_lines.append(stripped)

        cleaned = "\n".join(cleaned_lines).strip()
        return cleaned or caption.strip()

    def _prepare_image_bytes(self, image_path: str) -> Tuple[str, str]:
        """
        Load and standardize image for Vision API.
        Converts unsupported formats (e.g., PPM/PBM) to PNG to reduce refusals.
        Optionally downscales very large images to a safe max size.
        Returns base64 string and media_type.
        """
        try:
            from PIL import Image
        except ImportError:
            # Fallback: raw bytes with media type inferred from extension
            with open(image_path, "rb") as f:
                raw = base64.b64encode(f.read()).decode("utf-8")
            ext = Path(image_path).suffix.lower()
            media_type = "image/png" if ext == ".png" else "image/jpeg"
            return raw, media_type

        try:
            with Image.open(image_path) as img:
                # Convert to RGB for consistent encoding
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                elif img.mode == "L":
                    img = img.convert("RGB")

                # Downscale only if extremely large (keeps detail for most figures)
                if max(img.size) > MAX_VISION_IMAGE_DIM:
                    ratio = MAX_VISION_IMAGE_DIM / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                buf = io.BytesIO()
                img.save(buf, format="PNG")
                image_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                return image_data, "image/png"
        except Exception:
            # As a last resort, send raw bytes
            with open(image_path, "rb") as f:
                raw = base64.b64encode(f.read()).decode("utf-8")
            ext = Path(image_path).suffix.lower()
            media_type = "image/png" if ext == ".png" else "image/jpeg"
            return raw, media_type
    
    def _compute_hash(self, image_path: str) -> str:
        """Compute MD5 hash of image file."""
        return compute_image_hash(image_path)
    
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
