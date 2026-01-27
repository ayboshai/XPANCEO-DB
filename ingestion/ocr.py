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

# Valid image types - enforced across all layers
VALID_IMAGE_TYPES = {"text_scan", "table_scan", "chart", "diagram", "photo"}
MAX_CLASSIFY_SIZE = 512


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


def classify_image_content(image_path: str) -> str:
    """
    Classify image into 5 types using pure visual heuristics (NO OCR).

    PERFORMANCE: Downscales to max 512px on larger dimension before analysis.
    Thresholds remain deterministic and unchanged from original implementation.

    Types:
        'text_scan': Scanned text documents → OCR
        'table_scan': Scanned tables → OCR
        'chart': Charts/graphs → Vision
        'diagram': Technical diagrams → Vision
        'photo': Photos/screenshots → Vision

    Heuristics (all computed from image pixels only):
        - color_diversity: unique_gray_values / 256
        - edge_ratio: pixels with gradient >30 / sample_size
        - contrast: std(gray_pixels) / 255
        - line_density: horizontal/vertical projection peaks

    Fixed thresholds (no config):
        - text_scan: diversity <0.08, edges <0.15, contrast >0.3
        - table_scan: diversity <0.12, 0.15 <= edges <0.30, line_density >0.1
        - chart: 0.08 <= diversity <0.35, edges >=0.20
        - diagram: 0.08 <= diversity <0.40, edges >=0.25
        - photo: diversity >=0.35
        - Default: chart (safe for Vision)
    """
    try:
        img = Image.open(image_path)
        # Fast guard: very small images are likely artifacts.
        # Route to OCR to avoid Vision calls (preserve content if any).

        # PERFORMANCE: Downscale to max 512px (10-50x speedup for large images)
        # This preserves statistical properties needed for heuristics
        original_size = img.size
        if max(img.size) > MAX_CLASSIFY_SIZE:
            ratio = MAX_CLASSIFY_SIZE / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.BILINEAR)
            logger.debug(f"Downscaled {original_size} -> {img.size} for classification")

        width, height = img.size
        if width < 20 or height < 20:
            logger.debug(f"Small image ({width}x{height}), forcing OCR path: {image_path}")
            return 'text_scan'

        gray = img.convert('L')
        width, height = gray.size
        pixels = list(gray.getdata())
        total_pixels = width * height

        # Heuristic 1: Color diversity
        histogram = {}
        for px in pixels:
            histogram[px] = histogram.get(px, 0) + 1
        color_diversity = len(histogram) / 256

        # Heuristic 2: Contrast (standard deviation normalized)
        mean_val = sum(pixels) / total_pixels
        variance = sum((px - mean_val) ** 2 for px in pixels) / total_pixels
        contrast = (variance ** 0.5) / 255

        # Heuristic 3: Edge ratio (gradient magnitude sampling)
        sample_size = min(5000, total_pixels)
        step = max(1, total_pixels // sample_size)
        edge_count = 0
        sampled = 0
        for i in range(0, total_pixels, step):
            x, y = i % width, i // width
            if 0 < x < width - 1 and 0 < y < height - 1:
                dx = abs(pixels[i] - pixels[i - 1])
                dy = abs(pixels[i] - pixels[i - width])
                if dx > 30 or dy > 30:
                    edge_count += 1
                sampled += 1
        edge_ratio = edge_count / sampled if sampled > 0 else 0

        # Heuristic 4: Line density (horizontal/vertical projection peaks)
        # Count rows/cols with high edge activity (indicates table grid)
        h_proj = [0] * height
        v_proj = [0] * width
        for i in range(0, total_pixels, step):
            x, y = i % width, i // width
            if 0 < x < width - 1 and 0 < y < height - 1:
                dx = abs(pixels[i] - pixels[i - 1])
                dy = abs(pixels[i] - pixels[i - width])
                if dx > 50:
                    v_proj[x] += 1
                if dy > 50:
                    h_proj[y] += 1

        # Count projection peaks (lines)
        h_peaks = sum(1 for v in h_proj if v > sampled / height * 0.3)
        v_peaks = sum(1 for v in v_proj if v > sampled / width * 0.3)
        line_density = (h_peaks + v_peaks) / (height + width) if (height + width) > 0 else 0

        # Classification with FIXED thresholds
        # text_scan: low diversity, few edges, high contrast (black on white)
        if color_diversity < 0.08 and edge_ratio < 0.15 and contrast > 0.3:
            return 'text_scan'

        # table_scan: low diversity, medium edges (grid lines), visible line structure
        if color_diversity < 0.12 and 0.15 <= edge_ratio < 0.30 and line_density > 0.1:
            return 'table_scan'

        # photo: high color diversity
        if color_diversity >= 0.35:
            return 'photo'

        # diagram: medium diversity, high edge complexity
        if 0.08 <= color_diversity < 0.40 and edge_ratio >= 0.25:
            return 'diagram'

        # chart: medium diversity, structured edges
        if 0.08 <= color_diversity < 0.35 and edge_ratio >= 0.20:
            return 'chart'

        # Default: chart (safe for Vision - better to over-Vision than miss semantics)
        result = 'chart'

        # VALIDATION: Ensure only valid types are returned
        assert result in VALID_IMAGE_TYPES, f"Invalid image type: {result}"
        return result

    except Exception as e:
        logger.warning(f"Image classification failed for {image_path}: {e}")
        result = 'chart'  # Default to Vision (safer)
        assert result in VALID_IMAGE_TYPES
        return result




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
    
    def _preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image to improve OCR accuracy for text_scan/table_scan.

        Steps:
        1. Convert to grayscale
        2. Auto-contrast enhancement
        3. Mean threshold binarization (simple but effective for documents)
        4. Upscale if <1000px (improves small text recognition)

        Note: Uses mean threshold rather than Otsu's method.
        True Otsu would minimize intra-class variance via histogram analysis.
        """
        # Grayscale
        gray = image.convert('L')
        width, height = gray.size

        # Upscale if too small
        if width < 1000 or height < 1000:
            scale = max(1000 / width, 1000 / height)
            new_size = (int(width * scale), int(height * scale))
            gray = gray.resize(new_size, Image.Resampling.LANCZOS)

        # Auto-contrast
        from PIL import ImageOps
        gray = ImageOps.autocontrast(gray)

        # Mean threshold binarization
        pixels = list(gray.getdata())
        threshold = sum(pixels) // len(pixels)
        binary_pixels = [255 if p > threshold else 0 for p in pixels]
        gray.putdata(binary_pixels)

        return gray

    def _run_tesseract(self, image_path: str) -> OCRResult:
        """Execute Tesseract OCR on image with preprocessing and optimized config."""
        try:
            import pytesseract
            from pytesseract import Output
        except ImportError:
            raise ImportError("pytesseract not installed. Run: pip install pytesseract")

        try:
            image = Image.open(image_path)

            # Preprocess for better OCR quality
            preprocessed = self._preprocess_image_for_ocr(image)

            # Tesseract config: LSTM engine + uniform text block mode
            # --oem 1: LSTM only (more accurate than legacy)
            # --psm 6: Assume uniform block of text
            tesseract_config = '--oem 1 --psm 6'

            # Get detailed OCR data with confidence scores
            ocr_data = pytesseract.image_to_data(
                preprocessed,
                lang=self.lang,
                output_type=Output.DICT,
                config=tesseract_config,
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
