"""
Ingestion pipeline - orchestrates PDF → Chunks → Index flow.
Handles registry, error logging, and progress reporting.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Callable, Iterator, Optional

from .captioner import VisionCaptioner
from .chunker import Chunker
from .embedder import OpenAIEmbedder, create_embedder
from .index_faiss import FAISSIndex, create_index
from .models import Chunk, ChunkCounts, ErrorLogEntry, RegistryEntry, OCRResult
from .ocr import OCRProcessor, compute_image_hash, classify_image_content, VALID_IMAGE_TYPES
from .parser import PDFParser, Element

logger = logging.getLogger(__name__)

# Image routing constants (kept module-level for clarity and review).
TINY_MAX_DIM = 48
TINY_MAX_AREA = 2304  # 48x48
BLANK_UNIFORM_MAX_ENTROPY = 0.5
BLANK_UNIFORM_MAX_STDDEV = 5.0


class IngestionPipeline:
    """
    Full ingestion pipeline: PDF → Parse → Chunk → Embed → Index.
    Handles multi-PDF processing with registry and error logging.
    """
    
    def __init__(self, config: dict):
        self.config = config

        # Image deduplication cache (per-document, cleared on each ingest_document call)
        # Key: image_hash (MD5), Value: dict with caption, ocr_result, image_type
        self.image_cache = {}

        # IMPORTANT: initialize global sync limiter once with this run's config.
        # This avoids per-call limiter recreation and ensures iter configs apply.
        from shared import get_sync_limiter
        get_sync_limiter(config)

        # Initialize components
        self.parser = PDFParser(
            image_output_dir=os.path.join(config.get("cache_dir", "data/cache"), "images"),
            doc_id_strategy=config.get("doc_id_strategy", "hash"),
        )

        self.chunker = Chunker(
            chunk_size_tokens=config.get("chunk_size_tokens", 512),
            chunk_overlap_tokens=config.get("chunk_overlap_tokens", 50),
        )
        
        self.ocr = OCRProcessor(
            cache_dir=config.get("ocr_cache_dir"),
            config=config,
        )
        
        api_key = config.get("openai_api_key", os.getenv("OPENAI_API_KEY", ""))
        
        self.captioner = VisionCaptioner(
            api_key=api_key,
            model=config.get("model_vision", "gpt-4o-mini"),
            cache_dir=config.get("vision_cache_dir"),
            max_retries=config.get("api_max_retries", 3),
            backoff_base=config.get("api_backoff_base", 2.0),
            timeout=config.get("api_timeout", 30),
            config=config,
        )
        
        self.embedder = create_embedder(config)
        self.index = create_index(config)
        
        # Paths
        self.data_dir = config.get("data_dir", "data")
        self.chunks_file = os.path.join(self.data_dir, "chunks.jsonl")
        self.registry_file = os.path.join(self.data_dir, "pdf_registry.jsonl")
        self.error_log_file = os.path.join(self.data_dir, "error_log.jsonl")
        self.timing_file = os.path.join(self.data_dir, "timing.jsonl")

        os.makedirs(self.data_dir, exist_ok=True)
    
    def ingest_folder(
        self,
        folder_path: str,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        stage_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[RegistryEntry]:
        """
        Ingest all PDFs from a folder.
        
        Args:
            folder_path: Path to folder containing PDFs
            progress_callback: Optional callback(filename, current, total)
            stage_callback: Optional callback(stage, stage_index, stage_total)
            
        Returns:
            List of registry entries for processed documents
        """
        pdf_files = list(Path(folder_path).glob("**/*.pdf"))
        results = []
        
        for i, pdf_path in enumerate(pdf_files):
            if progress_callback:
                progress_callback(pdf_path.name, i + 1, len(pdf_files))
            
            try:
                entry = self.ingest_pdf(str(pdf_path), stage_callback=stage_callback)
                results.append(entry)
            except Exception as e:
                logger.error(f"Failed to ingest {pdf_path}: {e}")
                # Log error and continue
                self._log_error(
                    doc_id=pdf_path.stem,
                    page=None,
                    element_type="document",
                    error=str(e),
                )
        
        return results
    
    def ingest_pdf(
        self,
        pdf_path: str,
        stage_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> RegistryEntry:
        """
        Ingest single PDF file with stage timing telemetry.

        Args:
            pdf_path: Path to PDF file
            stage_callback: Optional callback(stage, stage_index, stage_total)

        Returns:
            Registry entry with processing statistics
        """
        logger.info(f"Ingesting: {pdf_path}")

        # Generate unique run_id for timing telemetry
        run_id = str(uuid.uuid4())

        stage_total = 6

        def _stage(name: str, idx: int) -> None:
            if stage_callback:
                stage_callback(name, idx, stage_total)

        # Clear image cache for new document (prevents cross-document pollution)
        self.image_cache.clear()

        # Parse PDF to get doc_id, elements, and page_count
        # Use detailed timing callback for parse substages
        def parse_timing_cb(stage: str, duration: float):
            self._log_timing(run_id, pdf_path, stage, duration)

        _stage("parse", 1)
        t0 = time.time()
        doc_id, elements, page_count = self.parser.parse(pdf_path, timing_callback=parse_timing_cb)
        total_parse = time.time() - t0
        self._log_timing(run_id, pdf_path, "parse_total", total_parse)
        
        # Handle reupload policy
        reupload_policy = self.config.get("reupload_policy", "new_version")
        existing_doc_ids = self._get_existing_doc_ids()
        
        if doc_id in existing_doc_ids:
            if reupload_policy == "overwrite":
                logger.info(f"Overwriting existing doc_id: {doc_id}")
                # CRITICAL ORDER:
                # 1. Collect image hashes BEFORE deletion (for cache cleanup)
                image_hashes = self._get_image_hashes_for_doc(doc_id)
                # 2. Delete from index FIRST (before chunks.jsonl is modified)
                self._remove_from_index_by_doc_id(doc_id)
                # 3. Delete from chunks.jsonl
                self._remove_doc_chunks(doc_id)
                # 4. Remove from registry
                self._remove_from_registry(doc_id)
                # 5. Clean up all related cache files (using collected hashes)
                self._cleanup_cache_for_doc(doc_id, image_hashes)
            else:  # new_version
                # Generate new unique doc_id
                import hashlib
                new_hash = hashlib.md5(f"{pdf_path}:{time.time()}".encode()).hexdigest()[:12]
                logger.info(f"Creating new version: {doc_id} -> {new_hash}")
                doc_id = new_hash
        
        # Process elements to chunks
        all_chunks: List[Chunk] = []
        image_count = 0
        ocr_failures = 0
        vision_calls = 0
        errors = 0
        
        # Separate images from text/tables
        text_table_elements = []
        image_elements = []
        
        for elem in elements:
            if elem.type == "image":
                image_elements.append(elem)
            else:
                text_table_elements.append(elem)
        
        # Process text and tables
        _stage("chunk_text_tables", 2)
        text_table_chunks = self.chunker.chunk(
            text_table_elements,
            doc_id=doc_id,
            source_path=pdf_path,
        )
        all_chunks.extend(text_table_chunks)
        
        # Process images sequentially. Tesseract performs poorly under thread load.
        t0 = time.time()
        _stage("image_processing", 3)
        if image_elements:
            low_info_hashes = self._identify_low_info_images(image_elements)
            if low_info_hashes:
                logger.info(f"Identified {len(low_info_hashes)} low-info decorative images")

            for idx, elem in enumerate(image_elements):
                if not elem.image_path or not os.path.exists(elem.image_path):
                    continue

                try:
                    result = self._process_image(
                        doc_id=doc_id,
                        page=elem.page,
                        image_index=idx,
                        image_path=elem.image_path,
                        source_path=pdf_path,
                        low_info_hashes=low_info_hashes,
                    )
                except Exception as e:
                    logger.error(f"Failed to process image on page {elem.page}: {e}")
                    self._log_error(doc_id, elem.page, "image", str(e))
                    errors += 1
                    continue

                if result:
                    all_chunks.extend(result)
                    for chunk in result:
                        if chunk.type == "image_ocr" and chunk.metadata.ocr_failed:
                            ocr_failures += 1
                        if chunk.type == "image_caption" and chunk.metadata.vision_used:
                            vision_calls += 1

                image_count += 1

        if image_count > 0:
            self._log_timing(run_id, pdf_path, "image_processing", time.time() - t0)

        # Save chunks to JSONL
        _stage("save_chunks", 4)
        self._save_chunks(all_chunks)

        # Embed and index
        t0 = time.time()
        _stage("embed_and_index", 5)
        self._embed_and_index(all_chunks)
        self._log_timing(run_id, pdf_path, "embed_and_index", time.time() - t0)
        
        # Create registry entry
        chunk_counts = ChunkCounts(
            text=sum(1 for c in all_chunks if c.type == "text"),
            table=sum(1 for c in all_chunks if c.type == "table"),
            image_ocr=sum(1 for c in all_chunks if c.type == "image_ocr"),
            image_caption=sum(1 for c in all_chunks if c.type == "image_caption"),
        )
        
        entry = RegistryEntry(
            doc_id=doc_id,
            filename=Path(pdf_path).name,
            filepath=pdf_path,
            pages=page_count,
            chunks=chunk_counts,
            ocr_failure_rate=ocr_failures / image_count if image_count > 0 else 0.0,
            vision_fallback_rate=vision_calls / image_count if image_count > 0 else 0.0,
            errors=errors,
        )
        
        self._save_registry(entry)
        
        logger.info(
            f"Ingested {pdf_path}: {len(all_chunks)} chunks "
            f"(text={chunk_counts.text}, table={chunk_counts.table}, "
            f"ocr={chunk_counts.image_ocr}, caption={chunk_counts.image_caption})"
        )

        _stage("done", 6)
        
        return entry
    
    def _process_image(
        self,
        doc_id: str,
        page: int,
        image_index: int,
        image_path: str,
        source_path: str,
        low_info_hashes: Optional[set] = None,
    ) -> List[Chunk]:
        """
        Process single image with type-based routing and deduplication.

        Routing summary:
        - All images run through OCR (for audit + text anchors).
        - text_scan/table_scan: Vision only on catastrophic OCR failure.
        - chart/diagram/photo: Vision by default, except blank/uniform or low-info.

        Deduplication: reuse OCR/Vision results for duplicate images by MD5 hash.
        """
        image_hash = compute_image_hash(image_path)
        is_low_info = bool(low_info_hashes and image_hash in low_info_hashes)

        # Reuse OCR/Vision results for duplicate images.
        if image_hash in self.image_cache:
            cached = self.image_cache[image_hash]
            logger.debug(f"Reusing cached result for image {image_hash[:8]} (type={cached['image_type']})")

            # Retrieve cached values
            image_type = cached['image_type']
            ocr_result = cached.get('ocr_result')
            ocr_text = cached.get('ocr_text', '')
            caption = cached.get('caption')
            ocr_catastrophic = cached.get('ocr_catastrophic')
            vision_error_override = cached.get('vision_error_override')
            if ocr_catastrophic is None:
                ocr_catastrophic = self._is_catastrophic_ocr(ocr_result)

            if image_type == 'table_scan' and ocr_text and ocr_text.strip():
                ocr_failed = self.ocr.is_ocr_failed(ocr_result) if ocr_result else False
                return self._create_table_chunks_from_ocr(
                    doc_id=doc_id,
                    page=page,
                    image_index=image_index,
                    source_path=source_path,
                    image_hash=image_hash,
                    image_type=image_type,
                    ocr_text=ocr_text,
                    ocr_result=ocr_result,
                    ocr_failed=ocr_failed,
                    ocr_catastrophic=ocr_catastrophic,
                    caption=caption,
                )

            # Create chunks from cached data
            return self.chunker.create_image_chunks(
                doc_id=doc_id,
                page=page,
                image_index=image_index,
                ocr_text=ocr_text,
                ocr_result=ocr_result,
                caption=caption,
                image_path=image_path,
                image_hash=image_hash,
                source_path=source_path,
                config=self.config,
                image_type=image_type,
                ocr_catastrophic=ocr_catastrophic,
                vision_error_override=vision_error_override,
            )

        # Classify image type (pure heuristics, no OCR).
        image_type = classify_image_content(image_path)
        assert image_type in VALID_IMAGE_TYPES, f"Invalid image type: {image_type}"
        logger.debug(f"Image {image_path} classified as: {image_type}")

        # Step 2: Route based on type
        ocr_result = None
        ocr_text = ""
        ocr_failed = False
        ocr_catastrophic = False
        caption = None
        vision_error_override = None

        # Low-info decorative images: run OCR first and skip Vision on catastrophic OCR.
        if is_low_info:
            logger.info(f"Low-info decorative image detected on page {page}: {image_path}")
            ocr_result = self.ocr.run(image_path)
            ocr_text = ocr_result.text if ocr_result else ""
            ocr_failed = self.ocr.is_ocr_failed(ocr_result)
            ocr_catastrophic = self._is_catastrophic_ocr(ocr_result)

            if ocr_catastrophic:
                logger.info(f"Skipping Vision for decorative low-info image (catastrophic OCR): {image_path}")
                vision_error_override = "Vision skipped: low-info catastrophic OCR"
                self.image_cache[image_hash] = {
                    'image_type': image_type,
                    'ocr_result': ocr_result,
                    'ocr_text': ocr_text,
                    'caption': None,
                    'ocr_catastrophic': ocr_catastrophic,
                    'vision_error_override': vision_error_override,
                }
                return self.chunker.create_image_chunks(
                    doc_id=doc_id,
                    page=page,
                    image_index=image_index,
                    ocr_text=ocr_text,
                    ocr_result=ocr_result,
                    caption=None,
                    image_path=image_path,
                    image_hash=image_hash,
                    source_path=source_path,
                    config=self.config,
                    image_type=image_type,
                    ocr_catastrophic=ocr_catastrophic,
                    vision_error_override=vision_error_override,
                )

        if image_type in ('text_scan', 'table_scan'):
            # OCR path for scanned text/tables
            if ocr_result is None:
                ocr_result = self.ocr.run(image_path)
                ocr_text = ocr_result.text if ocr_result else ""
                ocr_failed = self.ocr.is_ocr_failed(ocr_result)
                ocr_catastrophic = self._is_catastrophic_ocr(ocr_result)

            # Vision fallback only on catastrophic OCR failure to avoid excess token usage
            if ocr_failed and ocr_catastrophic:
                vision_success = False
                if self._should_skip_vision_blank_uniform(ocr_result, image_path):
                    vision_error_override = "Vision skipped: blank/uniform image"
                    logger.info(
                        f"Skipping Vision for blank/uniform image (catastrophic OCR): {image_path}"
                    )
                    caption = None
                else:
                    caption, vision_success = self.captioner.run(image_path)
                    if vision_success:
                        caption = self._filter_refusal_caption(caption)
                        if caption and self._is_low_information_caption(caption):
                            logger.info(f"Low-information caption detected, treating as failure: {image_path}")
                            caption = None
                            vision_success = False
                        if not caption:
                            vision_success = False
                    if not vision_success:
                        caption = None
                        logger.warning(f"Both OCR and Vision failed for {image_path}")
            elif ocr_failed:
                logger.info(
                    "OCR below thresholds but non-catastrophic; skipping Vision fallback "
                    f"for {image_path} (chars={getattr(ocr_result,'char_count',0)})"
                )
        else:
            # Visual types (chart/diagram/photo): Vision is required for semantics.
            # Skip Vision only for blank/uniform or known low-info images.
            if ocr_result is None:
                ocr_result = self.ocr.run(image_path)
                ocr_text = ocr_result.text if ocr_result else ""
                ocr_failed = self.ocr.is_ocr_failed(ocr_result)
                ocr_catastrophic = self._is_catastrophic_ocr(ocr_result)

            vision_success = False
            vision_reason = None  # "refused" | "low_info" | "error" | "skipped_*"

            if self._should_skip_vision_blank_uniform(ocr_result, image_path):
                vision_reason = "skipped_blank_uniform"
                vision_error_override = "Vision skipped: blank/uniform image"
                logger.info(f"Skipping Vision for blank/uniform image: {image_path}")
                caption = None
            elif is_low_info:
                vision_reason = "skipped_low_info"
                vision_error_override = "Vision skipped: low-info image"
                logger.info(f"Skipping Vision for low-info decorative image: {image_path}")
                caption = None
            else:
                caption, vision_success = self.captioner.run(image_path)
                if vision_success:
                    filtered = self._filter_refusal_caption(caption)
                    if filtered is None:
                        vision_reason = "refused"
                        caption = None
                        vision_success = False
                    elif self._is_low_information_caption(filtered):
                        vision_reason = "low_info"
                        caption = None
                        vision_success = False
                    else:
                        caption = filtered

            if not vision_success:
                if vision_reason == "skipped_blank_uniform":
                    logger.info(f"Vision skipped for blank/uniform image: {image_path}")
                elif vision_reason == "skipped_low_info":
                    logger.info(f"Vision skipped for low-info image: {image_path}")
                elif vision_reason == "low_info":
                    logger.info(f"Vision low-information caption, using OCR fallback: {image_path}")
                elif vision_reason == "refused":
                    logger.warning(f"Vision refused for visual content: {image_path}, trying OCR fallback")
                else:
                    logger.warning(f"Vision failed for visual content: {image_path}, using OCR fallback")
                caption = None
                if ocr_failed:
                    if vision_reason == "low_info":
                        logger.info(f"Both Vision and OCR low-signal for visual content: {image_path}")
                    else:
                        logger.warning(f"Both Vision and OCR failed for visual content: {image_path}")

        # Store in cache for reuse.
        self.image_cache[image_hash] = {
            'image_type': image_type,
            'ocr_result': ocr_result,
            'ocr_text': ocr_text,
            'caption': caption,
            'ocr_catastrophic': ocr_catastrophic,
            'vision_error_override': vision_error_override,
        }

        if image_type == 'table_scan' and ocr_text and ocr_text.strip():
            return self._create_table_chunks_from_ocr(
                doc_id=doc_id,
                page=page,
                image_index=image_index,
                source_path=source_path,
                image_hash=image_hash,
                image_type=image_type,
                ocr_text=ocr_text,
                ocr_result=ocr_result,
                ocr_failed=ocr_failed,
                ocr_catastrophic=ocr_catastrophic,
                caption=caption,
            )

        # Create chunks with image_type in metadata
        return self.chunker.create_image_chunks(
            doc_id=doc_id,
            page=page,
            image_index=image_index,
            ocr_text=ocr_text,
            ocr_result=ocr_result,
            caption=caption,
            image_path=image_path,
            image_hash=image_hash,
            source_path=source_path,
            config=self.config,
            image_type=image_type,
            ocr_catastrophic=ocr_catastrophic,
            vision_error_override=vision_error_override,
        )

    # Anchored refusal patterns - won't match "cannot be used" etc.
    _REFUSAL_PATTERNS = [
        re.compile(r"^I cannot\b", re.IGNORECASE),
        re.compile(r"^I'm unable\b", re.IGNORECASE),
        re.compile(r"^I can't\b", re.IGNORECASE),
        re.compile(r"^Sorry,\s+I\b", re.IGNORECASE),
        re.compile(r"^I'm sorry\b", re.IGNORECASE),
        re.compile(r"unable to (?:analyze|process|view|see)", re.IGNORECASE),
        re.compile(r"cannot (?:analyze|process|view|see) (?:this|the) image", re.IGNORECASE),
    ]

    def _compute_ahash(self, image_path: str, size: int = 8) -> Optional[int]:
        """Compute a simple average hash (aHash) for perceptual duplicate detection."""
        try:
            from PIL import Image
        except ImportError:
            return None

        try:
            with Image.open(image_path) as img:
                img = img.convert("L").resize((size, size))
                pixels = list(img.getdata())
        except Exception:
            return None

        if not pixels:
            return None

        mean_val = sum(pixels) / len(pixels)
        bits = ["1" if px >= mean_val else "0" for px in pixels]
        try:
            return int("".join(bits), 2)
        except ValueError:
            return None

    def _identify_low_info_images(self, image_elements: List[Element]) -> set:
        """
        Identify decorative low-information images to avoid wasting Vision calls.
        Rules:
        - Tiny images (very small max dimension/area) are always low-info.
        - Small images (<=200px max dimension) that repeat perceptually (aHash freq >= 3)
          are also low-info.
        Images are NOT dropped; Vision is only skipped when OCR is catastrophic.
        """
        if not image_elements:
            return set()

        ahash_counts: Dict[int, int] = {}
        small_records: List[Tuple[str, int]] = []
        low_info_hashes: set = set()

        try:
            from PIL import Image
        except ImportError:
            return set()

        for elem in image_elements:
            path = elem.image_path
            if not path or not os.path.exists(path):
                continue
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    max_dim = max(w, h)
                    area = w * h
            except Exception:
                continue

            # Unconditionally mark tiny images as low-info to avoid useless Vision calls
            if max_dim <= TINY_MAX_DIM or area <= TINY_MAX_AREA:
                low_info_hashes.add(compute_image_hash(path))
                continue

            if max_dim > 200:
                continue

            ah = self._compute_ahash(path)
            if ah is None:
                continue

            img_hash = compute_image_hash(path)
            small_records.append((img_hash, ah))
            ahash_counts[ah] = ahash_counts.get(ah, 0) + 1

        repeated_ahash = {ah for ah, count in ahash_counts.items() if count >= 3}
        if not repeated_ahash:
            return low_info_hashes

        repeated_hashes = {img_hash for img_hash, ah in small_records if ah in repeated_ahash}
        return low_info_hashes | repeated_hashes

    def _create_table_chunks_from_ocr(
        self,
        *,
        doc_id: str,
        page: int,
        image_index: int,
        source_path: str,
        image_hash: str,
        image_type: str,
        ocr_text: str,
        ocr_result: Optional[OCRResult],
        ocr_failed: bool,
        ocr_catastrophic: bool,
        caption: Optional[str],
    ) -> List[Chunk]:
        """Create table chunks from OCR text, optionally adding a Vision caption."""
        from .models import Chunk, ChunkMetadata
        from datetime import datetime

        if not ocr_text or not ocr_text.strip():
            return []

        ingest_ts = datetime.utcnow().isoformat() + "Z"
        processing_status = "failed" if ocr_catastrophic else "success"
        ocr_confidence = ocr_result.confidence if ocr_result else 0
        ocr_failed_reason = (
            ocr_result.get_failure_reason(self.config) if ocr_result and ocr_failed else None
        )

        chunks: List[Chunk] = []
        for idx, table_part in enumerate(self.chunker._split_table(ocr_text)):
            chunk_id = f"{doc_id}_p{page}_i{image_index}_table"
            if idx > 0:
                chunk_id += f"_{idx}"

            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    page=page,
                    chunk_id=chunk_id,
                    type="table",
                    content=table_part,
                    metadata=ChunkMetadata(
                        source_path=source_path,
                        table_source="ocr",
                        image_hash=image_hash,
                        image_type=image_type,
                        ocr_confidence=ocr_confidence,
                        ocr_failed=ocr_failed,
                        ocr_failed_reason=ocr_failed_reason,
                        processing_status=processing_status,
                        ocr_error="catastrophic_ocr" if ocr_catastrophic else None,
                        ingest_ts=ingest_ts,
                    ),
                )
            )

        if caption and caption.strip():
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    page=page,
                    chunk_id=f"{doc_id}_p{page}_i{image_index}_cap",
                    type="image_caption",
                    content=caption,
                    metadata=ChunkMetadata(
                        source_path=source_path,
                        image_hash=image_hash,
                        image_type=image_type,
                        ocr_failed=ocr_failed,
                        ocr_failed_reason=ocr_failed_reason,
                        vision_used=True,
                        ingest_ts=ingest_ts,
                    ),
                )
            )

        return chunks

    def _is_blank_ocr(self, result: Optional[OCRResult]) -> bool:
        """Blank OCR: no words and no characters extracted."""
        if result is None:
            return False
        return getattr(result, "char_count", 0) == 0 and getattr(result, "word_count", 0) == 0

    def _is_uniform_image(self, image_path: str) -> bool:
        """
        Detect near-uniform images (solid color / empty canvas).
        Used only as a guard when OCR is blank to avoid futile Vision calls.
        """
        try:
            from PIL import Image, ImageStat
            import math
        except ImportError:
            return False

        try:
            with Image.open(image_path) as img:
                gray = img.convert("L").resize((128, 128))
                stat = ImageStat.Stat(gray)
                stddev = float(stat.stddev[0]) if stat.stddev else 0.0
                hist = gray.histogram()
        except Exception:
            return False

        total = sum(hist)
        if total <= 0:
            return True

        entropy = 0.0
        for count in hist:
            if not count:
                continue
            p = count / total
            entropy -= p * math.log2(p)

        return entropy <= BLANK_UNIFORM_MAX_ENTROPY and stddev <= BLANK_UNIFORM_MAX_STDDEV

    def _should_skip_vision_blank_uniform(self, result: Optional[OCRResult], image_path: str) -> bool:
        """Skip Vision only when OCR is blank AND the image is near-uniform."""
        return self._is_blank_ocr(result) and self._is_uniform_image(image_path)

    def _filter_refusal_caption(self, caption: str) -> str:
        """
        Filter out Vision API refusal responses.

        Uses anchored patterns to avoid false positives like "cannot be used".
        Returns None if caption is a refusal, otherwise returns caption unchanged.
        """
        if not caption:
            return None

        for pattern in self._REFUSAL_PATTERNS:
            if pattern.search(caption):
                logger.debug(f"Filtered refusal caption: {caption[:50]}...")
                return None

        return caption

    def _is_low_information_caption(self, caption: Optional[str]) -> bool:
        """
        Detect captions that carry almost no usable information.
        These are stored for audit but not trusted for indexing.
        """
        if not caption:
            return True
        text = caption.strip().lower()
        if not text:
            return True
        if text == "unreadable":
            return True
        if text.startswith("unreadable") and len(text) < 120:
            return True
        has_digit = bool(re.search(r"\d", text))
        tokens = text.split()
        if len(text) < 12 and not has_digit and len(tokens) <= 2:
            return True
        return False

    def _is_catastrophic_ocr(self, result: Optional[OCRResult]) -> bool:
        """
        Detect catastrophic OCR failures (very little usable text).
        This gate controls when we spend extra tokens on Vision fallback
        for text/table scans.
        """
        if result is None:
            return True
        return (
            result.char_count < 30 or
            result.word_count < 5 or
            result.alpha_ratio < 0.20
        )

    def _is_strong_ocr(self, result: Optional[OCRResult]) -> bool:
        """
        Detect OCR results that are strong enough to skip Vision safely.
        Fixed thresholds, no extra config.
        """
        if result is None:
            return False
        return (
            result.char_count >= 120 and
            result.word_count >= 30 and
            result.alpha_ratio >= 0.45
        )
    
    def _embed_and_index(self, chunks: List[Chunk]) -> None:
        """Embed chunks and add to index. Skips failed chunks."""
        if not chunks:
            return

        # Prepare texts and metadata
        texts = [self._normalize_for_embedding(c) for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadata = [
            {
                "doc_id": c.doc_id,
                "page": c.page,
                "type": c.type,
                "content_preview": c.content[:100],
            }
            for c in chunks
        ]

        # Filter empty texts AND failed processing status
        valid_indices = [
            i for i, (t, c) in enumerate(zip(texts, chunks))
            if t.strip() and c.metadata.processing_status == "success"
        ]
        if not valid_indices:
            logger.debug(f"No valid chunks to index (all failed or empty)")
            return

        texts = [texts[i] for i in valid_indices]
        ids = [ids[i] for i in valid_indices]
        metadata = [metadata[i] for i in valid_indices]

        logger.debug(f"Indexing {len(texts)} chunks (filtered out {len(chunks) - len(texts)} failed/empty)")
        
        # Embed
        try:
            vectors = self.embedder.embed(texts)
            
            # Index
            self.index.upsert(ids, vectors, metadata)
            
        except Exception as e:
            logger.error(f"Failed to embed/index chunks: {e}")
            raise

    def _normalize_for_embedding(self, chunk: Chunk) -> str:
        """
        Normalize content for embeddings without altering stored chunk content.
        - Tables: append a flattened plain-text version to improve semantic retrieval.
        - Other types: return content as-is.
        """
        text = chunk.content or ""
        if chunk.type == "table":
            # Flatten table pipes to improve embedding similarity
            flat = re.sub(r"[|]+", " ", text)
            flat = re.sub(r"\s+", " ", flat).strip()
            if flat and flat not in text:
                return text + "\n\n" + flat
        elif chunk.type in ("image_caption", "image_ocr"):
            image_type = getattr(chunk.metadata, "image_type", None) or "image"
            return f"image {image_type}: {text}"
        return text
    
    def _save_chunks(self, chunks: List[Chunk]) -> None:
        """Append chunks to chunks.jsonl. Skips chunks with empty content."""
        with open(self.chunks_file, "a", encoding="utf-8") as f:
            for chunk in chunks:
                # Skip chunks with empty content (defensive check)
                if not chunk.content.strip():
                    logger.debug(f"Skipping empty chunk: {chunk.chunk_id}")
                    continue
                f.write(chunk.to_jsonl() + "\n")
    
    def _save_registry(self, entry: RegistryEntry) -> None:
        """Append entry to registry."""
        with open(self.registry_file, "a", encoding="utf-8") as f:
            f.write(entry.to_jsonl() + "\n")
    
    def _log_error(
        self,
        doc_id: str,
        page: Optional[int],
        element_type: str,
        error: str,
    ) -> None:
        """Log error to error_log.jsonl."""
        import traceback

        entry = ErrorLogEntry(
            doc_id=doc_id,
            page=page,
            element_type=element_type,
            error=error,
            traceback=traceback.format_exc(),
        )

        with open(self.error_log_file, "a", encoding="utf-8") as f:
            f.write(entry.to_jsonl() + "\n")

    def _log_timing(self, run_id: str, pdf_path: str, stage: str, duration_sec: float) -> None:
        """Log stage timing to timing.jsonl for p50/p90 measurement."""
        entry = {
            "run_id": run_id,
            "pdf_path": pdf_path,
            "stage": stage,
            "duration_ms": int(duration_sec * 1000),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        with open(self.timing_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def load_chunks(self) -> List[Chunk]:
        """Load all chunks from chunks.jsonl."""
        chunks = []
        if os.path.exists(self.chunks_file):
            with open(self.chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            chunks.append(Chunk.from_jsonl(line))
                        except Exception as e:
                            logger.warning(f"Failed to parse chunk: {e}")
        return chunks
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get single chunk by ID."""
        chunks = self.load_chunks()
        for chunk in chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
    
    def _get_existing_doc_ids(self) -> set[str]:
        """Get set of existing doc_ids from registry."""
        doc_ids = set()
        if os.path.exists(self.registry_file):
            with open(self.registry_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = RegistryEntry.from_jsonl(line)
                            doc_ids.add(entry.doc_id)
                        except Exception as e:
                            logger.warning(f"Failed to parse registry entry: {e}")
        return doc_ids
    
    def _remove_doc_chunks(self, doc_id: str) -> int:
        """Remove chunks for doc_id from chunks.jsonl. Returns count removed."""
        if not os.path.exists(self.chunks_file):
            return 0
        
        remaining = []
        removed = 0
        
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = Chunk.from_jsonl(line)
                    if chunk.doc_id == doc_id:
                        removed += 1
                    else:
                        remaining.append(line)
                except Exception as e:
                    logger.warning(f"Failed to parse chunk line during removal: {e}")
                    remaining.append(line)
        
        # Rewrite file without removed chunks
        with open(self.chunks_file, "w", encoding="utf-8") as f:
            for line in remaining:
                f.write(line + "\n")
        
        logger.info(f"Removed {removed} chunks for doc_id={doc_id}")
        return removed
    
    def _remove_from_index_by_doc_id(self, doc_id: str) -> int:
        """
        Remove vectors for doc_id from index.
        MUST be called BEFORE _remove_doc_chunks to read chunk IDs.
        """
        # Get chunk IDs from chunks.jsonl BEFORE it's modified
        chunk_ids_to_remove = []
        if os.path.exists(self.chunks_file):
            with open(self.chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = Chunk.from_jsonl(line)
                        if chunk.doc_id == doc_id:
                            chunk_ids_to_remove.append(chunk.chunk_id)
                    except Exception as e:
                        logger.warning(f"Failed to parse chunk line during index cleanup: {e}")
        
        if chunk_ids_to_remove:
            self.index.delete(chunk_ids_to_remove)
            logger.info(f"Removed {len(chunk_ids_to_remove)} vectors for doc_id={doc_id}")
        
        return len(chunk_ids_to_remove)
    
    def _remove_from_registry(self, doc_id: str) -> bool:
        """Remove entry for doc_id from pdf_registry.jsonl."""
        if not os.path.exists(self.registry_file):
            return False
        
        remaining = []
        removed = False
        
        with open(self.registry_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = RegistryEntry.from_jsonl(line)
                    if entry.doc_id == doc_id:
                        removed = True
                    else:
                        remaining.append(line)
                except Exception:
                    remaining.append(line)
        
        # Rewrite registry without the removed doc
        with open(self.registry_file, "w", encoding="utf-8") as f:
            for line in remaining:
                f.write(line + "\n")
        
        if removed:
            logger.info(f"Removed doc_id={doc_id} from registry")
        
        return removed

    def _get_image_hashes_for_doc(self, doc_id: str) -> List[str]:
        """
        Collect image hashes from chunks.jsonl for a doc_id.
        Must be called BEFORE chunks are deleted.
        
        Returns:
            List of md5 image hashes used for OCR/vision cache filenames.
        """
        hashes = []
        chunks_path = self.config.get("chunks_path", "data/chunks.jsonl")
        
        if not os.path.exists(chunks_path):
            return hashes
        
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    if chunk.get("doc_id") == doc_id:
                        # Extract actual image_hash from metadata (md5 of image file)
                        metadata = chunk.get("metadata", {})
                        image_hash = metadata.get("image_hash")
                        if image_hash:
                            hashes.append(image_hash)
                            logger.debug(f"Found image_hash={image_hash} for doc_id={doc_id}")
                except json.JSONDecodeError:
                    continue
        
        if hashes:
            logger.info(f"Collected {len(hashes)} image hashes for doc_id={doc_id}")
        
        return hashes
    
    def _cleanup_cache_for_doc(self, doc_id: str, image_hashes: List[str] = None) -> int:
        """
        Clean up cache files related to a doc_id on overwrite.
        Deletes cached OCR/vision files by image_hash (md5).
        
        Args:
            doc_id: Document ID to clean caches for
            image_hashes: List of md5 image hashes for OCR/vision cache cleanup
        """
        from pathlib import Path
        import shutil
        
        cleaned = 0
        image_hashes = image_hashes or []
        
        # 1. Clean images directory for this doc
        images_dir = os.path.join(self.config.get("cache_dir", "data/cache"), "images")
        doc_images_dir = os.path.join(images_dir, doc_id)
        
        if os.path.exists(doc_images_dir):
            try:
                shutil.rmtree(doc_images_dir)
                logger.info(f"Cleaned cache images for doc_id={doc_id}")
                cleaned += 1
            except Exception as e:
                logger.warning(f"Failed to clean images cache: {e}")
        
        # 2. Clean OCR cache files by image_hash
        ocr_cache_dir = self.config.get("ocr_cache_dir", "data/cache/ocr")
        if os.path.exists(ocr_cache_dir) and image_hashes:
            for img_hash in image_hashes:
                # OCR cache files are named: {image_hash}.json
                for ext in [".json", ".txt"]:
                    cache_file = os.path.join(ocr_cache_dir, f"{img_hash}{ext}")
                    if os.path.exists(cache_file):
                        try:
                            os.remove(cache_file)
                            logger.debug(f"Removed OCR cache: {cache_file}")
                            cleaned += 1
                        except Exception as e:
                            logger.warning(f"Failed to clean OCR cache {cache_file}: {e}")
        
        # 3. Clean Vision cache files by image_hash
        vision_cache_dir = self.config.get("vision_cache_dir", "data/cache/vision")
        if os.path.exists(vision_cache_dir) and image_hashes:
            for img_hash in image_hashes:
                # Vision cache files are named: {image_hash}.txt
                for ext in [".txt", ".json"]:
                    cache_file = os.path.join(vision_cache_dir, f"{img_hash}{ext}")
                    if os.path.exists(cache_file):
                        try:
                            os.remove(cache_file)
                            logger.debug(f"Removed vision cache: {cache_file}")
                            cleaned += 1
                        except Exception as e:
                            logger.warning(f"Failed to clean vision cache {cache_file}: {e}")
        
        # 4. Clean embedding cache for this doc
        # Policy: wipe all embeddings for consistency (conservative approach)
        # Embeddings use md5(model:text), so we cannot selectively delete
        # On overwrite, we clear entire embedding cache dir for this doc
        embedding_cache_dir = self.config.get("embedding_cache_dir", "data/cache/embeddings")
        if os.path.exists(embedding_cache_dir):
            # Full wipe on overwrite (conservative policy)
            import shutil
            try:
                shutil.rmtree(embedding_cache_dir)
                os.makedirs(embedding_cache_dir, exist_ok=True)
                logger.info(f"Wiped embedding cache directory on overwrite for doc_id={doc_id}")
                cleaned += 1
            except Exception as e:
                logger.warning(f"Failed to wipe embedding cache: {e}")
        
        if cleaned > 0:
            logger.info(f"Cleaned {cleaned} cache items for doc_id={doc_id} ({len(image_hashes)} image hashes)")
        elif image_hashes:
            logger.debug(f"No cache files found for {len(image_hashes)} image hashes")
        
        return cleaned
