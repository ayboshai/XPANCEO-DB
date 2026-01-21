"""
Ingestion pipeline - orchestrates PDF → Chunks → Index flow.
Handles registry, error logging, and progress reporting.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Optional

from .captioner import VisionCaptioner
from .chunker import Chunker
from .embedder import OpenAIEmbedder, create_embedder
from .index_faiss import FAISSIndex, create_index
from .models import Chunk, ChunkCounts, ErrorLogEntry, RegistryEntry
from .ocr import OCRProcessor, compute_image_hash
from .parser import PDFParser, Element

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Full ingestion pipeline: PDF → Parse → Chunk → Embed → Index.
    Handles multi-PDF processing with registry and error logging.
    """
    
    def __init__(self, config: dict):
        self.config = config
        
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
        )
        
        self.embedder = create_embedder(config)
        self.index = create_index(config)
        
        # Paths
        self.data_dir = config.get("data_dir", "data")
        self.chunks_file = os.path.join(self.data_dir, "chunks.jsonl")
        self.registry_file = os.path.join(self.data_dir, "pdf_registry.jsonl")
        self.error_log_file = os.path.join(self.data_dir, "error_log.jsonl")
        
        os.makedirs(self.data_dir, exist_ok=True)
    
    def ingest_folder(
        self,
        folder_path: str,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> list[RegistryEntry]:
        """
        Ingest all PDFs from a folder.
        
        Args:
            folder_path: Path to folder containing PDFs
            progress_callback: Optional callback(filename, current, total)
            
        Returns:
            List of registry entries for processed documents
        """
        pdf_files = list(Path(folder_path).glob("**/*.pdf"))
        results = []
        
        for i, pdf_path in enumerate(pdf_files):
            if progress_callback:
                progress_callback(pdf_path.name, i + 1, len(pdf_files))
            
            try:
                entry = self.ingest_pdf(str(pdf_path))
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
    
    def ingest_pdf(self, pdf_path: str) -> RegistryEntry:
        """
        Ingest single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Registry entry with processing statistics
        """
        logger.info(f"Ingesting: {pdf_path}")
        
        # Parse PDF to get doc_id
        doc_id, elements = self.parser.parse(pdf_path)
        page_count = self.parser.get_page_count(pdf_path)
        
        # Handle reupload policy
        reupload_policy = self.config.get("reupload_policy", "new_version")
        existing_doc_ids = self._get_existing_doc_ids()
        
        if doc_id in existing_doc_ids:
            if reupload_policy == "overwrite":
                logger.info(f"Overwriting existing doc_id: {doc_id}")
                # CRITICAL: Delete from index FIRST (before chunks.jsonl is modified)
                self._remove_from_index_by_doc_id(doc_id)
                # Then delete from chunks.jsonl
                self._remove_doc_chunks(doc_id)
                # Then remove from registry
                self._remove_from_registry(doc_id)
                # Clean up related cache files
                self._cleanup_cache_for_doc(doc_id)
            else:  # new_version
                # Generate new unique doc_id
                import hashlib
                import time
                new_hash = hashlib.md5(f"{pdf_path}:{time.time()}".encode()).hexdigest()[:12]
                logger.info(f"Creating new version: {doc_id} -> {new_hash}")
                doc_id = new_hash
        
        # Process elements to chunks
        all_chunks: list[Chunk] = []
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
        text_table_chunks = self.chunker.chunk(
            text_table_elements,
            doc_id=doc_id,
            source_path=pdf_path,
        )
        all_chunks.extend(text_table_chunks)
        
        # Process images
        for elem in image_elements:
            if not elem.image_path or not os.path.exists(elem.image_path):
                continue
            
            try:
                image_chunks = self._process_image(
                    doc_id=doc_id,
                    page=elem.page,
                    image_index=image_count,
                    image_path=elem.image_path,
                    source_path=pdf_path,
                )
                all_chunks.extend(image_chunks)
                
                # Track stats
                for chunk in image_chunks:
                    if chunk.type == "image_ocr" and chunk.metadata.ocr_failed:
                        ocr_failures += 1
                    if chunk.type == "image_caption" and chunk.metadata.vision_used:
                        vision_calls += 1
                
                image_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process image on page {elem.page}: {e}")
                self._log_error(doc_id, elem.page, "image", str(e))
                errors += 1
        
        # Save chunks to JSONL
        self._save_chunks(all_chunks)
        
        # Embed and index
        self._embed_and_index(all_chunks)
        
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
        
        return entry
    
    def _process_image(
        self,
        doc_id: str,
        page: int,
        image_index: int,
        image_path: str,
        source_path: str,
    ) -> list[Chunk]:
        """Process single image: OCR + optional Vision fallback."""
        image_hash = compute_image_hash(image_path)
        
        # Run OCR
        ocr_result = self.ocr.run(image_path)
        ocr_failed = self.ocr.is_ocr_failed(ocr_result)
        
        # Vision caption if OCR failed
        caption = None
        if ocr_failed:
            caption, vision_success = self.captioner.run(image_path)
            if not vision_success:
                caption = None
                logger.warning(f"Vision fallback failed for {image_path}")
        
        # Create chunks with config for proper OCR threshold checking
        return self.chunker.create_image_chunks(
            doc_id=doc_id,
            page=page,
            image_index=image_index,
            ocr_text=ocr_result.text,
            ocr_result=ocr_result,
            caption=caption,
            image_path=image_path,
            image_hash=image_hash,
            source_path=source_path,
            config=self.config,
        )
    
    def _embed_and_index(self, chunks: list[Chunk]) -> None:
        """Embed chunks and add to index."""
        if not chunks:
            return
        
        # Prepare texts and metadata
        texts = [c.content for c in chunks]
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
        
        # Filter empty texts
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        if not valid_indices:
            return
        
        texts = [texts[i] for i in valid_indices]
        ids = [ids[i] for i in valid_indices]
        metadata = [metadata[i] for i in valid_indices]
        
        # Embed
        try:
            vectors = self.embedder.embed(texts)
            
            # Index
            self.index.upsert(ids, vectors, metadata)
            
        except Exception as e:
            logger.error(f"Failed to embed/index chunks: {e}")
            raise
    
    def _save_chunks(self, chunks: list[Chunk]) -> None:
        """Append chunks to chunks.jsonl."""
        with open(self.chunks_file, "a", encoding="utf-8") as f:
            for chunk in chunks:
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
    
    def load_chunks(self) -> list[Chunk]:
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
                        except Exception:
                            pass
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
                except Exception:
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
                    except Exception:
                        pass
        
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
    
    # Keep old method for backwards compatibility (deprecated)
    def _remove_from_index(self, doc_id: str) -> int:
        """Deprecated: Use _remove_from_index_by_doc_id instead."""
        return self._remove_from_index_by_doc_id(doc_id)
    
    def _cleanup_cache_for_doc(self, doc_id: str) -> int:
        """
        Clean up cache files related to a doc_id on overwrite.
        Deletes cached OCR/vision files for images from that doc.
        """
        from pathlib import Path
        import shutil
        
        cleaned = 0
        
        # Get image hashes from chunks before they were deleted (already deleted)
        # So we clean by doc_id prefix in cache dirs
        cache_dirs = [
            self.config.get("ocr_cache_dir", "data/cache/ocr"),
            self.config.get("vision_cache_dir", "data/cache/vision"),
        ]
        
        # Clean images directory for this doc
        images_dir = os.path.join(self.config.get("cache_dir", "data/cache"), "images")
        doc_images_dir = os.path.join(images_dir, doc_id)
        
        if os.path.exists(doc_images_dir):
            try:
                shutil.rmtree(doc_images_dir)
                logger.info(f"Cleaned cache images for doc_id={doc_id}")
                cleaned += 1
            except Exception as e:
                logger.warning(f"Failed to clean images cache: {e}")
        
        # Note: OCR/vision caches are keyed by image_hash not doc_id
        # For full cleanup, we'd need to track image_hash -> doc_id mapping
        # This is a partial solution - full cleanup requires image_hash tracking
        
        return cleaned
