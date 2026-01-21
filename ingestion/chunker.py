"""
Chunking module - splits text into chunks, preserves tables, links chunks.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Iterator, Optional

from .models import Chunk, ChunkMetadata
from .parser import Element

logger = logging.getLogger(__name__)


def count_tokens_approx(text: str) -> int:
    """
    Approximate token count (words * 1.3).
    For exact count, use tiktoken, but this is faster for chunking.
    """
    return int(len(text.split()) * 1.3)


class Chunker:
    """
    Convert parsed elements into normalized chunks.
    - Text: split at chunk_size tokens with overlap
    - Tables: keep intact
    - Images: handled separately (OCR/caption)
    """
    
    def __init__(
        self,
        chunk_size_tokens: int = 512,
        chunk_overlap_tokens: int = 50,
    ):
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
    
    def chunk(
        self,
        elements: list[Element],
        doc_id: str,
        source_path: str,
    ) -> list[Chunk]:
        """
        Convert elements to chunks with linking.
        
        Args:
            elements: Parsed PDF elements
            doc_id: Document identifier
            source_path: Original PDF path
            
        Returns:
            List of Chunk objects with prev/next links
        """
        chunks: list[Chunk] = []
        
        # Group elements by page for chunk ID generation
        page_counters: dict[int, dict[str, int]] = {}
        
        for element in elements:
            page = element.page
            if page not in page_counters:
                page_counters[page] = {"text": 0, "table": 0, "image": 0}
            
            if element.type == "text":
                # Split text into smaller chunks
                text_chunks = list(self._split_text(element.content))
                for chunk_text in text_chunks:
                    chunk_id = f"{doc_id}_p{page}_t{page_counters[page]['text']}"
                    page_counters[page]["text"] += 1
                    
                    chunks.append(Chunk(
                        doc_id=doc_id,
                        page=page,
                        chunk_id=chunk_id,
                        type="text",
                        content=chunk_text,
                        metadata=ChunkMetadata(
                            source_path=source_path,
                            ingest_ts=datetime.utcnow().isoformat() + "Z",
                        ),
                    ))
            
            elif element.type == "table":
                # Keep tables intact
                chunk_id = f"{doc_id}_p{page}_tbl{page_counters[page]['table']}"
                page_counters[page]["table"] += 1
                
                chunks.append(Chunk(
                    doc_id=doc_id,
                    page=page,
                    chunk_id=chunk_id,
                    type="table",
                    content=element.content,
                    metadata=ChunkMetadata(
                        source_path=source_path,
                        ingest_ts=datetime.utcnow().isoformat() + "Z",
                    ),
                ))
            
            # Images are handled separately in pipeline (OCR + caption)
        
        # Link chunks (prev/next)
        chunks = self._link_chunks(chunks)
        
        return chunks
    
    def _split_text(self, text: str) -> Iterator[str]:
        """
        Split text into chunks of approximately chunk_size tokens.
        Preserves paragraph boundaries when possible.
        """
        if not text.strip():
            return
        
        # Split by paragraphs first
        paragraphs = re.split(r"\n\s*\n", text)
        
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_tokens = count_tokens_approx(para)
            
            # If paragraph itself is too large, split by sentences
            if para_tokens > self.chunk_size_tokens:
                # Flush current chunk
                if current_chunk:
                    yield "\n\n".join(current_chunk)
                    current_chunk = []
                    current_tokens = 0
                
                # Split large paragraph
                yield from self._split_paragraph(para)
            
            # If adding this paragraph exceeds limit
            elif current_tokens + para_tokens > self.chunk_size_tokens:
                # Yield current chunk
                if current_chunk:
                    yield "\n\n".join(current_chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_tokens = count_tokens_approx(" ".join(current_chunk))
            
            else:
                current_chunk.append(para)
                current_tokens += para_tokens
        
        # Yield remaining
        if current_chunk:
            yield "\n\n".join(current_chunk)
    
    def _split_paragraph(self, paragraph: str) -> Iterator[str]:
        """Split a large paragraph by sentences."""
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        
        current = []
        current_tokens = 0
        
        for sentence in sentences:
            sent_tokens = count_tokens_approx(sentence)
            
            if current_tokens + sent_tokens > self.chunk_size_tokens and current:
                yield " ".join(current)
                
                # Overlap from last sentence
                overlap = current[-1] if current else ""
                current = [overlap, sentence] if overlap else [sentence]
                current_tokens = count_tokens_approx(" ".join(current))
            else:
                current.append(sentence)
                current_tokens += sent_tokens
        
        if current:
            yield " ".join(current)
    
    def _get_overlap_text(self, chunks: list[str]) -> str:
        """Get overlap text from end of chunks."""
        if not chunks:
            return ""
        
        last_text = chunks[-1]
        words = last_text.split()
        
        # Take last N words for overlap
        overlap_words = int(self.chunk_overlap_tokens / 1.3)  # Approximate
        if len(words) <= overlap_words:
            return last_text
        
        return " ".join(words[-overlap_words:])
    
    def _link_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Add prev/next chunk links."""
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.metadata.prev_chunk_id = chunks[i - 1].chunk_id
            if i < len(chunks) - 1:
                chunk.metadata.next_chunk_id = chunks[i + 1].chunk_id
        
        return chunks
    
    def create_image_chunks(
        self,
        doc_id: str,
        page: int,
        image_index: int,
        ocr_text: str,
        ocr_result: "OCRResult",
        caption: Optional[str],
        image_path: str,
        image_hash: str,
        source_path: str,
        config: Optional[dict] = None,
    ) -> list[Chunk]:
        """
        Create image-related chunks (OCR and optionally caption).
        
        Args:
            doc_id: Document ID
            page: Page number
            image_index: Image index on page
            ocr_text: Text from OCR
            ocr_result: Full OCR result with metrics
            caption: Vision caption (if OCR failed)
            image_path: Path to image file
            image_hash: MD5 hash of image
            source_path: Original PDF path
            config: Optional config dict for OCR thresholds
            
        Returns:
            List of image chunks (1-2 depending on OCR success)
        """
        chunks = []
        now = datetime.utcnow().isoformat() + "Z"
        
        # Use config-based OCR failure check
        ocr_failed = ocr_result.is_failed_with_config(config)
        
        ocr_chunk = Chunk(
            doc_id=doc_id,
            page=page,
            chunk_id=f"{doc_id}_p{page}_i{image_index}_ocr",
            type="image_ocr",
            content=ocr_text,
            metadata=ChunkMetadata(
                source_path=source_path,
                image_hash=image_hash,
                ocr_confidence=ocr_result.confidence,
                ocr_failed=ocr_failed,
                ocr_failed_reason=ocr_result.get_failure_reason(config) if ocr_failed else None,
                vision_used=False,
                ingest_ts=now,
            ),
        )
        chunks.append(ocr_chunk)
        
        # Create caption chunk if OCR failed and caption provided
        if ocr_failed and caption:
            caption_chunk = Chunk(
                doc_id=doc_id,
                page=page,
                chunk_id=f"{doc_id}_p{page}_i{image_index}_cap",
                type="image_caption",
                content=caption,
                metadata=ChunkMetadata(
                    source_path=source_path,
                    image_hash=image_hash,
                    ocr_failed=True,
                    ocr_failed_reason=ocr_result.get_failure_reason(config),
                    vision_used=True,
                    ingest_ts=now,
                ),
            )
            chunks.append(caption_chunk)
        
        return chunks
