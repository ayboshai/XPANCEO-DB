"""
Chunking module - splits text into chunks, preserves tables, links chunks.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import List, Tuple, Dict,  Iterator, Optional

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
        elements: List[Element],
        doc_id: str,
        source_path: str,
    ) -> List[Chunk]:
        """
        Convert elements to chunks with linking.
        
        Args:
            elements: Parsed PDF elements
            doc_id: Document identifier
            source_path: Original PDF path
            
        Returns:
            List of Chunk objects with prev/next links
        """
        chunks: List[Chunk] = []
        
        # Group elements by page for chunk ID generation
        page_counters: Dict[int, Dict[str, int]] = {}
        
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
                # Split large tables into smaller chunks for better retrieval
                # Use smaller chunk size for tables (400 tokens vs 1000 for text)
                table_chunk_size = min(400, self.chunk_size_tokens // 2)
                table_chunks = list(self._split_table(element.content, table_chunk_size))

                for table_text in table_chunks:
                    chunk_id = f"{doc_id}_p{page}_tbl{page_counters[page]['table']}"
                    page_counters[page]["table"] += 1

                    chunks.append(Chunk(
                        doc_id=doc_id,
                        page=page,
                        chunk_id=chunk_id,
                        type="table",
                        content=table_text,
                        metadata=ChunkMetadata(
                            source_path=source_path,
                            table_source="pdfplumber",  # All tables from pdfplumber (or "ocr" for table_scan)
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

        # Split by double newlines (paragraphs) first
        paragraphs = re.split(r"\n\s*\n", text)

        # Track if we're using single-line mode (for PyPDF2 output)
        single_line_mode = False

        # If no paragraph breaks found (e.g., PyPDF2 output), try single newlines
        if len(paragraphs) == 1 and count_tokens_approx(text) > self.chunk_size_tokens:
            paragraphs = text.split('\n')
            single_line_mode = True

        # Choose separator based on mode
        separator = "\n" if single_line_mode else "\n\n"

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
                    yield separator.join(current_chunk)
                    current_chunk = []
                    current_tokens = 0

                # Split large paragraph
                yield from self._split_paragraph(para)

            # If adding this paragraph exceeds limit
            # Account for: 5% estimation variance + potential overlap in next chunk
            elif current_tokens + para_tokens > self.chunk_size_tokens - self.chunk_overlap_tokens - int(self.chunk_size_tokens * 0.05):
                # Yield current chunk and get overlap BEFORE clearing
                if current_chunk:
                    yield separator.join(current_chunk)
                    overlap_text = self._get_overlap_text(current_chunk)
                else:
                    overlap_text = ""

                # Start new chunk with overlap + current paragraph
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_tokens = count_tokens_approx(separator.join(current_chunk))

            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        # Yield remaining
        if current_chunk:
            yield separator.join(current_chunk)
    
    def _split_paragraph(self, paragraph: str) -> Iterator[str]:
        """Split a large paragraph by sentences."""
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        # For paragraphs, also try splitting by newlines if sentences don't work
        if len(sentences) == 1 and count_tokens_approx(paragraph) > self.chunk_size_tokens:
            sentences = paragraph.split('\n')

        current = []
        current_tokens = 0

        # Effective limit accounting for overlap added to next chunk
        effective_limit = self.chunk_size_tokens - self.chunk_overlap_tokens - int(self.chunk_size_tokens * 0.05)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sent_tokens = count_tokens_approx(sentence)

            # If a single sentence is too long, split by words.
            if sent_tokens > effective_limit:
                if current:
                    yield " ".join(current)
                    current = []
                    current_tokens = 0

                yield from self._split_long_sentence(sentence, effective_limit)
                continue

            if current_tokens + sent_tokens > effective_limit and current:
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

    def _split_long_sentence(self, sentence: str, effective_limit: int) -> Iterator[str]:
        """Split a long sentence into word chunks with overlap."""
        words = sentence.split()
        if not words:
            return

        words_per_chunk = max(1, int(effective_limit / 1.3))
        overlap_words = max(0, int(self.chunk_overlap_tokens / 1.3))

        start = 0
        while start < len(words):
            end = min(len(words), start + words_per_chunk)
            yield " ".join(words[start:end])
            if end == len(words):
                break
            start = max(0, end - overlap_words)
    
    def _split_table(self, table_text: str, max_tokens: int = 400) -> Iterator[str]:
        """
        Split table into chunks by rows, preserving header.

        For better retrieval, large tables are split into smaller chunks
        while keeping the header row for context.

        Args:
            table_text: Table content (may include header)
            max_tokens: Maximum tokens per chunk (default 400)

        Yields:
            Table chunks with header preserved
        """
        lines = table_text.strip().split('\n')

        if not lines:
            return

        # Estimate tokens (rough: 1 token ≈ 4 chars)
        total_tokens = len(table_text) // 4

        # If small enough, return as-is
        if total_tokens <= max_tokens:
            yield table_text
            return

        # Assume first line(s) might be header - keep with each chunk
        # Header heuristic: first 1-2 lines, usually contain column names
        header_lines = []
        data_lines = []

        for i, line in enumerate(lines):
            if i < 2 and not any(c.isdigit() for c in line[:20]):
                # First 2 lines without leading digits = likely header
                header_lines.append(line)
            else:
                data_lines.append(line)

        if not header_lines:
            # No clear header, just use first line
            header_lines = [lines[0]] if lines else []
            data_lines = lines[1:] if len(lines) > 1 else []

        header_text = '\n'.join(header_lines)
        header_tokens = len(header_text) // 4

        # Split data rows into chunks
        current_chunk = []
        current_tokens = header_tokens

        for line in data_lines:
            line_tokens = len(line) // 4

            if current_tokens + line_tokens > max_tokens and current_chunk:
                # Yield current chunk with header
                chunk_text = header_text + '\n' + '\n'.join(current_chunk) if header_text else '\n'.join(current_chunk)
                yield chunk_text
                current_chunk = [line]
                current_tokens = header_tokens + line_tokens
            else:
                current_chunk.append(line)
                current_tokens += line_tokens

        # Yield remaining
        if current_chunk:
            chunk_text = header_text + '\n' + '\n'.join(current_chunk) if header_text else '\n'.join(current_chunk)
            yield chunk_text

    def _get_overlap_text(self, chunks: List[str]) -> str:
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
    
    def _link_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
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
        ocr_result: Optional["OCRResult"],
        caption: Optional[str],
        image_path: str,
        image_hash: str,
        source_path: str,
        config: Optional[dict] = None,
        image_type: Optional[str] = None,
        ocr_catastrophic: bool = False,
        vision_error_override: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Create image-related chunks based on image type. ALWAYS returns at least one chunk.

        Routing:
        - text_scan, table_scan: OCR chunk (if text), caption chunk if OCR failed (both if OCR text exists but low quality)
        - chart, diagram, photo: caption chunk → OCR fallback if Vision failed → failure placeholder if both failed

        Failure handling:
        - If both OCR and Vision fail, creates a failure chunk with processing_status="failed"
        - Failed chunks are saved to chunks.jsonl but NOT indexed (filtered in pipeline)

        Args:
            doc_id: Document ID
            page: Page number
            image_index: Image index on page
            ocr_text: Text from OCR (may be empty)
            ocr_result: Full OCR result with metrics (may be None)
            caption: Vision caption (may be None)
            image_path: Path to image file
            image_hash: MD5 hash of image
            source_path: Original PDF path
            config: Optional config dict for OCR thresholds
            image_type: One of text_scan, table_scan, chart, diagram, photo
            ocr_catastrophic: OCR result is too poor to index (kept for audit only)
            vision_error_override: Explicit Vision skip/failure reason for audit accuracy

        Returns:
            List of image chunks (never empty - always at least one chunk)
        """
        chunks = []
        now = datetime.utcnow().isoformat() + "Z"

        # Determine OCR failure status
        ocr_failed = False
        if ocr_result is not None:
            ocr_failed = ocr_result.is_failed_with_config(config)
        ocr_processing_status = "failed" if ocr_catastrophic else "success"
        ocr_error = "catastrophic_ocr" if ocr_catastrophic else None

        # Visual types: chart, diagram, photo → caption ALWAYS, OCR fallback
        if image_type in ('chart', 'diagram', 'photo'):
            if caption and caption.strip():
                # Success: Vision caption available
                caption_chunk = Chunk(
                    doc_id=doc_id,
                    page=page,
                    chunk_id=f"{doc_id}_p{page}_i{image_index}_cap",
                    type="image_caption",
                    content=caption,
                    metadata=ChunkMetadata(
                        source_path=source_path,
                        image_hash=image_hash,
                        image_type=image_type,
                        ocr_failed=False,
                        vision_used=True,
                        ingest_ts=now,
                    ),
                )
                chunks.append(caption_chunk)
            if ocr_text and ocr_text.strip():
                # OCR text can add lexical recall even when Vision succeeds
                vision_err = None
                if not caption or not caption.strip():
                    vision_err = vision_error_override or "Vision API failed, used OCR fallback"
                ocr_chunk = Chunk(
                    doc_id=doc_id,
                    page=page,
                    chunk_id=f"{doc_id}_p{page}_i{image_index}_ocr",
                    type="image_ocr",
                    content=ocr_text,
                    metadata=ChunkMetadata(
                        source_path=source_path,
                        image_hash=image_hash,
                        image_type=image_type,
                        ocr_confidence=ocr_result.confidence if ocr_result else 0,
                        ocr_failed=ocr_failed,
                        ocr_failed_reason=ocr_result.get_failure_reason(config) if ocr_result and ocr_failed else None,
                        processing_status=ocr_processing_status,
                        ocr_error=ocr_error,
                        vision_used=False,
                        vision_error=vision_err,
                        ingest_ts=now,
                    ),
                )
                chunks.append(ocr_chunk)
            if chunks:
                return chunks

            # Total failure: Vision failed AND OCR failed/empty
            vision_err = vision_error_override or "Vision API failed or refused"
            if ocr_result:
                ocr_err = "OCR fallback also failed" if ocr_failed else "OCR returned no text"
            else:
                ocr_err = "OCR not attempted"

            failure_chunk = Chunk(
                doc_id=doc_id,
                page=page,
                chunk_id=f"{doc_id}_p{page}_i{image_index}_fail",
                type="image_caption",
                content=f"[Image processing failed: {image_type} on page {page}]",
                metadata=ChunkMetadata(
                    source_path=source_path,
                    image_hash=image_hash,
                    image_type=image_type,
                    processing_status="failed",
                    vision_error=vision_err,
                    ocr_error=ocr_err,
                    ingest_ts=now,
                ),
            )
            chunks.append(failure_chunk)
            return chunks

        # Text types: text_scan, table_scan → OCR chunk, caption on failure
        # OCR chunk if there's actual text content
        if ocr_text and ocr_text.strip():
            ocr_chunk = Chunk(
                doc_id=doc_id,
                page=page,
                chunk_id=f"{doc_id}_p{page}_i{image_index}_ocr",
                type="image_ocr",
                content=ocr_text,
                metadata=ChunkMetadata(
                    source_path=source_path,
                    image_hash=image_hash,
                    image_type=image_type,
                    ocr_confidence=ocr_result.confidence if ocr_result else 0,
                    ocr_failed=ocr_failed,
                    ocr_failed_reason=ocr_result.get_failure_reason(config) if ocr_result and ocr_failed else None,
                    processing_status=ocr_processing_status,
                    ocr_error=ocr_error,
                    vision_used=False,
                    ingest_ts=now,
                ),
            )
            chunks.append(ocr_chunk)

        # Caption chunk if OCR failed for text types (KEEP BOTH if OCR text exists but failed)
        if ocr_failed and caption and caption.strip():
            caption_chunk = Chunk(
                doc_id=doc_id,
                page=page,
                chunk_id=f"{doc_id}_p{page}_i{image_index}_cap",
                type="image_caption",
                content=caption,
                metadata=ChunkMetadata(
                    source_path=source_path,
                    image_hash=image_hash,
                    image_type=image_type,
                    ocr_failed=True,
                    ocr_failed_reason=ocr_result.get_failure_reason(config) if ocr_result else None,
                    vision_used=True,
                    ingest_ts=now,
                ),
            )
            chunks.append(caption_chunk)

        # Fallback: Total failure for text types (OCR failed AND Vision failed/empty)
        if not chunks:  # No chunks created yet
            # Determine accurate error messages
            ocr_err = ocr_result.get_failure_reason(config) if (ocr_result and ocr_failed) else "OCR failed"
            # caption being None after Vision call means either not attempted or failed/refused
            if vision_error_override:
                vision_err = vision_error_override
            elif caption is None and ocr_failed:
                vision_err = "Vision fallback failed or refused"
            else:
                vision_err = "Vision not attempted"

            failure_chunk = Chunk(
                doc_id=doc_id,
                page=page,
                chunk_id=f"{doc_id}_p{page}_i{image_index}_fail",
                type="image_ocr",
                content=f"[Image processing failed: {image_type} on page {page}]",
                metadata=ChunkMetadata(
                    source_path=source_path,
                    image_hash=image_hash,
                    image_type=image_type,
                    processing_status="failed",
                    ocr_error=ocr_err,
                    vision_error=vision_err,
                    ingest_ts=now,
                ),
            )
            chunks.append(failure_chunk)

        return chunks
