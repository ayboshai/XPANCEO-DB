"""
Pydantic models for XPANCEO DB data structures.
All data contracts defined here - single source of truth for schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Chunk Models (Core Data Unit)
# =============================================================================

class ChunkMetadata(BaseModel):
    """Metadata for a chunk - tracks source, OCR status, linking."""
    
    source_path: Optional[str] = None
    bbox: Optional[list[float]] = None
    image_hash: Optional[str] = None
    ocr_confidence: Optional[float] = None
    ocr_failed: bool = False
    ocr_failed_reason: Optional[str] = None
    vision_used: bool = False
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    ingest_ts: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class Chunk(BaseModel):
    """
    Core data unit - all PDF elements normalize to this format.
    Types: text, table, image_ocr, image_caption
    """
    
    doc_id: str
    page: int
    chunk_id: str
    type: Literal["text", "table", "image_ocr", "image_caption"]
    content: str
    metadata: ChunkMetadata
    
    @field_validator("content")
    @classmethod
    def content_not_empty_for_searchable(cls, v: str, info) -> str:
        """Allow empty content but log warning for searchable types."""
        return v
    
    def to_jsonl(self) -> str:
        """Serialize to JSONL format."""
        return self.model_dump_json()
    
    @classmethod
    def from_jsonl(cls, line: str) -> "Chunk":
        """Deserialize from JSONL line."""
        return cls.model_validate_json(line)


# =============================================================================
# OCR Result
# =============================================================================

class OCRResult(BaseModel):
    """Result from Tesseract OCR processing."""
    
    text: str
    confidence: float  # Mean confidence 0-100
    word_count: int
    char_count: int
    alpha_ratio: float  # Ratio of alphanumeric chars
    
    def is_failed_with_config(self, config: Optional[dict] = None) -> bool:
        """
        Check if OCR failed based on config thresholds.
        Use this method instead of is_failed property for config-aware checking.
        """
        from shared import get_ocr_checker
        checker = get_ocr_checker(config)
        return checker.is_failed(
            confidence=self.confidence,
            char_count=self.char_count,
            word_count=self.word_count,
            alpha_ratio=self.alpha_ratio,
        )
    
    @property
    def is_failed(self) -> bool:
        """
        Check if OCR failed using default config.
        For explicit config, use is_failed_with_config(config).
        """
        return self.is_failed_with_config(None)
    
    def get_failure_reason(self, config: Optional[dict] = None) -> Optional[str]:
        """Get specific reason for OCR failure using config thresholds."""
        from shared import get_ocr_checker
        checker = get_ocr_checker(config)
        return checker.get_failure_reason(
            confidence=self.confidence,
            char_count=self.char_count,
            word_count=self.word_count,
            alpha_ratio=self.alpha_ratio,
        )


# =============================================================================
# Registry Entry (Per-Document Stats)
# =============================================================================

class ChunkCounts(BaseModel):
    """Counts of chunks by type."""
    text: int = 0
    table: int = 0
    image_ocr: int = 0
    image_caption: int = 0


class RegistryEntry(BaseModel):
    """Entry in pdf_registry.jsonl - tracks per-document statistics."""
    
    doc_id: str
    filename: str
    filepath: str
    pages: int
    chunks: ChunkCounts
    ocr_failure_rate: float = 0.0
    vision_fallback_rate: float = 0.0
    errors: int = 0
    parse_failed: bool = False
    ingest_ts: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    def to_jsonl(self) -> str:
        return self.model_dump_json()
    
    @classmethod
    def from_jsonl(cls, line: str) -> "RegistryEntry":
        return cls.model_validate_json(line)


# =============================================================================
# Error Log Entry
# =============================================================================

class ErrorLogEntry(BaseModel):
    """Entry in error_log.jsonl - tracks element-level failures."""
    
    ts: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    doc_id: str
    page: Optional[int] = None
    element_type: str
    error: str
    traceback: Optional[str] = None
    
    def to_jsonl(self) -> str:
        return self.model_dump_json()


# =============================================================================
# Dataset & Evaluation Models
# =============================================================================

class DatasetEntry(BaseModel):
    """Entry in dataset.jsonl for evaluation."""
    
    question: str
    slice: Literal["overall", "table", "image", "no-answer"]
    has_answer: bool
    expected_answer: Optional[str] = None
    doc_id: Optional[str] = None
    
    def to_jsonl(self) -> str:
        return self.model_dump_json()
    
    @classmethod
    def from_jsonl(cls, line: str) -> "DatasetEntry":
        return cls.model_validate_json(line)


class SourceRef(BaseModel):
    """Reference to a source chunk in RAG output."""
    
    doc_id: str
    page: int
    chunk_id: str
    type: str
    score: float
    preview: str  # First 20 words or table head
    full_content: Optional[str] = None  # Full text for judge evaluation


class PredictionEntry(BaseModel):
    """Entry in predictions.jsonl - RAG system output."""
    
    question: str
    answer: str
    sources: list[SourceRef]
    retrieved_chunks: list[SourceRef]
    slice: str
    has_answer_pred: bool
    
    def to_jsonl(self) -> str:
        return self.model_dump_json()
    
    @classmethod
    def from_jsonl(cls, line: str) -> "PredictionEntry":
        return cls.model_validate_json(line)


class JudgeScores(BaseModel):
    """Scores from LLM judge."""
    
    faithfulness: float = Field(ge=0, le=1)
    relevancy: float = Field(ge=0, le=1)
    context_precision: float = Field(ge=0, le=1)
    context_recall: float = Field(ge=0, le=1)
    no_answer_correct: Optional[bool] = None
    notes: str = ""


class JudgeResponse(BaseModel):
    """Entry in judge_responses.jsonl."""
    
    question: str
    answer: str
    expected_answer: Optional[str]
    judge: JudgeScores
    
    def to_jsonl(self) -> str:
        return self.model_dump_json()
    
    @classmethod
    def from_jsonl(cls, line: str) -> "JudgeResponse":
        return cls.model_validate_json(line)


# =============================================================================
# Scored Result (for retrieval)
# =============================================================================

class ScoredChunk(BaseModel):
    """Chunk with similarity score from retrieval."""
    
    chunk: Chunk
    score: float
    
    def to_source_ref(self, preview_words: int = 20, include_full: bool = True) -> SourceRef:
        """Convert to SourceRef for output."""
        words = self.chunk.content.split()[:preview_words]
        preview = " ".join(words)
        if len(self.chunk.content.split()) > preview_words:
            preview += "..."
        
        return SourceRef(
            doc_id=self.chunk.doc_id,
            page=self.chunk.page,
            chunk_id=self.chunk.chunk_id,
            type=self.chunk.type,
            score=self.score,
            preview=preview,
            full_content=self.chunk.content if include_full else None,
        )
