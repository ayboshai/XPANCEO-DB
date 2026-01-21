"""
Smoke test for ingestion pipeline.
Verifies that ingestion runs and creates all chunk types.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_chunk_models_valid():
    """Test that Pydantic models are correctly defined."""
    from ingestion.models import Chunk, ChunkMetadata, DatasetEntry
    
    # Create valid chunk
    chunk = Chunk(
        doc_id="test123",
        page=1,
        chunk_id="test123_p1_c0",
        type="text",
        content="Test content",
        metadata=ChunkMetadata(),
    )
    
    assert chunk.doc_id == "test123"
    assert chunk.type == "text"
    
    # Test serialization
    jsonl = chunk.to_jsonl()
    assert "test123" in jsonl
    
    # Test deserialization
    chunk2 = Chunk.from_jsonl(jsonl)
    assert chunk2.doc_id == chunk.doc_id


def test_ocr_result_failure_detection():
    """Test OCR failure heuristics."""
    from ingestion.models import OCRResult
    
    # Good OCR
    good = OCRResult(
        text="This is a good OCR result with enough text",
        confidence=85.0,
        word_count=10,
        char_count=60,
        alpha_ratio=0.8,
    )
    assert not good.is_failed
    
    # Failed OCR - low confidence
    bad_conf = OCRResult(
        text="Some text",
        confidence=30.0,
        word_count=10,
        char_count=60,
        alpha_ratio=0.8,
    )
    assert bad_conf.is_failed
    
    # Failed OCR - too short
    bad_len = OCRResult(
        text="Short",
        confidence=90.0,
        word_count=2,
        char_count=10,
        alpha_ratio=0.9,
    )
    assert bad_len.is_failed


def test_embedder_initialization():
    """Test embedder can be initialized (without API call)."""
    from ingestion.embedder import OpenAIEmbedder
    
    embedder = OpenAIEmbedder(
        api_key="test-key",
        model="text-embedding-3-small",
    )
    
    assert embedder.model == "text-embedding-3-small"


def test_faiss_index_operations():
    """Test FAISS index basic operations."""
    import numpy as np
    from ingestion.index_faiss import FAISSIndex
    
    with tempfile.TemporaryDirectory() as tmpdir:
        index = FAISSIndex(index_dir=tmpdir, dimension=8)
        
        # Insert
        vectors = np.random.rand(3, 8).tolist()
        ids = ["chunk1", "chunk2", "chunk3"]
        metadata = [{"doc": "doc1"}, {"doc": "doc1"}, {"doc": "doc2"}]
        
        index.upsert(ids, vectors, metadata)
        
        assert index.count == 3
        
        # Search
        query = np.random.rand(8).tolist()
        results = index.search(query, top_k=2)
        
        assert len(results) == 2
        assert results[0][0] in ids  # chunk_id
        assert isinstance(results[0][1], float)  # score
        
        # Delete
        index.delete(["chunk1"])
        assert index.count == 2


def test_chunker_text_splitting():
    """Test text chunking logic."""
    from ingestion.chunker import Chunker
    from ingestion.parser import Element
    
    chunker = Chunker(chunk_size_tokens=50, chunk_overlap_tokens=10)
    
    # Short text - single chunk
    elements = [Element(type="text", content="Short text.", page=1)]
    chunks = chunker.chunk(elements, doc_id="test", source_path="test.pdf")
    
    assert len(chunks) >= 1
    assert chunks[0].type == "text"
    
    # Long text - multiple chunks
    long_text = " ".join(["word"] * 200)
    elements = [Element(type="text", content=long_text, page=1)]
    chunks = chunker.chunk(elements, doc_id="test", source_path="test.pdf")
    
    assert len(chunks) > 1
    # Check linking
    assert chunks[0].metadata.next_chunk_id == chunks[1].chunk_id
    assert chunks[1].metadata.prev_chunk_id == chunks[0].chunk_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
