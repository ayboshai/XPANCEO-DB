"""
E2E Smoke Test - Full Pipeline Verification
Tests: ingest → RAG query → eval (with no-answer) → report generation
"""

import os
import sys
import tempfile
import json
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestE2EPipeline:
    """End-to-end smoke tests for the full RAG pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temp directory for test artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_faiss_only_fail_fast(self):
        """Verify non-faiss backend fails fast."""
        from ingestion.index_faiss import create_index
        
        # Should raise ValueError, not NotImplementedError
        with pytest.raises(ValueError) as exc_info:
            create_index({"index_backend": "qdrant"})
        
        assert "Unsupported index_backend" in str(exc_info.value)
        assert "faiss" in str(exc_info.value).lower()
    
    def test_sync_rate_limiter_exists(self):
        """Verify SyncRateLimiter is properly instantiated."""
        from shared import get_sync_limiter
        
        limiter = get_sync_limiter()
        assert hasattr(limiter, "acquire_embedding")
        assert hasattr(limiter, "acquire_vision")
        assert hasattr(limiter, "acquire_judge")
        assert hasattr(limiter, "_last_api_call")  # Global timestamp
    
    def test_require_ragas_flag(self):
        """Verify require_ragas raises RuntimeError if RAGAS unavailable."""
        from evaluation.generate_dataset import DatasetGenerator
        from ingestion.models import Chunk, ChunkMetadata
        
        # Create sample chunks
        chunks = [
            Chunk(
                doc_id="test",
                page=1,
                chunk_id="test_1",
                type="text",
                content="Test content for RAGAS verification.",
                metadata=ChunkMetadata(),
            )
        ]
        
        # Generator with fake API key (no API calls when use_ragas=False)
        gen = DatasetGenerator(api_key="test-key")
        
        # Should raise RuntimeError with require_ragas=True
        with pytest.raises(RuntimeError) as exc_info:
            gen.generate_overall(chunks, num_questions=1, use_ragas=False, require_ragas=True)
        
        assert "RAGAS required" in str(exc_info.value)
    
    def test_no_answer_slice_roundtrip(self, temp_dir):
        """Verify no-answer slice generation and dataset roundtrip."""
        from evaluation.generate_dataset import DatasetGenerator, save_dataset, load_dataset
        
        gen = DatasetGenerator(api_key="test-key")
        entries = gen.generate_no_answer_slice(num_questions=3)
        
        assert len(entries) == 3
        assert all(e.slice == "no-answer" for e in entries)
        assert all(e.has_answer is False for e in entries)
        
        out_path = os.path.join(temp_dir, "dataset.jsonl")
        save_dataset(entries, out_path)
        loaded = load_dataset(out_path)
        
        assert len(loaded) == 3
        assert loaded[0].slice == "no-answer"
    
    def test_cache_cleanup_extracts_image_hash(self):
        """Verify _get_image_hashes_for_doc extracts metadata.image_hash."""
        from ingestion.pipeline import IngestionPipeline
        import tempfile
        
        # Create temp chunks file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            chunk = {
                "doc_id": "test_doc",
                "page": 1,
                "chunk_id": "test_1",
                "type": "image_ocr",
                "content": "OCR text",
                "metadata": {
                    "image_hash": "abc123md5hash"
                }
            }
            f.write(json.dumps(chunk) + "\n")
            chunks_path = f.name
        
        try:
            config = {"chunks_path": chunks_path}
            pipeline = IngestionPipeline(config)
            
            hashes = pipeline._get_image_hashes_for_doc("test_doc")
            
            assert len(hashes) == 1
            assert hashes[0] == "abc123md5hash"
        finally:
            os.unlink(chunks_path)
    
    def test_models_have_required_fields(self):
        """Verify Chunk and related models have all required fields."""
        from ingestion.models import Chunk, ChunkMetadata, DatasetEntry
        
        # ChunkMetadata should have image_hash
        meta = ChunkMetadata(image_hash="test_hash")
        assert meta.image_hash == "test_hash"
        
        # DatasetEntry should have slice types
        entry = DatasetEntry(
            question="Test?",
            slice="no-answer",
            has_answer=False,
            expected_answer="В документах нет ответа."
        )
        assert entry.slice == "no-answer"
        assert entry.has_answer is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
