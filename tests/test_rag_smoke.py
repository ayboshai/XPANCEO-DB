"""
Smoke test for RAG pipeline.
Verifies retrieval and generation produce structured output.
"""

import os
import sys

import pytest

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_generator_no_answer_detection():
    """Test no-answer phrase detection."""
    from rag.generator import Generator
    
    # Create generator (won't call API)
    gen = Generator(api_key="test", model="gpt-4o-mini")
    
    # Test no-answer detection
    assert gen._is_no_answer("В документах нет ответа / недостаточно данных.")
    assert gen._is_no_answer("I cannot answer this question. Not found in context.")
    assert gen._is_no_answer("Недостаточно данных для ответа.")
    
    # Should not flag as no-answer
    assert not gen._is_no_answer("The answer is 42.")
    assert not gen._is_no_answer("According to the document, X equals Y.")


def test_source_ref_creation():
    """Test SourceRef from ScoredChunk."""
    from ingestion.models import Chunk, ChunkMetadata, ScoredChunk
    
    chunk = Chunk(
        doc_id="doc1",
        page=5,
        chunk_id="doc1_p5_c0",
        type="text",
        content="The quick brown fox jumps over the lazy dog. This is additional text to make it longer.",
        metadata=ChunkMetadata(),
    )
    
    scored = ScoredChunk(chunk=chunk, score=0.95)
    source_ref = scored.to_source_ref(preview_words=5)
    
    assert source_ref.doc_id == "doc1"
    assert source_ref.page == 5
    assert source_ref.type == "text"
    assert source_ref.score == 0.95
    assert "quick" in source_ref.preview
    assert "..." in source_ref.preview


def test_retriever_bm25_disabled():
    """Test retriever initialization with BM25 disabled."""
    from ingestion.models import Chunk, ChunkMetadata
    from rag.retriever import Retriever
    
    # Mock embedder and index
    class MockEmbedder:
        def embed_single(self, text):
            return [0.1] * 8
    
    class MockIndex:
        def search(self, vector, top_k):
            return [("chunk1", 0.9, {"doc_id": "doc1"})]
    
    chunk = Chunk(
        doc_id="doc1",
        page=1,
        chunk_id="chunk1",
        type="text",
        content="Test",
        metadata=ChunkMetadata(),
    )
    
    retriever = Retriever(
        embedder=MockEmbedder(),
        index=MockIndex(),
        chunks_lookup={"chunk1": chunk},
        top_k=5,
        hybrid_enabled=False,
    )
    
    results = retriever.search("test query")
    
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk1"
    assert results[0].score == 0.9


def test_rag_response_to_prediction():
    """Test RAGResponse conversion to PredictionEntry."""
    from ingestion.models import SourceRef
    from rag.pipeline import RAGResponse
    
    response = RAGResponse(
        question="What is X?",
        answer="X is Y.",
        sources=[
            SourceRef(
                doc_id="doc1",
                page=1,
                chunk_id="c1",
                type="text",
                score=0.9,
                preview="Preview...",
            )
        ],
        retrieved_chunks=[],
        has_answer=True,
    )
    
    prediction = response.to_prediction_entry("overall")
    
    assert prediction.question == "What is X?"
    assert prediction.answer == "X is Y."
    assert prediction.slice == "overall"
    assert prediction.has_answer_pred is True
    assert len(prediction.sources) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
