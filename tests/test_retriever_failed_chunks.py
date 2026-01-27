"""
Regression tests: failed chunks must never appear in retrieval results.
These tests use the real FAISSIndex and retriever logic, with a deterministic
local embedder to avoid external API calls.
"""

import os
import sys
import tempfile
from typing import Dict, List

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.index_faiss import FAISSIndex
from ingestion.models import Chunk, ChunkMetadata
from rag.retriever import Retriever


class DeterministicEmbedder:
    """Local embedder for tests (no external calls)."""

    def __init__(self, mapping: Dict[str, List[float]], default: List[float]):
        self.mapping = mapping
        self.default = default

    def embed_single(self, text: str) -> List[float]:
        return self.mapping.get(text, self.default)


def _make_chunk(
    *,
    chunk_id: str,
    chunk_type: str,
    content: str,
    processing_status: str = "success",
) -> Chunk:
    return Chunk(
        doc_id="doc",
        page=1,
        chunk_id=chunk_id,
        type=chunk_type,  # type: ignore[arg-type]
        content=content,
        metadata=ChunkMetadata(processing_status=processing_status),
    )


def _build_index(index_dir: str, vectors: Dict[str, List[float]], dimension: int) -> FAISSIndex:
    """Create a fresh FAISS index populated with provided vectors."""
    index = FAISSIndex(index_dir=index_dir, dimension=dimension)
    ids = list(vectors.keys())
    vecs = [vectors[i] for i in ids]
    metadata = [{"doc_id": "doc", "chunk_id": i} for i in ids]
    index.upsert(ids=ids, vectors=vecs, metadata=metadata)
    return index


def test_dense_search_filters_failed_chunks():
    query = "dense_failed_query"
    dim = 4

    failed_vec = [1.0, 0.0, 0.0, 0.0]
    success_vec = [0.8, 0.2, 0.0, 0.0]
    default_vec = [0.0, 1.0, 0.0, 0.0]

    chunks = {
        "failed_text": _make_chunk(
            chunk_id="failed_text",
            chunk_type="text",
            content="failed dense",
            processing_status="failed",
        ),
        "success_text": _make_chunk(
            chunk_id="success_text",
            chunk_type="text",
            content="success dense",
            processing_status="success",
        ),
    }

    with tempfile.TemporaryDirectory(prefix="faiss_retriever_test_") as tmpdir:
        index = _build_index(
            tmpdir,
            {
                "failed_text": failed_vec,
                "success_text": success_vec,
            },
            dimension=dim,
        )
        embedder = DeterministicEmbedder({query: failed_vec}, default_vec)
        retriever = Retriever(
            embedder=embedder,
            index=index,
            chunks_lookup=chunks,
            top_k=2,
            hybrid_enabled=False,
        )

        results = retriever.search(query, top_k=2)

    assert results, "Expected at least one valid result after filtering"
    assert all(r.chunk.metadata.processing_status == "success" for r in results)
    assert any(r.chunk.chunk_id == "success_text" for r in results)
    assert all(r.chunk.chunk_id != "failed_text" for r in results)


def test_table_topup_filters_failed_chunks():
    query = "What does the table show?"
    dim = 4

    failed_table_vec = [1.0, 0.0, 0.0, 0.0]
    success_table_vec = [0.7, 0.3, 0.0, 0.0]
    text_vec = [0.2, 0.8, 0.0, 0.0]
    default_vec = [0.0, 1.0, 0.0, 0.0]

    chunks = {
        "failed_table": _make_chunk(
            chunk_id="failed_table",
            chunk_type="table",
            content="| bad | table |",
            processing_status="failed",
        ),
        "success_table": _make_chunk(
            chunk_id="success_table",
            chunk_type="table",
            content="| good | table |",
            processing_status="success",
        ),
        "success_text": _make_chunk(
            chunk_id="success_text",
            chunk_type="text",
            content="text context",
            processing_status="success",
        ),
    }

    with tempfile.TemporaryDirectory(prefix="faiss_retriever_test_") as tmpdir:
        index = _build_index(
            tmpdir,
            {
                "failed_table": failed_table_vec,
                "success_table": success_table_vec,
                "success_text": text_vec,
            },
            dimension=dim,
        )
        embedder = DeterministicEmbedder({query: failed_table_vec}, default_vec)
        retriever = Retriever(
            embedder=embedder,
            index=index,
            chunks_lookup=chunks,
            top_k=1,
            hybrid_enabled=False,
        )

        results = retriever.search(query, top_k=1)

    assert results, "Expected table top-up to recover a valid table chunk"
    assert all(r.chunk.metadata.processing_status == "success" for r in results)
    assert any(r.chunk.chunk_id == "success_table" for r in results)
    assert all(r.chunk.chunk_id != "failed_table" for r in results)


def test_image_topup_filters_failed_chunks():
    query = "In the figure, what is shown?"
    dim = 4

    failed_image_vec = [1.0, 0.0, 0.0, 0.0]
    success_image_vec = [0.75, 0.25, 0.0, 0.0]
    text_vec = [0.3, 0.7, 0.0, 0.0]
    default_vec = [0.0, 1.0, 0.0, 0.0]

    chunks = {
        "failed_image": _make_chunk(
            chunk_id="failed_image",
            chunk_type="image_caption",
            content="failed image caption",
            processing_status="failed",
        ),
        "success_image": _make_chunk(
            chunk_id="success_image",
            chunk_type="image_caption",
            content="success image caption",
            processing_status="success",
        ),
        "success_text": _make_chunk(
            chunk_id="success_text",
            chunk_type="text",
            content="text context",
            processing_status="success",
        ),
    }

    with tempfile.TemporaryDirectory(prefix="faiss_retriever_test_") as tmpdir:
        index = _build_index(
            tmpdir,
            {
                "failed_image": failed_image_vec,
                "success_image": success_image_vec,
                "success_text": text_vec,
            },
            dimension=dim,
        )
        embedder = DeterministicEmbedder({query: failed_image_vec}, default_vec)
        retriever = Retriever(
            embedder=embedder,
            index=index,
            chunks_lookup=chunks,
            top_k=1,
            hybrid_enabled=False,
        )

        results = retriever.search(query, top_k=1)

    assert results, "Expected image top-up to recover a valid image chunk"
    assert all(r.chunk.metadata.processing_status == "success" for r in results)
    assert any(r.chunk.chunk_id == "success_image" for r in results)
    assert all(r.chunk.chunk_id != "failed_image" for r in results)
