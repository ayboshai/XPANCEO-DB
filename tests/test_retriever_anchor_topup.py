"""
Regression test: anchor top-up must boost anchor chunks even if they are
already present in dense results with a low score.
Uses real FAISSIndex and retriever logic without external API calls.
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
    def __init__(self, mapping: Dict[str, List[float]], default: List[float]):
        self.mapping = mapping
        self.default = default

    def embed_single(self, text: str) -> List[float]:
        return self.mapping.get(text, self.default)


def _make_chunk(chunk_id: str, content: str) -> Chunk:
    return Chunk(
        doc_id="doc",
        page=1,
        chunk_id=chunk_id,
        type="text",
        content=content,
        metadata=ChunkMetadata(processing_status="success"),
    )


def _build_index(index_dir: str, vectors: Dict[str, List[float]], dimension: int) -> FAISSIndex:
    index = FAISSIndex(index_dir=index_dir, dimension=dimension)
    ids = list(vectors.keys())
    vecs = [vectors[i] for i in ids]
    metadata = [{"doc_id": "doc", "chunk_id": i} for i in ids]
    index.upsert(ids=ids, vectors=vecs, metadata=metadata)
    return index


def test_anchor_topup_boosts_existing_low_rank_anchor_chunk():
    dim = 4
    top_k = 3
    query = 'In the image, what phrase includes "RELL"?'

    # Query aligned with the first axis.
    query_vec = [1.0, 0.0, 0.0, 0.0]
    default_vec = [0.0, 1.0, 0.0, 0.0]

    # Many high-similarity distractors.
    vectors = {
        "d1": [1.0, 0.0, 0.0, 0.0],
        "d2": [0.99, 0.01, 0.0, 0.0],
        "d3": [0.98, 0.02, 0.0, 0.0],
        "d4": [0.97, 0.03, 0.0, 0.0],
        "d5": [0.96, 0.04, 0.0, 0.0],
        # Anchor chunk is present in dense results but ranks low by similarity.
        "anchor": [0.1, 0.99, 0.0, 0.0],
    }

    chunks = {
        "d1": _make_chunk("d1", "distractor one"),
        "d2": _make_chunk("d2", "distractor two"),
        "d3": _make_chunk("d3", "distractor three"),
        "d4": _make_chunk("d4", "distractor four"),
        "d5": _make_chunk("d5", "distractor five"),
        "anchor": _make_chunk("anchor", "This line contains the anchor token RELL clearly."),
    }

    with tempfile.TemporaryDirectory(prefix="faiss_anchor_topup_test_") as tmpdir:
        index = _build_index(tmpdir, vectors, dimension=dim)
        retriever = Retriever(
            embedder=DeterministicEmbedder({query: query_vec}, default_vec),
            index=index,
            chunks_lookup=chunks,
            top_k=top_k,
            hybrid_enabled=True,  # uses k*2 dense results
        )

        results = retriever.search(query, top_k=top_k)

    # Anchor must survive top-k truncation due to anchor top-up boosting.
    result_ids = [r.chunk.chunk_id for r in results]
    assert "anchor" in result_ids

