"""
RAG Pipeline - End-to-end question answering.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import yaml

from ingestion.models import Chunk, PredictionEntry, SourceRef
from ingestion.pipeline import IngestionPipeline

from .generator import Generator, GeneratorResponse, create_generator
from .retriever import Retriever, create_retriever

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Complete RAG response."""
    
    question: str
    answer: str
    sources: list[SourceRef]
    retrieved_chunks: list[SourceRef]
    has_answer: bool
    
    def to_prediction_entry(self, slice_name: str = "overall") -> PredictionEntry:
        """Convert to PredictionEntry for evaluation."""
        return PredictionEntry(
            question=self.question,
            answer=self.answer,
            sources=self.sources,
            retrieved_chunks=self.retrieved_chunks,
            slice=slice_name,
            has_answer_pred=self.has_answer,
        )


class RAGPipeline:
    """
    End-to-end RAG pipeline: Query → Retrieval → Generation.
    """
    
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
    
    def query(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            question: User question
            top_k: Override default top_k for retrieval
            
        Returns:
            RAGResponse with answer, sources, and metadata
        """
        k = top_k or self.top_k
        
        # Retrieve relevant chunks
        scored_chunks = self.retriever.search(question, k)
        
        # Generate answer
        response = self.generator.generate(question, scored_chunks)
        
        # Build retrieved chunks references
        retrieved = [chunk.to_source_ref() for chunk in scored_chunks]
        
        return RAGResponse(
            question=question,
            answer=response.answer,
            sources=response.sources,
            retrieved_chunks=retrieved,
            has_answer=response.has_answer,
        )
    
    def batch_query(self, questions: list[str]) -> list[RAGResponse]:
        """Answer multiple questions."""
        return [self.query(q) for q in questions]


def load_config(config_path: str = "config/master_config.yaml") -> dict:
    """Load configuration from YAML file."""
    # Resolve environment variables
    def resolve_env(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var, "")
        return value
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Resolve environment variables in config
    for key, value in config.items():
        config[key] = resolve_env(value)
    
    return config


def create_rag_pipeline(config_path: str = "config/master_config.yaml") -> RAGPipeline:
    """
    Create complete RAG pipeline from config.
    
    Args:
        config_path: Path to master_config.yaml
        
    Returns:
        Configured RAGPipeline instance
    """
    config = load_config(config_path)
    
    # Load chunks from ingestion pipeline
    ingestion = IngestionPipeline(config)
    chunks = ingestion.load_chunks()
    
    if not chunks:
        logger.warning("No chunks found. Please run ingestion first.")
    
    # Create components
    retriever = create_retriever(config, chunks)
    generator = create_generator(config)
    
    return RAGPipeline(
        retriever=retriever,
        generator=generator,
        top_k=config.get("top_k", 5),
    )
