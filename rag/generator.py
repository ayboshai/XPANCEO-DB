"""
Generator module - LLM answering with no-answer detection and source citations.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from ingestion.models import ScoredChunk, SourceRef

logger = logging.getLogger(__name__)

# Standard refusal phrase per specification
NO_ANSWER_PHRASE = "В документах нет ответа / недостаточно данных."

# System prompt per specification
SYSTEM_PROMPT = """You are a helpful assistant answering questions about technical documents.

RULES:
1. Answer ONLY based on the provided context.
2. If the context does not contain enough information to answer the question, respond exactly:
   "В документах нет ответа / недостаточно данных."
3. Always cite sources using format: [doc_id|page|chunk_id]
4. Do not speculate or add information not in the context.
5. For tables, reference specific cells/rows when applicable.
6. Be concise and direct in your answers.

CONTEXT:
{context}"""


@dataclass
class GeneratorResponse:
    """Response from the generator."""
    
    answer: str
    sources: list[SourceRef]
    has_answer: bool


class Generator:
    """
    Generates answers from retrieved context using LLM.
    Implements no-answer policy and source citations.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        timeout: int = 30,
    ):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai not installed. Run: pip install openai")
            self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    def generate(
        self,
        query: str,
        context_chunks: list[ScoredChunk],
    ) -> GeneratorResponse:
        """
        Generate answer for query using context chunks.
        
        Args:
            query: User question
            context_chunks: Retrieved chunks with scores
            
        Returns:
            GeneratorResponse with answer, sources, and has_answer flag
        """
        # Handle empty context
        if not context_chunks:
            return GeneratorResponse(
                answer=NO_ANSWER_PHRASE,
                sources=[],
                has_answer=False,
            )
        
        # Build context string
        context = self._build_context(context_chunks)
        
        # Generate answer
        answer = self._call_llm(query, context)
        
        # Check for no-answer response
        has_answer = not self._is_no_answer(answer)
        
        # Extract sources
        sources = [chunk.to_source_ref() for chunk in context_chunks]
        
        return GeneratorResponse(
            answer=answer,
            sources=sources,
            has_answer=has_answer,
        )
    
    def _build_context(self, chunks: list[ScoredChunk]) -> str:
        """Build context string from chunks with metadata."""
        context_parts = []
        
        for chunk in chunks:
            c = chunk.chunk
            header = f"[{c.doc_id}|{c.page}|{c.chunk_id}|{c.type}]"
            
            # Preview for context awareness
            preview = c.content[:50].replace("\n", " ") + "..." if len(c.content) > 50 else c.content
            
            context_parts.append(f"{header}\n{c.content}\n")
        
        return "\n---\n".join(context_parts)
    
    def _call_llm(self, query: str, context: str) -> str:
        """Call LLM to generate answer with retry and backoff."""
        import time
        from shared import load_config
        
        config = load_config()
        max_retries = config.get("api_max_retries", 3)
        backoff_base = config.get("api_backoff_base", 2.0)
        
        system_message = SYSTEM_PROMPT.format(context=context)
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": query},
                    ],
                    temperature=0.1,  # Low temperature for factual answers
                    max_tokens=1000,
                    timeout=self.timeout,
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                last_error = e
                wait_time = backoff_base ** attempt
                logger.warning(
                    f"LLM call attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retry in {wait_time:.1f}s..."
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
        
        # All retries exhausted
        logger.error(f"LLM call failed after {max_retries} attempts: {last_error}")
        return NO_ANSWER_PHRASE
    
    def _is_no_answer(self, answer: str) -> bool:
        """Check if answer indicates no information found."""
        # Check for standard phrase
        if NO_ANSWER_PHRASE.lower() in answer.lower():
            return True
        
        # Check for common no-answer patterns
        no_answer_patterns = [
            r"нет ответа",
            r"недостаточно данных",
            r"не содержит информации",
            r"не найдено",
            r"no answer",
            r"not found",
            r"insufficient information",
            r"cannot answer",
            r"don't have enough",
        ]
        
        answer_lower = answer.lower()
        for pattern in no_answer_patterns:
            if re.search(pattern, answer_lower):
                return True
        
        return False


def create_generator(config: dict) -> Generator:
    """Factory function to create generator from config."""
    return Generator(
        api_key=config.get("openai_api_key", os.getenv("OPENAI_API_KEY", "")),
        model=config.get("model_chat", "gpt-4o-mini"),
        max_retries=config.get("api_max_retries", 3),
        timeout=config.get("api_timeout", 30),
    )
