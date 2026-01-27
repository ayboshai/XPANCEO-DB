"""
Generator module - LLM answering with no-answer detection and source citations.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict,  Optional

from ingestion.models import ScoredChunk, SourceRef

logger = logging.getLogger(__name__)

# Standard refusal phrase per specification
NO_ANSWER_PHRASE = "В документах нет ответа / недостаточно данных."

# System prompt per specification
SYSTEM_PROMPT = """You are a precise technical assistant that answers questions based STRICTLY on provided context.

CRITICAL RULES:
1. **USE ONLY THE CONTEXT BELOW** to answer the question. If the answer is present in ANY part of the context, extract it EXACTLY as written.
   - DO NOT add information from your general knowledge.
   - DO NOT make assumptions or inferences beyond what's explicitly stated.
   - QUOTE or PARAPHRASE the context directly.

2. **ONLY refuse to answer** if the context is COMPLETELY UNRELATED to the question or if NO relevant information exists at all.
   - Refuse response format: "В документах нет ответа / недостаточно данных."
   - DO NOT refuse just because the answer seems incomplete or partial - provide what you find from the context!

3. **Cite sources** using the format shown in brackets: [doc_id|page|chunk_id]
   - Example: "The author is T. Wiens [9a239c804921|31|9a239c804921_p31_t6]"
   - ALWAYS include source citations for factual claims.

4. **For partial information**: If context has SOME relevant information but not complete details, provide ONLY what's in the context and note what's missing.
   - Example: "The study discusses X [source], but the full methodology is not detailed in the provided context."

5. **Be direct and concise** - extract facts from context word-for-word when possible. NO speculation, NO external knowledge.

CONTEXT CHUNKS (use ALL information below, nothing else):
{context}"""


@dataclass
class GeneratorResponse:
    """Response from the generator."""
    
    answer: str
    sources: List[SourceRef]
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
        context_chunks: List[ScoredChunk],
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
    
    def _build_context(self, chunks: List[ScoredChunk]) -> str:
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
                    temperature=0.3,  # Moderate temperature for balanced factual answers (0.1 was too conservative)
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
