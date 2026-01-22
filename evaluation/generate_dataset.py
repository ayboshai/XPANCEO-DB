"""
Dataset generator for evaluation.
Creates test questions across slices: overall, table, image, no-answer.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.models import Chunk, DatasetEntry

logger = logging.getLogger(__name__)


class DatasetGenerator:
    """
    Generates evaluation dataset with multiple slices.
    Supports reproducible generation via seed.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        seed: int = 42,
        max_retries: int = 3,
        backoff_base: float = 2.0,
    ):
        self.api_key = api_key
        self.model = model
        self.seed = seed
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._client = None
        
        # Set random seed for reproducibility
        random.seed(seed)
    
    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    def _call_llm_with_retry(self, prompt: str) -> Optional[str]:
        """Call LLM with retry and backoff."""
        import time
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=300,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                wait_time = self.backoff_base ** attempt
                logger.warning(
                    f"LLM call attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                    f"Retry in {wait_time:.1f}s..."
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
        
        logger.error(f"LLM call failed after {self.max_retries} attempts: {last_error}")
        return None
    
    def generate_overall(
        self,
        chunks: list[Chunk],
        num_questions: int = 20,
        use_ragas: bool = True,
    ) -> list[DatasetEntry]:
        """
        Generate overall questions from all chunks.
        Uses RAGAS TestsetGenerator if available, falls back to custom LLM.
        """
        entries = []
        
        # Try RAGAS first if requested
        if use_ragas:
            try:
                entries = self._generate_with_ragas(chunks, num_questions)
                if entries:
                    logger.info(f"Generated {len(entries)} questions via RAGAS")
                    return entries
            except ImportError:
                logger.warning("⚠️ RAGAS UNAVAILABLE: ragas package not installed. Using custom LLM generator. Run: pip install ragas langchain-openai")
            except Exception as e:
                logger.warning(f"⚠️ RAGAS FAILED: {e}. Falling back to custom LLM generator.")
        
        # Fallback to custom LLM generation
        logger.warning("📝 Using custom LLM generator (not RAGAS) for dataset generation")
        
        # Sample chunks
        sample_size = min(len(chunks), num_questions * 2)
        if sample_size == 0:
            logger.error("No chunks available for question generation")
            return []
        
        sampled = random.sample(chunks, sample_size)
        
        for chunk in sampled[:num_questions]:
            try:
                question, expected = self._generate_qa_pair(chunk)
                if question:
                    entries.append(DatasetEntry(
                        question=question,
                        slice="overall",
                        has_answer=True,
                        expected_answer=expected,
                        doc_id=chunk.doc_id,
                    ))
            except Exception as e:
                logger.warning(f"Failed to generate question: {e}")
        
        return entries
    
    def _generate_with_ragas(self, chunks: list[Chunk], num_questions: int) -> list[DatasetEntry]:
        """Generate questions using RAGAS TestsetGenerator."""
        from ragas.testset.generator import TestsetGenerator
        from ragas.testset.evolutions import simple, reasoning, multi_context
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain.docstore.document import Document
        
        # Convert chunks to LangChain documents
        documents = [
            Document(
                page_content=chunk.content,
                metadata={
                    "doc_id": chunk.doc_id,
                    "page": chunk.page,
                    "type": chunk.type,
                    "chunk_id": chunk.chunk_id,
                }
            )
            for chunk in chunks if chunk.content.strip()
        ]
        
        if not documents:
            raise ValueError("No valid documents for RAGAS")
        
        # Initialize generator with OpenAI models
        generator_llm = ChatOpenAI(model=self.model, api_key=self.api_key)
        critic_llm = ChatOpenAI(model=self.model, api_key=self.api_key)
        embeddings = OpenAIEmbeddings(api_key=self.api_key)
        
        generator = TestsetGenerator.from_langchain(
            generator_llm=generator_llm,
            critic_llm=critic_llm,
            embeddings=embeddings,
        )
        
        # Generate testset
        testset = generator.generate_with_langchain_docs(
            documents=documents,
            test_size=num_questions,
            distributions={simple: 0.5, reasoning: 0.3, multi_context: 0.2},
        )
        
        # Convert to DatasetEntry
        entries = []
        for row in testset.to_pandas().itertuples():
            entries.append(DatasetEntry(
                question=row.question,
                slice="overall",
                has_answer=True,
                expected_answer=row.ground_truth if hasattr(row, 'ground_truth') else row.answer,
                doc_id=None,  # RAGAS may not track this
            ))
        
        return entries
    
    def generate_table_slice(
        self,
        chunks: list[Chunk],
        num_questions: int = 10,
    ) -> list[DatasetEntry]:
        """Generate questions specifically about tables."""
        table_chunks = [c for c in chunks if c.type == "table"]
        
        if not table_chunks:
            logger.warning("No table chunks found")
            return []
        
        entries = []
        sampled = random.sample(table_chunks, min(len(table_chunks), num_questions))
        
        for chunk in sampled:
            try:
                question, expected = self._generate_table_question(chunk)
                if question:
                    entries.append(DatasetEntry(
                        question=question,
                        slice="table",
                        has_answer=True,
                        expected_answer=expected,
                        doc_id=chunk.doc_id,
                    ))
            except Exception as e:
                logger.warning(f"Failed to generate table question: {e}")
        
        return entries
    
    def generate_image_slice(
        self,
        chunks: list[Chunk],
        num_questions: int = 10,
    ) -> list[DatasetEntry]:
        """Generate questions about images."""
        image_chunks = [c for c in chunks if c.type in ("image_ocr", "image_caption")]
        
        if not image_chunks:
            logger.warning("No image chunks found")
            return []
        
        entries = []
        sampled = random.sample(image_chunks, min(len(image_chunks), num_questions))
        
        for chunk in sampled:
            try:
                question, expected = self._generate_image_question(chunk)
                if question:
                    entries.append(DatasetEntry(
                        question=question,
                        slice="image",
                        has_answer=True,
                        expected_answer=expected,
                        doc_id=chunk.doc_id,
                    ))
            except Exception as e:
                logger.warning(f"Failed to generate image question: {e}")
        
        return entries
    
    def generate_no_answer_slice(
        self,
        num_questions: int = 10,
    ) -> list[DatasetEntry]:
        """Generate questions that should NOT be answerable from the documents."""
        no_answer_prompts = [
            "What is the current stock price of Apple?",
            "Who won the FIFA World Cup in 2030?",
            "What is the weather forecast for tomorrow in Tokyo?",
            "How many employees does Google have as of today?",
            "What is the recipe for tiramisu?",
            "Who is the current president of Mars colony?",
            "What time does the next train to London depart?",
            "How much does a Tesla Model S cost in 2025?",
            "What is the population of Antarctica?",
            "Who invented the time machine?",
            "What are the symptoms of a fictional disease XYZ?",
            "How do you fix a broken quantum entanglement?",
            "What is the GDP of Atlantis?",
            "How many stars are visible from Proxima Centauri?",
            "What is the speed limit on Mars highways?",
        ]
        
        entries = []
        for prompt in random.sample(no_answer_prompts, min(len(no_answer_prompts), num_questions)):
            entries.append(DatasetEntry(
                question=prompt,
                slice="no-answer",
                has_answer=False,
                expected_answer="В документах нет ответа / недостаточно данных.",
                doc_id=None,
            ))
        
        return entries
    
    def _generate_qa_pair(self, chunk: Chunk) -> tuple[Optional[str], Optional[str]]:
        """Generate question-answer pair from chunk content."""
        prompt = f"""Based on the following text, generate a single factual question and its correct answer.
The question should be answerable ONLY from this text.

Text:
{chunk.content[:1500]}

Respond in JSON format:
{{"question": "...", "answer": "..."}}"""

        result = self._call_llm_with_retry(prompt)
        if not result:
            return None, None
        
        # Parse JSON
        try:
            # Handle markdown code blocks
            if "```" in result:
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            
            data = json.loads(result)
            return data.get("question"), data.get("answer")
        except json.JSONDecodeError:
            return None, None
    
    def _generate_table_question(self, chunk: Chunk) -> tuple[Optional[str], Optional[str]]:
        """Generate question about table content."""
        prompt = f"""Based on this table, generate a question that requires reading specific data from the table.

Table:
{chunk.content[:1500]}

Respond in JSON format:
{{"question": "...", "answer": "..."}}"""

        result = self._call_llm_with_retry(prompt)
        if not result:
            return None, None
        
        try:
            if "```" in result:
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            data = json.loads(result)
            return data.get("question"), data.get("answer")
        except json.JSONDecodeError:
            return None, None
    
    def _generate_image_question(self, chunk: Chunk) -> tuple[Optional[str], Optional[str]]:
        """Generate question about image content."""
        prompt = f"""Based on this image description/OCR text, generate a question about what the image shows.

Image content:
{chunk.content[:1500]}

Respond in JSON format:
{{"question": "...", "answer": "..."}}"""

        result = self._call_llm_with_retry(prompt)
        if not result:
            return None, None
        
        try:
            if "```" in result:
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            data = json.loads(result)
            return data.get("question"), data.get("answer")
        except json.JSONDecodeError:
            return None, None
    
    def generate_full_dataset(
        self,
        chunks: list[Chunk],
        overall_count: int = 20,
        table_count: int = 10,
        image_count: int = 10,
        no_answer_count: int = 10,
        min_ratio: float = 0.5,  # Warn if less than 50% of target achieved
        strict: bool = False,  # If True, raise error when slices below min_ratio
    ) -> list[DatasetEntry]:
        """
        Generate complete dataset with all slices.
        Logs warnings if target counts not achieved.
        
        Args:
            min_ratio: Minimum ratio of target to consider acceptable (0.5 = 50%)
            strict: If True, raise ValueError when any slice is below min_ratio
        """
        entries = []
        warnings = []
        
        logger.info("Generating overall questions...")
        overall_entries = self.generate_overall(chunks, overall_count)
        entries.extend(overall_entries)
        if len(overall_entries) < overall_count * min_ratio:
            warnings.append(f"overall: {len(overall_entries)}/{overall_count} (below {min_ratio*100:.0f}% target)")
        
        logger.info("Generating table questions...")
        table_entries = self.generate_table_slice(chunks, table_count)
        entries.extend(table_entries)
        if len(table_entries) < table_count * min_ratio:
            warnings.append(f"table: {len(table_entries)}/{table_count} (below {min_ratio*100:.0f}% target)")
        
        logger.info("Generating image questions...")
        image_entries = self.generate_image_slice(chunks, image_count)
        entries.extend(image_entries)
        if len(image_entries) < image_count * min_ratio:
            warnings.append(f"image: {len(image_entries)}/{image_count} (below {min_ratio*100:.0f}% target)")
        
        logger.info("Generating no-answer questions...")
        no_answer_entries = self.generate_no_answer_slice(no_answer_count)
        entries.extend(no_answer_entries)
        if len(no_answer_entries) < no_answer_count * min_ratio:
            warnings.append(f"no-answer: {len(no_answer_entries)}/{no_answer_count} (below {min_ratio*100:.0f}% target)")
        
        # Log slice statistics
        logger.info(f"Slice statistics: overall={len(overall_entries)}, table={len(table_entries)}, "
                   f"image={len(image_entries)}, no-answer={len(no_answer_entries)}")
        logger.info(f"Generated {len(entries)} total questions")
        
        # Warn about undercounts
        if warnings:
            logger.warning(f"Some slices below target: {'; '.join(warnings)}")
            
            # In strict mode, fail if any slice is below target
            if strict:
                raise ValueError(
                    f"Dataset generation failed in strict mode. "
                    f"Slices below {min_ratio*100:.0f}% target: {'; '.join(warnings)}"
                )
        
        return entries


def save_dataset(entries: list[DatasetEntry], output_path: str) -> None:
    """Save dataset to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry.to_jsonl() + "\n")
    
    logger.info(f"Saved {len(entries)} entries to {output_path}")


def load_dataset(input_path: str) -> list[DatasetEntry]:
    """Load dataset from JSONL file."""
    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(DatasetEntry.from_jsonl(line))
    return entries
