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
    ) -> list[DatasetEntry]:
        """Generate overall questions from all chunks."""
        entries = []
        
        # Sample chunks
        sample_size = min(len(chunks), num_questions * 2)
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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )
        
        result = response.choices[0].message.content.strip()
        
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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )
        
        result = response.choices[0].message.content.strip()
        
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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )
        
        result = response.choices[0].message.content.strip()
        
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
    ) -> list[DatasetEntry]:
        """Generate complete dataset with all slices."""
        entries = []
        
        logger.info("Generating overall questions...")
        entries.extend(self.generate_overall(chunks, overall_count))
        
        logger.info("Generating table questions...")
        entries.extend(self.generate_table_slice(chunks, table_count))
        
        logger.info("Generating image questions...")
        entries.extend(self.generate_image_slice(chunks, image_count))
        
        logger.info("Generating no-answer questions...")
        entries.extend(self.generate_no_answer_slice(no_answer_count))
        
        logger.info(f"Generated {len(entries)} total questions")
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
