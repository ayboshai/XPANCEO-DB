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
import re
from pathlib import Path
from typing import List, Tuple, Dict,  Optional

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
        chunks: List[Chunk],
        num_questions: int = 20,
        use_ragas: bool = False,
        require_ragas: bool = False,  # If True, fail if RAGAS unavailable
    ) -> List[DatasetEntry]:
        """
        Generate overall questions from all chunks.
        Uses RAGAS TestsetGenerator if available, falls back to custom LLM.
        
        Args:
            require_ragas: If True, raise error if RAGAS unavailable or fails
        """
        entries = []
        ragas_error = None
        
        # Try RAGAS first if requested
        if use_ragas:
            try:
                entries = self._generate_with_ragas(chunks, num_questions)
                if entries:
                    logger.info(f"✅ Generated {len(entries)} questions via RAGAS")
                    return entries
            except ImportError as e:
                ragas_error = f"RAGAS package not installed. Run: pip install ragas langchain-openai"
                logger.warning(f"⚠️ RAGAS UNAVAILABLE: {ragas_error}")
            except Exception as e:
                ragas_error = str(e)
                logger.warning(f"⚠️ RAGAS FAILED: {e}")
        
        # Fail if RAGAS required but unavailable
        if require_ragas:
            raise RuntimeError(
                f"RAGAS required but unavailable: {ragas_error or 'use_ragas=False'}. "
                f"Install RAGAS: pip install ragas langchain-openai"
            )
        
        # Fallback to custom LLM generation
        logger.warning("📝 Using custom LLM generator (not RAGAS) for dataset generation")
        
        def _is_clean_text_chunk(c: Chunk) -> bool:
            if getattr(c.metadata, "processing_status", "success") != "success":
                return False
            if c.type != "text":
                return False
            if not c.content or not c.content.strip():
                return False
            text = c.content.strip()
            lower = text.lower()
            if "unreadable" in lower:
                return False
            if lower.startswith("[image processing failed"):
                return False
            if len(text) < 200:
                return False
            pipe_ratio = text.count("|") / max(1, len(text))
            if pipe_ratio > 0.01:
                return False
            return True

        # Filter to clean, successful TEXT chunks only
        chunks = [c for c in chunks if _is_clean_text_chunk(c)]

        # Sample chunks (oversample to allow validation filtering)
        sample_size = min(len(chunks), num_questions * 15)
        if sample_size == 0:
            logger.error("No chunks available for question generation")
            return []
        
        sampled = random.sample(chunks, sample_size)
        
        for chunk in sampled:
            try:
                question, expected = self._generate_qa_pair(chunk)
                # Validate expected answer appears in source content
                if question and expected and self._answer_in_content(expected, chunk.content, strict=True):
                    entries.append(DatasetEntry(
                        question=question,
                        slice="overall",
                        has_answer=True,
                        expected_answer=expected,
                        doc_id=chunk.doc_id,
                    ))
                    if len(entries) >= num_questions:
                        break
            except Exception as e:
                logger.warning(f"Failed to generate question: {e}")
        
        return entries
    
    def _generate_with_ragas(self, chunks: List[Chunk], num_questions: int) -> List[DatasetEntry]:
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
        chunks: List[Chunk],
        num_questions: int = 10,
    ) -> List[DatasetEntry]:
        """Generate questions specifically about tables."""
        table_chunks = [
            c for c in chunks
            if c.type == "table"
            and c.content and c.content.strip()
            and getattr(c.metadata, "processing_status", "success") == "success"
        ]
        
        if not table_chunks:
            logger.warning("No table chunks found")
            return []
        
        entries = []
        sampled = random.sample(table_chunks, min(len(table_chunks), num_questions * 4))
        
        for chunk in sampled:
            try:
                question, expected = self._generate_table_question(chunk)
                # Validate expected answer appears in table content
                if question and expected and self._answer_in_content(expected, chunk.content):
                    entries.append(DatasetEntry(
                        question=question,
                        slice="table",
                        has_answer=True,
                        expected_answer=expected,
                        doc_id=chunk.doc_id,
                    ))
                    if len(entries) >= num_questions:
                        break
            except Exception as e:
                logger.warning(f"Failed to generate table question: {e}")
        
        return entries
    
    def generate_image_slice(
        self,
        chunks: List[Chunk],
        num_questions: int = 10,
    ) -> List[DatasetEntry]:
        """Generate questions about images."""
        image_chunks = [
            c for c in chunks
            if c.type in ("image_ocr", "image_caption")
            and c.content and c.content.strip()
            and getattr(c.metadata, "processing_status", "success") == "success"
        ]
        
        if not image_chunks:
            logger.warning("No image chunks found")
            return []
        
        entries = []
        seen_questions = set()
        # Oversample more aggressively because strict validation filters harder
        sampled = random.sample(image_chunks, min(len(image_chunks), num_questions * 8))
        
        for chunk in sampled:
            try:
                question, expected = self._generate_image_question(chunk)
                # Validate expected answer appears in caption/OCR content
                if question and expected and self._answer_in_content(expected, chunk.content, strict=True):
                    if question in seen_questions:
                        continue
                    seen_questions.add(question)
                    entries.append(DatasetEntry(
                        question=question,
                        slice="image",
                        has_answer=True,
                        expected_answer=expected,
                        doc_id=chunk.doc_id,
                    ))
                    if len(entries) >= num_questions:
                        break
            except Exception as e:
                logger.warning(f"Failed to generate image question: {e}")
        
        return entries

    def _answer_in_content(self, expected: str, content: str, strict: bool = False) -> bool:
        """
        Looser validation: accept if expected answer appears verbatim OR
        if there is token overlap (>=2 tokens length>3).
        """
        if not expected or not content:
            return False

        exp = expected.strip().lower()
        if not exp:
            return False
        cont = content.lower()
        if exp in cont:
            return True

        if strict:
            return False

        # Token overlap heuristic
        exp_tokens = [t for t in re.split(r"\W+", exp) if len(t) > 3]
        if not exp_tokens:
            return False

        overlap = sum(1 for t in exp_tokens if t in cont)
        return overlap >= 2
    
    def generate_no_answer_slice(
        self,
        num_questions: int = 10,
    ) -> List[DatasetEntry]:
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
    
    def _generate_qa_pair(self, chunk: Chunk) -> Tuple[Optional[str], Optional[str]]:
        """Generate question-answer pair from chunk content."""
        # Prefer deterministic, anchored questions to avoid generic/unsatisfiable prompts
        anchored = self._anchor_text_span(chunk.content)
        if anchored:
            anchor, span = anchored
            question = f'In the text, what phrase includes "{anchor}"?'
            return question, span

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
    
    def _generate_table_question(self, chunk: Chunk) -> Tuple[Optional[str], Optional[str]]:
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

    
    def _generate_image_question(self, chunk: Chunk) -> Tuple[Optional[str], Optional[str]]:
        """Generate question about image content."""
        # Prefer deterministic, anchored questions to avoid generic/unsatisfiable prompts
        anchored = self._anchor_image_span(chunk.content)
        if anchored:
            anchor, span = anchored
            question = f'In the image, what phrase includes "{anchor}"?'
            return question, span

        prompt = f"""You are generating an evaluation question from image content.
CRITICAL RULES:
- The answer MUST be an exact short quote copied from the provided content.
- Do NOT ask about anything that is not explicitly present in the content.
- Prefer concrete entities, labels, axis names, or short statements that can be quoted verbatim.
- The question MUST explicitly mention that it is about an image (include the word "image").
- If the content is unclear or too generic, return empty JSON: {{}}.

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

    def _anchor_image_span(self, content: str) -> Optional[Tuple[str, str]]:
        """
        Build a specific, answerable QA pair by anchoring on a concrete token.
        Returns (anchor_token, answer_span) where answer_span is a verbatim substring.
        """
        if not content or not content.strip():
            return None

        # Prefer anchors with digits or uppercase (more distinctive for retrieval)
        pattern = re.compile(r"[A-Za-zА-Яа-я0-9%µμ]{4,}")
        matches = list(pattern.finditer(content))
        if not matches:
            return None

        def is_strong_anchor(s: str) -> bool:
            return any(ch.isdigit() for ch in s) or (len(s) >= 4 and any(ch.isupper() for ch in s))

        strong = [m for m in matches if is_strong_anchor(m.group(0))]
        if not strong:
            return None
        match = strong[0]
        anchor = match.group(0)

        # Expand to a nearby sentence/line boundary to keep a clean span
        start = max(content.rfind("\n", 0, match.start()), content.rfind(".", 0, match.start()))
        start = 0 if start < 0 else start + 1

        end_candidates = [content.find("\n", match.end()), content.find(".", match.end())]
        end_candidates = [e for e in end_candidates if e >= 0]
        end = min(end_candidates) if end_candidates else len(content)

        span = content[start:end].strip()

        # Fallback to a tight window if the boundary span is too long/short
        if len(span) < len(anchor) or len(span) > 240:
            window = 120
            s = max(0, match.start() - window // 2)
            e = min(len(content), match.end() + window // 2)
            span = content[s:e].strip()

        if not span or anchor.lower() not in span.lower():
            return None

        return anchor, span

    def _anchor_text_span(self, content: str) -> Optional[Tuple[str, str]]:
        """
        Deterministically pick an anchor token and nearby span for general text chunks.
        Keeps overall questions grounded in the actual chunk content.
        """
        if not content:
            return None

        text = content.strip()
        if len(text) < 80:
            return None
        lower = text.lower()
        if "unreadable" in lower:
            return None
        pipe_ratio = text.count("|") / max(1, len(text))
        if pipe_ratio > 0.01:
            return None

        tokens = re.findall(r"[A-Za-zА-Яа-я0-9%µμ]{4,}", text)
        if not tokens:
            return None

        stopwords = {
            "table", "таблица", "figure", "рисунок", "chart", "diagram",
            "this", "that", "with", "from", "were", "have", "has",
            "using", "used", "into", "between", "within", "their",
        }

        def score(tok: str) -> int:
            t = tok.strip()
            lower = t.lower()
            if lower in stopwords:
                return -1
            s = 0
            if any(ch.isdigit() for ch in t):
                s += 4
            if sum(1 for ch in t if ch.isupper()) >= 2:
                s += 2
            if len(t) >= 8:
                s += 1
            return s

        ranked = sorted(tokens, key=score, reverse=True)
        anchor = ranked[0]
        if score(anchor) < 1:
            return None

        idx = text.lower().find(anchor.lower())
        if idx == -1:
            return None

        left_nl = text.rfind("\n", 0, idx)
        left_dot = text.rfind(".", 0, idx)
        left = max(left_nl, left_dot)
        left = 0 if left == -1 else left + 1

        right_nl = text.find("\n", idx)
        right_dot = text.find(".", idx)
        candidates = [p for p in (right_nl, right_dot) if p != -1]
        right = min(candidates) if candidates else -1
        right = len(text) if right == -1 else right + 1

        span = text[left:right].strip()
        if len(span) < 40:
            window = 180
            start = max(0, idx - window // 3)
            end = min(len(text), idx + window)
            span = text[start:end].strip()

        if anchor.lower() not in span.lower():
            return None

        if len(span) > 160:
            span = span[:160].rstrip()

        return anchor, span
    
    def generate_full_dataset(
        self,
        chunks: List[Chunk],
        overall_count: int = 20,
        table_count: int = 10,
        image_count: int = 10,
        no_answer_count: int = 10,
        use_ragas: bool = False,
        min_ratio: float = 0.5,  # Warn if less than 50% of target achieved
        strict: bool = False,  # If True, raise error when slices below min_ratio
    ) -> List[DatasetEntry]:
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
        overall_entries = self.generate_overall(chunks, overall_count, use_ragas=use_ragas)
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


def compute_chunks_hash(chunks_file: str) -> str:
    """Compute MD5 hash of chunks.jsonl for validation."""
    import hashlib

    hasher = hashlib.md5()
    with open(chunks_file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_dataset(entries: List[DatasetEntry], output_path: str, chunks_file: str = "data/chunks.jsonl") -> None:
    """
    Save dataset to JSONL file with metadata for validation.

    Creates dataset.meta.json alongside dataset.jsonl with:
    - chunks_hash: MD5 of chunks.jsonl
    - chunks_count: Number of chunks
    - ingest_ts: Timestamp from first chunk
    """
    import hashlib
    from datetime import datetime

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save dataset entries
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry.to_jsonl() + "\n")

    logger.info(f"Saved {len(entries)} entries to {output_path}")

    # Compute chunks hash for validation (if available)
    if os.path.exists(chunks_file):
        chunks_hash = compute_chunks_hash(chunks_file)
    else:
        chunks_hash = None
        logger.warning(f"chunks.jsonl not found at {chunks_file}; skipping hash in metadata")

    # Get chunks count and timestamp
    chunks_count = 0
    ingest_ts = None
    if os.path.exists(chunks_file):
        with open(chunks_file, "r") as f:
            for i, line in enumerate(f):
                chunks_count += 1
                if i == 0 and line.strip():
                    # Get timestamp from first chunk
                    try:
                        chunk_data = json.loads(line)
                        ingest_ts = chunk_data.get("metadata", {}).get("ingest_ts")
                    except:
                        pass

    # Save metadata
    meta_path = output_path.replace(".jsonl", ".meta.json")
    meta = {
        "dataset_path": output_path,
        "chunks_file": chunks_file,
        "chunks_hash": chunks_hash,
        "chunks_count": chunks_count,
        "ingest_ts": ingest_ts,
        "dataset_count": len(entries),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if chunks_hash:
        logger.info(f"Saved metadata to {meta_path} (chunks_hash={chunks_hash[:8]}...)")
    else:
        logger.info(f"Saved metadata to {meta_path} (chunks_hash=none)")


def load_dataset(input_path: str) -> List[DatasetEntry]:
    """Load dataset from JSONL file."""
    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(DatasetEntry.from_jsonl(line))
    return entries
