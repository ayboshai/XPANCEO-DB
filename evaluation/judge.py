"""
LLM-as-Judge for evaluation.
Scores predictions on faithfulness, relevancy, context precision/recall.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from ingestion.models import JudgeResponse, JudgeScores, PredictionEntry
from .generate_dataset import DatasetEntry

logger = logging.getLogger(__name__)

# Judge prompt per specification
JUDGE_PROMPT = """You are an evaluation judge for a RAG (Retrieval Augmented Generation) system.

Given:
- Question: {question}
- Expected Answer: {expected_answer}
- Actual Answer: {answer}
- Retrieved Context: {context}

Score the following on a 0-1 scale (0 = completely wrong, 1 = perfect):

1. **Faithfulness** (0-1): Does the answer ONLY use facts from the provided context? 
   - 1.0 = All facts in answer are found in context
   - 0.5 = Some facts are from context, some might be added
   - 0.0 = Answer contains information not in context

2. **Relevancy** (0-1): Does the answer address the question?
   - 1.0 = Directly and completely answers the question
   - 0.5 = Partially answers or tangentially related
   - 0.0 = Does not answer the question at all

3. **Context Precision** (0-1): Are the retrieved chunks relevant to the question?
   - 1.0 = All retrieved chunks are highly relevant
   - 0.5 = Some chunks are relevant, some are not
   - 0.0 = Retrieved chunks are not relevant

4. **Context Recall** (0-1): Does the context contain all information needed to answer?
   - 1.0 = Context has all needed information
   - 0.5 = Context has some but not all needed information
   - 0.0 = Context lacks the needed information

{no_answer_instruction}

Respond ONLY with a JSON object in this exact format:
{{"faithfulness": X.X, "relevancy": X.X, "context_precision": X.X, "context_recall": X.X, "no_answer_correct": true/false/null, "notes": "brief explanation"}}"""

NO_ANSWER_INSTRUCTION = """
5. **No-Answer Correct** (true/false): This question should NOT be answerable from the documents.
   - true = System correctly refused to answer (said no answer available)
   - false = System incorrectly provided an answer
"""


class LLMJudge:
    """
    LLM-as-Judge for evaluating RAG predictions.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key
        self.model = model
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    def judge(
        self,
        prediction: PredictionEntry,
        expected: Optional[DatasetEntry] = None,
    ) -> JudgeResponse:
        """
        Judge a single prediction.
        
        Args:
            prediction: RAG output
            expected: Expected answer (optional)
            
        Returns:
            JudgeResponse with scores
        """
        # Build context string from retrieved chunks
        # Use full_content when available for accurate faithfulness/recall scoring
        context_parts = []
        for chunk in prediction.retrieved_chunks:
            # Prefer full_content over preview for accurate evaluation
            content = chunk.full_content if chunk.full_content else chunk.preview
            context_parts.append(f"[{chunk.doc_id}|p{chunk.page}|{chunk.type}]:\n{content}")
        context = "\n\n".join(context_parts) if context_parts else "(no context retrieved)"
        
        # Determine if this is a no-answer question
        is_no_answer = prediction.slice == "no-answer" or (expected and not expected.has_answer)
        no_answer_instruction = NO_ANSWER_INSTRUCTION if is_no_answer else ""
        
        # Build prompt
        prompt = JUDGE_PROMPT.format(
            question=prediction.question,
            expected_answer=expected.expected_answer if expected else "(not provided)",
            answer=prediction.answer,
            context=context,
            no_answer_instruction=no_answer_instruction,
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            
            result = response.choices[0].message.content.strip()
            scores = self._parse_scores(result, is_no_answer)
            
        except Exception as e:
            logger.error(f"Judge API call failed: {e}")
            scores = JudgeScores(
                faithfulness=0.0,
                relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
                no_answer_correct=None,
                notes=f"Error: {e}",
            )
        
        return JudgeResponse(
            question=prediction.question,
            answer=prediction.answer,
            expected_answer=expected.expected_answer if expected else None,
            judge=scores,
        )
    
    def _parse_scores(self, result: str, is_no_answer: bool) -> JudgeScores:
        """Parse JSON scores from LLM response."""
        try:
            # Handle markdown code blocks
            if "```" in result:
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
                result = result.strip()
            
            data = json.loads(result)
            
            return JudgeScores(
                faithfulness=float(data.get("faithfulness", 0)),
                relevancy=float(data.get("relevancy", 0)),
                context_precision=float(data.get("context_precision", 0)),
                context_recall=float(data.get("context_recall", 0)),
                no_answer_correct=data.get("no_answer_correct") if is_no_answer else None,
                notes=str(data.get("notes", "")),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse judge response: {result[:100]}... - {e}")
            return JudgeScores(
                faithfulness=0.0,
                relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
                no_answer_correct=None,
                notes=f"Parse error: {e}",
            )
    
    def judge_all(
        self,
        predictions: list[PredictionEntry],
        dataset: Optional[list[DatasetEntry]] = None,
        output_dir: Optional[str] = None,
        show_progress: bool = True,
    ) -> list[JudgeResponse]:
        """
        Judge all predictions.
        
        Args:
            predictions: List of RAG outputs
            dataset: Corresponding dataset entries
            output_dir: Directory to save results
            show_progress: Show progress bar
            
        Returns:
            List of judge responses
        """
        # Build lookup for expected answers
        expected_lookup = {}
        if dataset:
            for entry in dataset:
                expected_lookup[entry.question] = entry
        
        responses = []
        iterator = tqdm(predictions, desc="Judging") if show_progress else predictions
        
        for pred in iterator:
            expected = expected_lookup.get(pred.question)
            response = self.judge(pred, expected)
            responses.append(response)
        
        # Save if output_dir provided
        if output_dir:
            judge_path = os.path.join(output_dir, "judge_responses.jsonl")
            with open(judge_path, "w", encoding="utf-8") as f:
                for resp in responses:
                    f.write(resp.to_jsonl() + "\n")
            logger.info(f"Saved judge responses to {judge_path}")
        
        return responses
