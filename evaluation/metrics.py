"""
Metrics calculation from judge responses.
Aggregates scores by slice and overall.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from ingestion.models import JudgeResponse

logger = logging.getLogger(__name__)


@dataclass
class SliceMetrics:
    """Metrics for a single slice."""
    
    slice_name: str
    count: int = 0
    faithfulness: float = 0.0
    relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    no_answer_accuracy: Optional[float] = None
    false_positive_rate: Optional[float] = None
    
    def to_dict(self) -> dict:
        d = {
            "slice": self.slice_name,
            "count": self.count,
            "faithfulness": round(self.faithfulness, 3),
            "relevancy": round(self.relevancy, 3),
            "context_precision": round(self.context_precision, 3),
            "context_recall": round(self.context_recall, 3),
        }
        if self.no_answer_accuracy is not None:
            d["no_answer_accuracy"] = round(self.no_answer_accuracy, 3)
        if self.false_positive_rate is not None:
            d["false_positive_rate"] = round(self.false_positive_rate, 3)
        return d


@dataclass
class EvalMetrics:
    """Complete evaluation metrics."""
    
    overall: SliceMetrics = field(default_factory=lambda: SliceMetrics("overall"))
    by_slice: dict[str, SliceMetrics] = field(default_factory=dict)
    
    # Counts
    total_questions: int = 0
    successful_judgments: int = 0
    failed_judgments: int = 0


def calculate_metrics(
    judge_responses: list[JudgeResponse],
    predictions_slices: Optional[dict[str, str]] = None,
) -> EvalMetrics:
    """
    Calculate metrics from judge responses.
    
    Args:
        judge_responses: List of judge responses
        predictions_slices: Optional mapping of question -> slice name
        
    Returns:
        EvalMetrics with overall and per-slice metrics
    """
    metrics = EvalMetrics()
    metrics.total_questions = len(judge_responses)
    
    # Group by slice
    slice_groups: dict[str, list[JudgeResponse]] = {
        "overall": [],
        "table": [],
        "image": [],
        "no-answer": [],
    }
    
    for resp in judge_responses:
        # Determine slice from predictions_slices or default to overall
        slice_name = "overall"
        if predictions_slices and resp.question in predictions_slices:
            slice_name = predictions_slices[resp.question]
        
        if slice_name in slice_groups:
            slice_groups[slice_name].append(resp)
        else:
            slice_groups["overall"].append(resp)
    
    # Calculate metrics per slice
    all_responses = []
    for slice_name, responses in slice_groups.items():
        if not responses:
            continue
        
        all_responses.extend(responses)
        
        slice_metrics = _calculate_slice_metrics(slice_name, responses)
        metrics.by_slice[slice_name] = slice_metrics
    
    # Calculate overall metrics (across all slices)
    if all_responses:
        metrics.overall = _calculate_slice_metrics("overall", all_responses)
        metrics.successful_judgments = sum(
            1 for r in all_responses if "error" not in r.judge.notes.lower()
        )
        metrics.failed_judgments = metrics.total_questions - metrics.successful_judgments
    
    return metrics


def _calculate_slice_metrics(slice_name: str, responses: list[JudgeResponse]) -> SliceMetrics:
    """Calculate metrics for a single slice."""
    n = len(responses)
    if n == 0:
        return SliceMetrics(slice_name)
    
    # Aggregate scores
    faithfulness = sum(r.judge.faithfulness for r in responses) / n
    relevancy = sum(r.judge.relevancy for r in responses) / n
    context_precision = sum(r.judge.context_precision for r in responses) / n
    context_recall = sum(r.judge.context_recall for r in responses) / n
    
    metrics = SliceMetrics(
        slice_name=slice_name,
        count=n,
        faithfulness=faithfulness,
        relevancy=relevancy,
        context_precision=context_precision,
        context_recall=context_recall,
    )
    
    # No-answer specific metrics
    if slice_name == "no-answer":
        # Count correct refusals
        correct = sum(
            1 for r in responses
            if r.judge.no_answer_correct is True
        )
        incorrect = sum(
            1 for r in responses
            if r.judge.no_answer_correct is False
        )
        
        if correct + incorrect > 0:
            metrics.no_answer_accuracy = correct / (correct + incorrect)
            metrics.false_positive_rate = incorrect / (correct + incorrect)
    
    return metrics


def format_metrics_summary(metrics: EvalMetrics) -> str:
    """Format metrics as human-readable summary."""
    lines = [
        "=" * 60,
        "EVALUATION METRICS SUMMARY",
        "=" * 60,
        "",
        f"Total questions: {metrics.total_questions}",
        f"Successful judgments: {metrics.successful_judgments}",
        f"Failed judgments: {metrics.failed_judgments}",
        "",
        "-" * 60,
        "OVERALL METRICS",
        "-" * 60,
        f"  Faithfulness:       {metrics.overall.faithfulness:.3f}",
        f"  Relevancy:          {metrics.overall.relevancy:.3f}",
        f"  Context Precision:  {metrics.overall.context_precision:.3f}",
        f"  Context Recall:     {metrics.overall.context_recall:.3f}",
        "",
    ]
    
    # Per-slice metrics
    for slice_name, slice_metrics in metrics.by_slice.items():
        if slice_metrics.count == 0:
            continue
        
        lines.extend([
            "-" * 60,
            f"SLICE: {slice_name.upper()} (n={slice_metrics.count})",
            "-" * 60,
            f"  Faithfulness:       {slice_metrics.faithfulness:.3f}",
            f"  Relevancy:          {slice_metrics.relevancy:.3f}",
            f"  Context Precision:  {slice_metrics.context_precision:.3f}",
            f"  Context Recall:     {slice_metrics.context_recall:.3f}",
        ])
        
        if slice_metrics.no_answer_accuracy is not None:
            lines.append(f"  No-Answer Accuracy: {slice_metrics.no_answer_accuracy:.3f}")
        if slice_metrics.false_positive_rate is not None:
            lines.append(f"  False Positive Rate: {slice_metrics.false_positive_rate:.3f}")
        
        lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)
