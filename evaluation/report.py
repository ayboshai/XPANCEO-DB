"""
Report generator - creates CSV and Markdown reports from evaluation results.
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from typing import Optional

from ingestion.models import JudgeResponse, PredictionEntry
from .generate_dataset import DatasetEntry
from .metrics import EvalMetrics, SliceMetrics

logger = logging.getLogger(__name__)


def generate_csv_report(
    metrics: EvalMetrics,
    output_path: str,
) -> None:
    """Generate CSV report with metrics by slice."""
    rows = [metrics.overall.to_dict()]
    
    for slice_name, slice_metrics in metrics.by_slice.items():
        if slice_metrics.count > 0:
            rows.append(slice_metrics.to_dict())
    
    # Determine all columns
    columns = ["slice", "count", "faithfulness", "relevancy", 
               "context_precision", "context_recall", 
               "no_answer_accuracy", "false_positive_rate"]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"Saved CSV report to {output_path}")


def generate_markdown_report(
    metrics: EvalMetrics,
    judge_responses: list[JudgeResponse],
    predictions: Optional[list[PredictionEntry]] = None,
    dataset: Optional[list[DatasetEntry]] = None,
    output_path: str = "report.md",
) -> None:
    """Generate detailed Markdown report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lines = [
        "# XPANCEO DB Evaluation Report",
        "",
        f"**Generated**: {now}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"- **Total Questions**: {metrics.total_questions}",
        f"- **Successful Judgments**: {metrics.successful_judgments}",
        f"- **Failed Judgments**: {metrics.failed_judgments}",
        "",
        "---",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Score |",
        "|--------|-------|",
        f"| Faithfulness | {metrics.overall.faithfulness:.3f} |",
        f"| Relevancy | {metrics.overall.relevancy:.3f} |",
        f"| Context Precision | {metrics.overall.context_precision:.3f} |",
        f"| Context Recall | {metrics.overall.context_recall:.3f} |",
        "",
        "---",
        "",
        "## Metrics by Slice",
        "",
    ]
    
    # Slice tables
    for slice_name, slice_metrics in metrics.by_slice.items():
        if slice_metrics.count == 0:
            continue
        
        lines.extend([
            f"### {slice_name.title()} (n={slice_metrics.count})",
            "",
            "| Metric | Score |",
            "|--------|-------|",
            f"| Faithfulness | {slice_metrics.faithfulness:.3f} |",
            f"| Relevancy | {slice_metrics.relevancy:.3f} |",
            f"| Context Precision | {slice_metrics.context_precision:.3f} |",
            f"| Context Recall | {slice_metrics.context_recall:.3f} |",
        ])
        
        if slice_metrics.no_answer_accuracy is not None:
            lines.append(f"| No-Answer Accuracy | {slice_metrics.no_answer_accuracy:.3f} |")
        if slice_metrics.false_positive_rate is not None:
            lines.append(f"| False Positive Rate | {slice_metrics.false_positive_rate:.3f} |")
        
        lines.extend(["", ""])
    
    # Add failure examples
    lines.extend([
        "---",
        "",
        "## Notable Cases",
        "",
    ])
    
    # Low faithfulness examples
    low_faith = [r for r in judge_responses if r.judge.faithfulness < 0.5]
    if low_faith:
        lines.extend([
            "### Low Faithfulness Examples",
            "",
        ])
        for resp in low_faith[:3]:
            lines.extend([
                f"**Q**: {resp.question[:100]}...",
                "",
                f"**A**: {resp.answer[:200]}...",
                "",
                f"**Score**: {resp.judge.faithfulness:.2f}",
                "",
                f"**Notes**: {resp.judge.notes}",
                "",
                "---",
                "",
            ])
    
    # No-answer failures
    no_answer_fails = [
        r for r in judge_responses 
        if r.judge.no_answer_correct is False
    ]
    if no_answer_fails:
        lines.extend([
            "### No-Answer False Positives",
            "",
        ])
        for resp in no_answer_fails[:3]:
            lines.extend([
                f"**Q**: {resp.question[:100]}...",
                "",
                f"**A**: {resp.answer[:200]}...",
                "",
                "*(System should have refused to answer)*",
                "",
                "---",
                "",
            ])
    
    # Write file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Saved Markdown report to {output_path}")


def generate_reports(
    metrics: EvalMetrics,
    judge_responses: list[JudgeResponse],
    output_dir: str,
    predictions: Optional[list[PredictionEntry]] = None,
    dataset: Optional[list[DatasetEntry]] = None,
) -> tuple[str, str]:
    """
    Generate both CSV and Markdown reports.
    
    Returns:
        Tuple of (csv_path, md_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "report.csv")
    md_path = os.path.join(output_dir, "report.md")
    
    generate_csv_report(metrics, csv_path)
    generate_markdown_report(
        metrics, judge_responses, predictions, dataset, md_path
    )
    
    return csv_path, md_path
