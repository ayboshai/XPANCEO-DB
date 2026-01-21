"""
Smoke test for evaluation pipeline.
Verifies metrics calculation and report generation.
"""

import os
import sys
import tempfile

import pytest

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_metrics_calculation():
    """Test metrics calculation from judge responses."""
    from ingestion.models import JudgeResponse, JudgeScores
    from evaluation.metrics import calculate_metrics
    
    responses = [
        JudgeResponse(
            question="Q1",
            answer="A1",
            expected_answer="A1",
            judge=JudgeScores(
                faithfulness=0.9,
                relevancy=0.8,
                context_precision=0.7,
                context_recall=0.6,
            ),
        ),
        JudgeResponse(
            question="Q2",
            answer="A2",
            expected_answer="A2",
            judge=JudgeScores(
                faithfulness=0.7,
                relevancy=0.9,
                context_precision=0.8,
                context_recall=0.7,
            ),
        ),
    ]
    
    metrics = calculate_metrics(responses)
    
    assert metrics.overall.count == 2
    assert metrics.overall.faithfulness == pytest.approx(0.8, abs=0.01)
    assert metrics.overall.relevancy == pytest.approx(0.85, abs=0.01)


def test_no_answer_metrics():
    """Test no-answer specific metrics."""
    from ingestion.models import JudgeResponse, JudgeScores
    from evaluation.metrics import calculate_metrics
    
    responses = [
        JudgeResponse(
            question="Q1",
            answer="No answer",
            expected_answer="No answer",
            judge=JudgeScores(
                faithfulness=1.0,
                relevancy=1.0,
                context_precision=0.5,
                context_recall=0.5,
                no_answer_correct=True,
            ),
        ),
        JudgeResponse(
            question="Q2",
            answer="Wrong answer given",
            expected_answer="Should refuse",
            judge=JudgeScores(
                faithfulness=0.3,
                relevancy=0.5,
                context_precision=0.5,
                context_recall=0.5,
                no_answer_correct=False,
            ),
        ),
    ]
    
    # Map questions to no-answer slice
    slices_mapping = {"Q1": "no-answer", "Q2": "no-answer"}
    
    metrics = calculate_metrics(responses, slices_mapping)
    
    assert "no-answer" in metrics.by_slice
    na_metrics = metrics.by_slice["no-answer"]
    assert na_metrics.no_answer_accuracy == 0.5
    assert na_metrics.false_positive_rate == 0.5


def test_csv_report_generation():
    """Test CSV report generation."""
    from evaluation.metrics import EvalMetrics, SliceMetrics
    from evaluation.report import generate_csv_report
    
    metrics = EvalMetrics()
    metrics.overall = SliceMetrics(
        slice_name="overall",
        count=10,
        faithfulness=0.8,
        relevancy=0.7,
        context_precision=0.6,
        context_recall=0.5,
    )
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = f.name
    
    try:
        generate_csv_report(metrics, csv_path)
        
        # Verify file exists and has content
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            content = f.read()
            assert "overall" in content
            assert "0.8" in content
    finally:
        os.unlink(csv_path)


def test_dataset_entry_serialization():
    """Test dataset entry JSONL serialization."""
    from ingestion.models import DatasetEntry
    
    entry = DatasetEntry(
        question="What is X?",
        slice="overall",
        has_answer=True,
        expected_answer="X is Y",
        doc_id="doc1",
    )
    
    jsonl = entry.to_jsonl()
    assert "What is X?" in jsonl
    
    # Deserialize
    entry2 = DatasetEntry.from_jsonl(jsonl)
    assert entry2.question == entry.question
    assert entry2.slice == entry.slice


def test_prediction_entry_serialization():
    """Test prediction entry JSONL serialization."""
    from ingestion.models import PredictionEntry, SourceRef
    
    entry = PredictionEntry(
        question="Q",
        answer="A",
        sources=[
            SourceRef(
                doc_id="d1",
                page=1,
                chunk_id="c1",
                type="text",
                score=0.9,
                preview="...",
            )
        ],
        retrieved_chunks=[],
        slice="overall",
        has_answer_pred=True,
    )
    
    jsonl = entry.to_jsonl()
    
    entry2 = PredictionEntry.from_jsonl(jsonl)
    assert entry2.question == "Q"
    assert len(entry2.sources) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
