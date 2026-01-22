"""
Evaluation runner - executes RAG on dataset and collects predictions.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from ingestion.models import PredictionEntry
from rag.pipeline import RAGPipeline, create_rag_pipeline
from .generate_dataset import DatasetEntry, load_dataset

logger = logging.getLogger(__name__)


class EvalRunner:
    """
    Runs RAG pipeline on evaluation dataset and saves predictions.
    """
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
    
    def run(
        self,
        dataset: List[DatasetEntry],
        output_dir: str,
        show_progress: bool = True,
    ) -> List[PredictionEntry]:
        """
        Run evaluation on dataset.
        
        Args:
            dataset: List of evaluation entries
            output_dir: Directory to save predictions
            show_progress: Show progress bar
            
        Returns:
            List of prediction entries
        """
        os.makedirs(output_dir, exist_ok=True)
        
        predictions = []
        iterator = tqdm(dataset, desc="Running evaluation") if show_progress else dataset
        
        for entry in iterator:
            try:
                # Run RAG
                response = self.rag_pipeline.query(entry.question)
                
                # Convert to prediction entry
                prediction = response.to_prediction_entry(entry.slice)
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Failed to evaluate question: {entry.question[:50]}... - {e}")
                # Create failed prediction
                predictions.append(PredictionEntry(
                    question=entry.question,
                    answer=f"ERROR: {e}",
                    sources=[],
                    retrieved_chunks=[],
                    slice=entry.slice,
                    has_answer_pred=False,
                ))
        
        # Save predictions
        predictions_path = os.path.join(output_dir, "predictions.jsonl")
        with open(predictions_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(pred.to_jsonl() + "\n")
        
        logger.info(f"Saved {len(predictions)} predictions to {predictions_path}")
        
        # Also save dataset copy for reference
        dataset_copy_path = os.path.join(output_dir, "dataset.jsonl")
        with open(dataset_copy_path, "w", encoding="utf-8") as f:
            for entry in dataset:
                f.write(entry.to_jsonl() + "\n")
        
        return predictions


def run_evaluation(
    dataset_path: str,
    output_dir: Optional[str] = None,
    config_path: str = "config/master_config.yaml",
) -> List[PredictionEntry]:
    """
    Run full evaluation pipeline.
    
    Args:
        dataset_path: Path to dataset.jsonl
        output_dir: Output directory (default: evaluation/runs/<timestamp>)
        config_path: Path to config file
        
    Returns:
        List of predictions
    """
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"evaluation/runs/{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)
    logger.info(f"Loaded {len(dataset)} questions")
    
    # Create RAG pipeline
    logger.info("Loading RAG pipeline...")
    rag_pipeline = create_rag_pipeline(config_path)
    
    # Run evaluation
    runner = EvalRunner(rag_pipeline)
    predictions = runner.run(dataset, output_dir)
    
    return predictions
