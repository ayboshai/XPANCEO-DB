#!/usr/bin/env python3
"""
CLI script for running evaluation.
Usage: python scripts/eval.py --dataset evaluation/dataset.jsonl
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml


def load_config(config_path: str = "config/master_config.yaml") -> dict:
    """Load configuration from YAML file."""
    def resolve_env(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return os.getenv(value[2:-1], "")
        return value
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    for key, value in config.items():
        config[key] = resolve_env(value)
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation on XPANCEO DB RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dataset first
  python scripts/eval.py --generate-dataset
  
  # Run evaluation on dataset
  python scripts/eval.py --dataset evaluation/dataset.jsonl
  
  # Full pipeline: generate + evaluate
  python scripts/eval.py --generate-dataset --run-eval
        """,
    )
    parser.add_argument(
        "--dataset",
        help="Path to dataset.jsonl file",
    )
    parser.add_argument(
        "--generate-dataset",
        action="store_true",
        help="Generate new dataset from indexed chunks",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run evaluation after generating dataset",
    )
    parser.add_argument(
        "--out-dir",
        help="Output directory for results (default: evaluation/runs/<timestamp>)",
    )
    parser.add_argument(
        "--config",
        default="config/master_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--overall-count",
        type=int,
        default=20,
        help="Number of overall questions to generate",
    )
    parser.add_argument(
        "--table-count",
        type=int,
        default=10,
        help="Number of table questions to generate",
    )
    parser.add_argument(
        "--image-count",
        type=int,
        default=10,
        help="Number of image questions to generate",
    )
    parser.add_argument(
        "--no-answer-count",
        type=int,
        default=10,
        help="Number of no-answer questions to generate",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Load config
    config = load_config(args.config)
    api_key = config.get("openai_api_key")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)
    
    # Create output directory
    if args.out_dir:
        out_dir = args.out_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"evaluation/runs/{timestamp}"
    
    os.makedirs(out_dir, exist_ok=True)
    
    dataset_path = args.dataset
    
    # Generate dataset if requested
    if args.generate_dataset:
        print("\n📊 Generating evaluation dataset...")
        
        from ingestion.pipeline import IngestionPipeline
        from evaluation.generate_dataset import DatasetGenerator, save_dataset
        
        # Load chunks
        pipeline = IngestionPipeline(config)
        chunks = pipeline.load_chunks()
        
        if not chunks:
            print("Error: No chunks found. Run ingestion first.")
            sys.exit(1)
        
        print(f"   Loaded {len(chunks)} chunks")
        
        # Generate dataset
        generator = DatasetGenerator(api_key, config.get("model_chat", "gpt-4o-mini"))
        entries = generator.generate_full_dataset(
            chunks,
            overall_count=args.overall_count,
            table_count=args.table_count,
            image_count=args.image_count,
            no_answer_count=args.no_answer_count,
        )
        
        # Save dataset
        dataset_path = os.path.join(out_dir, "dataset.jsonl")
        save_dataset(entries, dataset_path)
        
        print(f"✅ Generated {len(entries)} questions")
        print(f"   Saved to: {dataset_path}")
    
    # Run evaluation
    if args.run_eval or args.dataset:
        if not dataset_path:
            print("Error: No dataset specified. Use --dataset or --generate-dataset")
            sys.exit(1)
        
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset not found: {dataset_path}")
            sys.exit(1)
        
        print(f"\n🔍 Running evaluation...")
        print(f"   Dataset: {dataset_path}")
        print(f"   Output: {out_dir}")
        
        from evaluation.eval_runner import run_evaluation
        from evaluation.generate_dataset import load_dataset
        from evaluation.judge import LLMJudge
        from evaluation.metrics import calculate_metrics, format_metrics_summary
        from evaluation.report import generate_reports
        from ingestion.models import JudgeResponse
        
        # Run RAG on dataset
        predictions = run_evaluation(dataset_path, out_dir, args.config)
        print(f"   ✅ Generated {len(predictions)} predictions")
        
        # Load dataset for expected answers
        dataset = load_dataset(dataset_path)
        
        # Build slice mapping
        slices_mapping = {e.question: e.slice for e in dataset}
        
        # Judge predictions
        print("\n⚖️ Running LLM judge...")
        judge = LLMJudge(api_key, config.get("model_chat", "gpt-4o-mini"))
        judge_responses = judge.judge_all(predictions, dataset, out_dir)
        print(f"   ✅ Judged {len(judge_responses)} predictions")
        
        # Calculate metrics
        print("\n📈 Calculating metrics...")
        metrics = calculate_metrics(judge_responses, slices_mapping)
        
        # Generate reports
        csv_path, md_path = generate_reports(
            metrics, judge_responses, out_dir, predictions, dataset
        )
        
        # Print summary
        print("\n" + format_metrics_summary(metrics))
        
        print(f"\n📁 Reports saved to:")
        print(f"   CSV:  {csv_path}")
        print(f"   MD:   {md_path}")
        print(f"   Dir:  {out_dir}")
    
    if not args.generate_dataset and not args.run_eval and not args.dataset:
        parser.print_help()


if __name__ == "__main__":
    main()
