#!/usr/bin/env python3
"""
CLI script for generating evaluation dataset.
Usage: python scripts/generate_dataset.py
"""

import argparse
import logging
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import yaml
from dotenv import load_dotenv


def load_config(config_path: str = "config/master_config.yaml") -> dict:
    """Load configuration from YAML file. .env takes priority over system env."""
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)

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
        description="Generate evaluation dataset from indexed chunks",
    )
    parser.add_argument(
        "--config",
        default="config/master_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--out",
        default="evaluation/dataset.jsonl",
        help="Output path for dataset (default: evaluation/dataset.jsonl)",
    )
    parser.add_argument(
        "--overall-count",
        type=int,
        default=20,
        help="Number of overall questions",
    )
    parser.add_argument(
        "--table-count",
        type=int,
        default=10,
        help="Number of table questions",
    )
    parser.add_argument(
        "--image-count",
        type=int,
        default=10,
        help="Number of image questions",
    )
    parser.add_argument(
        "--no-answer-count",
        type=int,
        default=10,
        help="Number of no-answer questions",
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
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # Load config
    config = load_config(args.config)
    api_key = config.get("openai_api_key")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)
    
    # Import modules
    from ingestion.pipeline import IngestionPipeline
    from evaluation.generate_dataset import DatasetGenerator, save_dataset
    
    # Load chunks
    print("📄 Loading indexed chunks...")
    pipeline = IngestionPipeline(config)
    chunks = pipeline.load_chunks()
    
    if not chunks:
        print("❌ No chunks found. Run ingestion first.")
        sys.exit(1)
    
    print(f"   Found {len(chunks)} chunks")
    
    # Generate dataset
    print("\n📊 Generating evaluation dataset...")
    generator = DatasetGenerator(api_key, config.get("model_chat", "gpt-4o-mini"))
    
    entries = generator.generate_full_dataset(
        chunks,
        overall_count=args.overall_count,
        table_count=args.table_count,
        image_count=args.image_count,
        no_answer_count=args.no_answer_count,
        use_ragas=False,
    )
    
    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_dataset(entries, args.out, chunks_file=pipeline.chunks_file)
    
    # Summary
    slice_counts = {}
    for e in entries:
        slice_counts[e.slice] = slice_counts.get(e.slice, 0) + 1
    
    print(f"\n✅ Generated {len(entries)} questions:")
    for slice_name, count in slice_counts.items():
        print(f"   {slice_name}: {count}")
    
    print(f"\n📁 Saved to: {args.out}")


if __name__ == "__main__":
    main()
