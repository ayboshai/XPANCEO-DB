#!/usr/bin/env python3
"""
CLI script for reindexing from chunks.jsonl.
Usage: python scripts/reindex.py
"""

import argparse
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from tqdm import tqdm

from ingestion.embedder import create_embedder
from ingestion.index_faiss import create_index
from ingestion.pipeline import IngestionPipeline


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
        description="Reindex chunks from chunks.jsonl (skip PDF parsing)",
    )
    parser.add_argument(
        "--config",
        default="config/master_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing index before reindexing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for embedding",
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
    
    # Check API key
    if not config.get("openai_api_key"):
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)
    
    # Load chunks
    print("📄 Loading chunks from chunks.jsonl...")
    pipeline = IngestionPipeline(config)
    chunks = pipeline.load_chunks()
    
    if not chunks:
        print("❌ No chunks found in chunks.jsonl")
        sys.exit(1)
    
    print(f"   Found {len(chunks)} chunks")
    
    # Create embedder and index
    embedder = create_embedder(config)
    index = create_index(config)
    
    # Clear index if requested
    if args.clear:
        print("🗑️  Clearing existing index...")
        index.clear()
    
    # Process in batches
    print(f"🔄 Reindexing with batch size {args.batch_size}...")
    
    batch_size = args.batch_size
    for i in tqdm(range(0, len(chunks), batch_size), desc="Batches"):
        batch = chunks[i:i + batch_size]
        
        # Filter non-empty content
        valid_chunks = [c for c in batch if c.content.strip()]
        if not valid_chunks:
            continue
        
        # Embed
        texts = [c.content for c in valid_chunks]
        ids = [c.chunk_id for c in valid_chunks]
        metadata = [
            {
                "doc_id": c.doc_id,
                "page": c.page,
                "type": c.type,
                "content_preview": c.content[:100],
            }
            for c in valid_chunks
        ]
        
        try:
            vectors = embedder.embed(texts)
            index.upsert(ids, vectors, metadata)
        except Exception as e:
            logging.error(f"Failed to index batch {i}: {e}")
    
    print(f"\n✅ Reindexed {index.count} vectors")
    print(f"📁 Index saved to: {config.get('index_dir', 'data/index')}/")


if __name__ == "__main__":
    main()
