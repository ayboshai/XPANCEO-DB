#!/usr/bin/env python3
"""
CLI script for PDF ingestion.
Usage: python scripts/ingest.py <pdf_folder>
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
from tqdm import tqdm

from ingestion.pipeline import IngestionPipeline


def load_config(config_path: str = "config/master_config.yaml") -> dict:
    """
    Load configuration from YAML file.

    IMPORTANT: .env file takes priority over system environment variables
    to prevent accidental use of wrong API keys.
    """
    # Load .env file with override=True to prioritize local config
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
        logging.info(f"Loaded environment from: {env_path}")

    def resolve_env(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var, "")
        return value

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        config[key] = resolve_env(value)

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into XPANCEO DATABASE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest.py ./pdfs
  python scripts/ingest.py /path/to/documents --config custom_config.yaml
        """,
    )
    parser.add_argument(
        "pdf_folder",
        help="Path to folder containing PDF files",
    )
    parser.add_argument(
        "--config",
        default="config/master_config.yaml",
        help="Path to configuration file (default: config/master_config.yaml)",
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

    # CRITICAL: Silence noisy HTTP loggers (even in verbose mode)
    # These write base64-encoded images and bloat logs to gigabytes
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Validate folder
    if not os.path.isdir(args.pdf_folder):
        print(f"Error: {args.pdf_folder} is not a directory")
        sys.exit(1)
    
    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Check API key
    if not config.get("openai_api_key"):
        print("Error: OPENAI_API_KEY not set. Set it in environment or config.")
        sys.exit(1)
    
    # Create pipeline
    pipeline = IngestionPipeline(config)
    
    # Progress bar callback
    pbar = None
    
    def progress_callback(filename: str, current: int, total: int):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total, desc="Ingesting PDFs", unit="file")
        pbar.set_postfix_str(filename)
        pbar.update(1)
    
    # Run ingestion
    print(f"\n📄 Ingesting PDFs from: {args.pdf_folder}")
    print(f"📁 Output: {config.get('data_dir', 'data')}/")
    print()
    
    try:
        results = pipeline.ingest_folder(args.pdf_folder, progress_callback)
        
        if pbar:
            pbar.close()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Ingestion Summary")
        print("=" * 50)
        
        total_chunks = 0
        for entry in results:
            chunks = entry.chunks
            doc_total = chunks.text + chunks.table + chunks.image_ocr + chunks.image_caption
            total_chunks += doc_total
            
            print(f"\n📄 {entry.filename}")
            print(f"   Pages: {entry.pages}")
            print(f"   Chunks: {doc_total} (text={chunks.text}, table={chunks.table}, "
                  f"ocr={chunks.image_ocr}, caption={chunks.image_caption})")
            if entry.ocr_failure_rate > 0:
                print(f"   OCR failure rate: {entry.ocr_failure_rate:.1%}")
            if entry.vision_fallback_rate > 0:
                print(f"   Vision fallback rate: {entry.vision_fallback_rate:.1%}")
            if entry.errors > 0:
                print(f"   ⚠️  Errors: {entry.errors}")
        
        print("\n" + "-" * 50)
        print(f"✅ Total: {len(results)} documents, {total_chunks} chunks indexed")
        print(f"📁 Chunks saved to: {pipeline.chunks_file}")
        print(f"📁 Index saved to: {config.get('index_dir', 'data/index')}/")
        
    except Exception as e:
        if pbar:
            pbar.close()
        print(f"\n❌ Error: {e}")
        logging.exception("Ingestion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
