#!/usr/bin/env python3
"""
CLI script for chatting with indexed documents.
Usage: python scripts/chat.py --query "What is..." or python scripts/chat.py for interactive mode
"""

import argparse
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline import create_rag_pipeline, load_config


def format_response(response):
    """Format RAG response for display."""
    output = []
    
    # Status
    if response.has_answer:
        output.append("✅ Answer found")
    else:
        output.append("❌ No answer in documents")
    
    output.append("")
    output.append("📝 Answer:")
    output.append("-" * 40)
    output.append(response.answer)
    output.append("")
    
    # Sources
    if response.sources:
        output.append("📚 Sources:")
        output.append("-" * 40)
        for i, src in enumerate(response.sources, 1):
            output.append(f"{i}. [{src.doc_id}|p{src.page}|{src.chunk_id}] ({src.type})")
            output.append(f"   Score: {src.score:.3f}")
            output.append(f"   Preview: {src.preview[:80]}...")
            output.append("")
    
    return "\n".join(output)


def interactive_mode(rag_pipeline):
    """Run interactive chat session."""
    print("\n" + "=" * 50)
    print("🤖 XPANCEO DATABASE Chat")
    print("=" * 50)
    print("Type your questions. Enter 'quit' or 'exit' to end.")
    print()
    
    while True:
        try:
            query = input("❓ You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ("quit", "exit", "q"):
                print("\nGoodbye! 👋")
                break
            
            # Get response
            response = rag_pipeline.query(query)
            print()
            print(format_response(response))
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logging.exception("Query failed")


def main():
    parser = argparse.ArgumentParser(
        description="Chat with indexed PDF documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/chat.py                          # Interactive mode
  python scripts/chat.py --query "What is X?"     # Single query
  python scripts/chat.py -q "Explain Y" -k 10     # With top-k override
        """,
    )
    parser.add_argument(
        "--query", "-q",
        help="Single query to answer (skips interactive mode)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve (overrides config)",
    )
    parser.add_argument(
        "--config",
        default="config/master_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # Load config and create pipeline
    try:
        print("🔄 Loading RAG pipeline...")
        rag_pipeline = create_rag_pipeline(args.config)
        
        # Check if chunks exist
        if not rag_pipeline.retriever.chunks_lookup:
            print("⚠️  No documents indexed. Run 'python scripts/ingest.py <pdf_folder>' first.")
            sys.exit(1)
        
        print(f"✅ Loaded {len(rag_pipeline.retriever.chunks_lookup)} chunks")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to load pipeline: {e}")
        logging.exception("Pipeline initialization failed")
        sys.exit(1)
    
    # Single query or interactive mode
    if args.query:
        response = rag_pipeline.query(args.query, args.top_k)
        print()
        print(format_response(response))
    else:
        interactive_mode(rag_pipeline)


if __name__ == "__main__":
    main()
