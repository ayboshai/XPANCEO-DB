# XPANCEO DB: Multimodal RAG for Technical PDFs

> **Deterministic, modular RAG system** for answering questions about technical PDF documents containing text, tables, and images.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green)
![FAISS](https://img.shields.io/badge/Vector%20Index-FAISS-orange)

---

## 🎯 Features

- **📄 PDF Ingestion**: Parses text, tables, and images using Unstructured
- **🔍 OCR-First Strategy**: Tesseract OCR with GPT-4o-mini Vision fallback
- **🧠 RAG Pipeline**: Vector search (FAISS) + LLM generation with source citations
- **🚫 No-Answer Detection**: Explicitly refuses to answer when context insufficient
- **📊 Auto-Evaluation**: LLM-as-Judge with metrics per slice (overall/table/image/no-answer)
- **🖥️ Streamlit UI**: Modern dark theme chat interface

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install system dependencies
# Ubuntu/Debian
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus poppler-utils

# macOS
brew install tesseract poppler
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 4. Ingest PDFs

```bash
# Download sample PDFs
git clone https://github.com/xpanceo-team/x-trial /tmp/x-trial

# Ingest
python scripts/ingest.py /tmp/x-trial/pdf
```

### 5. Chat with Documents

```bash
# CLI mode
python scripts/chat.py

# Or single query
python scripts/chat.py --query "What is the main topic of the paper?"

# Or Streamlit UI
streamlit run ui/app.py
```

### 6. Run Evaluation

```bash
# Generate dataset + evaluate
python scripts/eval.py --generate-dataset --run-eval

# Check results
cat evaluation/runs/*/report.md
```

---

## 📁 Project Structure

```
xpanceo-db/
├── config/
│   └── master_config.yaml     # All tunables
├── data/
│   ├── chunks.jsonl           # Indexed chunks
│   ├── pdf_registry.jsonl     # Per-doc stats
│   └── cache/                 # OCR/Vision/Embedding caches
├── ingestion/
│   ├── models.py              # Pydantic schemas
│   ├── parser.py              # PDF parsing (Unstructured + PyPDF2 fallback)
│   ├── ocr.py                 # Tesseract OCR + heuristics
│   ├── captioner.py           # GPT-4o-mini Vision fallback
│   ├── chunker.py             # Text/table chunking
│   ├── embedder.py            # OpenAI embeddings
│   ├── index_faiss.py         # FAISS vector index
│   └── pipeline.py            # Orchestration
├── rag/
│   ├── retriever.py           # Vector search + optional BM25
│   ├── generator.py           # LLM answering with no-answer guard
│   └── pipeline.py            # End-to-end Q&A
├── ui/
│   └── app.py                 # Streamlit interface
├── evaluation/
│   ├── generate_dataset.py    # Create test questions
│   ├── eval_runner.py         # Run RAG on dataset
│   ├── judge.py               # LLM-as-Judge
│   ├── metrics.py             # Calculate metrics
│   └── report.py              # Generate reports
├── scripts/
│   ├── ingest.py              # CLI: ingest PDFs
│   ├── chat.py                # CLI: Q&A
│   ├── reindex.py             # CLI: rebuild index
│   ├── eval.py                # CLI: run evaluation
│   └── generate_dataset.py    # CLI: create dataset
├── tests/                     # Smoke tests
├── master_specification.md    # Authoritative spec
└── README.md
```

---

## 🎨 Design Decisions

### Why OCR-First + Vision Fallback?

- **Deterministic**: Tesseract OCR produces consistent results
- **Cost-efficient**: Vision API only used when OCR quality is low
- **Transparent**: Clear criteria for OCR failure (confidence < 60, chars < 50, tokens < 10, alpha_ratio < 0.4)

### Why FAISS?

- **Simplicity**: Local file-based index, no server required
- **Sufficient for MVP**: Up to 1M vectors on single machine
- **Zero dependencies**: No external services

> ℹ️ **Note**: Only FAISS is supported. Other backends (e.g., Qdrant) are not implemented.

### Why No Rerankers/Agents?

- **Predictable behavior**: Simple retrieval + generation
- **Debuggable**: Easy to trace what chunks were used
- **Faster iteration**: Less complexity = faster development

### Why Self-Implemented Judge?

- **Full control**: Prompts tuned for our specific use case
- **Consistent metrics**: Same scoring across all slices
- **No external dependencies**: Only OpenAI API

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Does answer only use facts from context? |
| **Relevancy** | Does answer address the question? |
| **Context Precision** | Are retrieved chunks relevant? |
| **Context Recall** | Does context have all needed info? |
| **No-Answer Accuracy** | Does system correctly refuse? |
| **False Positive Rate** | Does system wrongly answer no-answer questions? |

Metrics are calculated per slice: `overall`, `table`, `image`, `no-answer`

---

## 🔧 Configuration

All settings in `config/master_config.yaml`:

```yaml
# Models
model_chat: gpt-4o-mini
model_vision: gpt-4o-mini
model_embed: text-embedding-3-small

# Index
index_backend: faiss
top_k: 5

# OCR Thresholds
ocr_confidence_threshold: 60
ocr_min_chars: 50
ocr_min_tokens: 10
ocr_min_alpha_ratio: 0.4

# API Resilience
api_max_retries: 3
api_backoff_base: 2.0
```

---

## 🐳 Docker

```bash
# Build
docker build -t xpanceo-db .

# Run with API key
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -p 8501:8501 xpanceo-db
```

---

## 📝 Commands Reference

| Command | Description |
|---------|-------------|
| `python scripts/ingest.py <folder>` | Ingest PDFs from folder |
| `python scripts/chat.py` | Interactive Q&A |
| `python scripts/chat.py -q "..."` | Single query |
| `python scripts/reindex.py` | Rebuild index from chunks |
| `python scripts/generate_dataset.py` | Create eval dataset |
| `python scripts/eval.py --run-eval` | Run full evaluation |
| `streamlit run ui/app.py` | Launch web UI |

---

## 📄 License

MIT License

---

## 🔗 Links

- [Master Specification](./master_specification.md)
- [Test PDFs](https://github.com/xpanceo-team/x-trial)
