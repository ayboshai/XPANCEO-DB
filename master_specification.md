# XPANCEO DB — Master Specification

> **Authoritative Source of Truth** for XPANCEO DB: Multimodal RAG for PDF documents.
> All code must conform to this specification. Updates require analyst approval.
> 
> **Version**: 1.1.0  
> **Last Updated**: 2024-01-21  
> **Status**: Implementation Complete

---

## 0. Developer Handoff Notes

> [!IMPORTANT]
> **For Continuity**: If primary developer unavailable, follow these guidelines:

### Quick Orientation
1. **Start here**: Read this `master_specification.md` — it's the single source of truth
2. **Run smoke tests**: `pytest tests/ -v` — verify everything works
3. **Check example outputs**: `evaluation/runs/example/report.csv` and `report.md`
4. **Config location**: All settings in `config/master_config.yaml`

### File/Script Naming Convention
| Pattern | Purpose |
|---------|---------|
| `ingestion/*.py` | PDF → Chunks → Index |
| `rag/*.py` | Retrieval + Generation |
| `evaluation/*.py` | Dataset + Judge + Metrics |
| `scripts/*.py` | CLI entry points |
| `ui/app.py` | Streamlit interface |

### Key Entry Points
```bash
python scripts/ingest.py <folder>      # Ingest PDFs
python scripts/chat.py                  # Interactive Q&A
python scripts/eval.py --run-eval       # Full evaluation
streamlit run ui/app.py                 # Web UI
```

### Debugging Tips
- Check `data/error_log.jsonl` for element-level failures
- Check `data/pdf_registry.jsonl` for per-doc stats
- Enable verbose: `python scripts/ingest.py folder -v`

---

## 1. Scope & Principles

**Goal**: Deterministic, modular RAG system for technical PDFs (text/tables/images) with automatic evaluation.

**Core Principles**:
- **No agents, no rerankers** — simple retrieval + generation
- **OpenAI + local index only** — FAISS (default) or Qdrant
- **Deterministic OCR-first** — Vision API only as fallback
- **Explicit "no answer"** — never speculate beyond context
- **Single config source** — `master_config.yaml`
- **Fail-safe** — element errors don't crash pipeline

---

## 2. Configuration

**File**: `config/master_config.yaml`

### 2.1 Complete Configuration Reference

```yaml
# =============================================================================
# API Keys (from environment variables)
# =============================================================================
openai_api_key: ${OPENAI_API_KEY}

# =============================================================================
# Models
# =============================================================================
model_chat: gpt-4o-mini          # LLM for generation
model_vision: gpt-4o-mini        # Vision API for image captioning
model_embed: text-embedding-3-small  # Embedding model (dim=1536)

# =============================================================================
# Index Backend
# =============================================================================
index_backend: faiss             # faiss | qdrant
top_k: 5                         # Number of chunks to retrieve

# Hybrid retrieval (optional)
hybrid_enabled: false            # Enable BM25 union with dense search

# =============================================================================
# Chunking
# =============================================================================
chunk_size_tokens: 512           # Max tokens per chunk
chunk_overlap_tokens: 50         # Overlap between chunks

# =============================================================================
# OCR Thresholds (ANY condition triggers Vision fallback)
# =============================================================================
ocr_confidence_threshold: 60     # Tesseract confidence 0-100
ocr_min_chars: 50                # Minimum character count
ocr_min_tokens: 10               # Minimum word count
ocr_min_alpha_ratio: 0.4         # Alphanumeric ratio

# =============================================================================
# API Resilience
# =============================================================================
api_max_retries: 3               # Retry count for API failures
api_backoff_base: 2.0            # Exponential backoff base (seconds)
api_timeout: 30                  # Request timeout (seconds)
api_rate_limit_rpm: 500          # Rate limit (requests per minute)

# =============================================================================
# Paths
# =============================================================================
data_dir: data                   # Root data directory
cache_dir: data/cache            # Cache root
index_dir: data/index            # Vector index storage
reports_dir: evaluation/runs     # Evaluation outputs

# Cache subdirectories (auto-created)
ocr_cache_dir: data/cache/ocr           # OCR results by image_hash
vision_cache_dir: data/cache/vision     # Vision captions by image_hash
embedding_cache_dir: data/cache/embeddings  # Embeddings by chunk_id

# =============================================================================
# Parallelism
# =============================================================================
max_concurrent_embeddings: 5     # Parallel embedding requests
max_concurrent_vision: 2         # Parallel vision requests

# =============================================================================
# Document ID Policy
# =============================================================================
doc_id_strategy: hash            # hash | filename | uuid
# hash = md5(filepath + mtime) — deterministic, changes on file update
# filename = file stem — simple, may collide
# uuid = random uuid — always unique

# Re-upload behavior
reupload_policy: new_version     # new_version | overwrite
# new_version = generate new doc_id, keep old chunks
# overwrite = delete old chunks, replace with new

# =============================================================================
# Logging
# =============================================================================
log_level: INFO                  # DEBUG | INFO | WARNING | ERROR
log_format: json                 # json | text
```

**Rule**: No hardcoded values in code. All tunables loaded from config via `load_config()`.

---

## 3. Data Schemas (Pydantic)

> All schemas defined in `ingestion/models.py`. Validation on load with skip+log for invalid records.

### 3.1 Chunk (core data unit)

```python
class ChunkMetadata(BaseModel):
    source_path: Optional[str] = None      # Original PDF path
    bbox: Optional[list[float]] = None     # Bounding box [x1,y1,x2,y2]
    image_hash: Optional[str] = None       # MD5 of image bytes (for caching)
    ocr_confidence: Optional[float] = None # Tesseract mean confidence
    ocr_failed: bool = False               # OCR quality check failed
    ocr_failed_reason: Optional[str] = None # Why OCR failed
    vision_used: bool = False              # Was Vision API called
    prev_chunk_id: Optional[str] = None    # Link to previous chunk
    next_chunk_id: Optional[str] = None    # Link to next chunk
    ingest_ts: str = ""                    # ISO timestamp

class Chunk(BaseModel):
    doc_id: str                            # Document identifier
    page: int                              # Page number (1-indexed)
    chunk_id: str                          # Unique chunk ID
    type: Literal["text", "table", "image_ocr", "image_caption"]
    content: str                           # Text content
    metadata: ChunkMetadata
```

### 3.2 Storage Files

| File | Format | Purpose | Location |
|------|--------|---------|----------|
| `chunks.jsonl` | JSONL | All indexed chunks | `data/` |
| `pdf_registry.jsonl` | JSONL | Per-doc statistics | `data/` |
| `error_log.jsonl` | JSONL | Element-level failures | `data/` |
| `dataset.jsonl` | JSONL | Evaluation questions | `evaluation/` |
| `predictions.jsonl` | JSONL | RAG outputs | `evaluation/runs/<ts>/` |
| `judge_responses.jsonl` | JSONL | Judge scores | `evaluation/runs/<ts>/` |
| `report.csv` | CSV | Metrics summary | `evaluation/runs/<ts>/` |
| `report.md` | Markdown | Narrative report | `evaluation/runs/<ts>/` |

### 3.3 JSONL Examples

**chunks.jsonl** (one object per line, UTF-8):
```json
{"doc_id":"a1b2c3","page":1,"chunk_id":"a1b2c3_p1_t0","type":"text","content":"Introduction to neural networks...","metadata":{"source_path":"paper.pdf","ocr_failed":false,"vision_used":false,"prev_chunk_id":null,"next_chunk_id":"a1b2c3_p1_t1","ingest_ts":"2024-01-21T12:00:00Z"}}
{"doc_id":"a1b2c3","page":2,"chunk_id":"a1b2c3_p2_tbl0","type":"table","content":"| Layer | Params |\n|-------|--------|\n| Conv1 | 64 |","metadata":{"source_path":"paper.pdf","ocr_failed":false,"vision_used":false,"ingest_ts":"2024-01-21T12:00:00Z"}}
{"doc_id":"a1b2c3","page":3,"chunk_id":"a1b2c3_p3_i0_ocr","type":"image_ocr","content":"Figure 1: Loss curve","metadata":{"source_path":"paper.pdf","image_hash":"d41d8cd98f00b204","ocr_confidence":75.5,"ocr_failed":false,"vision_used":false,"ingest_ts":"2024-01-21T12:00:00Z"}}
{"doc_id":"a1b2c3","page":3,"chunk_id":"a1b2c3_p3_i0_cap","type":"image_caption","content":"Line graph showing training loss decreasing over 100 epochs","metadata":{"image_hash":"d41d8cd98f00b204","ocr_failed":true,"ocr_failed_reason":"confidence=45<60","vision_used":true,"ingest_ts":"2024-01-21T12:00:00Z"}}
```

**dataset.jsonl**:
```json
{"question":"What optimizer was used?","slice":"overall","has_answer":true,"expected_answer":"Adam optimizer with lr=0.001","doc_id":"a1b2c3"}
{"question":"What values are in Table 2?","slice":"table","has_answer":true,"expected_answer":"Conv1=64, Conv2=128","doc_id":"a1b2c3"}
{"question":"What does Figure 1 show?","slice":"image","has_answer":true,"expected_answer":"Training loss curve","doc_id":"a1b2c3"}
{"question":"What is the weather today?","slice":"no-answer","has_answer":false,"expected_answer":"В документах нет ответа / недостаточно данных.","doc_id":null}
```

**predictions.jsonl**:
```json
{"question":"What optimizer was used?","answer":"Adam optimizer with learning rate 0.001 [a1b2c3|1|a1b2c3_p1_t0]","sources":[{"doc_id":"a1b2c3","page":1,"chunk_id":"a1b2c3_p1_t0","type":"text","score":0.92,"preview":"Training used Adam..."}],"retrieved_chunks":[...],"slice":"overall","has_answer_pred":true}
```

**judge_responses.jsonl**:
```json
{"question":"What optimizer was used?","answer":"Adam...","expected_answer":"Adam optimizer...","judge":{"faithfulness":0.95,"relevancy":0.90,"context_precision":0.88,"context_recall":0.85,"no_answer_correct":null,"notes":"Answer matches context"}}
```

**pdf_registry.jsonl**:
```json
{"doc_id":"a1b2c3","filename":"paper.pdf","filepath":"/data/paper.pdf","pages":10,"chunks":{"text":45,"table":3,"image_ocr":5,"image_caption":2},"ocr_failure_rate":0.4,"vision_fallback_rate":0.4,"errors":0,"parse_failed":false,"ingest_ts":"2024-01-21T12:00:00Z"}
```

---

## 4. Ingestion Pipeline

### 4.1 Flow Diagram

```
PDF File
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  PARSER (parser.py)                                     │
│  ├─ Primary: Unstructured partition_pdf()               │
│  └─ Fallback: PyPDF2 (text-only if Unstructured fails)  │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  ELEMENTS: text | table | image                         │
└─────────────────────────────────────────────────────────┘
    │
    ├──► TEXT ──► Chunker (512 tokens, 50 overlap) ──► Chunk(type="text")
    │
    ├──► TABLE ──► Keep intact, Markdown format ──► Chunk(type="table")
    │
    └──► IMAGE ──► OCR (Tesseract) ──┬──► Quality OK ──► Chunk(type="image_ocr")
                                      │
                                      └──► Quality FAIL ──► Vision API ──► Chunk(type="image_caption")
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  EMBEDDER (embedder.py)                                 │
│  text-embedding-3-small, cached by chunk_id             │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  INDEX (index_faiss.py)                                 │
│  FAISS IndexFlatIP, normalized vectors                  │
└─────────────────────────────────────────────────────────┘
```

### 4.2 PDF Parsing

**Primary**: Unstructured `partition_pdf(strategy="hi_res")`
- Extracts: NarrativeText, Table, Image elements
- Saves images to `data/cache/images/<doc_id>/`

**Fallback**: PyPDF2 `PdfReader`
- Triggered on Unstructured exception
- Text-only extraction (no tables/images)
- Logged to `error_log.jsonl`

**doc_id Generation**:
- `hash`: `md5(filepath + mtime)[:12]` — default, deterministic
- `filename`: file stem — simple but may collide
- `uuid`: random — always unique

### 4.3 Text Chunking

- **Size**: ~512 tokens with 50 token overlap
- **Boundaries**: Preserve paragraphs when possible
- **Linking**: `prev_chunk_id` / `next_chunk_id` in metadata
- **Tables**: Kept intact, converted to Markdown

### 4.4 Image Processing (OCR-First Strategy)

**Step 1: Always run Tesseract OCR**
```python
ocr_data = pytesseract.image_to_data(image, lang="eng+rus", output_type=Output.DICT)
confidence = mean([c for c in ocr_data['conf'] if c > 0])
```

**Step 2: Check OCR quality (ANY triggers fallback)**
```python
ocr_failed = (
    confidence < 60 or      # Low confidence
    len(text) < 50 or       # Too short
    word_count < 10 or      # Too few words
    alpha_ratio < 0.4       # Too much noise
)
```

**Step 3: Always store image_ocr chunk**

**Step 4: If OCR failed → Vision API fallback**

Vision Prompt (Extract → Summarize):
```
Analyze this image from a technical PDF.

Step 1 (Extract): List all visible text, numbers, labels, axis titles, legend entries.
If text is unreadable, say "unreadable".

Step 2 (Summarize): Based ONLY on Step 1, briefly describe what the image shows.
Do NOT speculate or infer beyond visible content.
```

**Step 5: Store image_caption chunk with `vision_used=true`**

### 4.5 Caching Strategy

| Cache | Key | Location | Invalidation |
|-------|-----|----------|--------------|
| OCR | `image_hash` (MD5) | `data/cache/ocr/{hash}.json` | Manual delete |
| Vision | `image_hash` (MD5) | `data/cache/vision/{hash}.txt` | Manual delete |
| Embeddings | `chunk_id` | `data/cache/embeddings/{hash}.json` | On model change |

**Cache lookup**: Before API call, check cache. On hit, return cached. On miss, call API and save.

### 4.6 Registry & Error Logging

**pdf_registry.jsonl** — Per-document statistics:
- `doc_id`, `filename`, `filepath`, `pages`
- `chunks`: counts by type (text/table/image_ocr/image_caption)
- `ocr_failure_rate`: % of images where OCR failed
- `vision_fallback_rate`: % of images where Vision was called
- `errors`: count of element-level failures
- `parse_failed`: true if parsing completely failed

**error_log.jsonl** — Element-level failures:
- `ts`, `doc_id`, `page`, `element_type`, `error`, `traceback`

### 4.7 Reindex Command

`scripts/reindex.py` — Rebuild index from existing chunks:
```bash
python scripts/reindex.py --clear --batch-size 50
```
- Skips PDF parsing
- Loads chunks from `chunks.jsonl`
- Recomputes all embeddings
- Rebuilds FAISS index
- Use after: model change, index corruption, config update

---

## 5. Retrieval & Generation

### 5.1 Retrieval

1. **Embed query**: `text-embedding-3-small`
2. **Dense search**: FAISS top-k nearest neighbors
3. **Optional BM25 hybrid**: If `hybrid_enabled=true`, union with BM25 results, deduplicate by chunk_id

### 5.2 Context Packing

```
[doc_id|page|chunk_id|type] (score: 0.92)
Preview: First 20 words of chunk content...
---
Full chunk content here...
```

### 5.3 Generation Prompt

**System Message**:
```
You are a helpful assistant answering questions about technical documents.

RULES:
1. Answer ONLY based on the provided context.
2. If the context does not contain enough information, respond exactly:
   "В документах нет ответа / недостаточно данных."
3. Always cite sources using format: [doc_id|page|chunk_id]
4. Do not speculate or add information not in context.
5. For tables, reference specific cells/rows when applicable.
```

### 5.4 No-Answer Detection

**has_answer_pred = False** when:
- Retrieval returns empty results
- Model response contains refusal phrase: `"нет ответа"`, `"недостаточно данных"`, etc.

**Standard Refusal Phrase**:
```
В документах нет ответа / недостаточно данных.
```

---

## 6. UI (Streamlit)

**File**: `ui/app.py`

### Features
- **Upload**: PDF upload with drag-drop
- **Ingest Progress**: Chunk counts by type, OCR warnings
- **Query Input**: Text box for questions
- **Answer Panel**: Response with status indicator (✅ found / ❌ not found)
- **Sources**: Expandable cards with metadata + preview
- **Tables**: Rendered as Markdown
- **Images**: Caption snippet or thumbnail

### Design
- Dark theme inspired by xpanceo.com
- Gradient background, cyan accents
- Modern typography

---

## 7. Evaluation

### 7.1 Dataset Generation

| Slice | Count | Source | Method |
|-------|-------|--------|--------|
| `overall` | 20 | All chunks | LLM generates Q&A pairs |
| `table` | 10-20 | Table chunks | LLM generates from table content |
| `image` | 10-20 | Image chunks | LLM generates from OCR/caption |
| `no-answer` | 10-20 | Synthetic | Pre-defined unanswerable questions |

**Ground Truth**:
- `has_answer=true`: `expected_answer` generated by LLM from chunk
- `has_answer=false`: `expected_answer` = standard refusal phrase

### 7.2 Evaluation Flow

```bash
# 1. Generate dataset
python scripts/generate_dataset.py --out evaluation/dataset.jsonl

# 2. Run RAG on dataset
python scripts/eval.py --dataset evaluation/dataset.jsonl --out-dir evaluation/runs/20240121_153000

# Output structure:
evaluation/runs/20240121_153000/
├── dataset.jsonl          # Copy of input
├── predictions.jsonl      # RAG outputs
├── judge_responses.jsonl  # Judge scores
├── report.csv             # Metrics summary
└── report.md              # Narrative report
```

### 7.3 Judge Prompt

```
You are an evaluation judge for a RAG system.

Given:
- Question: {question}
- Expected Answer: {expected_answer}
- Actual Answer: {answer}
- Retrieved Context: {context}

Score the following (0-1 scale):
1. Faithfulness: Does the answer ONLY use facts from context?
2. Relevancy: Does the answer address the question?
3. Context Precision: Are retrieved chunks relevant?
4. Context Recall: Does context have all needed info?

For no-answer questions:
5. No-Answer Correct: Did system correctly refuse?

Return JSON:
{"faithfulness": X, "relevancy": X, "context_precision": X, "context_recall": X, "no_answer_correct": true/false/null, "notes": "..."}
```

**Parsing Failures**: Log error, use default scores (0.0), continue evaluation.

### 7.4 Metrics

| Metric | Slices | Formula |
|--------|--------|---------|
| Context Precision | all | mean(judge.context_precision) |
| Context Recall | all | mean(judge.context_recall) |
| Faithfulness | all | mean(judge.faithfulness) |
| Answer Relevancy | all | mean(judge.relevancy) |
| No-Answer Accuracy | no-answer | correct_refusals / total |
| False Positive Rate | no-answer | false_answers / total |

**Report includes**:
- Counts: total, successful, failed
- Metrics by slice
- Example failures (low faithfulness, no-answer false positives)

---

## 8. Error Handling

### 8.1 API Retry Policy

| API | Max Retries | Backoff | On Exhaustion |
|-----|-------------|---------|---------------|
| Embeddings | 3 | 2^n seconds (2, 4, 8s) | Skip chunk, log to error_log.jsonl |
| Vision | 3 | 2^n seconds | Skip caption, keep OCR-only, continue |
| LLM (generation) | 3 | 2^n seconds | Return "ERROR" answer, log |
| LLM (judge) | 3 | 2^n seconds | Use default scores (0.0), log |

### 8.2 Error Matrix

| Scenario | Action | Logged To |
|----------|--------|-----------|
| **Parser fails (Unstructured)** | Try PyPDF2 fallback | error_log.jsonl |
| **Both parsers fail** | Skip doc, mark `parse_failed=true` | registry, error_log |
| **OCR timeout/error** | Mark `ocr_failed=true`, store empty | error_log.jsonl |
| **OCR low quality** | Mark failed, trigger Vision fallback | metadata |
| **Vision API error (429/5xx)** | Retry 3x with backoff | error_log.jsonl |
| **Vision persistent failure** | Store reason in `ocr_failed_reason`, skip caption, continue | error_log.jsonl |
| **Embedding API error** | Retry 3x, skip chunk if persistent | error_log.jsonl |
| **Index unavailable** | Clear error message, abort command with guidance | console |
| **Corrupted PDF** | Log and skip doc | registry, error_log |
| **No answer in context** | Standard refusal phrase | response |
| **Invalid JSONL line** | Skip line, log warning, continue processing | console |
| **Judge parse error** | Use default scores: `{faithfulness: 0, relevancy: 0, ...}` | console |
| **Multi-PDF collision** | Follow `reupload_policy` in config | registry |

### 8.3 Behavior Details

**Parser Fallback Flow**:
```
Unstructured.partition_pdf() 
    │ Exception
    ▼
PyPDF2.PdfReader() [text-only]
    │ Exception  
    ▼
Log to error_log.jsonl + mark parse_failed=true + continue to next doc
```

**Vision Fallback Flow**:
```
Tesseract OCR → Quality Check FAIL
    │
    ▼
Vision API (attempt 1) → 429/5xx → wait 2s
Vision API (attempt 2) → 429/5xx → wait 4s  
Vision API (attempt 3) → FAIL
    │
    ▼
Log: ocr_failed_reason="vision_unavailable_after_3_retries"
Store: image_ocr chunk only (no caption)
Continue pipeline
```

**JSONL Validation**:
```python
for line in file:
    try:
        entry = Model.from_jsonl(line)
        process(entry)
    except ValidationError as e:
        logger.warning(f"Invalid line skipped: {e}")
        continue  # Never halt
```

**Principle**: Never crash on single element. Log and continue.

---

## 9. Extensibility

### 9.1 Provider Adapters

```python
class LLMProvider(Protocol):
    def chat(self, messages: list, **kwargs) -> str: ...
    def vision(self, image_path: str, prompt: str) -> str: ...

class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

# Implementations
class OpenAIProvider(LLMProvider, EmbeddingProvider): ...  # ✅ Implemented
class GeminiProvider(LLMProvider, EmbeddingProvider): ...  # 🔲 Placeholder
```

### 9.2 Index Adapters

```python
class IndexBackend(Protocol):
    def upsert(self, ids, vectors, metadata): ...
    def search(self, vector, top_k) -> list[tuple]: ...
    def delete(self, ids): ...
    def clear(self): ...

# Implementations
class FAISSIndex(IndexBackend): ...   # ✅ Implemented
class QdrantIndex(IndexBackend): ...  # 🔲 Placeholder
```

### 9.3 Adding New Provider

1. Create `ingestion/llm_gemini.py` implementing `LLMProvider`
2. Add `llm_provider: gemini` to config
3. Update factory in `rag/generator.py`

---

## 10. Repository Structure

```
xpanceo-db/
├── config/
│   └── master_config.yaml        # All tunables
├── data/                         # Runtime data (gitignored)
│   ├── chunks.jsonl
│   ├── pdf_registry.jsonl
│   ├── error_log.jsonl
│   ├── index/                    # FAISS files
│   └── cache/                    # OCR/Vision/Embedding caches
│       ├── ocr/
│       ├── vision/
│       ├── embeddings/
│       └── images/
├── ingestion/
│   ├── __init__.py
│   ├── models.py                 # Pydantic schemas
│   ├── parser.py                 # PDF parsing
│   ├── ocr.py                    # Tesseract OCR
│   ├── captioner.py              # Vision fallback
│   ├── chunker.py                # Text/table chunking
│   ├── embedder.py               # OpenAI embeddings
│   ├── index_faiss.py            # FAISS backend
│   ├── index_qdrant.py           # Qdrant placeholder
│   └── pipeline.py               # Orchestration
├── rag/
│   ├── __init__.py
│   ├── retriever.py              # Vector search
│   ├── generator.py              # LLM answering
│   └── pipeline.py               # End-to-end Q&A
├── ui/
│   └── app.py                    # Streamlit
├── evaluation/
│   ├── __init__.py
│   ├── generate_dataset.py       # Create test questions
│   ├── eval_runner.py            # Run RAG on dataset
│   ├── judge.py                  # LLM judge
│   ├── metrics.py                # Calculate metrics
│   ├── report.py                 # Generate reports
│   └── runs/                     # Evaluation outputs
│       └── example/              # Reference outputs
│           ├── report.csv
│           └── report.md
├── scripts/
│   ├── ingest.py                 # CLI: ingest PDFs
│   ├── chat.py                   # CLI: Q&A
│   ├── reindex.py                # CLI: rebuild index
│   ├── generate_dataset.py       # CLI: create dataset
│   └── eval.py                   # CLI: run evaluation
├── tests/
│   ├── __init__.py
│   ├── test_ingestion_smoke.py
│   ├── test_rag_smoke.py
│   └── test_eval_smoke.py
├── master_specification.md       # This document
├── README.md                     # Quick start
├── requirements.txt              # Python deps
└── Dockerfile                    # Container setup
```

---

## 11. System Dependencies

```bash
# Ubuntu/Debian
apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus \
                   poppler-utils libmagic1

# macOS
brew install tesseract poppler libmagic
```

**Python** (see requirements.txt):
- Core: openai, pydantic, pyyaml
- PDF: unstructured[pdf], pytesseract, PyPDF2
- Index: faiss-cpu (qdrant-client optional)
- UI: streamlit
- Eval: ragas, datasets
- Hybrid: rank-bm25 (optional)

---

## 12. Acceptance Criteria

- [x] Ingestion runs on sample PDF, creates all chunk types
- [x] OCR fallback to Vision works when confidence low
- [x] Retrieval returns relevant chunks with scores
- [x] Generation produces answer with source citations
- [x] No-answer case handled with standard refusal
- [x] Evaluation produces report.csv and report.md
- [x] UI displays upload/query/answer flow
- [x] No crashes on arbitrary technical PDFs
- [x] Smoke tests pass: `pytest tests/ -v`

---

## 13. Changelog

### v1.1.0 (2024-01-21)
- Added developer handoff notes
- Expanded config documentation with all parameters
- Added JSONL format examples
- Documented caching strategy and invalidation
- Added reindex flow documentation
- Clarified eval ground truth and judge prompt
- Enhanced error handling table

### v1.0.0 (2024-01-21)
- Initial implementation complete
- Core pipeline: ingestion → RAG → evaluation
- FAISS index, Tesseract OCR, GPT-4o-mini Vision
- Streamlit UI, CLI scripts, smoke tests

---

*This specification is the single source of truth. All development must conform to it.*
