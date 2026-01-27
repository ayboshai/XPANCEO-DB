# Multimodal RAG — Master Specification (2026-01-27)

This document is the current engineering contract for the system. Keep it small, testable, and aligned with the code.

## 1) Non-Negotiable Invariants

1. No content loss: text, tables, images, pages, captions.
2. No hard timeouts or limits that truncate data.
3. Minimal switches and fallbacks. Prefer fixed, simple modes.
4. Failed chunks:
   - must be written to `chunks.jsonl` (audit trail),
   - must NOT be indexed.
5. Tests must run against real code (integration-style), not mocks.

## 2) System Shape (3 Modules)

1. Ingestion pipeline: PDF -> chunks -> index.
2. RAG pipeline: query -> retrieval -> generation.
3. Evaluation pipeline: dataset generation -> automatic judging -> metrics.

Key entry points:
- `scripts/ingest.py`
- `scripts/generate_dataset.py`
- `scripts/eval.py`
- `scripts/chat.py`

## 3) Ingestion Contract

### 3.1 Parser (`ingestion/parser.py`)

Required behavior:
- Parse is PyPDF2-first with coverage fallback.
- Tables are extracted only via `pdfplumber` (single fixed mode).
- Images are extracted via `pdfimages -png -p`.
- Tables and images are extracted unconditionally (independent of text parser choice).
- Dedup is allowed only when it is clearly safe (e.g., exact duplicates on the same page).

### 3.2 OCR + Routing (`ingestion/ocr.py`, `ingestion/pipeline.py`)

Routing decisions must be deterministic and content-safe:
- Each image is classified into one of:
  - `text_scan`, `table_scan`, `chart`, `diagram`, `photo`.
- OCR is always executed for auditability and anchor text.
- For `text_scan` and `table_scan`:
  - OCR-first,
  - Vision only when OCR is catastrophic.
- For `chart`, `diagram`, `photo`:
  - Vision is the default path,
  - Vision is skipped only when:
    - OCR is blank AND the image is near-uniform, or
    - the image is low-info/tiny decorative content.

Low-info policy:
- Low-info never deletes content.
- Low-info only changes whether Vision is called.
- OCR still runs and failed chunks still go to audit.

Performance safety note:
- Image processing is intentionally sequential.
- Rationale: Tesseract OCR becomes dramatically slower under thread load in this environment.

### 3.3 Vision Captioning (`ingestion/captioner.py`)

Required behavior:
- Vision calls are rate-limited by a singleton sync limiter.
- Limiter is initialized once per run with the run config.
- Vision client must not rely on hidden SDK retries.
- Image bytes must be normalized to supported formats.

### 3.4 Chunking (`ingestion/chunker.py`, `ingestion/pipeline.py`)

Required behavior:
- Chunking must be "never empty" in normal cases.
- Tables must stay LLM-readable (markdown-like text is acceptable).
- Table scans from OCR may be split into multiple table chunks.
- Image chunks must preserve:
  - OCR text,
  - Vision caption (when available),
  - image type,
  - audit status fields.

### 3.5 Embedding + Indexing (`ingestion/pipeline.py`, `ingestion/index_faiss.py`)

Required behavior:
- Only success + non-empty chunks are indexed.
- Failed chunks remain in `chunks.jsonl` and are excluded from the index.

## 4) Retrieval Contract (`rag/retriever.py`)

Required behavior:
- Dense + BM25 hybrid retrieval is supported.
- Failed chunks must be filtered in all retrieval paths:
  - dense search,
  - BM25/hybrid merge,
  - anchor fast-path,
  - table and image top-ups.
- Anchor-aware lexical top-up must work for quoted phrases.

## 5) Evaluation Contract (`evaluation/*`, `scripts/eval.py`)

Required behavior:
- Dataset generation is custom and cheap by default.
- RAGAS is disabled by default.
- Dataset metadata must be tied to the actual `chunks.jsonl` used by the pipeline.
- The evaluation report must be reproducible via scripts.

Core metrics to track:
- Faithfulness
- Relevancy
- Context Precision
- Context Recall

## 6) Minimal Runbook (Stable Commands)

### 6.1 Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...
```

### 6.2 Ingest

```bash
python3 scripts/ingest.py <pdf_folder> --config config/iter_10a.yaml
```

### 6.3 Generate Dataset

```bash
python3 scripts/generate_dataset.py \
  --config config/iter_10a.yaml \
  --out evaluation/datasets/iter_10a_small.jsonl \
  --overall-count 6 --table-count 4 --image-count 4 --no-answer-count 6
```

### 6.4 Evaluate

```bash
python3 scripts/eval.py \
  --config config/iter_10a.yaml \
  --dataset evaluation/datasets/iter_10a_small.jsonl
```

## 7) Required Sanity Checks

### 7.1 Unsupported formats must be gone

```bash
find data/iter_10a/cache/images -type f \
  \( -name "*.jb2e" -o -name "*.jp2" -o -name "*.ppm" -o -name "*.pbm" -o -name "*.pgm" \)
```

### 7.2 Index must match success non-empty chunks

```bash
python3 - <<'PY'
import json
from pathlib import Path
from ingestion.index_faiss import FAISSIndex

chunks = [json.loads(l) for l in Path("data/iter_10a/chunks.jsonl").read_text().splitlines() if l.strip()]
success_nonempty = sum(
    1 for c in chunks
    if c.get("metadata", {}).get("processing_status") == "success" and (c.get("content") or "").strip()
)
idx = FAISSIndex(index_dir="data/iter_10a/index", dimension=1536)
print({"success_nonempty": success_nonempty, "index_ntotal": idx.index.ntotal, "diff": idx.index.ntotal - success_nonempty})
PY
```

## 8) What "Good" Looks Like

The system is considered stable when:
- ingestion completes without crashes,
- index integrity check shows `diff = 0`,
- failed chunks remain bounded and explainable,
- Faithfulness and Relevancy stay high,
- image and table slices remain strong.

If a change improves speed but risks visual semantics, prefer the semantics.

## 9) UI Contract (Optional)

- UI uses a single pinned config: `config/iter_10a.yaml`.
- UI shows evaluation metrics by slice (overall/table/image/no-answer).
- UI can trigger dataset generation + evaluation with visible progress.
- Ingestion progress shows per-PDF stage updates.
