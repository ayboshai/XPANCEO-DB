# Multimodal RAG for Technical PDFs

A small, deterministic RAG system that can answer questions over PDFs with text, tables, and images, plus an automatic evaluation pipeline.

Design goals:
- no data loss,
- simple architecture,
- reproducible metrics,
- easy to run and debug.

Authoritative engineering contract: `master_specification.md`.

## 1) What It Does

The system has three modules:
1. Ingestion: PDF -> chunks -> vector index.
2. RAG: query -> retrieval -> answer with citations.
3. Evaluation: dataset generation -> automatic judging -> metrics.

Key behavior:
- tables stay readable for LLMs,
- visual content uses Vision by default,
- failed chunks are kept for audit but excluded from the index.

## 2) Requirements

System packages (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus
```

Python:
- Python 3.10+

## 3) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=YOUR_KEY
```

## 4) Quick Run

```bash
python3 - <<'PY'
import shutil
from pathlib import Path
path = Path('data/iter_10a')
if path.exists():
    shutil.rmtree(path)
print('cleaned', not path.exists())
PY

python3 scripts/ingest.py <pdf_folder> --config config/iter_10a.yaml

python3 scripts/generate_dataset.py \
  --config config/iter_10a.yaml \
  --out evaluation/datasets/iter_10a_small.jsonl \
  --overall-count 6 --table-count 4 --image-count 4 --no-answer-count 6

python3 scripts/eval.py \
  --config config/iter_10a.yaml \
  --dataset evaluation/datasets/iter_10a_small.jsonl
```

## 5) Sanity Checks (Required)

Unsupported formats must be gone:

```bash
find data/iter_10a/cache/images -type f \
  \( -name "*.jb2e" -o -name "*.jp2" -o -name "*.ppm" -o -name "*.pbm" -o -name "*.pgm" \)
```

Index integrity must hold:

```bash
python3 - <<'PY'
import json
from pathlib import Path
from ingestion.index_faiss import FAISSIndex

chunks = [json.loads(l) for l in Path('data/iter_10a/chunks.jsonl').read_text().splitlines() if l.strip()]
success_nonempty = sum(
    1 for c in chunks
    if c.get('metadata', {}).get('processing_status') == 'success' and (c.get('content') or '').strip()
)
idx = FAISSIndex(index_dir='data/iter_10a/index', dimension=1536)
print({'success_nonempty': success_nonempty, 'index_ntotal': idx.index.ntotal, 'diff': idx.index.ntotal - success_nonempty})
PY
```

## 6) Chat / UI

CLI chat:

```bash
python3 scripts/chat.py --config config/iter_10a.yaml
```

Streamlit UI (optional):

```bash
streamlit run ui/app.py
```

UI notes:
- The UI is pinned to a single stable config (`config/iter_10a.yaml`).
- It shows evaluation metrics by slice: overall / table / image / no-answer.
- It can generate a dataset and run evaluation from the sidebar.
- Ingestion shows live progress by stage (parse → images → embed/index).

Evaluation metrics (what they mean):
- **Faithfulness**: answer is grounded in the retrieved context (hallucination control).
- **Relevancy**: answer matches the question intent.
- **Context Precision**: retrieved context is mostly useful (noise control).
- **Context Recall**: retrieved context covers the needed facts (coverage).
- **No‑Answer Accuracy**: correct refusal when the answer is not in the documents.
- **False Positive Rate**: answers given when the system should refuse.

Metrics are reported by slice: `overall`, `table`, `image`, `no‑answer`.

## 7) Project Map

Core modules:
- `ingestion/parser.py`
- `ingestion/ocr.py`
- `ingestion/captioner.py`
- `ingestion/pipeline.py`
- `rag/retriever.py`
- `rag/generator.py`
- `evaluation/*`
- `scripts/*`

If you only read two files, read:
- `master_specification.md`
- `ingestion/pipeline.py`
