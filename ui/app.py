"""
XPANCEO DATABASE - Streamlit UI
Modern chat interface for PDF Q&A with source visualization.
Enhanced with Settings and Status panels.
"""

import os
import sys
import json
import csv
import tempfile
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv

# Load local .env (API key persistence; not tracked in Git)
load_dotenv(override=False)

# Logger for UI diagnostics
logger = logging.getLogger(__name__)

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="XPANCEO DATABASE",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for XPANCEO styling
st.markdown(
    """
<style>
    :root {
        --xp-bg: #0b0f17;
        --xp-panel: rgba(18, 26, 38, 0.9);
        --xp-panel-2: rgba(15, 22, 34, 0.85);
        --xp-accent: #00d4ff;
        --xp-accent-2: #7b2cbf;
        --xp-text: #e6f1ff;
        --xp-border: rgba(0, 212, 255, 0.18);
    }

    .stApp, body {
        background: radial-gradient(1200px 600px at 20% -10%, rgba(0, 212, 255, 0.12), transparent 60%),
                    radial-gradient(1000px 500px at 100% 0%, rgba(123, 44, 191, 0.10), transparent 55%),
                    var(--xp-bg);
        color: var(--xp-text);
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--xp-panel-2);
        border-right: 1px solid var(--xp-border);
    }

    /* Headers */
    h1, h2, h3 {
        color: var(--xp-accent) !important;
        font-family: Inter, sans-serif;
        letter-spacing: 0.2px;
    }

    /* Chat messages */
    .stChatMessage {
        background: var(--xp-panel) !important;
        border-radius: 12px !important;
        border: 1px solid var(--xp-border) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, var(--xp-accent), var(--xp-accent-2)) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover {
        box-shadow: 0 6px 22px rgba(0, 212, 255, 0.35) !important;
        transform: translateY(-1px);
    }

    /* Metrics and expanders */
    div[data-testid="stMetric"],
    div[data-testid="stMetric"] > div {
        background: var(--xp-panel);
        border: 1px solid var(--xp-border);
        padding: 8px 10px;
        border-radius: 10px;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: var(--xp-text) !important;
    }

    div[data-testid="stMetric"] * {
        color: var(--xp-text) !important;
    }

    /* Inputs */
    input, textarea, select, [data-testid="stTextInput"] input {
        background: #0f1522 !important;
        color: var(--xp-text) !important;
        border: 1px solid var(--xp-border) !important;
    }

    [data-testid="stSelectbox"] > div {
        background: #0f1522 !important;
        border: 1px solid var(--xp-border) !important;
    }

    [data-testid="stExpander"] {
        background: var(--xp-panel);
        border: 1px solid var(--xp-border);
        border-radius: 10px;
    }

    /* Source cards */
    .source-card {
        background: rgba(0, 212, 255, 0.10);
        border: 1px solid rgba(0, 212, 255, 0.28);
        border-radius: 10px;
        padding: 12px;
        margin: 8px 0;
    }

    /* Status indicators */
    .status-found {
        color: #00ff9c;
        font-weight: 700;
    }

    .status-not-found {
        color: #ff7b7b;
        font-weight: 700;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--xp-accent), var(--xp-accent-2)) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize session state variables."""
    default_config_path = "config/iter_10a.yaml" if Path("config/iter_10a.yaml").exists() else "config/master_config.yaml"
    defaults = {
        "messages": [],
        "rag_pipeline": None,
        "chunks_count": 0,
        "ingestion_results": [],
        "config_path": default_config_path,
        "active_config": None,
        "active_paths": {},
        "eval_status": {},
        "api_key_input": os.getenv("OPENAI_API_KEY", ""),
        "model_chat": "gpt-4o-mini",
        "model_vision": "gpt-4o-mini",
        "model_embed": "text-embedding-3-small",
        "top_k": 5,
        "hybrid_enabled": True,
        "reupload_policy": "overwrite",
        "chunk_size_tokens": 400,
        "chunk_overlap_tokens": 50,
        "max_concurrent_vision": 6,
        "max_concurrent_embeddings": 5,
        "rpm": 500,
        "max_retries": 3,
        "eval_overall_count": 6,
        "eval_table_count": 4,
        "eval_image_count": 4,
        "eval_no_answer_count": 6,
        "_last_config_path": None,
        "cancel_requested": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sanitize_error_message(message: str) -> str:
    """Redact API keys and other sensitive tokens from error messages."""
    if not message:
        return message
    # Mask OpenAI keys like sk-... or sk-proj-...
    message = re.sub(r"sk-[A-Za-z0-9_\-]{6,}", "sk-***", message)
    message = re.sub(r"sk-proj-[A-Za-z0-9_\-]{6,}", "sk-proj-***", message)
    return message


def format_duration(seconds: float) -> str:
    """Format seconds as M:SS."""
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}:{secs:02d}"


def save_api_key_to_env(api_key: str) -> None:
    """Persist API key to local .env (not tracked in Git)."""
    env_path = Path(".env")
    lines = []
    if env_path.exists():
        lines = env_path.read_text().splitlines()

    found = False
    new_lines = []
    for line in lines:
        if line.strip().startswith("OPENAI_API_KEY="):
            new_lines.append(f"OPENAI_API_KEY={api_key}")
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append(f"OPENAI_API_KEY={api_key}")

    env_path.write_text("\n".join(new_lines) + "\n")


def clear_api_key_from_env() -> None:
    """Remove API key from local .env."""
    env_path = Path(".env")
    if not env_path.exists():
        return

    lines = env_path.read_text().splitlines()
    new_lines = [line for line in lines if not line.strip().startswith("OPENAI_API_KEY=")]
    env_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))


def list_config_paths() -> List[str]:
    """List available YAML configs, preferring master_config first."""
    preferred = Path("config/iter_10a.yaml")
    if preferred.exists():
        return [str(preferred)]

    fallback = Path("config/master_config.yaml")
    return [str(fallback)]


def _apply_config_defaults(config: dict, config_path: str) -> None:
    """Apply config defaults to session state when config changes."""
    if st.session_state.get("_last_config_path") == config_path:
        return

    mapping = {
        "model_chat": "model_chat",
        "model_vision": "model_vision",
        "model_embed": "model_embed",
        "top_k": "top_k",
        "hybrid_enabled": "hybrid_enabled",
        "reupload_policy": "reupload_policy",
        "chunk_size_tokens": "chunk_size_tokens",
        "chunk_overlap_tokens": "chunk_overlap_tokens",
        "api_rate_limit_rpm": "rpm",
        "api_max_retries": "max_retries",
        "max_concurrent_vision": "max_concurrent_vision",
        "max_concurrent_embeddings": "max_concurrent_embeddings",
    }

    for config_key, state_key in mapping.items():
        if config_key in config and config[config_key] is not None:
            st.session_state[state_key] = config[config_key]

    # Config change invalidates current pipeline and cached eval view.
    st.session_state.rag_pipeline = None
    st.session_state.chunks_count = 0
    st.session_state._last_config_path = config_path


def get_active_paths(config: dict) -> Dict[str, str]:
    """Resolve key paths from config."""
    data_dir = config.get("data_dir", "data")
    chunks_file = os.path.join(data_dir, "chunks.jsonl")
    index_dir = config.get("index_dir", os.path.join(data_dir, "index"))
    reports_dir = config.get("reports_dir", "evaluation/runs")
    return {
        "data_dir": data_dir,
        "chunks_file": chunks_file,
        "index_dir": index_dir,
        "reports_dir": reports_dir,
    }


def load_active_config(config_path: str) -> Tuple[dict, Dict[str, str]]:
    """Load selected config with env resolution and update session state."""
    from shared import load_config as shared_load_config

    config = shared_load_config(config_path, force_reload=True)
    _apply_config_defaults(config, config_path)
    paths = get_active_paths(config)

    st.session_state.active_config = config
    st.session_state.active_paths = paths

    return config, paths


def _eval_status_path(data_dir: str) -> str:
    return os.path.join(data_dir, "eval_status.json")


def load_eval_status(data_dir: str) -> dict:
    """Load persistent eval status for a data_dir."""
    status_path = _eval_status_path(data_dir)
    if not os.path.exists(status_path):
        return {}
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load eval status from {status_path}: {e}")
        return {}


def save_eval_status(data_dir: str, status: dict) -> None:
    """Persist eval status alongside the data_dir."""
    os.makedirs(data_dir, exist_ok=True)
    status_path = _eval_status_path(data_dir)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


def compute_chunks_hash(chunks_file: str) -> Optional[str]:
    """Compute chunks hash if file exists."""
    if not os.path.exists(chunks_file):
        return None
    from evaluation.generate_dataset import compute_chunks_hash as _compute_chunks_hash

    try:
        return _compute_chunks_hash(chunks_file)
    except Exception as e:
        logger.warning(f"Failed to compute chunks hash for {chunks_file}: {e}")
        return None


def _file_md5(path: str) -> Optional[str]:
    """Compute MD5 hash of a file."""
    import hashlib

    if not os.path.exists(path):
        return None

    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _auto_detect_eval_status(paths: Dict[str, str], current_hash: str) -> dict:
    """
    Try to find the latest eval run that matches the current chunks hash.
    This runs only when no eval_status.json is present.
    """
    datasets_dir = Path("evaluation/datasets")
    runs_dir = Path(paths["reports_dir"])

    if not datasets_dir.exists() or not runs_dir.exists():
        return {}

    meta_files = sorted(datasets_dir.glob("*.meta.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:40]
    run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:60]

    for meta_path in meta_files:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        if meta.get("chunks_hash") != current_hash:
            continue

        dataset_path = str(meta_path).replace(".meta.json", ".jsonl")
        dataset_hash = _file_md5(dataset_path)
        if not dataset_hash:
            continue

        for run_dir in run_dirs:
            run_dataset = run_dir / "dataset.jsonl"
            run_hash = _file_md5(str(run_dataset))
            if run_hash and run_hash == dataset_hash:
                detected = {
                    "chunks_hash": current_hash,
                    "last_run_dir": str(run_dir),
                    "updated_at": datetime.utcnow().isoformat(),
                    "auto_detected": True,
                }
                save_eval_status(paths["data_dir"], detected)
                return detected

    return {}


def get_eval_state(paths: Dict[str, str]) -> dict:
    """Determine whether eval is up to date for the active data_dir."""
    chunks_file = paths["chunks_file"]
    data_dir = paths["data_dir"]
    current_hash = compute_chunks_hash(chunks_file)
    status = load_eval_status(data_dir)

    if not status and current_hash:
        status = _auto_detect_eval_status(paths, current_hash)

    last_hash = status.get("chunks_hash")
    last_run_dir = status.get("last_run_dir")
    has_run = bool(last_run_dir and os.path.exists(last_run_dir))

    needs_eval = current_hash is not None and (not has_run or current_hash != last_hash)

    return {
        "current_hash": current_hash,
        "last_hash": last_hash,
        "last_run_dir": last_run_dir,
        "needs_eval": needs_eval,
    }


def parse_report_csv(report_path: str) -> Dict[str, dict]:
    """Parse report.csv into a dict keyed by slice name."""
    metrics_by_slice: Dict[str, dict] = {}
    if not os.path.exists(report_path):
        return metrics_by_slice

    with open(report_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slice_name = row.get("slice", "")
            if not slice_name:
                continue
            try:
                metrics_by_slice[slice_name] = {
                    "count": int(float(row.get("count", 0) or 0)),
                    "faithfulness": float(row.get("faithfulness", 0) or 0),
                    "relevancy": float(row.get("relevancy", 0) or 0),
                    "context_precision": float(row.get("context_precision", 0) or 0),
                    "context_recall": float(row.get("context_recall", 0) or 0),
                    "no_answer_accuracy": float(row.get("no_answer_accuracy", 0) or 0),
                    "false_positive_rate": float(row.get("false_positive_rate", 0) or 0),
                }
            except Exception as e:
                logger.warning(f"Failed to parse metrics row {row}: {e}")
    return metrics_by_slice


def get_last_eval_metrics(paths: Dict[str, str]) -> Optional[dict]:
    """Get last eval metrics for the active data_dir."""
    reports_dir = paths["reports_dir"]
    state = get_eval_state(paths)

    # Prefer the run recorded for this data_dir.
    run_dir = state.get("last_run_dir")
    fallback = False

    if not run_dir or not os.path.exists(run_dir):
        if not os.path.exists(reports_dir):
            return None
        runs = sorted(os.listdir(reports_dir), reverse=True)
        if not runs:
            return None
        run_dir = os.path.join(reports_dir, runs[0])
        fallback = True

    report_path = os.path.join(run_dir, "report.csv")
    metrics_by_slice = parse_report_csv(report_path)
    if not metrics_by_slice:
        return None

    return {
        "run_dir": run_dir,
        "run_name": Path(run_dir).name,
        "report_path": report_path,
        "metrics_by_slice": metrics_by_slice,
        "fallback": fallback,
        "needs_eval": state["needs_eval"],
    }


def mark_eval_stale(paths: Dict[str, str]) -> None:
    """Mark eval status as stale after ingestion changes chunks."""
    current_hash = compute_chunks_hash(paths["chunks_file"])
    save_eval_status(
        paths["data_dir"],
        {
            "chunks_hash": current_hash,
            "last_run_dir": None,
            "updated_at": datetime.utcnow().isoformat(),
        },
    )


def update_eval_status(paths: Dict[str, str], run_dir: str) -> None:
    """Persist successful eval run for the current data_dir."""
    current_hash = compute_chunks_hash(paths["chunks_file"])
    save_eval_status(
        paths["data_dir"],
        {
            "chunks_hash": current_hash,
            "last_run_dir": run_dir,
            "updated_at": datetime.utcnow().isoformat(),
        },
    )


def load_pipeline(config_path: Optional[str] = None):
    """Load RAG pipeline for the selected config with UI overrides."""
    try:
        from rag.pipeline import create_rag_pipeline

        config_path = config_path or st.session_state.get("config_path", "config/master_config.yaml")
        config, paths = load_active_config(config_path)

        # Collect settings from UI session_state
        config_override = {
            "openai_api_key": st.session_state.get("api_key_input") or None,
            "model_chat": st.session_state.get("model_chat"),
            "model_vision": st.session_state.get("model_vision"),
            "model_embed": st.session_state.get("model_embed"),
            "top_k": st.session_state.get("top_k"),
            "hybrid_enabled": st.session_state.get("hybrid_enabled"),
            "api_rate_limit_rpm": st.session_state.get("rpm"),
            "api_max_retries": st.session_state.get("max_retries"),
        }

        with st.spinner(f"🔄 Loading RAG pipeline ({Path(config_path).name})..."):
            pipeline = create_rag_pipeline(
                config_path=config_path,
                config_override=config_override,
            )
            chunks_count = len(pipeline.retriever.chunks_lookup)

            st.session_state.rag_pipeline = pipeline
            st.session_state.chunks_count = chunks_count
            st.session_state.active_config = config
            st.session_state.active_paths = paths

            return pipeline, chunks_count
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        return None, 0


def run_ingestion(pdf_folder: str, config_path: Optional[str] = None):
    """Run ingestion on PDF folder with UI settings applied."""
    from ingestion.pipeline import IngestionPipeline

    config_path = config_path or st.session_state.get("config_path", "config/master_config.yaml")
    config, paths = load_active_config(config_path)

    # Apply UI settings to ingestion config (only real config keys).
    ui_ingestion_settings = {
        "openai_api_key": st.session_state.get("api_key_input") or None,
        "reupload_policy": st.session_state.get("reupload_policy"),
        "chunk_size_tokens": st.session_state.get("chunk_size_tokens"),
        "chunk_overlap_tokens": st.session_state.get("chunk_overlap_tokens"),
        "model_chat": st.session_state.get("model_chat"),
        "model_vision": st.session_state.get("model_vision"),
        "model_embed": st.session_state.get("model_embed"),
        "api_rate_limit_rpm": st.session_state.get("rpm"),
        "api_max_retries": st.session_state.get("max_retries"),
        "max_concurrent_vision": st.session_state.get("max_concurrent_vision"),
        "max_concurrent_embeddings": st.session_state.get("max_concurrent_embeddings"),
    }

    for key, value in ui_ingestion_settings.items():
        if value is not None:
            config[key] = value

    pipeline = IngestionPipeline(config)

    pdf_files = list(Path(pdf_folder).glob("**/*.pdf"))
    if not pdf_files:
        st.warning("No PDF files found in folder")
        return []

    pdf_progress = st.progress(0.0)
    stage_progress = st.progress(0.0)
    status_text = st.empty()
    stage_text = st.empty()
    time_text = st.empty()

    stage_labels = {
        "parse": "Parse PDF",
        "chunk_text_tables": "Chunk Text/Tables",
        "image_processing": "Process Images",
        "save_chunks": "Save Chunks",
        "embed_and_index": "Embed & Index",
        "done": "Done",
    }

    results = []
    total_pdfs = len(pdf_files)
    start_time = time.time()
    completed_durations: List[float] = []
    for i, pdf_path in enumerate(pdf_files, start=1):
        if st.session_state.get("cancel_requested"):
            status_text.text("Cancelled by user.")
            break

        status_text.text(f"[{i}/{total_pdfs}] Processing: {pdf_path.name}")
        stage_progress.progress(0.0)
        stage_text.text("Stage: starting...")
        pdf_start = time.time()

        def stage_callback(stage: str, stage_idx: int, stage_total: int) -> None:
            if st.session_state.get("cancel_requested"):
                return
            label = stage_labels.get(stage, stage)
            stage_progress.progress(stage_idx / max(stage_total, 1))
            stage_text.text(f"Stage {stage_idx}/{stage_total}: {label}")
            # Update ETA using completed PDFs as baseline.
            elapsed = time.time() - start_time
            elapsed_current = time.time() - pdf_start

            if completed_durations:
                avg_pdf = sum(completed_durations) / len(completed_durations)
                remaining_current = max(avg_pdf - elapsed_current, 0)
                remaining = remaining_current + avg_pdf * max(total_pdfs - i, 0)
                time_text.text(
                    f"Elapsed: {format_duration(elapsed)} | ETA: {format_duration(remaining)}"
                )
            else:
                time_text.text(f"Elapsed: {format_duration(elapsed)} | ETA: calculating...")

        try:
            entry = pipeline.ingest_pdf(str(pdf_path), stage_callback=stage_callback)
            results.append(entry)
        except Exception as e:
            st.error(f"Failed to ingest {pdf_path.name}: {sanitize_error_message(str(e))}")
            if "invalid_api_key" in str(e).lower() or "incorrect api key" in str(e).lower():
                st.session_state.cancel_requested = True

        pdf_progress.progress(i / total_pdfs)
        pdf_elapsed = time.time() - pdf_start
        completed_durations.append(pdf_elapsed)
        elapsed = time.time() - start_time
        avg_per_pdf = sum(completed_durations) / len(completed_durations)
        remaining = avg_per_pdf * max(total_pdfs - i, 0)
        time_text.text(f"Elapsed: {format_duration(elapsed)} | ETA: {format_duration(remaining)}")

    pdf_progress.empty()
    stage_progress.empty()
    status_text.empty()
    stage_text.empty()
    time_text.empty()

    # Ingestion changes chunks → eval is now stale for this data_dir.
    if results:
        mark_eval_stale(paths)

    return results


def run_dataset_and_eval(
    config_path: Optional[str] = None,
    overall_count: int = 6,
    table_count: int = 4,
    image_count: int = 4,
    no_answer_count: int = 6,
) -> Optional[dict]:
    """Generate dataset and run full evaluation with UI progress."""
    from ingestion.pipeline import IngestionPipeline
    from evaluation.generate_dataset import DatasetGenerator, save_dataset, load_dataset
    from evaluation.eval_runner import run_evaluation
    from evaluation.judge import LLMJudge
    from evaluation.metrics import calculate_metrics
    from evaluation.report import generate_reports

    config_path = config_path or st.session_state.get("config_path", "config/master_config.yaml")
    config, paths = load_active_config(config_path)

    # Apply UI overrides to config and run eval against that effective config.
    overrides = {
        "openai_api_key": st.session_state.get("api_key_input") or None,
        "model_chat": st.session_state.get("model_chat"),
        "model_vision": st.session_state.get("model_vision"),
        "model_embed": st.session_state.get("model_embed"),
        "top_k": st.session_state.get("top_k"),
        "hybrid_enabled": st.session_state.get("hybrid_enabled"),
        "api_rate_limit_rpm": st.session_state.get("rpm"),
        "api_max_retries": st.session_state.get("max_retries"),
        "max_concurrent_vision": st.session_state.get("max_concurrent_vision"),
        "max_concurrent_embeddings": st.session_state.get("max_concurrent_embeddings"),
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    api_key = config.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OPENAI_API_KEY not set")
        return None

    pipeline = IngestionPipeline(config)
    chunks = pipeline.load_chunks()
    if not chunks:
        st.error(f"No chunks found at {paths['chunks_file']}. Run ingestion first.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = Path(config_path).stem
    dataset_path = f"evaluation/datasets/{config_name}_{timestamp}.jsonl"
    output_dir = os.path.join(paths["reports_dir"], timestamp)

    stage_bar = st.progress(0.0)
    step_bar = st.progress(0.0)
    stage_text = st.empty()
    step_text = st.empty()
    time_text = st.empty()

    def set_stage(idx: int, total: int, label: str) -> None:
        stage_bar.progress(idx / max(total, 1))
        stage_text.text(f"Stage {idx}/{total}: {label}")
        step_bar.progress(0.0)

    start_time = time.time()
    try:
        # Stage 1/4: dataset generation
        stage_total = 4
        set_stage(1, stage_total, "Generate Dataset")
        generator = DatasetGenerator(api_key, config.get("model_chat", "gpt-4o-mini"))

        entries = []
        slice_plan = [
            ("overall", overall_count, lambda: generator.generate_overall(chunks, overall_count, use_ragas=False)),
            ("table", table_count, lambda: generator.generate_table_slice(chunks, table_count)),
            ("image", image_count, lambda: generator.generate_image_slice(chunks, image_count)),
            ("no-answer", no_answer_count, lambda: generator.generate_no_answer_slice(no_answer_count)),
        ]

        for i, (slice_name, target, fn) in enumerate(slice_plan, start=1):
            if st.session_state.get("cancel_requested"):
                raise RuntimeError("Evaluation cancelled by user.")
            step_text.text(f"Dataset slice {i}/{len(slice_plan)}: {slice_name} (target={target})")
            step_bar.progress((i - 1) / len(slice_plan))
            new_entries = fn()
            entries.extend(new_entries)
            step_bar.progress(i / len(slice_plan))

        save_dataset(entries, dataset_path, chunks_file=pipeline.chunks_file)

        # Stage 2/4: RAG predictions
        set_stage(2, stage_total, "Run RAG")

        def eval_progress(current: int, total: int, entry) -> None:
            if st.session_state.get("cancel_requested"):
                raise RuntimeError("Evaluation cancelled by user.")
            step_bar.progress(current / max(total, 1))
            step_text.text(f"RAG {current}/{total}: {entry.slice}")
            elapsed = time.time() - start_time
            remaining = (elapsed / max(current, 1)) * max(total - current, 0)
            time_text.text(f"Elapsed: {format_duration(elapsed)} | ETA: {format_duration(remaining)}")

        predictions = run_evaluation(
            dataset_path,
            output_dir=output_dir,
            config_path=_write_temp_config(config, config_name, timestamp),
            progress_callback=eval_progress,
        )

        # Stage 3/4: judge
        set_stage(3, stage_total, "LLM Judge")
        dataset_entries = load_dataset(dataset_path)
        slices_mapping = {e.question: e.slice for e in dataset_entries}

        judge = LLMJudge(api_key, config.get("model_chat", "gpt-4o-mini"))

        def judge_progress(current: int, total: int, pred) -> None:
            if st.session_state.get("cancel_requested"):
                raise RuntimeError("Evaluation cancelled by user.")
            step_bar.progress(current / max(total, 1))
            step_text.text(f"Judge {current}/{total}: {pred.slice}")
            elapsed = time.time() - start_time
            remaining = (elapsed / max(current, 1)) * max(total - current, 0)
            time_text.text(f"Elapsed: {format_duration(elapsed)} | ETA: {format_duration(remaining)}")

        judge_responses = judge.judge_all(
            predictions,
            dataset_entries,
            output_dir=output_dir,
            show_progress=False,
            progress_callback=judge_progress,
        )

        # Stage 4/4: metrics + reports
        set_stage(4, stage_total, "Compute Metrics")
        metrics = calculate_metrics(judge_responses, slices_mapping)
        csv_path, md_path = generate_reports(metrics, judge_responses, output_dir, predictions, dataset_entries)
        step_bar.progress(1.0)
        step_text.text("Reports generated")

        update_eval_status(paths, output_dir)

        return {
            "output_dir": output_dir,
            "dataset_path": dataset_path,
            "csv_path": csv_path,
            "md_path": md_path,
            "metrics_by_slice": metrics.by_slice,
            "overall": metrics.overall,
        }
    except Exception as e:
        st.error(f"Evaluation failed: {sanitize_error_message(str(e))}")
        logger.exception("Evaluation failed")
        return None
    finally:
        stage_bar.empty()
        step_bar.empty()
        stage_text.empty()
        step_text.empty()
        time_text.empty()


def _write_temp_config(config: dict, config_name: str, timestamp: str) -> str:
    """Write effective config to a temp file for evaluation runs."""
    import yaml
    from shared import load_config as shared_load_config

    temp_config_dir = Path("config/.ui_tmp")
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = temp_config_dir / f"{config_name}_{timestamp}.yaml"

    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # Prime shared config cache to keep judge/limiters consistent.
    shared_load_config(str(temp_config_path), force_reload=True)

    return str(temp_config_path)


def get_document_stats(chunks_file: str) -> dict:
    """Get document statistics from the active chunks.jsonl."""
    stats = {
        "docs": 0,
        "total": 0,
        "text": 0,
        "table": 0,
        "image_ocr": 0,
        "image_caption": 0,
        "ocr_failed": 0,
        "vision_used": 0,
    }
    
    if not os.path.exists(chunks_file):
        return stats
    
    docs = set()
    with open(chunks_file, "r") as f:
        for line in f:
            try:
                chunk = json.loads(line)
                stats["total"] += 1
                chunk_type = chunk.get("type", "text")
                if chunk_type in stats:
                    stats[chunk_type] += 1
                docs.add(chunk.get("doc_id", ""))
                meta = chunk.get("metadata", {})
                if meta.get("ocr_failed"):
                    stats["ocr_failed"] += 1
                if meta.get("vision_used"):
                    stats["vision_used"] += 1
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse chunk JSONL line: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error while parsing chunks: {e}")
    
    stats["docs"] = len(docs)
    return stats


def format_source_card(source):
    """Format source as HTML card."""
    # Handle dict (from history) vs object (from live run)
    if isinstance(source, dict):
        doc_id = source.get("doc_id", "???")
        page = source.get("page", "?")
        stype = source.get("type", "text")
        score = source.get("score", 0.0)
        preview = source.get("preview", "")
    else:
        doc_id = getattr(source, "doc_id", "???")
        page = getattr(source, "page", "?")
        stype = getattr(source, "type", "text")
        score = getattr(source, "score", 0.0)
        preview = getattr(source, "preview", "")

    return f"""
    <div class="source-card">
        <strong>📄 {doc_id}</strong> | Page {page} | {stype}<br>
        <small>Score: {score:.3f}</small><br>
        <em>{preview}</em>
    </div>
    """


def render_sidebar():
    """Render sidebar with Settings and Status panels."""
    with st.sidebar:
        st.header("⚙️ Settings")

        config_paths = list_config_paths()
        current_config_path = st.session_state.get("config_path", config_paths[0])
        if current_config_path not in config_paths:
            current_config_path = config_paths[0]
            st.session_state.config_path = current_config_path

        def safe_index(options: List[str], value: str) -> int:
            try:
                return options.index(value)
            except ValueError:
                return 0

        if len(config_paths) == 1:
            st.text(f"Active Config: {Path(config_paths[0]).name}")
            st.session_state.config_path = config_paths[0]
        else:
            selected_config_path = st.selectbox(
                "Active Config",
                config_paths,
                index=safe_index(config_paths, current_config_path),
                format_func=lambda p: Path(p).name,
                help="All UI actions run against this config/data_dir",
            )
            if selected_config_path != st.session_state.config_path:
                st.session_state.config_path = selected_config_path

        config, paths = load_active_config(st.session_state.config_path)
        eval_state = get_eval_state(paths)

        st.caption(f"Config: `{Path(st.session_state.config_path).name}`")
        st.caption(f"Data dir: `{paths['data_dir']}`")
        if eval_state["needs_eval"]:
            st.warning("Evaluation is stale for this data_dir")
        elif eval_state["last_run_dir"]:
            st.success("Evaluation is up to date")

        with st.expander("🔑 API Configuration", expanded=False):
            api_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.api_key_input or "",
                type="password",
                help="OpenAI key used for embeddings, vision, and judge",
            )
            col_use, col_clear = st.columns(2)
            with col_use:
                if st.button("✅ Use This Key", use_container_width=True):
                    if api_key:
                        st.session_state.api_key_input = api_key
                        os.environ["OPENAI_API_KEY"] = api_key
                        save_api_key_to_env(api_key)
                        from shared import load_config as _reload_config
                        _reload_config(st.session_state.config_path, force_reload=True)
                        st.success("API key saved locally and loaded")
                    else:
                        st.warning("Enter a key first")
            with col_clear:
                if st.button("🧽 Clear Stored Key", use_container_width=True):
                    st.session_state.api_key_input = ""
                    os.environ.pop("OPENAI_API_KEY", None)
                    clear_api_key_from_env()
                    st.success("Cleared API key from session and .env")

            if os.getenv("OPENAI_API_KEY"):
                st.info("API key is loaded in this session")
            else:
                st.warning("No API key set for this session")

        with st.expander("🤖 Model Settings", expanded=False):
            chat_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
            vision_options = ["gpt-4o-mini", "gpt-4o"]
            embed_options = ["text-embedding-3-small", "text-embedding-3-large"]

            st.session_state.model_chat = st.selectbox(
                "Chat Model",
                chat_options,
                index=safe_index(chat_options, st.session_state.model_chat),
            )
            st.session_state.model_vision = st.selectbox(
                "Vision Model",
                vision_options,
                index=safe_index(vision_options, st.session_state.model_vision),
            )
            st.session_state.model_embed = st.selectbox(
                "Embedding Model",
                embed_options,
                index=safe_index(embed_options, st.session_state.model_embed),
            )

        with st.expander("🔍 Retrieval Settings", expanded=False):
            st.session_state.top_k = st.slider(
                "Top K Results",
                min_value=1,
                max_value=20,
                value=int(st.session_state.top_k),
                help="Number of chunks retrieved for context",
            )
            st.session_state.hybrid_enabled = st.checkbox(
                "Hybrid Search (BM25 + Vector)",
                value=bool(st.session_state.hybrid_enabled),
            )

        with st.expander("📄 Ingestion Settings", expanded=False):
            st.session_state.reupload_policy = st.selectbox(
                "Reupload Policy",
                ["overwrite", "new_version"],
                index=safe_index(["overwrite", "new_version"], st.session_state.reupload_policy),
                help="overwrite replaces existing doc_id; new_version keeps both",
            )
            st.session_state.chunk_size_tokens = st.slider(
                "Chunk Size (tokens)",
                min_value=200,
                max_value=1200,
                step=50,
                value=int(st.session_state.chunk_size_tokens),
            )
            st.session_state.chunk_overlap_tokens = st.slider(
                "Chunk Overlap (tokens)",
                min_value=0,
                max_value=200,
                step=10,
                value=int(st.session_state.chunk_overlap_tokens),
            )

        with st.expander("⚡ Rate Limits & Concurrency", expanded=False):
            st.session_state.rpm = st.number_input(
                "API Rate Limit (RPM)",
                min_value=1,
                max_value=10000,
                value=int(st.session_state.rpm),
            )
            st.session_state.max_retries = st.number_input(
                "Max Retries",
                min_value=1,
                max_value=10,
                value=int(st.session_state.max_retries),
            )
            st.session_state.max_concurrent_vision = st.slider(
                "Max Concurrent Vision",
                min_value=1,
                max_value=16,
                value=int(st.session_state.max_concurrent_vision),
            )
            st.session_state.max_concurrent_embeddings = st.slider(
                "Max Concurrent Embeddings",
                min_value=1,
                max_value=16,
                value=int(st.session_state.max_concurrent_embeddings),
            )

        st.divider()

        # ===== STATUS PANEL =====
        st.header("📊 Status")

        if os.getenv("OPENAI_API_KEY"):
            st.success("🔑 API Key: Loaded")
        else:
            st.error("🔑 API Key: Not set")

        if st.session_state.rag_pipeline:
            try:
                vectors = st.session_state.rag_pipeline.retriever.index.index.ntotal
            except Exception:
                vectors = st.session_state.chunks_count
            st.success(f"📦 Index: {vectors} vectors")
        else:
            st.warning("📦 Index: Not loaded")
            if st.button("🔄 Load Pipeline", use_container_width=True):
                load_pipeline(st.session_state.config_path)
                st.rerun()

        with st.expander("📄 Documents", expanded=True):
            stats = get_document_stats(paths["chunks_file"])
            if stats["total"] > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("PDFs", stats["docs"])
                    st.metric("Text", stats["text"])
                    st.metric("Image OCR", stats["image_ocr"])
                with col2:
                    st.metric("Chunks", stats["total"])
                    st.metric("Table", stats["table"])
                    st.metric("Caption", stats["image_caption"])

                ocr_rate = stats["ocr_failed"] / stats["total"] * 100 if stats["total"] else 0
                vision_rate = stats["vision_used"] / stats["total"] * 100 if stats["total"] else 0
                st.caption(f"OCR fail: {ocr_rate:.1f}% | Vision used: {vision_rate:.1f}%")
            else:
                st.info(f"No chunks found at `{paths['chunks_file']}`")

        with st.expander("📈 Evaluation", expanded=False):
            metrics = get_last_eval_metrics(paths)
            if metrics:
                st.caption(f"Run: {metrics['run_name']}")
                if metrics["fallback"]:
                    st.warning("Showing latest run (may not match current data_dir)")
                if metrics["needs_eval"]:
                    st.warning("Chunks changed since the last recorded eval")

                rows = []
                for slice_name, m in metrics["metrics_by_slice"].items():
                    rows.append(
                        {
                            "slice": slice_name,
                            "n": m["count"],
                            "faith": round(m["faithfulness"], 3),
                            "relev": round(m["relevancy"], 3),
                            "ctx_prec": round(m["context_precision"], 3),
                            "ctx_rec": round(m["context_recall"], 3),
                            "no_ans_acc": round(m["no_answer_accuracy"], 3),
                            "fpr": round(m["false_positive_rate"], 3),
                        }
                    )
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.info("No evaluation runs found")

            st.subheader("Run Evaluation")
            st.session_state.eval_overall_count = st.number_input(
                "Overall Questions",
                min_value=2,
                max_value=50,
                value=int(st.session_state.eval_overall_count),
            )
            st.session_state.eval_table_count = st.number_input(
                "Table Questions",
                min_value=2,
                max_value=50,
                value=int(st.session_state.eval_table_count),
            )
            st.session_state.eval_image_count = st.number_input(
                "Image Questions",
                min_value=2,
                max_value=50,
                value=int(st.session_state.eval_image_count),
            )
            st.session_state.eval_no_answer_count = st.number_input(
                "No-Answer Questions",
                min_value=2,
                max_value=50,
                value=int(st.session_state.eval_no_answer_count),
            )

            if st.button("🧪 Generate Dataset + Evaluate", use_container_width=True):
                st.session_state.cancel_requested = False
                result = run_dataset_and_eval(
                    config_path=st.session_state.config_path,
                    overall_count=int(st.session_state.eval_overall_count),
                    table_count=int(st.session_state.eval_table_count),
                    image_count=int(st.session_state.eval_image_count),
                    no_answer_count=int(st.session_state.eval_no_answer_count),
                )
                if result:
                    st.success(f"Evaluation complete: {Path(result['output_dir']).name}")
                    st.rerun()

        with st.expander("⚠️ Warnings", expanded=False):
            warnings: List[str] = []
            if not os.getenv("OPENAI_API_KEY"):
                warnings.append("API key not set")
            if not os.path.exists(paths["chunks_file"]):
                warnings.append(f"No chunks: {paths['chunks_file']}")
            index_path = os.path.join(paths["index_dir"], "index.faiss")
            if not os.path.exists(index_path):
                warnings.append(f"Index not found: {index_path}")
            if eval_state["needs_eval"]:
                warnings.append("Evaluation is stale for this data_dir")

            if warnings:
                for w in warnings:
                    st.warning(w)
            else:
                st.success("✅ All systems OK")

        st.divider()

        st.subheader("📥 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        col_cancel, col_reset = st.columns(2)
        with col_cancel:
            if st.button("⛔ Cancel Run", use_container_width=True):
                st.session_state.cancel_requested = True
                st.warning("Cancel requested. Stopping after current step.")
        with col_reset:
            if st.button("🧹 Reset UI", use_container_width=True):
                st.session_state.cancel_requested = False
                st.session_state.messages = []
                st.session_state.ingestion_results = []
                st.session_state.rag_pipeline = None
                st.session_state.chunks_count = 0
                st.success("UI state reset")

        if uploaded_files and st.button("🚀 Ingest", use_container_width=True):
            st.session_state.cancel_requested = False
            temp_dir = tempfile.mkdtemp(prefix="xpanceo_")
            for file in uploaded_files:
                with open(os.path.join(temp_dir, file.name), "wb") as f:
                    f.write(file.getvalue())

            results = run_ingestion(temp_dir, config_path=st.session_state.config_path)
            st.session_state.ingestion_results = results

            if results:
                st.success(f"✅ Ingested {len(results)} documents")
                load_pipeline(st.session_state.config_path)
                st.rerun()

        if st.session_state.ingestion_results:
            with st.expander("📋 Last Ingest"):
                for entry in st.session_state.ingestion_results:
                    st.write(
                        f"**{entry.filename}**: {entry.pages}p, "
                        f"{entry.chunks.text} text / {entry.chunks.table} tables / "
                        f"{entry.chunks.image_ocr} image OCR / {entry.chunks.image_caption} captions"
                    )


def main():
    init_session_state()
    
    # Header
    st.title("🔮 XPANCEO DATABASE")
    st.markdown("*Multimodal RAG for Technical PDFs*")
    
    # Render sidebar
    render_sidebar()
    
    # Main chat area
    st.divider()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.markdown(format_source_card(source), unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            if not st.session_state.rag_pipeline:
                st.error("Please load the pipeline first")
                response_text = "Pipeline not loaded."
                sources = []
            else:
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_pipeline.query(prompt)
                    response_text = response.answer
                    sources = response.sources
                    
                    if response.has_answer:
                        st.markdown('<span class="status-found">✅ Answer found</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="status-not-found">❌ No answer</span>', unsafe_allow_html=True)
            
            st.markdown(response_text)
            
            if sources:
                with st.expander("📚 Sources"):
                    for source in sources:
                        st.markdown(format_source_card(source), unsafe_allow_html=True)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "sources": [{"doc_id": s.doc_id, "page": s.page, "type": s.type, "score": s.score, "preview": s.preview} for s in sources] if sources else [],
        })


if __name__ == "__main__":
    main()
