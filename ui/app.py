"""
XPANCEO DB - Streamlit UI
Modern chat interface for PDF Q&A with source visualization.
Enhanced with Settings and Status panels.
"""

import os
import sys
import json
import csv
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="XPANCEO DB",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for XPANCEO styling
st.markdown("""
<style>
    /* Dark theme inspired by xpanceo.com */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d4ff !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff, #7b2cbf) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.4) !important;
    }
    
    /* Source cards */
    .source-card {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    
    /* Status indicators */
    .status-found {
        color: #00ff88;
        font-weight: bold;
    }
    
    .status-not-found {
        color: #ff6b6b;
        font-weight: bold;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00d4ff, #7b2cbf) !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "messages": [],
        "rag_pipeline": None,
        "chunks_count": 0,
        "ingestion_results": [],
        "model_chat": "gpt-4o-mini",
        "model_embed": "text-embedding-3-small",
        "top_k": 5,
        "hybrid_enabled": False,
        "reupload_policy": "overwrite",
        "ocr_threshold": 60,
        "rpm": 500,
        "max_retries": 3,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_pipeline():
    """Load RAG pipeline."""
    try:
        from rag.pipeline import create_rag_pipeline
        
        with st.spinner("🔄 Loading RAG pipeline..."):
            pipeline = create_rag_pipeline()
            chunks_count = len(pipeline.retriever.chunks_lookup)
            
            st.session_state.rag_pipeline = pipeline
            st.session_state.chunks_count = chunks_count
            
            return pipeline, chunks_count
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        return None, 0


def run_ingestion(pdf_folder: str):
    """Run ingestion on PDF folder."""
    import yaml
    from ingestion.pipeline import IngestionPipeline
    
    # Load config
    config_path = "config/master_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Resolve env vars
    for key, value in config.items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            config[key] = os.getenv(value[2:-1], "")
    
    pipeline = IngestionPipeline(config)
    
    # Get PDF list
    from pathlib import Path
    pdf_files = list(Path(pdf_folder).glob("**/*.pdf"))
    
    if not pdf_files:
        st.warning("No PDF files found in folder")
        return []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    for i, pdf_path in enumerate(pdf_files):
        status_text.text(f"Processing: {pdf_path.name}")
        try:
            entry = pipeline.ingest_pdf(str(pdf_path))
            results.append(entry)
        except Exception as e:
            st.error(f"Failed to ingest {pdf_path.name}: {e}")
        
        progress_bar.progress((i + 1) / len(pdf_files))
    
    progress_bar.empty()
    status_text.empty()
    
    return results


def get_document_stats():
    """Get document statistics from chunks.jsonl."""
    chunks_file = "data/chunks.jsonl"
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
            except:
                pass
    
    stats["docs"] = len(docs)
    return stats


def get_last_eval_metrics():
    """Get metrics from last evaluation run."""
    eval_runs_dir = "evaluation/runs"
    if not os.path.exists(eval_runs_dir):
        return None
    
    runs = sorted(os.listdir(eval_runs_dir), reverse=True)
    if not runs:
        return None
    
    latest_run = runs[0]
    report_path = os.path.join(eval_runs_dir, latest_run, "report.csv")
    
    if not os.path.exists(report_path):
        return None
    
    metrics = {"run": latest_run, "path": report_path}
    
    with open(report_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slice_name = row.get("slice", "")
            if slice_name == "overall":
                metrics["faithfulness"] = float(row.get("faithfulness", 0))
                metrics["relevancy"] = float(row.get("relevancy", 0))
                metrics["context_precision"] = float(row.get("context_precision", 0))
                metrics["context_recall"] = float(row.get("context_recall", 0))
            if slice_name == "no-answer":
                metrics["no_answer_accuracy"] = float(row.get("no_answer_accuracy", 0))
                metrics["no_answer_fpr"] = float(row.get("no_answer_fpr", 0))
    
    return metrics


def format_source_card(source):
    """Format source as HTML card."""
    return f"""
    <div class="source-card">
        <strong>📄 {source.doc_id}</strong> | Page {source.page} | {source.type}<br>
        <small>Score: {source.score:.3f}</small><br>
        <em>{source.preview}</em>
    </div>
    """


def render_sidebar():
    """Render sidebar with Settings and Status panels."""
    with st.sidebar:
        # ===== SETTINGS PANEL =====
        st.header("⚙️ Settings")
        
        with st.expander("🔑 API Configuration", expanded=False):
            api_key = st.text_input(
                "OpenAI API Key",
                value=os.getenv("OPENAI_API_KEY", ""),
                type="password",
                help="Your OpenAI API key for embeddings and chat"
            )
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                st.success("✅ API key set")
            else:
                st.warning("⚠️ No API key")
        
        with st.expander("🤖 Model Settings", expanded=False):
            st.session_state.model_chat = st.selectbox(
                "Chat Model",
                ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                index=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"].index(st.session_state.model_chat),
                help="GPT model for generating answers"
            )
            st.session_state.model_embed = st.selectbox(
                "Embedding Model",
                ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
                help="Model for vector embeddings"
            )
        
        with st.expander("🔍 Retrieval Settings", expanded=False):
            st.session_state.top_k = st.slider(
                "Top K Results", min_value=1, max_value=20,
                value=st.session_state.top_k,
                help="Number of chunks to retrieve for context"
            )
            st.session_state.hybrid_enabled = st.checkbox(
                "Hybrid Search (BM25 + Vector)",
                value=st.session_state.hybrid_enabled,
                help="Enable BM25 keyword matching + vector search"
            )
        
        with st.expander("📄 Ingestion Settings", expanded=False):
            st.session_state.reupload_policy = st.selectbox(
                "Reupload Policy",
                ["overwrite", "new_version"],
                index=["overwrite", "new_version"].index(st.session_state.reupload_policy),
                help="overwrite: clears old data. new_version: adds copy"
            )
            st.session_state.ocr_threshold = st.slider(
                "OCR Confidence Threshold",
                min_value=0, max_value=100, value=st.session_state.ocr_threshold,
                help="Below threshold, Vision API is used"
            )
        
        with st.expander("⚡ Rate Limits", expanded=False):
            st.session_state.rpm = st.number_input(
                "API Rate Limit (RPM)",
                min_value=1, max_value=10000, value=st.session_state.rpm,
                help="Max requests per minute"
            )
            st.session_state.max_retries = st.number_input(
                "Max Retries",
                min_value=1, max_value=10, value=st.session_state.max_retries,
                help="Retry attempts on API errors"
            )
        
        st.divider()
        
        # ===== STATUS PANEL =====
        st.header("📊 Status")
        
        # API Status
        if os.getenv("OPENAI_API_KEY"):
            st.success("🔑 API Key: Loaded")
        else:
            st.error("🔑 API Key: Not set")
        
        # Index Status
        if st.session_state.rag_pipeline:
            st.success(f"📦 FAISS: {st.session_state.chunks_count} vectors")
        else:
            st.warning("📦 Index: Not loaded")
            if st.button("🔄 Load Pipeline"):
                load_pipeline()
                st.rerun()
        
        # Document Stats
        with st.expander("📄 Documents", expanded=True):
            stats = get_document_stats()
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
                st.caption(f"⚠️ OCR fail: {ocr_rate:.1f}% | 👁️ Vision: {vision_rate:.1f}%")
            else:
                st.info("No documents ingested")
        
        # Eval Results
        with st.expander("📈 Last Eval", expanded=False):
            metrics = get_last_eval_metrics()
            if metrics:
                st.caption(f"Run: {metrics['run']}")
                st.write(f"Faithfulness: {metrics.get('faithfulness', 0):.1%}")
                st.write(f"Relevancy: {metrics.get('relevancy', 0):.1%}")
                st.write(f"Context Precision: {metrics.get('context_precision', 0):.1%}")
                st.write(f"Context Recall: {metrics.get('context_recall', 0):.1%}")
                if "no_answer_accuracy" in metrics:
                    st.write(f"No-Answer Acc: {metrics['no_answer_accuracy']:.1%}")
            else:
                st.info("No evaluation runs")
        
        # Warnings
        with st.expander("⚠️ Warnings", expanded=False):
            warnings = []
            if not os.getenv("OPENAI_API_KEY"):
                warnings.append("❌ API key not set")
            if not os.path.exists("data/chunks.jsonl"):
                warnings.append("📭 No PDFs ingested")
            if not os.path.exists("data/index/faiss.index"):
                warnings.append("📦 Index not found")
            
            if warnings:
                for w in warnings:
                    st.warning(w)
            else:
                st.success("✅ All systems OK")
        
        st.divider()
        
        # Upload
        st.subheader("📥 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files and st.button("🚀 Ingest"):
            temp_dir = tempfile.mkdtemp(prefix="xpanceo_")
            for file in uploaded_files:
                with open(os.path.join(temp_dir, file.name), "wb") as f:
                    f.write(file.getvalue())
            
            with st.spinner("Ingesting..."):
                results = run_ingestion(temp_dir)
                st.session_state.ingestion_results = results
            
            if results:
                st.success(f"✅ {len(results)} docs")
                load_pipeline()
                st.rerun()
        
        # Last ingestion
        if st.session_state.ingestion_results:
            with st.expander("📋 Last Ingest"):
                for entry in st.session_state.ingestion_results:
                    st.write(f"**{entry.filename}**: {entry.pages}p, {entry.chunks.text}t/{entry.chunks.table}tb/{entry.chunks.image_ocr}img")


def main():
    init_session_state()
    
    # Header
    st.title("🔮 XPANCEO DB")
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
