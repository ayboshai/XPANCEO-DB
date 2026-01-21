"""
XPANCEO DB - Streamlit UI
Modern chat interface for PDF Q&A with source visualization.
"""

import os
import sys
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
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None
    if "chunks_count" not in st.session_state:
        st.session_state.chunks_count = 0
    if "ingestion_results" not in st.session_state:
        st.session_state.ingestion_results = []


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


def format_source_card(source):
    """Format source as HTML card."""
    return f"""
    <div class="source-card">
        <strong>📄 {source.doc_id}</strong> | Page {source.page} | {source.type}<br>
        <small>Score: {source.score:.3f}</small><br>
        <em>{source.preview}</em>
    </div>
    """


def main():
    init_session_state()
    
    # Header
    st.title("🔮 XPANCEO DB")
    st.markdown("*Multimodal RAG for Technical PDFs*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Status
        st.subheader("📊 Status")
        if st.session_state.rag_pipeline:
            st.success(f"✅ Loaded: {st.session_state.chunks_count} chunks")
        else:
            st.warning("⚠️ Pipeline not loaded")
            if st.button("🔄 Load Pipeline"):
                load_pipeline()
                st.rerun()
        
        st.divider()
        
        # Ingestion
        st.subheader("📥 Upload & Ingest")
        
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )
        
        if uploaded_files:
            if st.button("🚀 Ingest Uploaded PDFs"):
                # Save to temp folder
                temp_dir = tempfile.mkdtemp(prefix="xpanceo_upload_")
                for file in uploaded_files:
                    with open(os.path.join(temp_dir, file.name), "wb") as f:
                        f.write(file.getvalue())
                
                # Run ingestion
                with st.spinner("Ingesting PDFs..."):
                    results = run_ingestion(temp_dir)
                    st.session_state.ingestion_results = results
                
                if results:
                    st.success(f"✅ Ingested {len(results)} documents")
                    # Reload pipeline
                    load_pipeline()
                    st.rerun()
        
        st.divider()
        
        # Ingestion results
        if st.session_state.ingestion_results:
            st.subheader("📋 Last Ingestion")
            for entry in st.session_state.ingestion_results:
                with st.expander(f"📄 {entry.filename}"):
                    st.write(f"Pages: {entry.pages}")
                    st.write(f"Text chunks: {entry.chunks.text}")
                    st.write(f"Table chunks: {entry.chunks.table}")
                    st.write(f"Image OCR: {entry.chunks.image_ocr}")
                    st.write(f"Image caption: {entry.chunks.image_caption}")
                    if entry.ocr_failure_rate > 0:
                        st.write(f"OCR failure rate: {entry.ocr_failure_rate:.1%}")
    
    # Main chat area
    st.divider()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.markdown(format_source_card(source), unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            if not st.session_state.rag_pipeline:
                st.error("Please load the pipeline first (sidebar)")
                response_text = "Pipeline not loaded. Please click 'Load Pipeline' in the sidebar."
                sources = []
            else:
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_pipeline.query(prompt)
                    response_text = response.answer
                    sources = response.sources
                    
                    # Status indicator
                    if response.has_answer:
                        st.markdown('<span class="status-found">✅ Answer found</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="status-not-found">❌ No answer in documents</span>', unsafe_allow_html=True)
            
            st.markdown(response_text)
            
            # Sources
            if sources:
                with st.expander("📚 Sources"):
                    for source in sources:
                        st.markdown(format_source_card(source), unsafe_allow_html=True)
        
        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "sources": [{"doc_id": s.doc_id, "page": s.page, "type": s.type, "score": s.score, "preview": s.preview} for s in sources] if sources else [],
        })


if __name__ == "__main__":
    main()
