import os
import sys
import time
from pathlib import Path
import httpx
import streamlit as st

# Ensure project root is in Python path for package imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ==============================================================================
# 1. STREAMLIT CONFIGURATION & CUSTOM THEMED CSS INJECTION
# ==============================================================================
st.set_page_config(
    page_title="Technical AI Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS targeting modern AI-chat layout with dark/light mode adaptivity
st.markdown("""
<style>
    /* Main Content Container Alignment */
    .main .block-container {
        max-width: 880px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 4rem !important;
    }

    /* Top Brand Navigation Header */
    .brand-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 0.75rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .brand-title {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
        color: #F3F4F6 !important;
        margin: 0 !important;
    }
    .brand-subtitle {
        font-size: 0.85rem !important;
        color: #9CA3AF !important;
        margin-top: 2px !important;
    }

    /* Status Badges */
    .badge-online {
        font-size: 0.75rem;
        font-weight: 500;
        color: #34D399;
        background-color: rgba(52, 211, 153, 0.1);
        padding: 3px 10px;
        border-radius: 12px;
        border: 1px solid rgba(52, 211, 153, 0.25);
    }
    .badge-offline {
        font-size: 0.75rem;
        font-weight: 500;
        color: #F87171;
        background-color: rgba(248, 113, 113, 0.1);
        padding: 3px 10px;
        border-radius: 12px;
        border: 1px solid rgba(248, 113, 113, 0.25);
    }

    /* Source Citation Pill Badges */
    .source-pill {
        display: inline-flex;
        align-items: center;
        font-size: 0.78rem;
        font-family: monospace;
        padding: 3px 8px;
        margin-right: 6px;
        margin-bottom: 6px;
        border-radius: 6px;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #D1D5DB;
    }

    /* Suggestion Section Header */
    .suggestion-header {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #9CA3AF !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.5rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Pipeline Flow UI */
    .pipeline-step {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 4px;
        font-size: 0.85rem;
    }
    .pipeline-arrow {
        text-align: center;
        color: #6B7280;
        font-size: 0.8rem;
        margin: 2px 0;
    }

    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. BACKEND API SERVICE HELPERS
# ==============================================================================
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")


@st.cache_data(ttl=5)
def fetch_api_health():
    """Fetches system health status and vector counts from FastAPI backend."""
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=2.5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None


@st.cache_data(ttl=30)
def fetch_system_stats():
    """Fetches system model settings and statistics from FastAPI backend."""
    try:
        response = httpx.get(f"{API_BASE_URL}/stats", timeout=3.0)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None


def submit_query_api(question: str):
    """Submits technical query to FastAPI /query endpoint."""
    try:
        response = httpx.post(
            f"{API_BASE_URL}/query",
            json={"question": question},
            timeout=120.0
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error ({response.status_code}): {response.text}"
    except Exception as e:
        return None, "The assistant is temporarily unavailable. Please make sure the FastAPI server is running."


def upload_document_api(uploaded_file):
    """Sends uploaded file bytes to FastAPI /upload endpoint."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = httpx.post(f"{API_BASE_URL}/upload", files=files, timeout=90.0)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Upload failed: {response.text}"
    except Exception as e:
        return None, f"Connection error during file upload: {str(e)}"


# ==============================================================================
# 3. SESSION STATE MANAGEMENT
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []


def clear_conversation():
    st.session_state.messages = []


# ==============================================================================
# 4. COMPONENT RENDERING FUNCTIONS
# ==============================================================================
def render_header(health_data):
    """Renders compact top brand header with status badge."""
    col1, col2 = st.columns([0.8, 0.2])
    
    with col1:
        st.markdown("""
            <div class="brand-header-left">
                <div class="brand-title">✦ Technical AI</div>
                <div class="brand-subtitle">Ask questions, explore concepts, and get answers grounded in your documentation.</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div style='text-align: right; margin-top: 4px;'>", unsafe_allow_html=True)
        if health_data:
            st.markdown('<span class="badge-online">● Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-offline">○ Backend unavailable</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr style='margin-top:0.5rem; margin-bottom:1.5rem; border-color:rgba(128,128,128,0.15);'>", unsafe_allow_html=True)


def render_sidebar(health_data, stats_data):
    """Renders clean Workspace sidebar for document knowledge base."""
    with st.sidebar:
        st.title("Workspace")
        st.caption("Manage documentation & assistant settings.")

        if st.button("＋ New conversation", use_container_width=True):
            clear_conversation()
            st.rerun()

        st.markdown("---")

        # Knowledge Base Overview
        st.subheader("Knowledge base")
        indexed_count = health_data.get("indexed_vectors", 0) if health_data else 0
        st.markdown(f"**{indexed_count}** indexed context chunks available.")

        with st.expander("Add documents", expanded=False):
            st.caption("Upload PDF, Markdown, or TXT files to expand knowledge.")
            uploaded_file = st.file_uploader(
                "Choose file",
                type=["pdf", "md", "txt", "markdown"],
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                if st.button("Process & index", type="primary", use_container_width=True):
                    with st.spinner("Processing document..."):
                        res, err = upload_document_api(uploaded_file)
                        if res:
                            st.success(f"✓ Added `{res.get('filename')}`")
                            st.info(f"Indexed **{res.get('chunks_created')}** new chunk(s).")
                            st.cache_data.clear()
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(err)

        st.markdown("---")

        # System & Model Architecture Info
        with st.expander("About this assistant", expanded=False):
            if stats_data:
                st.markdown(f"**LLM:** `{stats_data.get('llm_model')}`")
                st.markdown(f"**Re-Ranker:** `{stats_data.get('reranker_model')}`")
                st.markdown(f"**Embedding Model:** `{stats_data.get('embedding_model')}`")
                st.markdown(f"**Retrieval Top-K:** `{stats_data.get('retrieval_top_k')}`")
                st.markdown(f"**Re-Rank Top-K:** `{stats_data.get('rerank_top_k')}`")
            else:
                st.caption("System information unavailable.")

        st.caption("Technical AI Assistant v1.0")


def render_welcome_screen():
    """Renders elegant initial landing state when no conversation exists."""
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; padding: 1.5rem 0 2rem 0;'>
            <h2 style='font-weight: 500; font-size: 1.5rem; margin-bottom: 0.4rem;'>What would you like to know?</h2>
            <p style='font-size: 0.9rem; opacity: 0.7;'>Ask questions about your uploaded technical documentation or select a suggestion below.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="suggestion-header">Suggested Questions</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    prompt_selected = None
    with col1:
        if st.button("How do I create a POST endpoint in FastAPI?", use_container_width=True):
            prompt_selected = "How do I create a POST endpoint in FastAPI?"
        if st.button("How do path parameters work in FastAPI?", use_container_width=True):
            prompt_selected = "How do path parameters work in FastAPI?"

    with col2:
        if st.button("What is the difference between Docker COPY and ADD?", use_container_width=True):
            prompt_selected = "What is the difference between Docker COPY and ADD?"
        if st.button("Explain vector similarity search thresholds.", use_container_width=True):
            prompt_selected = "Explain vector similarity search thresholds."

    return prompt_selected


def render_sources(sources):
    """Renders source attribution pills below assistant response."""
    if not sources:
        return

    st.markdown("<div style='font-size:0.8rem; font-weight:600; opacity:0.8; margin-top:0.8rem; margin-bottom:0.3rem;'>Sources</div>", unsafe_allow_html=True)
    pills_html = ""
    for src in sources:
        file_name = src.get("file_name", "Document")
        chunk_id = src.get("chunk_id", "chunk")
        pills_html += f'<span class="source-pill">📄 {file_name} · {chunk_id}</span> '
    
    st.markdown(pills_html, unsafe_allow_html=True)


def render_technical_details(question, latency, retrieved_context, sources, stats_data):
    """Renders expandable advanced RAG execution pipeline view."""
    with st.expander("How this answer was generated", expanded=False):
        # 1. Visual Pipeline Flow
        st.markdown("#### Pipeline Execution Flow")

        # Step 1: Question
        st.markdown(f'<div class="pipeline-step"><b>1. User Question:</b> "{question}"</div>', unsafe_allow_html=True)
        st.markdown('<div class="pipeline-arrow">↓</div>', unsafe_allow_html=True)

        # Step 2: Query Embedding
        emb_model = stats_data.get("embedding_model", "bge-small-en-v1.5") if stats_data else "Embedding Engine"
        st.markdown(f'<div class="pipeline-step"><b>2. Query Embedding:</b> Vectorized query using <code>{emb_model}</code></div>', unsafe_allow_html=True)
        st.markdown('<div class="pipeline-arrow">↓</div>', unsafe_allow_html=True)

        # Step 3 & 4: Dense Retrieval & Candidate Count
        cand_top_k = stats_data.get("retrieval_top_k", 15) if stats_data else 15
        st.markdown(
            f'<div class="pipeline-step"><b>3 & 4. Dense Retrieval:</b> FAISS vector index retrieved <b>{cand_top_k} candidate chunks</b>'
            f' (Latency: <code>{latency.get("retrieval", 0):.1f} ms</code>)</div>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="pipeline-arrow">↓</div>', unsafe_allow_html=True)

        # Step 5: Similarity Filtering
        st.markdown(f'<div class="pipeline-step"><b>5. Similarity Filtering:</b> Filtered candidate chunks by similarity threshold</div>', unsafe_allow_html=True)
        st.markdown('<div class="pipeline-arrow">↓</div>', unsafe_allow_html=True)

        # Step 6: Cross-Encoder Reranking
        rerank_model = stats_data.get("reranker_model", "bge-reranker-base") if stats_data else "Cross-Encoder"
        selected_count = len(retrieved_context) if retrieved_context else 0
        st.markdown(
            f'<div class="pipeline-step"><b>6. Cross-Encoder Reranking:</b> Applied joint attention re-ranking using <code>{rerank_model}</code>'
            f' (Latency: <code>{latency.get("reranking", 0):.1f} ms</code>)</div>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="pipeline-arrow">↓</div>', unsafe_allow_html=True)

        # Step 7: Selected Context
        st.markdown(f'<div class="pipeline-step"><b>7. Selected Context:</b> Passed <b>{selected_count} high-confidence chunk(s)</b> into the prompt template</div>', unsafe_allow_html=True)
        st.markdown('<div class="pipeline-arrow">↓</div>', unsafe_allow_html=True)

        # Step 8: LLM Generation
        llm_model = stats_data.get("llm_model", "Qwen2.5-3B-Instruct") if stats_data else "LLM"
        st.markdown(
            f'<div class="pipeline-step"><b>8. LLM Generation:</b> Synthesized grounded answer using <code>{llm_model}</code>'
            f' (Latency: <code>{latency.get("generation", 0):.1f} ms</code>)</div>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="pipeline-arrow">↓</div>', unsafe_allow_html=True)

        # Step 9: Sources
        source_files = list(set([src.get("file_name") for src in sources])) if sources else []
        sources_str = ", ".join([f"<code>{f}</code>" for f in source_files]) if source_files else "None"
        st.markdown(f'<div class="pipeline-step"><b>9. Sources Attributed:</b> {sources_str}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # 2. Performance & Latency Breakdown
        st.markdown("#### Execution Latency Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Retrieval Latency", f"{latency.get('retrieval', 0):.1f} ms")
        m2.metric("Reranking Latency", f"{latency.get('reranking', 0):.1f} ms")
        m3.metric("Generation Latency", f"{latency.get('generation', 0):.1f} ms")
        m4.metric("Total Latency", f"{latency.get('total', 0):.1f} ms")

        st.markdown("---")

        # 3. Selected Context Chunks with Reranking Scores
        st.markdown("#### Selected Context Chunks")
        if retrieved_context:
            for idx, chunk in enumerate(retrieved_context, start=1):
                st.markdown(
                    f"**Chunk #{idx}** (`{chunk.get('chunk_id')}`) — Score: `{chunk.get('score', 0.0):.4f}`"
                )
                st.code(chunk.get("content", ""), language="text")
        else:
            st.caption("No context chunks passed relevance threshold.")


def render_chat_message(role, content, question=None, sources=None, latency=None, context=None, stats_data=None):
    """Renders individual chat message bubble."""
    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant":
            if sources:
                render_sources(sources)
            if latency and context is not None:
                render_technical_details(question, latency, context, sources, stats_data)


# ==============================================================================
# 5. MAIN CONTROLLER PIPELINE
# ==============================================================================
def main():
    # Fetch status
    health_data = fetch_api_health()
    stats_data = fetch_system_stats()

    # Render Header & Sidebar
    render_header(health_data)
    render_sidebar(health_data, stats_data)

    # Render Active Chat or Landing Screen
    prompt_from_welcome = None
    if not st.session_state.messages:
        prompt_from_welcome = render_welcome_screen()
    else:
        for msg in st.session_state.messages:
            render_chat_message(
                role=msg["role"],
                content=msg["content"],
                question=msg.get("question"),
                sources=msg.get("sources"),
                latency=msg.get("latency"),
                context=msg.get("context"),
                stats_data=stats_data
            )

    # Capture Chat Input
    chat_input = st.chat_input("Ask anything about your documentation...")
    active_prompt = chat_input or prompt_from_welcome

    if active_prompt:
        if not health_data:
            st.error("The assistant backend is currently unavailable. Please ensure the FastAPI server is running.")
            return

        # Append & Render User Question
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        render_chat_message("user", active_prompt)

        # Submit & Render Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Searching your documentation..."):
                response_data, error_msg = submit_query_api(active_prompt)

            if error_msg:
                st.error("Something went wrong while generating the answer.")
                with st.expander("Technical details"):
                    st.code(error_msg)
            elif response_data:
                answer_text = response_data.get("answer", "")
                sources = response_data.get("sources", [])
                latency = response_data.get("latency_ms", {})
                context = response_data.get("retrieved_context", [])

                # Out-of-Domain Guardrail Fallback formatting
                if not sources and "couldn't find sufficient information" in answer_text.lower():
                    answer_text = "I couldn't find relevant information about this in your indexed documentation.\n\nTry asking a question directly related to the documents in your knowledge base."

                st.markdown(answer_text)
                render_sources(sources)
                render_technical_details(active_prompt, latency, context, sources, stats_data)

                # Save message to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "question": active_prompt,
                    "sources": sources,
                    "latency": latency,
                    "context": context
                })

        if prompt_from_welcome:
            st.rerun()


if __name__ == "__main__":
    main()