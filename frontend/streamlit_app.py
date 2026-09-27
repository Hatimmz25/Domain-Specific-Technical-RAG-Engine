import html
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import streamlit as st

# Ensure project root is importable when Streamlit runs this file directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="Technical AI Assistant",
    page_icon="TA",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# Brand mark: a plain document outline. Inline SVG, no external assets.
BRAND_ICON = (
    '<svg viewBox="0 0 24 24" width="100%" height="100%" fill="none" stroke="currentColor" '
    'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/>'
    '<path d="M14 3v5h5"/><path d="M9 13h6M9 17h4"/></svg>'
)


ASSISTANT_META = (
    f'<div class="ta-assistant-meta"><span class="ta-assistant-mark">{BRAND_ICON}</span>'
    "<span>Technical AI</span></div>"
)


def esc(value: Any) -> str:
    """Escape backend-provided text before placing it inside raw HTML."""
    return html.escape(str(value), quote=True)

# -----------------------------------------------------------------------------
# Design system
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
:root {
    --ta-bg: #0b0c0f;
    --ta-sidebar: #090a0d;
    --ta-surface: #111318;
    --ta-surface-2: #15171d;
    --ta-surface-3: #1a1d24;
    --ta-border: #252932;
    --ta-border-strong: #343944;
    --ta-text: #f4f4f5;
    --ta-text-soft: #a1a6b0;
    --ta-text-faint: #686e79;
    --ta-accent: #f4f4f5;
    --ta-success: #4ade80;
    --ta-danger: #f87171;
    --ta-font: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    --ta-mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

/* ---------- Streamlit chrome ---------- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
    background: var(--ta-bg) !important;
    color: var(--ta-text) !important;
}
body, .stApp { font-family: var(--ta-font); }
.stApp *:not([data-testid="stIconMaterial"]):not([translate="no"]):not(code):not(pre):not(pre *) {
    font-family: var(--ta-font);
}
/* Icons are a font (ligatures). Overriding it prints icon names as text. */
.stApp [data-testid="stIconMaterial"],
.stApp [translate="no"]:not(code):not(pre) {
    font-family: "Material Symbols Rounded" !important;
}
.stApp code, .stApp pre, .stApp pre *, .stApp kbd, .stApp .ta-source-meta {
    font-family: var(--ta-mono) !important;
}
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
/* The header stays: Streamlit renders the "open sidebar" button inside it.
   Only the deploy button and main menu are hidden. */
header[data-testid="stHeader"] {
    background: var(--ta-bg) !important;
    height: 48px !important;
}
[data-testid="stToolbarActions"],
[data-testid="stMainMenu"],
[data-testid="stAppDeployButton"] { display: none !important; }
[data-testid="stExpandSidebarButton"] {
    border-radius: 8px !important;
    color: var(--ta-text-soft) !important;
}
[data-testid="stExpandSidebarButton"] * { color: #c4c8d0 !important; }
[data-testid="stExpandSidebarButton"]:hover {
    background: var(--ta-surface-3) !important;
    color: var(--ta-text) !important;
}

/* ---------- Theme independence ----------
   Streamlit follows the OS light/dark setting; this UI is dark-only, so every
   surface and text color it depends on is set explicitly. */
.stApp { color: var(--ta-text); }
.block-container [data-testid="stMarkdownContainer"] { color: var(--ta-text); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #8a909b !important; }
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] { background: var(--ta-bg) !important; }

/* ---------- Main canvas ---------- */
[data-testid="stAppViewContainer"] {
    background: var(--ta-bg) !important;
}
[data-testid="stAppViewContainer"] .block-container {
    max-width: 940px;
    padding: 64px 28px 130px;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--ta-sidebar) !important;
    border-right: 1px solid #1c1f25 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 28px 18px 20px !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: var(--ta-text-soft);
}
.ta-brand {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 2px 6px 24px;
}
.ta-brand-mark {
    width: 30px;
    height: 30px;
    border: 1px solid #30343d;
    border-radius: 8px;
    display: grid;
    place-items: center;
    background: #111318;
    color: #f5f5f5;
    padding: 7px;
}
.ta-brand-name { color: #f5f5f5; font-size: 14px; font-weight: 650; }
.ta-brand-sub { color: #737984; font-size: 11px; margin-top: 2px; }
.ta-section-label {
    color: #5f6570;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin: 24px 7px 8px;
}
.ta-sidebar-note {
    color: #777d88 !important;
    font-size: 11px;
    line-height: 1.5;
    padding: 5px 7px;
}
.ta-status {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #a1a6b0;
    font-size: 11px;
    padding: 7px;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { padding: 0 7px; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { font-size: 11px !important; }
.ta-status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--ta-success);
    box-shadow: 0 0 0 3px rgba(74,222,128,.08);
}
.ta-status-dot.offline { background: var(--ta-danger); box-shadow: none; }

/* ---------- Sidebar buttons ---------- */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    min-height: 39px;
    border-radius: 8px;
    border: 1px solid transparent;
    background: transparent;
    color: #b8bdc7;
    text-align: left;
    padding: 0 12px;
    font-size: 12px;
    box-shadow: none;
}
[data-testid="stSidebar"] .stButton > button { justify-content: flex-start; }
[data-testid="stSidebar"] .stButton > button > div { justify-content: flex-start; width: 100%; }
[data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] .stButton > button span { color: inherit; text-align: left; }
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[kind="primary"] > div { justify-content: center; }
[data-testid="stSidebar"] .stButton > button:hover {
    background: #111318;
    border-color: #20232a;
    color: #f4f4f5;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #f2f2f3;
    color: #101114;
    border-color: #f2f2f3;
    font-weight: 600;
    text-align: center;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: #ffffff;
    border-color: #ffffff;
}

/* ---------- Header ---------- */
.ta-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 42px;
    margin-bottom: 24px;
    border-bottom: 1px solid #1d2026;
}
.ta-header-title { color: #e9eaec; font-size: 13px; font-weight: 600; }
.ta-header-sub { color: #666c76; font-size: 10px; margin-top: 2px; }
.ta-header-status { color: #686e79; font-size: 10px; }

/* ---------- Empty state ---------- */
.ta-empty {
    max-width: 700px;
    margin: clamp(0px, 7vh, 64px) auto 0;
    text-align: center;
}
.ta-empty h1 {
    margin: 0;
    color: #f3f4f6;
    font-size: 32px;
    line-height: 1.15;
    letter-spacing: -.04em;
    font-weight: 650;
}
.ta-empty p {
    margin: 13px auto 34px;
    max-width: 570px;
    color: #858b96;
    font-size: 13px;
    line-height: 1.65;
}
.ta-suggestion-label {
    color: #5f6570;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin: 0 0 8px 1px;
}

/* ---------- Suggestion buttons ---------- */
.block-container .stButton > button p, .block-container .stButton > button span { color: inherit; }
.block-container .stButton > button {
    min-height: 42px;
    border-radius: 9px;
    border: 1px solid #242830;
    background: #101216;
    color: #c8ccd3;
    font-size: 11px;
    box-shadow: none;
    transition: background .12s ease, border-color .12s ease, color .12s ease;
}
.block-container .stButton > button:hover {
    background: #161920;
    border-color: #3a3f49;
    color: #ffffff;
}

/* ---------- Chat ---------- */
/* Streamlit's built-in avatars (robot / face) are hidden; the assistant is
   identified by the brand mark in .ta-assistant-meta instead. */
[data-testid^="stChatMessageAvatar"] { display: none !important; }
[data-testid="stChatMessage"] {
    border: 0 !important;
    background: transparent !important;
    padding: 12px 0 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    color: #dedfe2;
    font-size: 14px;
    line-height: 1.72;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    color: #dedfe2;
}
/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    margin-left: auto;
    width: fit-content;
    max-width: 76%;
    background: #17191f !important;
    border: 1px solid #242830 !important;
    border-radius: 14px !important;
    padding: 3px 15px !important;
}
/* Assistant stays open and document-like */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    max-width: 100%;
}
.ta-assistant-meta {
    display: flex;
    align-items: center;
    gap: 7px;
    color: #707681;
    font-size: 10px;
    margin-bottom: 7px;
}
.ta-assistant-mark {
    width: 20px;
    height: 20px;
    border: 1px solid #30343c;
    border-radius: 6px;
    display: inline-grid;
    place-items: center;
    color: #c9cbd0;
    background: #111318;
    padding: 4px;
}

/* Markdown/code */
[data-testid="stChatMessage"] pre {
    background: #0a0b0e !important;
    border: 1px solid #252932 !important;
    border-radius: 9px !important;
}
[data-testid="stChatMessage"] code {
    color: #d7dae0 !important;
}
[data-testid="stChatMessage"] table {
    border-color: #2a2e36 !important;
}
[data-testid="stChatMessage"] th,
[data-testid="stChatMessage"] td {
    border-color: #2a2e36 !important;
}

/* ---------- Expanding technical information ---------- */
[data-testid="stExpander"] {
    border: 1px solid #23262d !important;
    border-radius: 9px !important;
    background: #0f1115 !important;
}
[data-testid="stExpander"] details, [data-testid="stExpander"] summary { background: transparent !important; }
[data-testid="stExpander"] summary:hover { background: var(--ta-surface-2) !important; }
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span {
    color: #969ca6 !important;
    font-size: 11px !important;
}
[data-testid="stExpander"] summary [data-testid="stIconMaterial"] { font-size: 18px !important; }
.ta-source-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #20232a;
}
.ta-source-row:last-child { border-bottom: 0; }
.ta-source-file { color: #e3e4e7; font-size: 12px; font-weight: 550; overflow-wrap: anywhere; }
.ta-source-meta { color: #676d77; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 9px; }
.ta-kicker { color: #626873; font-size: 9px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; margin: 8px 0 6px; }
.ta-stat {
    border: 1px solid #252932;
    border-radius: 9px;
    padding: 12px;
    background: #111318;
}
.ta-stat-value { color: #f1f2f3; font-size: 17px; font-weight: 600; letter-spacing: -.02em; }
.ta-stat-label { color: #707681; font-size: 9px; margin-top: 2px; }
.ta-doc-item {
    border-bottom: 1px solid #20232a;
    padding: 14px 2px;
}
.ta-doc-name { color: #e3e4e7; font-size: 12px; font-weight: 550; }
.ta-doc-meta { color: #6e747e; font-size: 10px; margin-top: 3px; }

/* ---------- Chat composer ---------- */
[data-testid="stChatInput"] {
    background: transparent !important;
    border-top: 0 !important;
}
[data-testid="stChatInput"] > div {
    background: #111318 !important;
    border: 1px solid #30343d !important;
    border-radius: 15px !important;
    box-shadow: 0 10px 35px rgba(0,0,0,.28) !important;
}
[data-testid="stChatInput"] textarea {
    color: #eceef1 !important;
    caret-color: #ffffff !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #626873 !important; }
[data-testid="stChatInput"] button {
    background: transparent !important;
    color: var(--ta-text-soft) !important;
    border: 0 !important;
}
[data-testid="stChatInput"] button:hover { background: var(--ta-surface-3) !important; color: var(--ta-text) !important; }
[data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"] {
    background: #e8e9eb !important;
    color: #101114 !important;
}
[data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"]:hover { background: #ffffff !important; color: #101114 !important; }
[data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"]:disabled { opacity: .35; }
[data-testid="stChatInput"] button:focus-visible { outline: 2px solid #8a93a6; outline-offset: 2px; }

/* ---------- File uploader / forms ---------- */
[data-testid="stFileUploader"] section {
    background: #101216 !important;
    border: 1px dashed #343943 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] section:hover { border-color: #4a505b !important; }
[data-testid="stFileUploader"] button {
    background: #181b21 !important;
    color: #d9dbe0 !important;
    border: 1px solid #30343c !important;
}
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
    background: #111318 !important;
    color: #f1f2f3 !important;
    border-color: #2d3139 !important;
}

/* ---------- Alerts ---------- */
[data-testid="stAlert"] {
    background: #111318 !important;
    border: 1px solid #2b2f37 !important;
    color: #c7cad0 !important;
}

/* ---------- Secondary pages ---------- */
.block-container h2, .block-container h3 { color: #f1f2f3 !important; }

.block-container hr { border-color: #20232a !important; }

@media (max-width: 700px) {
    [data-testid="stAppViewContainer"] .block-container { padding: 64px 16px 120px; }
    .ta-empty { margin-top: 9vh; }
    .ta-empty h1 { font-size: 26px; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { max-width: 88%; }
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Backend API helpers
# -----------------------------------------------------------------------------
@st.cache_data(ttl=5)
def fetch_api_health() -> Optional[Dict[str, Any]]:
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=2.5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=30)
def fetch_system_stats() -> Optional[Dict[str, Any]]:
    try:
        response = httpx.get(f"{API_BASE_URL}/stats", timeout=3.0)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def submit_query_api(question: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        response = httpx.post(
            f"{API_BASE_URL}/query",
            json={"question": question},
            timeout=120.0,
        )
        if response.status_code == 200:
            return response.json(), None
        return None, f"API Error ({response.status_code}): {response.text}"
    except Exception:
        return None, "The assistant is temporarily unavailable. Please make sure the FastAPI server is running."


def upload_document_api(uploaded_file) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = httpx.post(f"{API_BASE_URL}/upload", files=files, timeout=90.0)
        if response.status_code == 200:
            return response.json(), None
        return None, f"Upload failed: {response.text}"
    except Exception as exc:
        return None, f"Connection error during file upload: {exc}"


ALLOWED_UPLOAD_TYPES = ["pdf", "md", "txt", "markdown"]


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_view" not in st.session_state:
    st.session_state.active_view = "chat"
if "session_documents" not in st.session_state:
    st.session_state.session_documents = {}


def clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.active_view = "chat"


def set_view(view: str) -> None:
    st.session_state.active_view = view


# -----------------------------------------------------------------------------
# UI components
# -----------------------------------------------------------------------------
def render_sidebar(health_data: Optional[Dict[str, Any]]) -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="ta-brand">
                <div class="ta-brand-mark">{BRAND_ICON}</div>
                <div>
                    <div class="ta-brand-name">Technical AI</div>
                    <div class="ta-brand-sub">Knowledge assistant</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("＋  New conversation", use_container_width=True, type="primary"):
            clear_conversation()
            st.rerun()

        st.markdown('<div class="ta-section-label">Workspace</div>', unsafe_allow_html=True)

        if st.button("⌂  Chat", use_container_width=True):
            set_view("chat")
            st.rerun()
        if st.button("▣  Knowledge", use_container_width=True):
            set_view("knowledge")
            st.rerun()
        if st.button("⚙  Settings", use_container_width=True):
            set_view("settings")
            st.rerun()

        st.markdown('<div class="ta-section-label">Conversation</div>', unsafe_allow_html=True)
        if st.session_state.messages:
            first_user = next(
                (m["content"] for m in st.session_state.messages if m.get("role") == "user"),
                "Current conversation",
            )
            label = first_user.replace("\n", " ").strip()
            if len(label) > 34:
                label = label[:34].rstrip() + "…"
            st.markdown(f'<div class="ta-sidebar-note">{esc(label)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ta-sidebar-note">No active conversation</div>', unsafe_allow_html=True)

        st.markdown('<div class="ta-section-label">System</div>', unsafe_allow_html=True)
        online = health_data is not None
        dot_class = "" if online else "offline"
        label = "System operational" if online else "Backend unavailable"
        st.markdown(
            f'<div class="ta-status"><span class="ta-status-dot {dot_class}"></span>{label}</div>',
            unsafe_allow_html=True,
        )

        if health_data:
            indexed = health_data.get("indexed_vectors", 0)
            st.caption(f"{indexed:,} indexed chunks")

        st.markdown(
            '<div class="ta-sidebar-note" style="margin-top:20px;">Session history is kept locally in this browser session.</div>',
            unsafe_allow_html=True,
        )


def render_header(health_data: Optional[Dict[str, Any]]) -> None:
    status = "Operational" if health_data else "API unavailable"
    st.markdown(
        f"""
        <div class="ta-header">
            <div>
                <div class="ta-header-title">Technical AI</div>
                <div class="ta-header-sub">Knowledge assistant</div>
            </div>
            <div class="ta-header-status">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> Optional[str]:
    st.markdown(
        """
        <div class="ta-empty">
            <h1>How can I help?</h1>
            <p>Ask questions about your indexed technical documentation. Answers are grounded in the retrieved knowledge base.</p>
        </div>
        <div class="ta-suggestion-label">Try a question</div>
        """,
        unsafe_allow_html=True,
    )

    suggestions = [
        "How do I create a POST endpoint in FastAPI?",
        "How does the retrieval pipeline work?",
        "Explain the document ingestion process",
        "What is the difference between Docker COPY and ADD?",
    ]

    selected = None
    cols = st.columns(2)
    for index, suggestion in enumerate(suggestions):
        with cols[index % 2]:
            if st.button(suggestion, use_container_width=True, key=f"suggestion_{index}"):
                selected = suggestion
    return selected


def render_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        return

    with st.expander(f"Sources · {len(sources)}", expanded=False):
        for source in sources:
            file_name = source.get("file_name") or source.get("source") or "Document"
            chunk_id = source.get("chunk_id", "chunk")
            score = source.get("score")
            score_text = f"Relevance {float(score):.4f}" if score is not None else "Relevance unavailable"
            st.markdown(
                f"""
                <div class="ta-source-row">
                    <div style="min-width:0; flex:1;">
                        <div class="ta-source-file">{esc(file_name)}</div>
                        <div class="ta-source-meta">{esc(chunk_id)}</div>
                    </div>
                    <div class="ta-source-meta">{esc(score_text)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_run_details(
    question: str,
    latency: Dict[str, Any],
    retrieved_context: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    stats_data: Optional[Dict[str, Any]],
) -> None:
    with st.expander("Run details", expanded=False):
        retrieval = latency.get("retrieval", 0)
        reranking = latency.get("reranking", 0)
        generation = latency.get("generation", 0)
        total = latency.get("total", 0)
        candidates = stats_data.get("retrieval_top_k", "—") if stats_data else "—"
        selected = len(retrieved_context) if retrieved_context else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="ta-stat"><div class="ta-stat-value">{float(retrieval):.1f}</div><div class="ta-stat-label">Retrieval · ms</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="ta-stat"><div class="ta-stat-value">{float(reranking):.1f}</div><div class="ta-stat-label">Reranking · ms</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="ta-stat"><div class="ta-stat-value">{float(generation):.1f}</div><div class="ta-stat-label">Generation · ms</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="ta-stat"><div class="ta-stat-value">{float(total):.1f}</div><div class="ta-stat-label">Total · ms</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        a, b = st.columns(2)
        a.caption(f"Retrieved candidates: {candidates}")
        b.caption(f"Selected context chunks: {selected}")

        if retrieved_context:
            st.markdown('<div class="ta-kicker">Selected context</div>', unsafe_allow_html=True)
            for index, chunk in enumerate(retrieved_context, start=1):
                chunk_id = chunk.get("chunk_id", f"chunk_{index}")
                score = chunk.get("score")
                score_text = f" · score {float(score):.4f}" if score is not None else ""
                with st.expander(f"Chunk {index} · {chunk_id}{score_text}", expanded=False):
                    st.code(chunk.get("content", ""), language="text")


def render_assistant_message(msg: Dict[str, Any], stats_data: Optional[Dict[str, Any]]) -> None:
    with st.chat_message("assistant"):
        st.markdown(
            ASSISTANT_META,
            unsafe_allow_html=True,
        )
        st.markdown(msg.get("content", ""))
        sources = msg.get("sources") or []
        if sources:
            render_sources(sources)
        if msg.get("latency") is not None:
            render_run_details(
                msg.get("question", ""),
                msg.get("latency") or {},
                msg.get("context") or [],
                sources,
                stats_data,
            )


def render_composer() -> Tuple[str, List[Any]]:
    """Chat composer. Returns (text, attached_files); the attach button feeds /upload."""
    placeholder = "Ask about your documentation…"
    try:
        value = st.chat_input(placeholder, accept_file="multiple", file_type=ALLOWED_UPLOAD_TYPES)
    except TypeError:  # older Streamlit without attachment support
        value = st.chat_input(placeholder)
    if value is None:
        return "", []
    if isinstance(value, str):
        return value.strip(), []
    return (value.get("text") or "").strip(), list(value.get("files") or [])


def handle_attachments(files: List[Any]) -> None:
    """Send each attached file to the existing upload endpoint and record the outcome."""
    for uploaded in files:
        size = f"{uploaded.size} B" if uploaded.size < 1024 else f"{uploaded.size / 1024:.0f} KB"
        note = f"Uploaded {uploaded.name} · {size}"
        st.session_state.messages.append({"role": "user", "content": note})
        with st.chat_message("user"):
            st.markdown(note)

        with st.chat_message("assistant"):
            st.markdown(ASSISTANT_META, unsafe_allow_html=True)
            with st.spinner(f"Indexing {uploaded.name}…"):
                result, error = upload_document_api(uploaded)

            if result:
                filename = result.get("filename") or uploaded.name
                chunks = result.get("chunks_created")
                st.session_state.session_documents[filename] = chunks
                st.cache_data.clear()  # refresh indexed-chunk counts
                detail = f" ({chunks} chunks)" if chunks is not None else ""
                reply = f"**{filename}** is indexed{detail}. You can ask questions about it now."
                st.markdown(reply)
            else:
                reply = f"Document indexing failed for **{uploaded.name}**. Check the file type and that the API is running, then try again."
                st.markdown(reply)
                with st.expander("Technical details", expanded=False):
                    st.code(error or "Unknown error")
            st.session_state.messages.append({"role": "assistant", "content": reply})


def render_chat_view(
    health_data: Optional[Dict[str, Any]],
    stats_data: Optional[Dict[str, Any]],
) -> None:
    # Read the composer first so the welcome screen is skipped as soon as
    # something is submitted (otherwise it would linger above the new messages).
    typed_text, attachments = render_composer()
    prompt_from_welcome = None

    if not st.session_state.messages and not (typed_text or attachments):
        prompt_from_welcome = render_empty_state()
    else:
        for msg in st.session_state.messages:
            if msg.get("role") == "user":
                with st.chat_message("user"):
                    st.markdown(msg.get("content", ""))
            else:
                render_assistant_message(msg, stats_data)

    active_prompt = typed_text or prompt_from_welcome

    if not active_prompt and not attachments:
        return

    if not health_data:
        st.error("The assistant couldn't connect to the API. Please make sure the FastAPI server is running.")
        if st.button("Try again", key="retry_connection"):
            st.cache_data.clear()
            st.rerun()
        return

    if attachments:
        handle_attachments(attachments)
    if not active_prompt:
        return

    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)

    with st.chat_message("assistant"):
        st.markdown(
            ASSISTANT_META,
            unsafe_allow_html=True,
        )
        with st.spinner("Searching documentation and generating an answer…"):
            response_data, error_msg = submit_query_api(active_prompt)

        if error_msg:
            st.error("Something went wrong while generating the answer.")
            with st.expander("Technical details", expanded=False):
                st.code(error_msg)
            return

        if not response_data:
            st.error("The API returned an empty response.")
            return

        answer_text = response_data.get("answer", "")
        sources = response_data.get("sources", []) or []
        latency = response_data.get("latency_ms", {}) or {}
        context = response_data.get("retrieved_context", []) or []

        if not sources and "couldn't find sufficient information" in answer_text.lower():
            answer_text = (
                "I couldn't find relevant information about this in your indexed documentation.\n\n"
                "Try asking a question directly related to the documents in your knowledge base."
            )

        st.markdown(answer_text)
        render_sources(sources)
        render_run_details(active_prompt, latency, context, sources, stats_data)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer_text,
                "question": active_prompt,
                "sources": sources,
                "latency": latency,
                "context": context,
            }
        )

    if prompt_from_welcome:
        st.rerun()


def render_knowledge_view(health_data: Optional[Dict[str, Any]]) -> None:
    st.markdown("## Knowledge")
    st.caption("Manage the documents available to the RAG pipeline.")

    indexed_count = health_data.get("indexed_vectors", 0) if health_data else 0
    st.markdown(
        f"""
        <div class="ta-stat" style="margin: 14px 0 22px 0;">
            <div class="ta-stat-value">{indexed_count:,}</div>
            <div class="ta-stat-label">Indexed context chunks</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Add documentation")
    st.caption("Upload PDF, Markdown, or TXT files. The backend handles ingestion, chunking, embeddings, and indexing.")
    uploaded_file = st.file_uploader(
        "Drop a document here",
        type=["pdf", "md", "txt", "markdown"],
        label_visibility="collapsed",
        key="knowledge_uploader",
    )

    if uploaded_file is not None:
        st.markdown(f"**{uploaded_file.name}** · {uploaded_file.size / 1024:.0f} KB")
        if st.button("Upload and index", type="primary", use_container_width=False):
            with st.spinner("Uploading and indexing document…"):
                result, error = upload_document_api(uploaded_file)
            if result:
                filename = result.get("filename") or uploaded_file.name
                chunks = result.get("chunks_created")
                st.session_state.session_documents[filename] = chunks
                st.cache_data.clear()
                if chunks is not None:
                    st.success(f"{filename} indexed successfully · {chunks} chunks created.")
                else:
                    st.success(f"{filename} indexed successfully.")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(error or "Document indexing failed.")

    st.markdown("### Documents referenced in this session")
    if st.session_state.session_documents:
        for filename, chunks in st.session_state.session_documents.items():
            chunk_text = f"{chunks} chunks indexed" if chunks is not None else "Indexed"
            st.markdown(
                f'<div class="ta-doc-item"><div class="ta-doc-name">{esc(filename)}</div><div class="ta-doc-meta">{esc(chunk_text)}</div></div>',
                unsafe_allow_html=True,
            )
    else:
        source_names = []
        for message in st.session_state.messages:
            for source in message.get("sources", []) or []:
                name = source.get("file_name") or source.get("source")
                if name and name not in source_names:
                    source_names.append(name)
        if source_names:
            for name in source_names:
                st.markdown(
                    f'<div class="ta-doc-item"><div class="ta-doc-name">{esc(name)}</div><div class="ta-doc-meta">Referenced by this conversation</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No document has been uploaded or referenced in this session yet.")

    st.markdown("### How it works")
    st.caption("Documents are processed by the existing backend ingestion pipeline. The frontend does not duplicate indexing logic or maintain a second knowledge store.")


def render_settings_view(stats_data: Optional[Dict[str, Any]], health_data: Optional[Dict[str, Any]]) -> None:
    st.markdown("## Settings")
    st.caption("Read-only configuration exposed by the RAG backend.")

    if not stats_data:
        st.warning("System configuration is currently unavailable.")
    else:
        st.markdown("### Model configuration")
        values = [
            ("LLM model", stats_data.get("llm_model", "—")),
            ("Embedding model", stats_data.get("embedding_model", "—")),
            ("Reranker model", stats_data.get("reranker_model", "—")),
            ("Retrieval Top-K", stats_data.get("retrieval_top_k", "—")),
            ("Rerank Top-K", stats_data.get("rerank_top_k", "—")),
        ]
        for label, value in values:
            st.markdown(
                f"**{label}**  \n`{value}`",
            )
            st.divider()

    st.markdown("### System status")
    if health_data:
        st.success("API and RAG health endpoint are reachable.")
        st.caption(f"Indexed vectors: {health_data.get('indexed_vectors', 0):,}")
        subsystem_data = health_data.get("subsystems") or health_data.get("components")
        if isinstance(subsystem_data, dict):
            for name, state in subsystem_data.items():
                st.write(f"**{name}** — {state}")
    else:
        st.error("The FastAPI backend is unavailable.")

    st.markdown("### About")
    st.caption("Technical AI Assistant · RAG knowledge interface")
    st.caption(f"API endpoint: {API_BASE_URL}")


def main() -> None:
    health_data = fetch_api_health()
    stats_data = fetch_system_stats()

    render_sidebar(health_data)
    render_header(health_data)

    if st.session_state.active_view == "knowledge":
        render_knowledge_view(health_data)
    elif st.session_state.active_view == "settings":
        render_settings_view(stats_data, health_data)
    else:
        render_chat_view(health_data, stats_data)


if __name__ == "__main__":
    main()