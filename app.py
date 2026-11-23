import os
import time
import tempfile
from datetime import datetime
import streamlit as st

from data_ingestion import ingest_pdf
from vector_engine import build_vector_store, retrieve_relevant_chunks
from llm_generator import generate_rag_answer
from auth_voice import (
    init_auth_voice_state,
    logout_user,
    render_login_screen,
    render_temp_chat_toggle,
    transcribe_audio_bytes,
)

st.set_page_config(
    page_title="RAGMentor",
    page_icon="📘",
    layout="wide",
)


def _apply_workspace_styles():
    """Apply dark navy sidebar + clean workspace styling."""
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: #0b2545;
            color: #ffffff;
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0;
        }
        [data-testid="stSidebar"] .stButton > button {
            background-color: rgba(255, 255, 255, 0.08);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 8px;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background-color: #2563eb;
            border-color: #2563eb;
            color: #ffffff;
        }
        .doc-banner {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 12px 16px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .metric-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 10px 12px;
            margin-bottom: 8px;
        }
        .fallback-box {
            background: #fef2f2;
            border: 1px solid #fecaca;
            border-radius: 12px;
            padding: 14px;
            margin-top: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_app_state():
    """Initialize session state keys for chat, sessions, documents, and settings."""
    init_auth_voice_state()
    defaults = {
        "messages": [],
        "chat_history": [],
        "active_nav": "Home",
        "doc_summary": "",
        "total_pages": 0,
        "file_size_mb": 0.0,
        "uploaded_at": "",
        "suggested_questions": [],
        "vectorstore": None,
        "uploaded_filename": None,
        "pending_prompt": None,
        "show_upload_panel": False,
        "show_settings_panel": False,
        "top_k": 4,
        "voice_lang": "en-IN",
        "search_query": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _save_current_chat_to_history():
    """Save current messages into chat_history if Temporary Chat is OFF."""
    if st.session_state.temp_chat:
        return
    if not st.session_state.messages:
        return

    first_user_msg = next(
        (m["content"] for m in st.session_state.messages if m["role"] == "user"),
        "Document Session",
    )
    title = first_user_msg[:36] + ("..." if len(first_user_msg) > 36 else "")
    timestamp = datetime.now().strftime("Today, %I:%M %p")

    entry = {
        "title": title,
        "time": timestamp,
        "messages": list(st.session_state.messages),
        "filename": st.session_state.uploaded_filename,
        "doc_summary": st.session_state.doc_summary,
        "total_pages": st.session_state.total_pages,
    }
    st.session_state.chat_history.insert(0, entry)
    st.session_state.chat_history = st.session_state.chat_history[:12]


def _start_new_chat():
    """Archive current chat (if not temporary) and clear active messages."""
    _save_current_chat_to_history()
    st.session_state.messages = []
    st.session_state.voice_question = ""
    st.session_state.active_nav = "Home"


def _save_uploaded_pdf(uploaded_file):
    """Persist uploaded PDF to temporary file path and return path."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def _process_user_query(question_text):
    """Retrieve chunks, generate LLM response, and record response time & sources."""
    cleaned_q = (question_text or "").strip()
    if not cleaned_q:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": cleaned_q,
            "timestamp": datetime.now().strftime("%I:%M %p"),
        }
    )

    start_ts = time.perf_counter()
    vectorstore = st.session_state.vectorstore

    if vectorstore is None:
        elapsed = round(time.perf_counter() - start_ts, 1)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "No document uploaded yet. Upload a PDF to get context-grounded answers.",
                "page_numbers": [],
                "response_time": elapsed,
                "is_fallback": True,
                "options": [
                    "Upload a relevant PDF from My Documents / Upload panel.",
                    "Ask a question once your document is processed.",
                    "Try a shorter PDF if needed.",
                ],
            }
        )
        return

    retrieval = retrieve_relevant_chunks(vectorstore, cleaned_q, k=st.session_state.top_k)
    result = generate_rag_answer(
        query=cleaned_q,
        context_text=retrieval.get("context", ""),
        page_numbers=retrieval.get("page_numbers", []),
        suggested_questions=st.session_state.suggested_questions,
    )
    elapsed = round(time.perf_counter() - start_ts, 1)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.get("answer", ""),
            "page_numbers": result.get("page_numbers", []),
            "response_time": elapsed,
            "is_fallback": result.get("is_fallback", False),
            "options": result.get("options", []),
        }
    )


def _render_sidebar():
    """Render left navy sidebar with navigation, temporary chat, recent chats, and logout."""
    with st.sidebar:
        st.markdown("### RAGMentor")
        st.caption("Learn Smarter with Your Documents")

        if st.button("+ New Chat", type="primary", use_container_width=True):
            _start_new_chat()
            st.rerun()

        render_temp_chat_toggle()

        nav_items = ["Home", "My Documents", "Chat History", "Document Summary", "Settings"]
        for item in nav_items:
            if st.button(item, key=f"nav_{item}", use_container_width=True):
                st.session_state.active_nav = item
                if item == "My Documents":
                    st.session_state.show_upload_panel = True
                elif item == "Settings":
                    st.session_state.show_settings_panel = not st.session_state.show_settings_panel
                st.rerun()

        st.markdown("---")
        st.markdown("**Recent Chats**")
        if st.session_state.temp_chat:
            st.caption("Temporary Chat is ON — history paused.")
        elif not st.session_state.chat_history:
            st.caption("No saved chats yet.")
        else:
            for idx, item in enumerate(st.session_state.chat_history[:6]):
                label = f"{item['title']} ({item['time']})"
                if st.button(label, key=f"hist_btn_{idx}", use_container_width=True):
                    st.session_state.messages = list(item["messages"])
                    if item.get("filename"):
                        st.session_state.uploaded_filename = item["filename"]
                    if item.get("doc_summary"):
                        st.session_state.doc_summary = item["doc_summary"]
                    if item.get("total_pages"):
                        st.session_state.total_pages = item["total_pages"]
                    st.rerun()

        st.markdown("---")
        st.caption(f"User: **{st.session_state.user_label}**")
        if st.button("Logout", key="sidebar_logout_btn", use_container_width=True):
            _save_current_chat_to_history()
            logout_user()
            st.session_state.messages = []
            st.rerun()


def _render_top_header():
    """Render top navigation bar with search and user profile + Logout."""
    c1, c2, c3 = st.columns([2.2, 2.2, 1.4])
    with c1:
        st.markdown("**RAGMentor**  •  Features  •  How It Works  •  About")
    with c2:
        st.session_state.search_query = st.text_input(
            "Search in chat",
            value=st.session_state.search_query,
            placeholder="Search in your messages...",
            label_visibility="collapsed",
        )
    with c3:
        p_col, l_col = st.columns([1.4, 1])
        with p_col:
            st.markdown(f"**{st.session_state.user_label}**")
        with l_col:
            if st.button("Logout", key="top_logout_btn", use_container_width=True):
                _save_current_chat_to_history()
                logout_user()
                st.session_state.messages = []
                st.rerun()


def _render_upload_and_settings_drawers():
    """Render collapsible Upload PDF panel and Settings panel."""
    if st.session_state.show_upload_panel or st.session_state.active_nav == "My Documents":
        with st.expander("Upload & Process PDF Document", expanded=True):
            uploaded_file = st.file_uploader("Select PDF file", type=["pdf"], key="main_pdf_uploader")
            if uploaded_file is not None:
                if st.button("Process PDF", type="primary", key="main_process_pdf_btn"):
                    with st.spinner("Extracting pages, generating summary, and indexing..."):
                        tmp_path = None
                        try:
                            tmp_path = _save_uploaded_pdf(uploaded_file)
                            ingestion = ingest_pdf(tmp_path)
                            vs = build_vector_store(ingestion["chunks"])

                            st.session_state.vectorstore = vs
                            st.session_state.uploaded_filename = uploaded_file.name
                            st.session_state.total_pages = ingestion["total_pages"]
                            st.session_state.file_size_mb = round(uploaded_file.size / (1024 * 1024), 2)
                            st.session_state.uploaded_at = datetime.now().strftime("Uploaded %b %d, %Y")
                            st.session_state.doc_summary = ingestion["summary"]
                            st.session_state.suggested_questions = ingestion["relevant_questions"]
                            if not st.session_state.temp_chat:
                                st.session_state.messages = []
                            st.session_state.show_upload_panel = False
                            st.success(f"Processed **{uploaded_file.name}** successfully.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Error processing PDF: {exc}")
                        finally:
                            if tmp_path and os.path.exists(tmp_path):
                                try:
                                    os.remove(tmp_path)
                                except OSError:
                                    pass

    if st.session_state.show_settings_panel or st.session_state.active_nav == "Settings":
        with st.expander("Workspace Settings", expanded=True):
            s1, s2 = st.columns(2)
            with s1:
                st.session_state.top_k = st.slider(
                    "Retrieved Chunks (Top-K)",
                    min_value=1,
                    max_value=8,
                    value=st.session_state.top_k,
                )
            with s2:
                st.session_state.voice_lang = st.selectbox(
                    "Voice Recognition Language",
                    options=["en-IN", "en-US", "te-IN", "hi-IN"],
                    index=["en-IN", "en-US", "te-IN", "hi-IN"].index(st.session_state.voice_lang),
                )


def _render_right_panel():
    """Render Document Overview, Document Summary, Metrics, and Grounding Verification."""
    st.markdown("### Document Overview")
    if st.session_state.uploaded_filename:
        st.markdown(
            f"""
            <div class="metric-card">
                <strong>{st.session_state.uploaded_filename}</strong><br/>
                <small>{st.session_state.total_pages} pages • {st.session_state.file_size_mb} MB</small><br/>
                <small>{st.session_state.uploaded_at}</small><br/>
                <span style="color:#16a34a;font-weight:600;">● Document ready</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No document loaded yet.")

    st.markdown("### Document Summary")
    if st.session_state.doc_summary:
        st.write(st.session_state.doc_summary)
    else:
        st.caption("Summary appears here after processing a PDF.")

    st.markdown("### Performance Metrics")
    avg_rt = 0.0
    rt_values = [
        m.get("response_time")
        for m in st.session_state.messages
        if m["role"] == "assistant" and m.get("response_time") is not None
    ]
    if rt_values:
        avg_rt = round(sum(rt_values) / len(rt_values), 1)

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Top-K Chunks", st.session_state.top_k)
    with m2:
        st.metric("Avg Response", f"{avg_rt} s")

    st.markdown("### Grounding Verification")
    st.markdown("- Answer grounded in uploaded document context")
    st.markdown("- Page numbers attached when retrieved")
    st.markdown("- Actionable options shown when not found")


def main():
    _init_app_state()

    if not st.session_state.authenticated:
        render_login_screen()
        return

    _apply_workspace_styles()
    _render_sidebar()
    _render_top_header()
    _render_upload_and_settings_drawers()

    center_area, right_area = st.columns([2.4, 1.15])

    with center_area:
        if st.session_state.uploaded_filename:
            st.markdown(
                f"""
                <div class="doc-banner">
                    <div>
                        <strong>{st.session_state.uploaded_filename}</strong><br/>
                        <small>{st.session_state.uploaded_at} • {st.session_state.total_pages} pages • {st.session_state.file_size_mb} MB</small>
                    </div>
                    <span style="color:#16a34a;font-weight:600;">● Document ready</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            if st.button("+ Upload a PDF Document", type="primary"):
                st.session_state.show_upload_panel = True
                st.rerun()

        # Filter messages if search query entered
        q_filter = (st.session_state.search_query or "").strip().lower()
        visible_messages = [
            m for m in st.session_state.messages
            if not q_filter or q_filter in (m.get("content") or "").lower()
        ]

        for msg in visible_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    pages = msg.get("page_numbers") or []
                    rt = msg.get("response_time")
                    meta_bits = []
                    if pages:
                        meta_bits.append("Sources: " + ", ".join(f"Page {p}" for p in pages))
                    if rt is not None:
                        meta_bits.append(f"Response Time: {rt} s")
                    if meta_bits:
                        st.caption("  |  ".join(meta_bits))

                    if msg.get("is_fallback") and msg.get("options"):
                        st.markdown(
                            """
                            <div class="fallback-box">
                                <strong>No relevant information found in document.</strong><br/>
                                What would you like to do?
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        for idx, opt in enumerate(msg["options"]):
                            st.markdown(f"- {opt}")
                        if st.session_state.suggested_questions:
                            st.caption("Try a suggested question below.")

        # Relevant starter questions chips
        if st.session_state.suggested_questions:
            st.markdown("**Relevant Questions**")
            q_cols = st.columns(2)
            for idx, sq in enumerate(st.session_state.suggested_questions[:4]):
                with q_cols[idx % 2]:
                    if st.button(sq, key=f"rel_q_{idx}", use_container_width=True):
                        st.session_state.pending_prompt = sq
                        st.rerun()

        # Audio / Voice input bar
        if hasattr(st, "audio_input"):
            audio_val = st.audio_input("Audio / Voice input", key="workspace_audio_input")
            if audio_val is not None:
                if st.button("Convert Voice to Question", key="voice_to_q_btn"):
                    with st.spinner("Transcribing audio..."):
                        res = transcribe_audio_bytes(audio_val, language=st.session_state.voice_lang)
                    if res.get("error"):
                        st.warning(res["error"])
                    elif res.get("text"):
                        st.session_state.voice_question = res["text"]
                        st.success(f"Transcribed: \"{res['text']}\"")

            if st.session_state.voice_question:
                st.info(f"Voice question: **{st.session_state.voice_question}**")
                if st.button("Ask Voice Question", type="primary", key="ask_voice_q_btn"):
                    q_text = st.session_state.voice_question
                    st.session_state.voice_question = ""
                    st.session_state.pending_prompt = q_text
                    st.rerun()

        # Handle pending prompt from button or voice
        if st.session_state.pending_prompt:
            prompt_run = st.session_state.pending_prompt
            st.session_state.pending_prompt = None
            _process_user_query(prompt_run)
            st.rerun()

        # Text chat input
        typed_q = st.chat_input("Ask a question about your document...")
        if typed_q:
            _process_user_query(typed_q)
            st.rerun()

    with right_area:
        _render_right_panel()


if __name__ == "__main__":
    main()