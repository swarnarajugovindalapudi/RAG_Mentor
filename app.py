import os
import time
import tempfile
import uuid
from datetime import datetime
from urllib.parse import quote_plus
from html import escape

import streamlit as st
from langchain_core.documents import Document

from data_ingestion import ingest_pdf
from vector_engine import build_vector_store, retrieve_relevant_chunks
from llm_generator import generate_rag_answer
from auth_voice import (
    init_auth_voice_state,
    logout_user,
    render_temp_chat_toggle,
    transcribe_audio_bytes,
)
from auth_flow import get_logged_in_user, logout_auth_flow, require_login


st.set_page_config(
    page_title="RAGMentor",
    page_icon="📘",
    layout="wide",
)


# ---------------------------------------------------------------------
# STYLING
# ---------------------------------------------------------------------

def _apply_workspace_styles():
    theme = st.session_state.get("settings", {}).get("theme", "Light")
    dark = theme == "Dark"

    if dark:
        bg = "#0b1220"
        panel = "#111c2e"
        panel_alt = "#17243a"
        text = "#f8fafc"
        muted = "#cbd5e1"
        border = "#334155"
        input_bg = "#1e293b"
        sidebar = "#071226"
        user_bg = "#1d4ed8"
        user_text = "#ffffff"
        ai_bg = "#17243a"

    else:
        bg = "#f5f7fb"
        panel = "#ffffff"
        panel_alt = "#f1f5fb"
        text = "#172033"
        muted = "#64748b"
        border = "#dbe3ef"
        input_bg = "#ffffff"
        sidebar = "#0b2545"
        user_bg = "#eaf2ff"
        user_text = "#172554"
        ai_bg = "#ffffff"


    st.markdown(
        f"""
        <style>
        :root {{ color-scheme: {"dark" if dark else "light"}; }}
        html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"],
        [data-testid="stMain"], .main, .block-container {{
            background: {bg} !important; color: {text} !important;
        }}
        [data-testid="stHeader"] {{ background: {bg} !important; }}
        [data-testid="stSidebar"] {{ background: {sidebar} !important; }}
        [data-testid="stSidebar"] * {{ color: #eef4ff !important; }}
        [data-testid="stSidebar"] .stButton > button {{
            width:100%; text-align:left; background:rgba(255,255,255,.07) !important;
            color:#eef4ff !important; border:1px solid rgba(255,255,255,.12) !important;
            border-radius:10px; padding:.55rem .75rem;
        }}
        [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
            background: {panel} !important;
            border: 1px solid {border} !important;
            border-radius: 12px !important;
            padding: 8px !important;
        }}
        [data-testid="stSidebar"] [data-testid="stToggle"] label,
        [data-testid="stSidebar"] [data-testid="stToggle"] p {{
            color: #eef4ff !important;
            font-weight: 600 !important;
        }}
        [data-testid="stSidebar"] .stButton > button:hover {{
            background:#1d4ed8 !important; border-color:#60a5fa !important;
        }}
        .stMarkdown, .stMarkdown p, p, label, span, small,
        [data-testid="stCaptionContainer"], [data-testid="stText"],
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{ color:{text}; }}
        [data-testid="stCaptionContainer"] p, .stCaption,
        [data-testid="stWidgetLabel"] {{ color:{muted} !important; }}
        .rm-doc-card, .rm-panel {{ border:1px solid {border}; border-radius:12px;
            padding:14px 16px; margin:8px 0 14px 0; background:{panel}; color:{text}; }}
        .rm-message-row {{ display:flex; align-items:flex-start; gap:10px; width:100%; margin:12px 0 4px 0; }}
        .rm-user-row {{ justify-content:flex-end; }} .rm-assistant-row {{ justify-content:flex-start; }}
        .rm-avatar {{ display:inline-flex; align-items:center; justify-content:center;
            min-width:34px; width:34px; height:34px; border-radius:50%; font-size:18px; flex-shrink:0; }}
        .rm-user-avatar {{ background:#2563eb; color:white !important; }}
        .rm-ai-avatar {{ background:#e0e7ff; color:#172554 !important; }}
        .rm-user-bubble {{ max-width:88%; padding:12px 16px; line-height:1.55;
            overflow-wrap:anywhere; white-space:normal; background:{user_bg}; color:{user_text} !important;
            border:1px solid {border}; border-radius:16px 16px 4px 16px; }}
        .rm-assistant-bubble {{ max-width:92%; padding:12px 16px; line-height:1.6;
            overflow-wrap:anywhere; white-space:normal; background:{ai_bg}; color:{text} !important;
            border:1px solid {border}; border-radius:16px 16px 16px 4px; }}
        .rm-fallback {{ border:1px solid #d99; background:{panel}; color:{text}; border-radius:12px; padding:13px 15px; margin:8px 0; }}
        [data-testid="stMetric"] {{ background:{panel}; color:{text}; border:1px solid {border}; padding:10px; border-radius:10px; }}
        [data-testid="stVerticalBlockBorderWrapper"] {{ border-color:{border} !important; background:{panel}; }}
        [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
        [data-testid="stSelectbox"] div[data-baseweb="select"], [data-testid="stNumberInput"] input {{
            background:{input_bg} !important; color:{text} !important; border-color:{border} !important;
        }}
        [data-testid="stFileUploader"] section {{ background:{panel_alt} !important; border-color:{border} !important; }}
        [data-testid="stAudioInput"], [data-testid="stExpander"] {{ background:{panel}; border-color:{border} !important; }}
        .stButton > button, .stDownloadButton > button, .stLinkButton > a {{ border-radius:10px; }}
        div[data-testid="stChatInput"] {{ border-radius:18px; }}
        [data-testid="stButton"] button[kind="secondary"] {{
            border-radius:12px !important;
        }}
        /* Paperclip attachment control */
        button[kind="secondary"] {{
            color:{text} !important;
        }}
        [data-testid="stAudioInput"] {{
            min-width:48px !important; width:100% !important;
            display:flex !important; align-items:center !important;
            justify-content:center !important;
        }}
        [data-testid="stAudioInput"] button {{
            min-width:48px !important; min-height:48px !important;
            width:48px !important; height:48px !important;
            border-radius:12px !important; background:{panel_alt} !important;
            color:{text} !important; border:1px solid {border} !important;
        }}
        [data-testid="stAudioInput"] label {{
            display:none !important;
        }}
        [data-testid="stTextInput"] input {{
            min-height:48px !important; border-radius:12px !important;
        }}
        .stButton > button[kind="primary"] {{
            min-height:48px !important; border-radius:12px !important;
        }}
        [data-testid="stAudioInput"] button {{ min-height:42px; border-radius:12px; }}
        /* Composer controls: keep attachment and voice controls visible and aligned */
        [data-testid="stButton"] button {{
            min-height:46px !important;
        }}
        [data-testid="stAudioInput"] {{
            min-height:46px !important;
            padding-top:0 !important;
        }}
        [data-testid="stAudioInput"] button {{
            min-width:46px !important;
            min-height:46px !important;
            width:46px !important;
            height:46px !important;
            border-radius:12px !important;
        }}
        [data-testid="stSidebar"] [data-testid="stToggle"] {{
            padding:4px 0 2px 0 !important;
        }}
        [data-testid="stSidebar"] [data-testid="stToggle"] label {{
            font-size:0.92rem !important;
        }}
        [data-testid="stSidebar"] [data-testid="stToggle"] [data-testid="stWidgetLabel"] {{
            color:#eef4ff !important;
        }}
        [data-testid="stSidebar"] [data-testid="stCheckbox"] label,
        [data-testid="stSidebar"] [data-testid="stCheckbox"] p,
        [data-testid="stSidebar"] [data-testid="stCheckbox"] span {{
            color:#eef4ff !important;
            opacity:1 !important;
            visibility:visible !important;
        }}
        [data-testid="stSidebar"] [data-testid="stCheckbox"] input {{
            accent-color:#60a5fa !important;
        }}

        [data-testid="stFileUploader"] {{ color:{text} !important; }}
        [data-testid="stTabs"] button {{ color:{text} !important; }}
        [data-testid="stExpander"] summary {{ color:{text} !important; }}
        [data-testid="stAlert"] {{ border-radius:12px; }}
        .rm-user-bubble, .rm-assistant-bubble {{ box-shadow:0 1px 3px rgba(15,23,42,.06); }}

        /* Theme-aware visibility fixes for Streamlit widgets */
        h1, h2, h3, h4, h5, h6,
        .stMarkdown, .stMarkdown p, .stMarkdown li,
        [data-testid="stText"], [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stCaptionContainer"] p {{
            color: {text} !important;
        }}
        [data-testid="stCaptionContainer"], .stCaption {{
            color: {muted} !important;
        }}
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input,
        [data-testid="stChatInput"] textarea {{
            background: {input_bg} !important;
            color: {text} !important;
            border-color: {border} !important;
            caret-color: {text} !important;
        }}
        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stTextArea"] textarea::placeholder,
        [data-testid="stChatInput"] textarea::placeholder {{
            color: {muted} !important;
            opacity: 1 !important;
        }}
        [data-testid="stFileUploader"] section {{
            background: {panel_alt} !important;
            color: {text} !important;
            border-color: {border} !important;
        }}
        [data-testid="stFileUploader"] section *,
        [data-testid="stFileUploader"] small {{
            color: {text} !important;
        }}
        [data-testid="stFileUploader"] button {{
            background: {panel} !important;
            color: {text} !important;
            border-color: {border} !important;
        }}
        [data-testid="stSelectbox"] div[data-baseweb="select"],
        [data-testid="stMultiSelect"] div[data-baseweb="select"] {{
            background: {input_bg} !important;
            color: {text} !important;
        }}
        [data-baseweb="popover"], [data-baseweb="menu"],
        [role="listbox"], [role="option"] {{
            background: {panel} !important;
            color: {text} !important;
        }}
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary *,
        [data-testid="stTabs"] button,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {{
            color: {text} !important;
        }}
        .stButton > button, .stDownloadButton > button,
        .stLinkButton > a {{
            color: {text} !important;
            border-color: {border} !important;
        }}

        /* Popovers and profile menu: theme-aware surface and readable text */
        [data-testid="stPopover"],
        [data-testid="stPopoverBody"],
        [data-baseweb="popover"],
        [data-baseweb="popover"] > div {{
            background: {panel} !important;
            color: {text} !important;
            border-color: {border} !important;
        }}
        [data-testid="stPopover"] *,
        [data-testid="stPopoverBody"] *,
        [data-baseweb="popover"] * {{
            color: {text} !important;
        }}
        [data-testid="stPopover"] input,
        [data-testid="stPopoverBody"] input,
        [data-baseweb="popover"] input {{
            background: {input_bg} !important;
            color: {text} !important;
            border: 1px solid {border} !important;
            caret-color: {text} !important;
        }}
        [data-testid="stPopover"] button,
        [data-testid="stPopoverBody"] button,
        [data-baseweb="popover"] button {{
            border-color: {border} !important;
        }}
        .stButton > button[kind="primary"],
        .stDownloadButton > button[kind="primary"] {{
            color: #ffffff !important;
        }}
        [data-testid="stPopover"] [data-testid="stMarkdownContainer"],
        [data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] {{
            color: {text} !important;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------

def _init_app_state():
    init_auth_voice_state()

    auth_user = get_logged_in_user()
    user_label = (
        (auth_user or {}).get("name")
        or st.session_state.get("user_label")
        or "User"
    )

    defaults = {
        "messages": [],
        "chat_history": [],
        "uploaded_documents": [],
        "active_document_id": None,
        "active_nav": "Home",
        "doc_summary": "",
        "total_pages": 0,
        "file_size_mb": 0.0,
        "uploaded_at": "",
        "suggested_questions": [],
        "vectorstore": None,
        "uploaded_filename": None,
        "pending_prompt": None,
        "top_k": 4,
        "voice_lang": "en-IN",
        "search_query": "",
        "user_label": user_label,
        "show_question_hint": False,
        "voice_question": "",
        "temp_chat": False,
        "settings": {
            "theme": "Light",
            "compact_mode": False,
            "notifications": True,
            "save_history": True,
            "confirm_delete": True,
            "show_sources": True,
            "auto_suggestions": True,
        },
        "last_metrics": {},
        "feedback": {},
        "composer_text": "",
        "last_voice_audio_hash": None,
        "last_uploaded_file_hash": None,
        "composer_upload_version": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if auth_user and auth_user.get("name"):
        st.session_state.user_label = auth_user["name"]


def _active_document():
    active_id = st.session_state.active_document_id
    for document in st.session_state.uploaded_documents:
        if document["id"] == active_id:
            return document
    return None


def _activate_document(document_id):
    for document in st.session_state.uploaded_documents:
        if document["id"] == document_id:
            st.session_state.active_document_id = document_id
            st.session_state.vectorstore = document["vectorstore"]
            st.session_state.uploaded_filename = document["filename"]
            st.session_state.doc_summary = document["summary"]
            st.session_state.total_pages = document["total_pages"]
            st.session_state.file_size_mb = document["file_size_mb"]
            st.session_state.uploaded_at = document["uploaded_at"]
            st.session_state.suggested_questions = document["suggested_questions"]
            return


# ---------------------------------------------------------------------
# CHAT HISTORY / SESSION
# ---------------------------------------------------------------------

def _save_current_chat_to_history():
    if st.session_state.get("temp_chat"):
        return
    if not st.session_state.get("settings", {}).get("save_history", True):
        return

    messages = st.session_state.messages
    if not messages:
        return

    first_question = next(
        (
            message.get("content", "")
            for message in messages
            if message.get("role") == "user"
        ),
        "Document Session",
    )

    title = first_question[:48] + ("…" if len(first_question) > 48 else "")
    document = _active_document()

    entry = {
        "id": uuid.uuid4().hex,
        "title": title or "New conversation",
        "time": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "messages": list(messages),
        "document_id": st.session_state.active_document_id,
        "document_name": document["filename"] if document else None,
    }

    st.session_state.chat_history.insert(0, entry)
    st.session_state.chat_history = st.session_state.chat_history[:50]


def _start_new_chat():
    """Start from a completely blank chat, without carrying over the active PDF.

    Previously uploaded PDFs remain listed in My Documents, but none is selected
    for this new conversation until the user uploads or explicitly opens one.
    """
    _save_current_chat_to_history()
    st.session_state.messages = []
    st.session_state.pending_prompt = None
    st.session_state.show_question_hint = False
    st.session_state.voice_question = ""
    st.session_state.composer_text = ""
    st.session_state.active_document_id = None
    st.session_state.vectorstore = None
    st.session_state.uploaded_filename = None
    st.session_state.doc_summary = ""
    st.session_state.total_pages = 0
    st.session_state.file_size_mb = 0.0
    st.session_state.uploaded_at = ""
    st.session_state.suggested_questions = []
    st.session_state.last_metrics = {}
    st.session_state.last_uploaded_file_hash = None
    st.session_state.composer_upload_version += 1
    st.session_state.active_nav = "New Chat"


def _open_history_chat(chat_id):
    target = next(
        (chat for chat in st.session_state.chat_history if chat["id"] == chat_id),
        None,
    )
    if target is None:
        return
    current_messages = st.session_state.get("messages", [])
    if current_messages and current_messages != target.get("messages", []):
        _save_current_chat_to_history()

    for item in st.session_state.chat_history:
        if item["id"] == chat_id:
            st.session_state.messages = list(item["messages"])
            st.session_state.active_document_id = None
            st.session_state.vectorstore = None
            st.session_state.uploaded_filename = None
            st.session_state.doc_summary = ""
            st.session_state.total_pages = 0
            st.session_state.file_size_mb = 0.0
            st.session_state.uploaded_at = ""
            st.session_state.suggested_questions = []
            if item.get("document_id"):
                _activate_document(item["document_id"])
            st.session_state.last_metrics = {}
            for message in reversed(st.session_state.messages):
                if message.get("role") == "assistant" and message.get("metrics"):
                    st.session_state.last_metrics = message["metrics"]
                    break
            st.session_state.active_nav = "New Chat"
            return


def _handle_logout():
    _save_current_chat_to_history()
    try:
        logout_auth_flow()
    except Exception:
        pass
    try:
        logout_user()
    except Exception:
        pass
    st.session_state.messages = []
    st.rerun()


# ---------------------------------------------------------------------
# PDF INGESTION
# ---------------------------------------------------------------------

def _save_uploaded_pdf(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def _convert_chunks(raw_chunks):
    documents = []

    for item in raw_chunks or []:
        if isinstance(item, Document):
            doc = item
            metadata = dict(doc.metadata or {})
            if "page" not in metadata and metadata.get("page_number") is not None:
                metadata["page"] = metadata["page_number"]
            if "page_number" not in metadata and metadata.get("page") is not None:
                metadata["page_number"] = metadata["page"]
            documents.append(
                Document(page_content=doc.page_content, metadata=metadata)
            )
            continue

        if isinstance(item, dict):
            text = (
                item.get("page_content")
                or item.get("text")
                or item.get("content")
                or ""
            )
            metadata = dict(item.get("metadata") or {})
            if "page" not in metadata:
                if item.get("page") is not None:
                    metadata["page"] = item["page"]
                elif metadata.get("page_number") is not None:
                    metadata["page"] = metadata["page_number"]
            if "page_number" not in metadata and metadata.get("page") is not None:
                metadata["page_number"] = metadata["page"]
            if str(text).strip():
                documents.append(
                    Document(page_content=str(text), metadata=metadata)
                )
        elif str(item).strip():
            documents.append(Document(page_content=str(item), metadata={}))

    return documents


def _generate_summary_from_chunks(chunks):
    """Generate a document-grounded summary when ingestion did not provide one."""
    context = "\n\n".join(
        chunk.page_content for chunk in chunks if chunk.page_content.strip()
    )
    if not context:
        return ""

    context = context[:24000]
    result = generate_rag_answer(
        query=(
            "Create a clear, well-structured summary of this PDF. Cover its main "
            "topic, objectives, important concepts, key points, and conclusion. "
            "Use headings and bullet points where useful. Do not add outside information."
        ),
        context_text=context,
        page_numbers=[],
        suggested_questions=[],
    )
    if result.get("is_error") or result.get("is_fallback"):
        return ""
    return (result.get("answer") or "").strip()


def _process_pdf_file(uploaded_file):
    temp_path = None

    try:
        with st.spinner(
            f"Processing {uploaded_file.name}: extracting, chunking and indexing..."
        ):
            temp_path = _save_uploaded_pdf(uploaded_file)
            ingestion = ingest_pdf(temp_path)
            chunks = _convert_chunks(ingestion.get("chunks", []))

            if not chunks:
                raise ValueError(
                    "No readable text was extracted. If this is a scanned PDF, "
                    "OCR support is required."
                )

            vectorstore = build_vector_store(chunks)

            # Generate a summary here if the ingestion module did not provide one.
            summary = (ingestion.get("summary") or "").strip()
            if not summary:
                with st.spinner("Generating document summary..."):
                    summary = _generate_summary_from_chunks(chunks)

            # Preserve the current conversation before switching to another PDF.
            _save_current_chat_to_history()

            document_record = {
                "id": uuid.uuid4().hex,
                "filename": uploaded_file.name,
                "total_pages": ingestion.get("total_pages", 0),
                "file_size_mb": round(uploaded_file.size / (1024 * 1024), 2),
                "uploaded_at": datetime.now().strftime("Uploaded %b %d, %Y"),
                "summary": summary,
                "suggested_questions": ingestion.get("relevant_questions", []),
                "vectorstore": vectorstore,
            }

            # Keep all uploaded documents available in this Streamlit session.
            st.session_state.uploaded_documents.insert(0, document_record)
            _activate_document(document_record["id"])
            st.session_state.messages = []
            st.session_state.active_nav = "New Chat"

        st.success(
            f"'{uploaded_file.name}' is ready. "
            f"{document_record['total_pages']} pages indexed."
        )
        st.rerun()

    except Exception as exc:
        st.error(f"Could not process PDF: {exc}")

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------
# RESPONSE METRICS
# ---------------------------------------------------------------------

def _calculate_response_metrics(question, answer, context, retrieval, elapsed):
    """Calculate transparent, lightweight diagnostics for this response.

    These are operational indicators, not benchmark accuracy. True model
    accuracy requires a labelled evaluation dataset and human-verified answers.
    """
    answer_words = {
        word.lower().strip(".,:;!?()[]{}\"'`")
        for word in (answer or "").split()
        if len(word.strip(".,:;!?()[]{}\"'`")) > 3
    }
    context_words = {
        word.lower().strip(".,:;!?()[]{}\"'`")
        for word in (context or "").split()
        if len(word.strip(".,:;!?()[]{}\"'`")) > 3
    }
    if answer_words:
        grounding = round(100 * len(answer_words & context_words) / len(answer_words))
    else:
        grounding = 0

    question_words = {
        word.lower().strip(".,:;!?()[]{}\"'`")
        for word in (question or "").split()
        if len(word.strip(".,:;!?()[]{}\"'`")) > 3
    }
    if question_words:
        relevance = round(100 * len(question_words & context_words) / len(question_words))
    else:
        relevance = 0

    question_coverage = round(
        100 * len(question_words & answer_words) / len(question_words)
    ) if question_words else 0

    docs = retrieval.get("docs", []) if isinstance(retrieval, dict) else []
    page_count = len(set(retrieval.get("page_numbers", []) or [])) if isinstance(retrieval, dict) else 0
    return {
        "grounding": grounding,
        "relevance": relevance,
        "coverage": question_coverage,
        "source_pages": page_count,
        "retrieved_chunks": len(docs),
        "response_time": round(float(elapsed or 0), 2),
        # Accuracy is deliberately not fabricated without a labelled benchmark.
        "accuracy": None,
    }


# ---------------------------------------------------------------------
# RAG QUESTION ANSWERING
# ---------------------------------------------------------------------

def _append_assistant(
    content,
    question="",
    pages=None,
    elapsed=None,
    fallback=False,
    error=False,
    options=None,
):
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": content,
            "original_question": question,
            "page_numbers": pages or [],
            "response_time": elapsed,
            "is_fallback": fallback,
            "is_error": error,
            "options": options or [],
            "timestamp": datetime.now().strftime("%I:%M %p"),
        }
    )


def _process_user_query(question):
    question = (question or "").strip()
    if not question:
        return

    st.session_state.show_question_hint = False
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
            "timestamp": datetime.now().strftime("%I:%M %p"),
        }
    )

    start = time.perf_counter()
    vectorstore = st.session_state.vectorstore

    if vectorstore is None:
        _append_assistant(
            "Please upload a PDF first so I can answer using its contents.",
            question=question,
            elapsed=round(time.perf_counter() - start, 1),
            fallback=True,
            options=[
                "Upload a Relevant PDF",
                "Ask Something from Your Document",
                "Search the Web",
            ],
        )
        return

    try:
        retrieval = retrieve_relevant_chunks(
            vectorstore,
            question,
            k=st.session_state.top_k,
        )

        result = generate_rag_answer(
            query=question,
            context_text=retrieval.get("context", ""),
            page_numbers=retrieval.get("page_numbers", []),
            suggested_questions=st.session_state.suggested_questions,
        )

        elapsed = round(time.perf_counter() - start, 2)
        _append_assistant(
            result.get("answer", ""),
            question=question,
            pages=result.get("page_numbers", []),
            elapsed=elapsed,
            fallback=result.get("is_fallback", False),
            error=result.get("is_error", False),
            options=result.get("options", []),
        )
        metrics = _calculate_response_metrics(
            question,
            result.get("answer", ""),
            retrieval.get("context", ""),
            retrieval,
            elapsed,
        )
        st.session_state.messages[-1]["metrics"] = metrics
        st.session_state.last_metrics = metrics

    except Exception as exc:
        _append_assistant(
            f"Document processing error: {exc}",
            question=question,
            elapsed=round(time.perf_counter() - start, 1),
            error=True,
        )


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------

def _render_sidebar():
    with st.sidebar:
        st.markdown("## RAGMentor")
        st.caption("Learn Smarter with Your Documents")

        if st.button("＋  New Chat", key="new_chat_sidebar", use_container_width=True):
            _start_new_chat()
            st.rerun()

        # Use a checkbox instead of st.toggle: its checked state and label stay
        # visible even with custom sidebar themes / different Streamlit versions.
        with st.container(border=True):
            st.markdown("**🕒 Temporary Chat**")
            temp_enabled = st.checkbox(
                "Don't save this conversation",
                value=bool(st.session_state.get("temp_chat", False)),
                key="sidebar_temp_chat_checkbox",
                help="When enabled, this conversation will not be added to Chat History or Recent Chats.",
            )
            st.session_state["temp_chat"] = temp_enabled
            if temp_enabled:
                st.caption("🔒 Temporary mode is ON — this chat won't be saved.")
            else:
                st.caption("History saving follows your Settings.")

        st.markdown("---")

        nav_items = [
            ("Home", "⌂"),
            ("My Documents", "▤"),
            ("Chat History", "◷"),
            ("Document Summary", "▧"),
            ("Settings", "⚙"),
        ]

        for item, symbol in nav_items:
            is_active = st.session_state.active_nav == item
            label = f"{'● ' if is_active else ''}{symbol}  {item}"
            if st.button(
                label,
                key=f"nav_{item}",
                use_container_width=True,
            ):
                st.session_state.active_nav = item
                st.rerun()


        st.markdown("---")
        st.caption("RECENT CHATS · THIS SESSION")
        recent_chats = st.session_state.chat_history[:5]
        if not recent_chats:
            st.caption("Your saved chats will appear here.")
        for chat in recent_chats:
            if st.button(
                f"💬 {chat['title'][:28]}",
                key=f"sidebar_recent_{chat['id']}",
                use_container_width=True,
                help=f"{chat['title']} · {chat['time']}",
            ):
                _open_history_chat(chat["id"])
                st.rerun()


# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------

def _render_header():
    brand, search_col, theme_col, profile_col = st.columns([1.25, 3.0, 0.55, 1.55], vertical_alignment="center")
    with brand:
        st.markdown("### 📘 RAGMentor")
    with search_col:
        st.session_state.search_query = st.text_input(
            "Search messages", value=st.session_state.search_query,
            placeholder="Search in your messages...", label_visibility="collapsed",
        )
    with theme_col:
        is_dark = st.session_state.settings.get("theme") == "Dark"
        icon = "☀️" if is_dark else "🌙"
        if st.button(icon, key="header_theme_toggle", help="Switch to light mode" if is_dark else "Switch to dark mode"):
            new_theme = "Light" if is_dark else "Dark"
            st.session_state.settings["theme"] = new_theme
            if "settings_appearance" in st.session_state:
                st.session_state.settings_appearance = new_theme
            st.rerun()
    with profile_col:
        with st.popover(f"👤 {st.session_state.user_label}", use_container_width=True):
            st.write(f"Signed in as **{st.session_state.user_label}**")
            if st.button("⚙ Account settings", key="profile_settings", use_container_width=True):
                st.session_state.active_nav = "Settings"
                st.rerun()
            if st.button("↪ Logout", key="profile_logout", type="primary", use_container_width=True):
                _handle_logout()


# ---------------------------------------------------------------------
# DOCUMENTS PAGE
# ---------------------------------------------------------------------

def _render_my_documents_page():
    st.markdown("## My Documents")
    st.caption("All PDFs uploaded during this app session.")

    with st.container(border=True):
        st.markdown("### Upload a new PDF")
        file = st.file_uploader(
            "Choose a PDF document",
            type=["pdf"],
            key="my_documents_uploader",
            label_visibility="collapsed",
        )
        if file and st.button("Process PDF", type="primary", key="my_docs_process"):
            _process_pdf_file(file)

    st.markdown("### Uploaded documents")

    documents = st.session_state.uploaded_documents
    if not documents:
        st.info("No documents uploaded yet.")
        return

    for doc in documents:
        with st.container(border=True):
            col_info, col_action = st.columns([4, 1.25])

            with col_info:
                active = doc["id"] == st.session_state.active_document_id
                st.markdown(f"**{doc['filename']}**")
                st.caption(
                    f"{doc['total_pages']} pages · "
                    f"{doc['file_size_mb']} MB · {doc['uploaded_at']}"
                )
                if active:
                    st.success("Currently active")

            with col_action:
                if st.button(
                    "Open",
                    key=f"open_doc_{doc['id']}",
                    use_container_width=True,
                    disabled=(doc["id"] == st.session_state.active_document_id),
                ):
                    _save_current_chat_to_history()
                    _activate_document(doc["id"])
                    st.session_state.messages = []
                    st.session_state.active_nav = "New Chat"
                    st.rerun()


# ---------------------------------------------------------------------
# CHAT HISTORY PAGE
# ---------------------------------------------------------------------

def _render_chat_history_page():
    st.markdown("## Chat History")
    st.caption("Open a previous conversation.")

    if not st.session_state.chat_history:
        st.info(
            "No saved conversations yet. Start a chat and use New Chat "
            "to save the current conversation."
        )
        return

    for item in st.session_state.chat_history:
        with st.container(border=True):
            col_details, col_open = st.columns([4, 1])

            with col_details:
                st.markdown(f"**{item['title']}**")
                st.caption(
                    f"{item['time']} · "
                    f"{item.get('document_name') or 'No document attached'}"
                )
                st.caption(
                    f"{len(item.get('messages', []))} messages"
                )

            with col_open:
                if st.button(
                    "Open chat",
                    key=f"open_chat_{item['id']}",
                    use_container_width=True,
                ):
                    _open_history_chat(item["id"])
                    st.rerun()


# ---------------------------------------------------------------------
# SETTINGS PAGE
# ---------------------------------------------------------------------

def _render_settings_page():
    st.markdown("## Settings")
    st.caption("Manage your RAGMentor account, appearance, voice and privacy.")

    settings = st.session_state.settings
    tabs = st.tabs([
        "Account", "Customize", "General", "Usage & Limits",
        "Notifications", "Voice", "Safety", "Security & Login",
        "Storage", "Privacy Center", "History & Data", "About",
    ])

    with tabs[0]:
        st.markdown("### Account")
        with st.container(border=True):
            st.write(f"**Name:** {st.session_state.user_label}")
            st.write("**Status:** Signed in")
            st.caption("Your account is managed by the RAGMentor authentication module.")
        st.markdown("### Switch account")
        st.caption("Sign out to return to the login screen and sign in with another account.")
        if st.button("↔ Switch account", key="settings_switch_account"):
            _handle_logout()
        if st.button("↪ Logout", key="settings_logout", type="primary"):
            _handle_logout()

    with tabs[1]:
        st.markdown("### Customize RAGMentor")
        theme_options = ["Light", "Dark", "System"]
        current_theme = settings.get("theme", "Light")
        if current_theme not in theme_options:
            current_theme = "Light"
        settings["theme"] = st.selectbox(
            "Appearance", theme_options,
            index=theme_options.index(current_theme),
            help="Choose the appearance of the complete RAGMentor workspace.",
            key="settings_appearance",
        )
        if settings["theme"] == "System":
            st.caption("System appearance currently follows the light theme. Choose Light or Dark to set it explicitly.")
        st.divider()
        settings["compact_mode"] = st.toggle(
            "Compact chat layout", value=settings.get("compact_mode", False),
            key="settings_compact_mode",
        )
        settings["show_sources"] = st.toggle(
            "Show source citations below answers", value=settings.get("show_sources", True),
            key="settings_show_sources",
        )
        settings["auto_suggestions"] = st.toggle(
            "Show suggested questions", value=settings.get("auto_suggestions", True),
            key="settings_auto_suggestions",
        )

    with tabs[2]:
        st.markdown("### General")
        st.session_state.top_k = st.slider(
            "Retrieved chunks per question", min_value=1, max_value=8,
            value=st.session_state.top_k,
            help="Controls how many matching PDF sections are passed to Gemini.",
            key="settings_top_k",
        )
        st.caption("New Chat always starts without an active PDF. Previously uploaded PDFs remain available in My Documents.")
        st.caption("Use Privacy Center to control whether conversations are saved to Chat History.")

    with tabs[3]:
        st.markdown("### Usage & Limits")
        docs = st.session_state.uploaded_documents
        chats = st.session_state.chat_history
        c1, c2, c3 = st.columns(3)
        c1.metric("Uploaded documents", len(docs))
        c2.metric("Saved conversations", len(chats))
        c3.metric("Retrieved chunks", st.session_state.top_k)
        st.caption("These counters reflect this active Streamlit session, not account-wide billing or API quota.")

    with tabs[4]:
        st.markdown("### Notifications")
        settings["notifications"] = st.toggle(
            "Enable in-app notifications", value=settings.get("notifications", True),
            key="settings_notifications",
        )
        st.caption("Email and push notifications are not connected in this version.")

    with tabs[5]:
        st.markdown("### Voice")
        languages = ["en-IN", "te-IN", "hi-IN", "en-US"]
        current = st.session_state.voice_lang if st.session_state.voice_lang in languages else "en-IN"
        st.session_state.voice_lang = st.selectbox(
            "Speech recognition language", languages,
            index=languages.index(current),
            key="settings_voice_language",
        )
        st.info("Use the microphone button beside the Send arrow in the message composer. Your speech is converted to text before sending.")

    with tabs[6]:
        st.markdown("### Safety")
        st.success("Document-only answering is enabled.")
        st.write("- Answers are generated from retrieved PDF context.")
        st.write("- If relevant information is missing, RAGMentor should abstain instead of inventing an answer.")
        st.write("- Source page references are shown when available.")
        st.caption("These safeguards are part of the current RAG generation flow and cannot be disabled from Settings.")

    with tabs[7]:
        st.markdown("### Security & Login")
        with st.container(border=True):
            st.write("**Login status:** Active")
            st.write("**Authentication:** Managed by the existing RAGMentor login module")
        st.caption("Password changes and multi-factor authentication are not configured in this version.")
        if st.button("Logout from this device", key="settings_security_logout"):
            _handle_logout()

    with tabs[8]:
        st.markdown("### Storage")
        st.metric("PDFs in this session", len(st.session_state.uploaded_documents))
        st.caption("PDF indexes are currently held in the active Streamlit session. They are not yet stored in permanent cloud storage.")
        if st.session_state.uploaded_documents:
            st.markdown("**Stored documents**")
            for doc in st.session_state.uploaded_documents:
                st.caption(f"📄 {doc['filename']} · {doc['total_pages']} pages")

    with tabs[9]:
        st.markdown("### Privacy Center")
        settings["save_history"] = st.toggle(
            "Save conversations to Chat History",
            value=settings.get("save_history", True),
            key="privacy_save_history",
        )
        settings["confirm_delete"] = st.toggle(
            "Confirm before clearing session data",
            value=settings.get("confirm_delete", True),
            key="privacy_confirm_delete",
        )
        st.info("Questions and retrieved document text are sent to the configured Gemini API to generate answers.")

    with tabs[10]:
        st.markdown("### History & Data")
        st.metric("Saved chats", len(st.session_state.chat_history))
        if st.button("Clear chat history", key="clear_chat_history"):
            if settings.get("confirm_delete", True):
                st.session_state.confirm_clear_history = True
            else:
                st.session_state.chat_history = []
                st.success("Chat history cleared for this session.")
        if st.session_state.get("confirm_clear_history"):
            st.warning("Are you sure you want to clear all saved chats in this session?")
            yes, no = st.columns(2)
            with yes:
                if st.button("Yes, clear history", key="confirm_clear_history_yes"):
                    st.session_state.chat_history = []
                    st.session_state.confirm_clear_history = False
                    st.rerun()
            with no:
                if st.button("Cancel", key="confirm_clear_history_no"):
                    st.session_state.confirm_clear_history = False
                    st.rerun()

    with tabs[11]:
        st.markdown("### About RAGMentor")
        st.write("RAGMentor is a document-grounded learning assistant that retrieves relevant PDF sections, generates answers, and displays source-page references.")
        st.caption("Version 1.0 · Retrieval-Augmented Generation with grounding verification")

    st.session_state.settings = settings



# DOCUMENT SUMMARY PAGE
# ---------------------------------------------------------------------

def _render_summary_page():
    st.markdown("## Document Summary")
    st.caption("Choose any PDF uploaded during this app session.")

    documents = st.session_state.get("uploaded_documents", [])
    if not documents:
        st.info("No PDFs uploaded in this session. Upload a PDF from My Documents.")
        return

    document_ids = [doc["id"] for doc in documents]
    active_id = st.session_state.get("active_document_id")
    selected_index = document_ids.index(active_id) if active_id in document_ids else 0

    selected_id = st.selectbox(
        "Select document",
        options=document_ids,
        index=selected_index,
        format_func=lambda doc_id: next(
            (doc["filename"] for doc in documents if doc["id"] == doc_id),
            "Uploaded PDF",
        ),
        key="document_summary_selector",
    )

    _activate_document(selected_id)
    doc = _active_document()
    if not doc:
        st.error("Unable to load the selected PDF.")
        return

    with st.container(border=True):
        st.markdown("### 📄 Document Details")
        st.write(f"**{doc['filename']}**")
        st.caption(
            f"{doc['total_pages']} pages · {doc['file_size_mb']} MB · "
            f"{doc['uploaded_at']}"
        )
        st.divider()

        st.markdown("### 📝 Summary")
        summary = (doc.get("summary") or "").strip()
        if summary:
            st.markdown(summary)
        else:
            st.warning(
                "A summary is not available for this PDF. It may have been "
                "uploaded before summary generation was added, or Gemini may "
                "not have been available."
            )
            if st.button(
                "Generate summary now",
                type="primary",
                key=f"generate_summary_{doc['id']}",
            ):
                with st.spinner("Generating summary from this PDF..."):
                    try:
                        retrieval = retrieve_relevant_chunks(
                            doc["vectorstore"],
                            "Summarize the main topics and important points in this document",
                            k=8,
                        )
                        result = generate_rag_answer(
                            query=(
                                "Summarize this document with its main topic, "
                                "objectives, key concepts, important points and conclusion. "
                                "Use headings and bullet points. Only use the supplied context."
                            ),
                            context_text=retrieval.get("context", ""),
                            page_numbers=retrieval.get("page_numbers", []),
                            suggested_questions=[],
                        )
                        if result.get("is_error") or result.get("is_fallback"):
                            st.error(
                                result.get("answer")
                                or "Summary generation failed. Check Gemini API key and quota."
                            )
                        else:
                            doc["summary"] = (result.get("answer") or "").strip()
                            st.session_state.doc_summary = doc["summary"]
                            st.success("Summary generated successfully.")
                            st.rerun()
                    except Exception as exc:
                        st.error(f"Summary generation failed: {exc}")

        st.divider()
        st.markdown("### 💡 Suggested Questions")
        questions = doc.get("suggested_questions") or []
        if questions:
            for question in questions:
                st.markdown(f"- {question}")
        else:
            st.caption("No suggested questions were generated for this PDF.")


# CHAT DISPLAY: QUESTION LEFT, RAGMENTOR RIGHT
# ---------------------------------------------------------------------

def _render_fallback_actions(message, index):
    st.markdown(
        """
        <div class="rm-fallback">
            <strong>No relevant information found in your document.</strong><br/>
            What would you like to do?
        </div>
        """,
        unsafe_allow_html=True,
    )

    question = message.get("original_question", "")
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button(
            "📄 Upload a Relevant PDF",
            key=f"fallback_upload_{index}",
            use_container_width=True,
        ):
            st.session_state.active_nav = "My Documents"
            st.rerun()

    with c2:
        if st.button(
            "💬 Ask Something from Your Document",
            key=f"fallback_ask_{index}",
            use_container_width=True,
        ):
            st.session_state.show_question_hint = True
            st.rerun()

    with c3:
        url = "https://www.google.com/search?q=" + quote_plus(question)
        st.link_button(
            "🌐 Search the Web",
            url,
            use_container_width=True,
        )


def _render_chat_pair(user_message, assistant_message, pair_index):
    # User message: right aligned, with a dedicated user avatar.
    spacer, user_col = st.columns([1.2, 5])
    with user_col:
        st.markdown(
            f'<div class="rm-message-row rm-user-row">'
            f'<div class="rm-user-bubble">{escape(user_message.get("content", ""))}</div>'
            f'<span class="rm-avatar rm-user-avatar">👤</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(user_message.get("timestamp", ""))

    # RAGMentor response: separate row below the user message.
    assistant_col, empty = st.columns([5, 1.2])
    with assistant_col:
        safe_answer = escape(assistant_message.get("content", "")).replace("\n", "<br>")
        st.markdown(
            f'<div class="rm-message-row rm-assistant-row"><span class="rm-avatar rm-ai-avatar">🤖</span>'
            f'<div class="rm-assistant-bubble">{safe_answer}</div></div>',
            unsafe_allow_html=True,
        )
        if assistant_message.get("is_error"):
            st.error("The request could not be completed. Check Gemini API availability, model access, internet connection and quota.")

        pages = assistant_message.get("page_numbers") or []
        elapsed = assistant_message.get("response_time")
        if pages and st.session_state.settings.get("show_sources", True):
            st.caption("📚 Sources: " + ", ".join(f"Page {p}" for p in pages))
        if elapsed is not None:
            st.caption(f"Response time: {elapsed} s")

        metrics = assistant_message.get("metrics") or {}
        if metrics:
            with st.expander("📊 Response diagnostics", expanded=False):
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", "Not benchmarked")
                m2.metric("Grounding score*", f"{metrics.get('grounding', 0)}%")
                m3.metric("Query relevance*", f"{metrics.get('relevance', 0)}%")
                m4, m5, m6 = st.columns(3)
                m4.metric("Question coverage*", f"{metrics.get('coverage', 0)}%")
                m5.metric("Source pages", metrics.get("source_pages", 0))
                m6.metric("Response time", f"{metrics.get('response_time', 0)} s")
                st.caption("*Grounding, relevance and coverage are lightweight text-overlap estimates. They are useful diagnostics, not verified model accuracy. True accuracy requires a labelled evaluation dataset.")

        fb1, fb2, fb3 = st.columns([1, 1, 5])
        with fb1:
            if st.button("👍", key=f"like_{pair_index}", help="Helpful answer"):
                st.session_state.feedback[pair_index] = "positive"
                st.toast("Thanks for your feedback")
        with fb2:
            if st.button("👎", key=f"dislike_{pair_index}", help="Not helpful"):
                st.session_state.feedback[pair_index] = "negative"
                st.toast("Feedback recorded")
        with fb3:
            if st.button("📋 Copy answer", key=f"copy_{pair_index}"):
                st.code(assistant_message.get("content", ""), language=None)

    if assistant_message.get("is_fallback"):
        _render_fallback_actions(assistant_message, pair_index)
    st.divider()


def _render_chat_messages():
    messages = st.session_state.messages
    search_term = (st.session_state.search_query or "").strip().lower()
    if search_term:
        messages = [msg for msg in messages if search_term in (msg.get("content") or "").lower()]

    if not messages:
        st.markdown("### Ask questions about your document")
        st.caption("Your conversation will appear here. User messages are on the right and RAGMentor replies are on the left.")
        return

    idx = 0
    pair_index = 0
    while idx < len(messages):
        current = messages[idx]
        if current.get("role") == "user" and idx + 1 < len(messages) and messages[idx + 1].get("role") == "assistant":
            _render_chat_pair(current, messages[idx + 1], pair_index)
            idx += 2
            pair_index += 1
        else:
            if current.get("role") == "user":
                st.markdown(f'<div class="rm-message-row rm-user-row"><span class="rm-avatar rm-user-avatar">👤</span><div class="rm-user-bubble">{escape(current.get("content", ""))}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="rm-message-row rm-assistant-row"><span class="rm-avatar rm-ai-avatar">🤖</span><div class="rm-assistant-bubble">{escape(current.get("content", ""))}</div></div>', unsafe_allow_html=True)
            idx += 1


# ---------------------------------------------------------------------
# SUGGESTED QUESTIONS AND VOICE
# ---------------------------------------------------------------------

def _render_suggested_questions():
    questions = st.session_state.suggested_questions
    if not st.session_state.settings.get("auto_suggestions", True):
        return
    if not questions:
        return

    with st.expander("Suggested questions", expanded=False):
        for index, question in enumerate(questions[:6]):
            if st.button(
                question,
                key=f"suggested_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_prompt = question
                st.rerun()


def _transcribe_audio_if_new(audio):
    """Transcribe a newly recorded audio clip once and place text in composer."""
    if audio is None:
        return
    try:
        audio_bytes = audio.getvalue()
        import hashlib
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        if audio_hash == st.session_state.get("last_voice_audio_hash"):
            return
        st.session_state.last_voice_audio_hash = audio_hash
        with st.spinner("Transcribing your voice question..."):
            result = transcribe_audio_bytes(audio, language=st.session_state.voice_lang)
        if result.get("error"):
            st.error(result["error"])
        elif result.get("text"):
            st.session_state.composer_text = result["text"]
            st.toast("Voice converted to text. Press Send to ask RAGMentor.")
    except Exception as exc:
        st.error(f"Voice transcription failed: {exc}")


# ---------------------------------------------------------------------
# HOME / CHAT COMPOSER
# ---------------------------------------------------------------------

def _render_active_document_strip():
    doc = _active_document()

    if not doc:
        st.info("No active PDF. Use the attachment button in the message box to upload one.")
        return

    with st.container(border=True):
        col_name, col_status = st.columns([4, 1.2])
        with col_name:
            st.markdown(f"**📄 {doc['filename']}**")
            st.caption(
                f"{doc['total_pages']} pages · {doc['file_size_mb']} MB · "
                f"{doc['uploaded_at']}"
            )
        with col_status:
            st.success("Document ready")


def _render_dashboard_page():
    """Dedicated Home dashboard; separate from the blank New Chat screen."""
    st.markdown("# Welcome to RAGMentor 👋")
    st.markdown("### Your AI learning workspace")
    st.caption("Upload a document, explore your saved PDFs, or continue a previous conversation.")

    upload_col, info_col = st.columns([1.65, 1], gap="large")
    with upload_col:
        with st.container(border=True):
            st.markdown("### 📄 Start with a document")
            dashboard_pdf = st.file_uploader(
                "Choose a PDF to begin", type=["pdf"],
                key="dashboard_pdf_upload", help="Upload a PDF to create a document-grounded chat.",
            )
            if dashboard_pdf is not None:
                if st.button("Upload and start learning", type="primary", key="dashboard_process_pdf", use_container_width=True):
                    _process_pdf_file(dashboard_pdf)

    with info_col:
        with st.container(border=True):
            st.markdown("### ✨ What you can do")
            st.markdown("- Ask questions grounded in your PDF\n- See source page citations\n- Review document summaries\n- Check response diagnostics")

    st.markdown("## Your workspace")
    stat1, stat2, stat3 = st.columns(3)
    stat1.metric("Uploaded documents", len(st.session_state.uploaded_documents))
    stat2.metric("Saved conversations", len(st.session_state.chat_history))
    stat3.metric("Active document", "None" if not _active_document() else "Selected")

    recent_col, chats_col = st.columns(2, gap="large")
    with recent_col:
        st.markdown("### Recent documents")
        docs = st.session_state.uploaded_documents[:5]
        if not docs:
            st.caption("Your uploaded PDFs will appear here.")
        for doc in docs:
            with st.container(border=True):
                st.markdown(f"**📄 {escape(doc['filename'])}**")
                st.caption(f"{doc['total_pages']} pages · {doc['file_size_mb']} MB · {doc['uploaded_at']}")
                if st.button("Open document", key=f"dashboard_open_doc_{doc['id']}", use_container_width=True):
                    _save_current_chat_to_history()
                    _activate_document(doc["id"])
                    st.session_state.messages = []
                    st.session_state.active_nav = "New Chat"
                    st.rerun()
    with chats_col:
        st.markdown("### Recent conversations")
        chats = st.session_state.chat_history[:5]
        if not chats:
            st.caption("Your saved conversations will appear here.")
        for item in chats:
            with st.container(border=True):
                st.markdown(f"**💬 {escape(item['title'])}**")
                st.caption(item['time'])
                if st.button("Continue chat", key=f"dashboard_open_chat_{item['id']}", use_container_width=True):
                    _open_history_chat(item["id"])
                    st.rerun()


def _submit_composer_question():
    """Submit the composer text through a Streamlit callback.

    Callbacks run before the next script render, so it is safe to clear the
    widget's session-state value here (unlike changing it after st.text_input).
    """
    question = (st.session_state.get("composer_text") or "").strip()
    if not question:
        st.session_state.composer_notice = "Type a question or record your voice first."
        return

    st.session_state.composer_text = ""
    st.session_state.composer_notice = ""
    _process_user_query(question)


def _render_home_page():
    """Chat workspace used by New Chat and opened document conversations."""
    _render_active_document_strip()

    left_panel, right_panel = st.columns([2.25, 1], gap="large")

    with left_panel:
        if st.session_state.show_question_hint:
            st.info("Type another question about the active PDF below.")

        _render_chat_messages()
        _render_suggested_questions()

        # Message composer: visible attachment, text, microphone and send controls.
        attach_col, text_col, voice_col, send_col = st.columns(
            [0.58, 5.0, 0.72, 0.72],
            vertical_alignment="bottom",
            gap="small",
        )

        # A plain button is used instead of st.popover so Streamlit does not
        # append its default chevron/dropdown indicator to the paperclip icon.
        if "show_composer_attachment" not in st.session_state:
            st.session_state["show_composer_attachment"] = False

        with attach_col:
            if st.button(
                "📎",
                key="composer_attach_toggle",
                help="Attach a PDF document",
                use_container_width=True,
            ):
                st.session_state["show_composer_attachment"] = (
                    not st.session_state["show_composer_attachment"]
                )

        if st.session_state.get("show_composer_attachment", False):
            with st.container(border=True):
                st.markdown("**📎 Attach a PDF document**")
                attached_pdf = st.file_uploader(
                    "Choose a PDF",
                    type=["pdf"],
                    key=f"composer_pdf_upload_{st.session_state.composer_upload_version}",
                    label_visibility="visible",
                )
                if attached_pdf is not None and st.button(
                    "Upload and process",
                    key=f"composer_process_pdf_{st.session_state.composer_upload_version}",
                    type="primary",
                    use_container_width=True,
                ):
                    import hashlib
                    pdf_hash = hashlib.sha256(attached_pdf.getvalue()).hexdigest()
                    if pdf_hash != st.session_state.get("last_uploaded_file_hash"):
                        st.session_state.last_uploaded_file_hash = pdf_hash
                        _process_pdf_file(attached_pdf)
                        st.session_state["show_composer_attachment"] = False
                    else:
                        st.info("This PDF has already been processed in this session.")

        # Capture voice input in its own clearly visible microphone control.
        with voice_col:
            audio = st.audio_input(
                "🎙️ Voice",
                key="voice_audio_input",
                help="Record your question",
                label_visibility="collapsed",
            )

        _transcribe_audio_if_new(audio)

        with text_col:
            st.text_input(
                "Message RAGMentor",
                key="composer_text",
                placeholder="Ask a question about your document...",
                label_visibility="collapsed",
            )

        with send_col:
            st.button(
                "➤",
                key="composer_send",
                type="primary",
                use_container_width=True,
                help="Send message",
                on_click=_submit_composer_question,
            )

        composer_notice = st.session_state.pop("composer_notice", "")
        if composer_notice:
            st.info(composer_notice)

        if st.session_state.pending_prompt:
            prompt = st.session_state.pending_prompt
            st.session_state.pending_prompt = None
            _process_user_query(prompt)
            st.rerun()

    with right_panel:
        doc = _active_document()

        with st.container(border=True):
            st.markdown("### 📄 Document Overview")
            if doc:
                st.markdown(f"**{escape(doc['filename'])}**")
                st.caption(
                    f"{doc['total_pages']} pages · {doc['file_size_mb']} MB"
                )
                st.caption(doc["uploaded_at"])
                st.success("Document ready")
            else:
                st.caption("Upload a PDF to see document details.")

        with st.container(border=True):
            st.markdown("### 📝 Document Summary")
            if doc and doc.get("summary"):
                st.write(doc["summary"])
                if st.button("View full summary", key="view_full_summary"):
                    st.session_state.active_nav = "Document Summary"
                    st.rerun()
            else:
                st.caption("Your PDF summary will appear here.")

        with st.container(border=True):
            st.markdown("### Performance Metrics")
            metrics = st.session_state.get("last_metrics", {})
            metric_a, metric_b = st.columns(2)
            with metric_a:
                st.metric("Accuracy", "Not benchmarked")
                st.metric("Grounding score*", f"{metrics.get('grounding', 0)}%" if metrics else "—")
                st.metric("Query relevance*", f"{metrics.get('relevance', 0)}%" if metrics else "—")
            with metric_b:
                st.metric("Question coverage*", f"{metrics.get('coverage', 0)}%" if metrics else "—")
                st.metric("Response time", f"{metrics.get('response_time', 0)} s" if metrics else "—")
                st.metric("Source pages", metrics.get("source_pages", 0) if metrics else "—")
            st.caption("*Grounding, relevance and coverage are heuristic text-overlap estimates, not verified accuracy. True accuracy requires a labelled evaluation dataset.")

        with st.container(border=True):
            st.markdown("### Grounding Verification")
            if st.session_state.messages and st.session_state.messages[-1].get("role") == "assistant":
                last = st.session_state.messages[-1]
                if last.get("is_fallback"):
                    st.warning("Answer not verified from the selected document.")
                elif last.get("is_error"):
                    st.error("Verification unavailable because generation failed.")
                else:
                    st.success("Answer generated using retrieved document context.")
                    st.caption("Review the source page citations for confirmation.")
            else:
                st.caption("Verification status will appear after your first question.")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    require_login()
    _init_app_state()
    # Sync the Settings appearance widget before injecting the theme CSS.
    if st.session_state.active_nav == "Settings" and "settings_appearance" in st.session_state:
        st.session_state.settings["theme"] = st.session_state.settings_appearance
    _apply_workspace_styles()
    _render_sidebar()
    _render_header()

    page = st.session_state.active_nav

    if page == "Home":
        _render_dashboard_page()
    elif page == "New Chat":
        _render_home_page()
    elif page == "My Documents":
        _render_my_documents_page()
    elif page == "Chat History":
        _render_chat_history_page()
    elif page == "Settings":
        _render_settings_page()
    elif page == "Document Summary":
        _render_summary_page()
    else:
        _render_home_page()


if __name__ == "__main__":
    main()
