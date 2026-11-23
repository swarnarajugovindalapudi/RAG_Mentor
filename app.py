import os
import tempfile
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


def _init_app_state():
    """Initialize application session state keys."""
    init_auth_voice_state()
    defaults = {
        "messages": [],
        "doc_summary": "",
        "total_pages": 0,
        "suggested_questions": [],
        "vectorstore": None,
        "uploaded_filename": None,
        "pending_prompt": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _save_uploaded_pdf(uploaded_file):
    """Persist uploaded PDF to a temporary file path and return the path."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def _process_user_query(question_text):
    """Run retrieval + LLM generation and append user & assistant turns."""
    cleaned_q = (question_text or "").strip()
    if not cleaned_q:
        return

    st.session_state.messages.append({"role": "user", "content": cleaned_q})

    vectorstore = st.session_state.vectorstore
    if vectorstore is None:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "Please upload and process a PDF document first so I can search its pages "
                    "and answer your question."
                ),
                "page_numbers": [],
                "is_fallback": True,
                "options": [
                    "Upload a PDF from the sidebar and click 'Process PDF'.",
                    "Try a shorter document if you have multiple files.",
                    "Ask a general question after your PDF is indexed.",
                ],
            }
        )
        return

    retrieval = retrieve_relevant_chunks(vectorstore, cleaned_q, k=4)
    result = generate_rag_answer(
        query=cleaned_q,
        context_text=retrieval.get("context", ""),
        page_numbers=retrieval.get("page_numbers", []),
        suggested_questions=st.session_state.suggested_questions,
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.get("answer", ""),
            "page_numbers": result.get("page_numbers", []),
            "is_fallback": result.get("is_fallback", False),
            "options": result.get("options", []),
        }
    )


def _render_sidebar():
    """Render sidebar controls for user profile, temporary chat, and PDF upload."""
    with st.sidebar:
        st.markdown("## RAGMentor")
        st.caption(
            f"Signed in as: **{st.session_state.user_label}** ({st.session_state.auth_mode})"
        )

        render_temp_chat_toggle()

        col_clear, col_logout = st.columns(2)
        with col_clear:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.voice_question = ""
                st.rerun()
        with col_logout:
            if st.button("Logout", use_container_width=True):
                logout_user()
                st.session_state.messages = []
                st.session_state.vectorstore = None
                st.session_state.doc_summary = ""
                st.session_state.suggested_questions = []
                st.rerun()

        st.divider()
        st.markdown("### 1. Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

        if uploaded_file is not None:
            if st.button("Process PDF", type="primary", use_container_width=True):
                with st.spinner("Extracting pages, building summary, and indexing chunks..."):
                    tmp_path = None
                    try:
                        tmp_path = _save_uploaded_pdf(uploaded_file)
                        ingestion = ingest_pdf(tmp_path)
                        vs = build_vector_store(ingestion["chunks"])

                        st.session_state.vectorstore = vs
                        st.session_state.uploaded_filename = uploaded_file.name
                        st.session_state.total_pages = ingestion["total_pages"]
                        st.session_state.doc_summary = ingestion["summary"]
                        st.session_state.suggested_questions = ingestion["relevant_questions"]
                        if not st.session_state.temp_chat:
                            st.session_state.messages = []

                        st.success(
                            f"Indexed **{uploaded_file.name}** ({ingestion['total_pages']} pages)."
                        )
                    except Exception as exc:
                        st.error(f"Failed to process PDF: {exc}")
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass

        if st.session_state.doc_summary:
            st.divider()
            st.markdown("### Quick Summary")
            st.write(st.session_state.doc_summary)


def _render_starter_questions():
    """Render clickable relevant questions generated from the document."""
    questions = st.session_state.suggested_questions
    if not questions:
        return

    st.markdown("#### Suggested Questions from Your Document")
    cols = st.columns(2)
    for idx, q in enumerate(questions):
        with cols[idx % 2]:
            if st.button(q, key=f"starter_q_{idx}", use_container_width=True):
                st.session_state.pending_prompt = q
                st.rerun()


def _render_voice_input():
    """Render voice recording input and transcribe to a question."""
    if hasattr(st, "audio_input"):
        audio_val = st.audio_input("Speak your question")
        if audio_val is not None:
            if st.button("Transcribe Voice to Question", key="transcribe_btn"):
                with st.spinner("Transcribing audio..."):
                    res = transcribe_audio_bytes(audio_val)
                if res.get("error"):
                    st.warning(res["error"])
                elif res.get("text"):
                    st.session_state.voice_question = res["text"]
                    st.success(f"Heard: \"{res['text']}\"")

        if st.session_state.voice_question:
            st.info(f"Ready to ask: **{st.session_state.voice_question}**")
            if st.button("Ask Transcribed Question", type="primary", key="ask_voice_btn"):
                q = st.session_state.voice_question
                st.session_state.voice_question = ""
                st.session_state.pending_prompt = q
                st.rerun()


def main():
    _init_app_state()

    if not st.session_state.authenticated:
        render_login_screen()
        return

    _render_sidebar()

    st.title("RAGMentor — PDF Study & Q&A Assistant")
    if st.session_state.uploaded_filename:
        st.caption(
            f"Active Document: **{st.session_state.uploaded_filename}** "
            f"({st.session_state.total_pages} pages)"
        )
    else:
        st.caption("Upload a PDF in the sidebar to get a quick summary, starter questions, and cited answers.")

    _render_starter_questions()
    _render_voice_input()

    st.divider()

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                pages = msg.get("page_numbers") or []
                if pages:
                    page_str = ", ".join(str(p) for p in pages)
                    st.caption(f"Source Pages: {page_str}")

                if msg.get("is_fallback") and msg.get("options"):
                    st.markdown("**Try one of these options:**")
                    for opt in msg["options"]:
                        st.markdown(f"- {opt}")

    # Handle pending question from starter buttons or voice transcription
    if st.session_state.pending_prompt:
        prompt_to_run = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        _process_user_query(prompt_to_run)
        st.rerun()

    # Standard text chat input
    typed_prompt = st.chat_input("Ask a question about your uploaded PDF...")
    if typed_prompt:
        _process_user_query(typed_prompt)
        st.rerun()


if __name__ == "__main__":
    main()