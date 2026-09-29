import streamlit as st


def render_metrics():
    st.markdown("## 📊 Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Documents", 0)

    with col2:
        st.metric("Chats", 0)

    st.markdown("---")
    st.markdown("### 📈 RAG Performance")

    st.metric("Retrieved Chunks", 0)
    st.metric("Response Time", "0s")