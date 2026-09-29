import streamlit as st
from frontend.chat_interface import render_chat
from frontend.metrics_panel import render_metrics

# Interface setup[cite: 2]
st.set_page_config(page_title="RAGMentor", layout="wide", page_icon="📚")

def main():
    # Left Sidebar Navigation[cite: 2]
    with st.sidebar:
        st.markdown("## 📚 RAGMentor")
        st.caption("Learn Smarter with Your Documents")
        st.button("＋ New Chat", use_container_width=True)
        st.markdown("---")
        st.markdown("🏠 Home\n\n📄 My Documents\n\n🕒 Chat History\n\n📑 Document Summary\n\n⚙️ Settings")
        st.markdown("---")
        st.markdown("### Recent Chats")
        st.markdown("💬 Capital of France?\n\n💬 Key findings of the paper\n\n💬 Explain methodology")
        
    # Main split layout: Center Chat (66%) and Right Metrics (33%)[cite: 2]
    chat_col, metrics_col = st.columns([2, 1])
    
    with chat_col:
        render_chat()
        
    with metrics_col:
        render_metrics()

if __name__ == "__main__":
    main()