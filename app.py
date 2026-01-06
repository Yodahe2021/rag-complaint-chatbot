import streamlit as st
import sys
import os
from pathlib import Path

# Fix for LangChain 2026: Use community vectorstores
from langchain_community.vectorstores import Chroma, FAISS

# Ensure project root is in path so 'src' can be imported
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import local modules
try:
    from src.config import setup_hf_cache
    from src.rag_pipeline import RAGPipeline
    from src import config
except ImportError as e:
    st.error(f"Failed to import local modules from 'src'. Ensure an __init__.py exists in that folder. Error: {e}")
    st.stop()

# Set up HuggingFace cache
setup_hf_cache()

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Support Ticket RAG",
    page_icon="🎫",
    layout="wide",
)

@st.cache_resource
def load_rag_pipeline():
    """
    Load the RAG pipeline (cached to avoid reloading on each interaction).
    """
    with st.spinner("Loading RAG pipeline... (this may take a minute on first run)"):
        # Pass the required argument using the path from your config file
        rag = RAGPipeline(
            vector_store_path=config.VECTOR_STORE_PATH, # Added this line
            retrieval_k=5
        )
    return rag

def main():
    st.title("🎫 Support Ticket RAG Assistant")
    st.markdown("Ask questions about customer support tickets and get answers with sources.")
    st.divider()
    
    try:
        rag = load_rag_pipeline()
        st.success("✓ RAG pipeline loaded successfully!")
    except Exception as e:
        st.error(f"Error loading RAG pipeline: {e}")
        st.info("Tip: Ensure you have run your indexing notebooks to create the vector store folder.")
        return

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(f"**Vector Store:** {getattr(config, 'VECTOR_STORE_TYPE', 'FAISS')}")
        st.code(f"Retrieval k: {config.RETRIEVAL_K}\nChunk size: {config.CHUNK_SIZE}")
    
    # Question Input
    if "question" not in st.session_state:
        st.session_state.question = ""
    
    question = st.text_area("Enter your question:", value=st.session_state.question, height=100)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🚀 Ask", type="primary")
    
    if ask_button and question.strip():
        with st.spinner("Searching..."):
            response = rag.ask(question)
            st.subheader("💬 Answer")
            st.info(response.answer)
            
            st.subheader(f"📚 Sources ({len(response.sources)} documents)")
            for i, doc in enumerate(response.sources, 1):
                with st.expander(f"Source {i}: {doc.metadata.get('product', 'General')}"):
                    st.write(doc.page_content)

if __name__ == "__main__":
    main()
