# =============================================================================
# app.py - Streamlit RAG Demo Application
# =============================================================================
"""
A simple Streamlit web app for interactive RAG question answering.

Run with: streamlit run app.py

Features:
- Text input for questions
- Answer display with sources
- Expandable source details
- Clear button to reset
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up HuggingFace cache BEFORE importing other modules
from src.config import setup_hf_cache
setup_hf_cache()

from src.rag_pipeline import RAGPipeline
from src import config


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Support Ticket RAG",
    page_icon="🎫",
    layout="wide",
)


# =============================================================================
# CACHING - Load models once
# =============================================================================

@st.cache_resource
def load_rag_pipeline():
    """
    Load the RAG pipeline (cached to avoid reloading on each interaction).
    
    @st.cache_resource ensures this only runs once per session.
    """
    with st.spinner("Loading RAG pipeline... (this may take a minute on first run)"):
        rag = RAGPipeline(retrieval_k=5)
    return rag


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.title("🎫 Support Ticket RAG Assistant")
    st.markdown("""
    Ask questions about customer support tickets and get answers with sources.
    
    **How it works:**
    1. Your question is used to find relevant support tickets
    2. The retrieved tickets are used as context for the LLM
    3. The LLM generates an answer based on the context
    """)
    
    st.divider()
    
    # Load RAG pipeline
    try:
        rag = load_rag_pipeline()
        st.success("✓ RAG pipeline loaded successfully!")
    except FileNotFoundError as e:
        st.error(f"""
        **Vector store not found!**
        
        Please run the notebooks first:
        1. `00_eda_and_cleaning.ipynb` - Preprocess the data
        2. `01_chunk_embed_index.ipynb` - Create the vector store
        
        Error: {e}
        """)
        return
    except Exception as e:
        st.error(f"Error loading RAG pipeline: {e}")
        return
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This demo uses:
        - **Embeddings**: all-MiniLM-L6-v2
        - **LLM**: FLAN-T5-small
        - **Vector Store**: FAISS
        
        **Configuration:**
        """)
        st.code(f"""
Retrieval k: {config.RETRIEVAL_K}
Chunk size: {config.CHUNK_SIZE}
Chunk overlap: {config.CHUNK_OVERLAP}
        """)
        
        st.divider()
        
        st.header("💡 Sample Questions")
        sample_questions = [
            "What are common billing issues?",
            "What problems do customers have with smart TVs?",
            "How are refund requests handled?",
            "What technical issues are reported?",
            "What are critical priority issues?",
        ]
        for q in sample_questions:
            if st.button(q, key=f"sample_{q[:20]}"):
                st.session_state.question = q
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Question input
        st.subheader("🔍 Ask a Question")
        
        # Initialize session state for question
        if "question" not in st.session_state:
            st.session_state.question = ""
        
        question = st.text_area(
            "Enter your question about support tickets:",
            value=st.session_state.question,
            height=100,
            placeholder="e.g., What are the most common issues customers report?",
            key="question_input"
        )
        
        # Buttons
        button_col1, button_col2 = st.columns(2)
        with button_col1:
            ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)
        with button_col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.question = ""
                st.session_state.pop("response", None)
                st.rerun()
    
    with col2:
        st.subheader("⚙️ Settings")
        retrieval_k = st.slider(
            "Number of sources to retrieve (k)",
            min_value=1,
            max_value=10,
            value=5,
            help="More sources = more context, but may include noise"
        )
    
    st.divider()
    
    # Process question
    if ask_button and question.strip():
        with st.spinner("🔄 Searching and generating answer..."):
            try:
                # Update retrieval k if changed
                rag.retrieval_k = retrieval_k
                rag.retriever = rag.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": retrieval_k}
                )
                
                # Get response
                response = rag.ask(question)
                st.session_state.response = response
                
            except Exception as e:
                st.error(f"Error generating answer: {e}")
                return
    
    # Display response
    if "response" in st.session_state:
        response = st.session_state.response
        
        # Answer section
        st.subheader("💬 Answer")
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
        {response.answer}
        </div>
        """, unsafe_allow_html=True)
        
        # Sources section
        st.subheader(f"📚 Sources ({len(response.sources)} documents)")
        
        for i, doc in enumerate(response.sources, 1):
            with st.expander(
                f"Source {i}: Ticket {doc.metadata.get('ticket_id', 'N/A')} - "
                f"{doc.metadata.get('product', 'Unknown Product')}"
            ):
                # Metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Type:** {doc.metadata.get('ticket_type', 'N/A')}")
                with col2:
                    st.markdown(f"**Priority:** {doc.metadata.get('priority', 'N/A')}")
                with col3:
                    st.markdown(f"**Status:** {doc.metadata.get('status', 'N/A')}")
                
                st.divider()
                
                # Content
                st.markdown("**Content:**")
                st.text(doc.page_content)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 12px;">
    Built with LangChain, FAISS, and Streamlit | 
    RAG Pipeline Demo for Customer Support Tickets
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# RUN APP
# =============================================================================

if __name__ == "__main__":
    main()
