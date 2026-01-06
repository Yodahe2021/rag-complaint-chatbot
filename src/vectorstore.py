# =============================================================================
# vectorstore.py - FAISS Vector Store Operations
# =============================================================================
"""
This module handles the vector store (FAISS) for storing and retrieving embeddings.

KEY CONCEPTS FOR BEGINNERS:

WHAT IS A VECTOR STORE?
- A database optimized for storing and searching vectors (embeddings)
- Each document is converted to a vector (list of numbers)
- Similar documents have similar vectors (close in vector space)
- We can find relevant documents by searching for similar vectors

WHY FAISS?
- Fast similarity search library
- Works fully locally (no server)
- Easy to persist via LangChain (save/load)
- Great for learning and small-to-medium projects

EMBEDDINGS:
- Text → Numbers (vector of 384 dimensions for MiniLM)
- "billing problem" → [0.12, -0.34, 0.56, ...]
- Similar texts have similar vectors
- This enables semantic search (meaning-based, not just keyword)
"""

from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from . import config


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load the embedding model.
    
    We use sentence-transformers/all-MiniLM-L6-v2:
    - Small and fast (~80MB)
    - Good quality embeddings
    - 384-dimensional vectors
    - Works well on CPU
    
    Returns:
        HuggingFaceEmbeddings object ready to use
        
    Note:
        First call downloads the model (~80MB) to the models/hf/ folder.
        Subsequent calls load from cache (fast).
    """
    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
    print(f"  (First run will download ~80MB to {config.MODELS_DIR})")
    
    # Create embeddings object
    # model_kwargs: passed to the underlying model
    # encode_kwargs: passed when encoding text
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},  # Force CPU (works everywhere)
        encode_kwargs={"normalize_embeddings": True},  # Normalize for cosine similarity
        cache_folder=str(config.MODELS_DIR),  # Cache in our project folder
    )
    
    print("✓ Embedding model loaded")
    return embeddings


def create_vector_store(
    documents: List[Document],
    persist_directory: Optional[Path] = None
) -> FAISS:
    """
    Create a new FAISS vector store from documents.
    
    This function:
    1. Loads the embedding model
    2. Embeds all documents (converts text to vectors)
    3. Stores vectors in FAISS
    4. Persists to disk for later use
    
    Args:
        documents: List of LangChain Documents to embed
        persist_directory: Where to save the database (default from config)
        
    Returns:
        FAISS vector store object
        
    Example:
        >>> chunks = chunk_documents(docs)
        >>> vectorstore = create_vector_store(chunks)
        >>> # Now you can search!
    """
    if persist_directory is None:
        persist_directory = config.VECTOR_STORE_DIR
    
    persist_directory = Path(persist_directory)
    
    # Create directory if needed
    persist_directory.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating vector store with {len(documents):,} documents...")
    print(f"  Persist directory: {persist_directory}")
    
    # Get embedding model
    embeddings = get_embedding_model()
    
    # Create FAISS vector store
    # This embeds all documents and stores them in an in-memory FAISS index.
    # We'll then persist to disk with save_local().
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    # Persist FAISS index and docstore to disk
    # This creates files like:
    # - index.faiss
    # - index.pkl
    vectorstore.save_local(str(persist_directory))

    # Helpful info
    try:
        n_total = int(vectorstore.index.ntotal)
        print(f"✓ FAISS index built (ntotal={n_total:,})")
    except Exception:
        print("✓ FAISS index built")

    print(f"✓ Vector store persisted to {persist_directory}")
    return vectorstore


def load_vector_store(persist_directory: Optional[Path] = None) -> FAISS:
    """
    Load an existing FAISS vector store from disk.
    
    Use this after you've already created and persisted a vector store.
    Much faster than re-embedding all documents!
    
    Args:
        persist_directory: Where the database is saved (default from config)
        
    Returns:
        FAISS vector store object
        
    Raises:
        FileNotFoundError: If the vector store doesn't exist
        
    Example:
        >>> vectorstore = load_vector_store()
        >>> results = vectorstore.similarity_search("billing issue", k=3)
    """
    if persist_directory is None:
        persist_directory = config.VECTOR_STORE_DIR
    
    persist_directory = Path(persist_directory)
    
    # Check if directory exists
    if not persist_directory.exists():
        raise FileNotFoundError(
            f"Vector store not found at: {persist_directory}\n"
            f"Please run the indexing notebook (01_chunk_embed_index.ipynb) first."
        )
    
    print(f"Loading vector store from {persist_directory}...")
    
    # Get embedding model (needed for queries)
    embeddings = get_embedding_model()
    
    # Load existing FAISS index
    # allow_dangerous_deserialization=True is required because FAISS stores
    # the docstore in a pickle file (index.pkl).
    vectorstore = FAISS.load_local(
        str(persist_directory),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    try:
        n_total = int(vectorstore.index.ntotal)
        print(f"✓ Vector store loaded (ntotal={n_total:,})")
    except Exception:
        print("✓ Vector store loaded")

    return vectorstore


def get_retriever(vectorstore: FAISS, k: int = None):
    """
    Create a retriever from the vector store.
    
    A retriever is a simple interface for searching:
    - Input: query string
    - Output: list of relevant documents
    
    Args:
        vectorstore: FAISS vector store
        k: Number of documents to retrieve (default from config)
        
    Returns:
        LangChain retriever object
        
    Example:
        >>> retriever = get_retriever(vectorstore, k=5)
        >>> docs = retriever.invoke("refund request")
    """
    if k is None:
        k = config.RETRIEVAL_K
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    
    print(f"✓ Created retriever (k={k})")
    return retriever


def search_similar(
    vectorstore: FAISS,
    query: str,
    k: int = None
) -> List[Document]:
    """
    Search for documents similar to a query.
    
    This is a convenience function for quick searches.
    
    Args:
        vectorstore: FAISS vector store
        query: Search query (natural language)
        k: Number of results to return
        
    Returns:
        List of similar Documents with metadata
        
    Example:
        >>> results = search_similar(vectorstore, "device not turning on", k=3)
        >>> for doc in results:
        ...     print(doc.metadata['ticket_id'], doc.page_content[:100])
    """
    if k is None:
        k = config.RETRIEVAL_K
    
    results = vectorstore.similarity_search(query, k=k)
    return results


def search_with_scores(
    vectorstore: FAISS,
    query: str,
    k: int = None
) -> List[tuple]:
    """
    Search for documents and return similarity scores.
    
    Scores help you understand how relevant each result is.
    Lower score = more similar (for distance-based metrics).
    
    Args:
        vectorstore: FAISS vector store
        query: Search query
        k: Number of results
        
    Returns:
        List of (Document, score) tuples
        
    Example:
        >>> results = search_with_scores(vectorstore, "billing", k=3)
        >>> for doc, score in results:
        ...     print(f"Score: {score:.3f} - {doc.metadata['ticket_id']}")
    """
    if k is None:
        k = config.RETRIEVAL_K
    
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results


def print_search_results(results: List[Document], query: str) -> None:
    """
    Pretty print search results.
    
    Args:
        results: List of Documents from search
        query: The original query (for display)
    """
    print("=" * 70)
    print(f"SEARCH RESULTS for: '{query}'")
    print("=" * 70)
    
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Ticket ID: {doc.metadata.get('ticket_id', 'N/A')}")
        print(f"Product: {doc.metadata.get('product', 'N/A')}")
        print(f"Type: {doc.metadata.get('ticket_type', 'N/A')}")
        print(f"Priority: {doc.metadata.get('priority', 'N/A')}")
        print(f"Content preview:\n{doc.page_content[:300]}...")
    
    print("\n" + "=" * 70)
