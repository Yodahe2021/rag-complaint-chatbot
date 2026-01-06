# =============================================================================
# chunking.py - Text Splitting for Embeddings
# =============================================================================
"""
This module handles text chunking (splitting) for the RAG pipeline.

KEY CONCEPTS FOR BEGINNERS:

WHY CHUNK TEXT?
- Embedding models have input limits (typically 256-512 tokens)
- Smaller chunks = more precise retrieval
- Larger chunks = more context but less precise
- We need to find a balance!

CHUNK SIZE vs OVERLAP:
- chunk_size: Maximum characters per chunk
- chunk_overlap: Characters shared between consecutive chunks
- Overlap helps preserve context at boundaries

EXAMPLE:
  Original: "The quick brown fox jumps over the lazy dog."
  chunk_size=20, overlap=5 might give:
  - Chunk 1: "The quick brown fox"
  - Chunk 2: "brown fox jumps over"  (overlaps "brown fox")
  - Chunk 3: "jumps over the lazy"
  - Chunk 4: "the lazy dog."

RecursiveCharacterTextSplitter:
- Tries to split on natural boundaries (paragraphs, sentences, words)
- Falls back to character-level if needed
- Better than naive character splitting!
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def create_text_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter with specified parameters.
    
    Args:
        chunk_size: Max characters per chunk (default from config)
        chunk_overlap: Overlap between chunks (default from config)
        
    Returns:
        Configured RecursiveCharacterTextSplitter
        
    Example:
        >>> splitter = create_text_splitter(chunk_size=300, chunk_overlap=30)
    """
    # Use config defaults if not specified
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = config.CHUNK_OVERLAP
    
    # Create the splitter
    # separators: Try to split on these in order (paragraph, newline, sentence, word, char)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # Use character count
        separators=["\n\n", "\n", ". ", " ", ""],  # Natural boundaries
        is_separator_regex=False,
    )
    
    print(f"✓ Created text splitter (chunk_size={chunk_size}, overlap={chunk_overlap})")
    return splitter


def chunk_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Document]:
    """
    Split documents into smaller chunks.
    
    Each chunk inherits the metadata from its parent document.
    We also add chunk-specific metadata (chunk_index).
    
    Args:
        documents: List of LangChain Documents to chunk
        chunk_size: Max characters per chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunked Documents (usually more than input)
        
    Example:
        >>> docs = dataframe_to_documents(df)  # 8000 docs
        >>> chunks = chunk_documents(docs)      # Maybe 12000 chunks
    """
    # Create splitter
    splitter = create_text_splitter(chunk_size, chunk_overlap)
    
    # Split all documents
    # split_documents preserves metadata automatically!
    chunked_docs = splitter.split_documents(documents)
    
    # Add chunk index to metadata for traceability
    # Group by ticket_id to number chunks within each ticket
    ticket_chunk_counts = {}
    for doc in chunked_docs:
        ticket_id = doc.metadata.get("ticket_id", "unknown")
        
        # Initialize or increment chunk counter for this ticket
        if ticket_id not in ticket_chunk_counts:
            ticket_chunk_counts[ticket_id] = 0
        
        doc.metadata["chunk_index"] = ticket_chunk_counts[ticket_id]
        ticket_chunk_counts[ticket_id] += 1
    
    # Report statistics
    original_count = len(documents)
    chunked_count = len(chunked_docs)
    expansion_ratio = chunked_count / original_count if original_count > 0 else 0
    
    print(f"✓ Chunking complete:")
    print(f"  Original documents: {original_count:,}")
    print(f"  After chunking: {chunked_count:,}")
    print(f"  Expansion ratio: {expansion_ratio:.2f}x")
    
    return chunked_docs


def get_chunk_stats(chunks: List[Document]) -> dict:
    """
    Calculate statistics about chunk sizes.
    
    Useful for understanding your chunking results.
    
    Args:
        chunks: List of chunked Documents
        
    Returns:
        Dictionary with min, max, mean, median chunk lengths
    """
    lengths = [len(doc.page_content) for doc in chunks]
    
    if not lengths:
        return {"error": "No chunks provided"}
    
    stats = {
        "total_chunks": len(chunks),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": round(sum(lengths) / len(lengths), 1),
        "median_length": sorted(lengths)[len(lengths) // 2],
    }
    
    return stats


def print_chunk_samples(chunks: List[Document], n_samples: int = 2) -> None:
    """
    Print sample chunks for inspection.
    
    Args:
        chunks: List of chunked Documents
        n_samples: Number of samples to show
    """
    print("=" * 60)
    print(f"CHUNK SAMPLES (showing {n_samples} of {len(chunks):,})")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks[:n_samples]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Length: {len(chunk.page_content)} chars")
        print(f"Ticket ID: {chunk.metadata.get('ticket_id', 'N/A')}")
        print(f"Chunk Index: {chunk.metadata.get('chunk_index', 'N/A')}")
        print(f"Content preview:\n{chunk.page_content[:200]}...")
        print()
