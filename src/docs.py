# =============================================================================
# docs.py - Convert Data to LangChain Documents
# =============================================================================
"""
This module converts pandas DataFrames to LangChain Document objects.

KEY CONCEPTS FOR BEGINNERS:
- LangChain uses Document objects as a standard format
- Each Document has:
  - page_content: The text that gets embedded and searched
  - metadata: Additional info (ticket ID, product, etc.) for filtering/display
- Metadata is NOT embedded, but is stored alongside for reference
"""

import pandas as pd
from typing import List, Dict, Any
from langchain_core.documents import Document


def row_to_document(row: pd.Series) -> Document:
    """
    Convert a single DataFrame row to a LangChain Document.
    
    Args:
        row: A pandas Series (one row from DataFrame)
        
    Returns:
        LangChain Document object
        
    Example:
        >>> doc = row_to_document(df.iloc[0])
        >>> print(doc.page_content[:50])
        'Subject: Product setup...'
    """
    # The text content that will be embedded
    # This should be the 'document_text' field we created in preprocessing
    page_content = str(row.get("document_text", ""))
    
    # Metadata: information we want to keep but NOT embed
    # This is useful for:
    # - Filtering results (e.g., only show high priority tickets)
    # - Displaying context to users (e.g., show ticket ID)
    # - Traceability (linking back to original data)
    metadata = {
        "ticket_id": row.get("Ticket ID", ""),
        "product": row.get("Product Purchased", ""),
        "ticket_type": row.get("Ticket Type", ""),
        "ticket_subject": row.get("Ticket Subject", ""),
        "status": row.get("Ticket Status", ""),
        "priority": row.get("Ticket Priority", ""),
        "channel": row.get("Ticket Channel", ""),
        "date_of_purchase": str(row.get("Date of Purchase", "")),
        "satisfaction_rating": row.get("Customer Satisfaction Rating", None),
    }
    
    # Clean up metadata: convert NaN to None, ensure strings
    cleaned_metadata = {}
    for key, value in metadata.items():
        if pd.isna(value):
            cleaned_metadata[key] = None
        elif isinstance(value, (int, float)):
            cleaned_metadata[key] = value
        else:
            cleaned_metadata[key] = str(value)
    
    return Document(page_content=page_content, metadata=cleaned_metadata)


def dataframe_to_documents(df: pd.DataFrame) -> List[Document]:
    """
    Convert entire DataFrame to list of LangChain Documents.
    
    This is the main function you'll use to prepare data for the vector store.
    
    Args:
        df: DataFrame with 'document_text' column and metadata columns
        
    Returns:
        List of LangChain Document objects
        
    Example:
        >>> df = load_processed_tickets()
        >>> docs = dataframe_to_documents(df)
        >>> print(f"Created {len(docs)} documents")
        Created 8000 documents
    """
    documents = []
    
    for idx, row in df.iterrows():
        doc = row_to_document(row)
        
        # Skip empty documents
        if doc.page_content.strip():
            documents.append(doc)
    
    print(f"✓ Converted {len(documents):,} rows to LangChain Documents")
    
    # Show sample
    if documents:
        print(f"  Sample metadata keys: {list(documents[0].metadata.keys())}")
    
    return documents


def print_document_sample(doc: Document, max_content_length: int = 200) -> None:
    """
    Pretty print a Document for inspection.
    
    Useful for debugging and understanding your data.
    
    Args:
        doc: LangChain Document to display
        max_content_length: Truncate content to this many characters
    """
    print("=" * 60)
    print("DOCUMENT SAMPLE")
    print("=" * 60)
    
    # Show content (truncated)
    content = doc.page_content
    if len(content) > max_content_length:
        content = content[:max_content_length] + "..."
    print(f"Content:\n{content}")
    
    print("-" * 60)
    print("Metadata:")
    for key, value in doc.metadata.items():
        print(f"  {key}: {value}")
    print("=" * 60)
