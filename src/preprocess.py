# =============================================================================
# preprocess.py - Privacy-Safe Data Preprocessing
# =============================================================================
"""
This module handles data cleaning with a focus on PRIVACY.

KEY CONCEPTS FOR BEGINNERS:
- PII (Personally Identifiable Information) must NEVER be embedded
- We drop names and emails completely
- We create a 'document_text' field that combines safe text fields
- This document_text is what gets embedded and searched

PRIVACY RULES:
1. Customer Name - DROPPED (never embedded)
2. Customer Email - DROPPED (never embedded)  
3. Customer Age/Gender - Kept as metadata only, NOT in embedded text
4. Ticket ID - Kept for traceability
"""

import re
import pandas as pd
from typing import List, Optional

from . import config


def drop_pii_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns containing Personally Identifiable Information.
    
    This is the FIRST and MOST IMPORTANT step in preprocessing.
    We completely remove Customer Name and Customer Email.
    
    Args:
        df: Raw DataFrame with all columns
        
    Returns:
        DataFrame with PII columns removed
        
    Example:
        >>> df_safe = drop_pii_columns(df_raw)
        >>> 'Customer Name' in df_safe.columns
        False
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Drop PII columns
    columns_to_drop = []
    for col in config.PII_COLUMNS_TO_DROP:
        if col in df_clean.columns:
            columns_to_drop.append(col)
    
    if columns_to_drop:
        df_clean = df_clean.drop(columns=columns_to_drop)
        print(f"✓ Dropped PII columns: {columns_to_drop}")
    else:
        print("⚠ No PII columns found to drop")
    
    return df_clean


def create_document_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the 'document_text' field that will be embedded.
    
    This combines:
    - Ticket Subject (the title/summary)
    - Ticket Description (the main content)
    - Resolution (if available - contains solution info)
    
    IMPORTANT: This text does NOT include any PII!
    
    Args:
        df: DataFrame (should already have PII removed)
        
    Returns:
        DataFrame with new 'document_text' column
        
    Example:
        >>> df = create_document_text(df)
        >>> print(df['document_text'].iloc[0][:100])
        'Subject: Product setup. Description: I'm having an issue...'
    """
    df = df.copy()
    
    def combine_text_fields(row) -> str:
        """Combine text fields for a single row."""
        parts = []
        
        # Add subject if available
        subject = row.get("Ticket Subject", "")
        if pd.notna(subject) and str(subject).strip():
            parts.append(f"Subject: {str(subject).strip()}")
        
        # Add description if available
        description = row.get("Ticket Description", "")
        if pd.notna(description) and str(description).strip():
            parts.append(f"Description: {str(description).strip()}")
        
        # Add resolution if available (very useful for RAG!)
        resolution = row.get("Resolution", "")
        if pd.notna(resolution) and str(resolution).strip():
            parts.append(f"Resolution: {str(resolution).strip()}")
        
        # Join with newlines for readability
        return "\n".join(parts)
    
    # Apply to all rows
    df["document_text"] = df.apply(combine_text_fields, axis=1)
    
    print(f"✓ Created 'document_text' column")
    return df


def normalize_text(text: str) -> str:
    """
    Apply basic text normalization.
    
    This cleans up the text without being too aggressive:
    - Strip leading/trailing whitespace
    - Normalize multiple spaces to single space
    - Normalize multiple newlines to double newline
    
    Args:
        text: Raw text string
        
    Returns:
        Normalized text string
        
    Example:
        >>> normalize_text("  Hello   world  ")
        'Hello world'
    """
    if not isinstance(text, str):
        return ""
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Replace 3+ newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text


def apply_text_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text normalization to the document_text column.
    
    Args:
        df: DataFrame with 'document_text' column
        
    Returns:
        DataFrame with normalized text
    """
    df = df.copy()
    
    if "document_text" in df.columns:
        df["document_text"] = df["document_text"].apply(normalize_text)
        print("✓ Applied text normalization")
    else:
        print("⚠ No 'document_text' column found to normalize")
    
    return df


def verify_no_emails_in_text(df: pd.DataFrame, column: str = "document_text") -> bool:
    """
    Verify that no email addresses appear in the specified column.
    
    This is a SAFETY CHECK to ensure we didn't accidentally include emails.
    
    Args:
        df: DataFrame to check
        column: Column name to check
        
    Returns:
        True if no emails found (safe), False if emails found (unsafe!)
        
    Example:
        >>> is_safe = verify_no_emails_in_text(df)
        >>> assert is_safe, "Found emails in document text!"
    """
    # Simple email regex pattern
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    
    # Check each row
    emails_found = []
    for idx, text in df[column].items():
        if pd.notna(text):
            matches = re.findall(email_pattern, str(text))
            if matches:
                emails_found.extend(matches)
    
    if emails_found:
        print(f"⚠ WARNING: Found {len(emails_found)} email(s) in '{column}'!")
        print(f"  Examples: {emails_found[:3]}")
        return False
    else:
        print(f"✓ No emails found in '{column}' - safe to embed")
        return True


def preprocess_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline.
    
    This is the main function that combines all preprocessing steps:
    1. Drop PII columns (names, emails)
    2. Create document_text field
    3. Normalize text
    4. Verify no emails leaked into text
    
    Args:
        df: Raw DataFrame from CSV
        
    Returns:
        Cleaned DataFrame ready for embedding
        
    Example:
        >>> df_raw = load_raw_tickets()
        >>> df_clean = preprocess_tickets(df_raw)
        >>> save_processed_tickets(df_clean)
    """
    print("=" * 50)
    print("Starting preprocessing pipeline...")
    print("=" * 50)
    
    # Step 1: Drop PII columns
    df = drop_pii_columns(df)
    
    # Step 2: Create document text
    df = create_document_text(df)
    
    # Step 3: Normalize text
    df = apply_text_normalization(df)
    
    # Step 4: Safety check - verify no emails
    is_safe = verify_no_emails_in_text(df)
    if not is_safe:
        print("⚠ WARNING: Emails found in document text!")
        print("  Please review the data before proceeding.")
    
    print("=" * 50)
    print(f"✓ Preprocessing complete. {len(df):,} tickets processed.")
    print("=" * 50)
    
    return df


def get_text_stats(df: pd.DataFrame, column: str = "document_text") -> dict:
    """
    Calculate statistics about text length.
    
    Useful for understanding your data and choosing chunk sizes.
    
    Args:
        df: DataFrame with text column
        column: Name of text column
        
    Returns:
        Dictionary with min, max, mean, median word counts
    """
    # Calculate word counts
    word_counts = df[column].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    
    stats = {
        "min_words": int(word_counts.min()),
        "max_words": int(word_counts.max()),
        "mean_words": round(word_counts.mean(), 1),
        "median_words": int(word_counts.median()),
        "total_documents": len(df),
        "empty_documents": int((word_counts == 0).sum())
    }
    
    return stats
