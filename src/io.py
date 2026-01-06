# =============================================================================
# io.py - Data Input/Output Utilities
# =============================================================================
"""
This module handles loading and saving data files.

KEY CONCEPTS FOR BEGINNERS:
- Separating I/O from processing makes code more testable
- Using pandas for CSV operations is standard in data science
- Always check if files exist before loading to give clear errors
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from . import config


def load_raw_tickets(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the raw tickets CSV file.
    
    Args:
        filepath: Path to the CSV file. If None, uses config.RAW_TICKETS_PATH
        
    Returns:
        pandas DataFrame with raw ticket data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        
    Example:
        >>> df = load_raw_tickets()
        >>> print(df.shape)
        (8000, 17)
    """
    # Use default path if none provided
    if filepath is None:
        filepath = config.RAW_TICKETS_PATH
    
    # Convert to Path object if string
    filepath = Path(filepath)
    
    # Check if file exists and give helpful error message
    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw tickets file not found at: {filepath}\n"
            f"Please place your Kaggle dataset at: {config.RAW_TICKETS_PATH}"
        )
    
    # Load the CSV
    # low_memory=False prevents mixed type warnings for large files
    df = pd.read_csv(filepath, low_memory=False)
    
    print(f"✓ Loaded {len(df):,} tickets from {filepath.name}")
    return df


def save_processed_tickets(df: pd.DataFrame, filepath: Optional[Path] = None) -> Path:
    """
    Save the processed (cleaned) tickets to CSV.
    
    Args:
        df: DataFrame with cleaned ticket data
        filepath: Where to save. If None, uses config.PROCESSED_TICKETS_PATH
        
    Returns:
        Path where the file was saved
        
    Example:
        >>> save_processed_tickets(df_clean)
        ✓ Saved 8000 tickets to tickets_clean.csv
    """
    # Use default path if none provided
    if filepath is None:
        filepath = config.PROCESSED_TICKETS_PATH
    
    filepath = Path(filepath)
    
    # Create parent directories if they don't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    # index=False prevents adding an extra index column
    df.to_csv(filepath, index=False)
    
    print(f"✓ Saved {len(df):,} tickets to {filepath.name}")
    return filepath


def load_processed_tickets(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the processed (cleaned) tickets CSV file.
    
    Args:
        filepath: Path to the CSV file. If None, uses config.PROCESSED_TICKETS_PATH
        
    Returns:
        pandas DataFrame with cleaned ticket data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if filepath is None:
        filepath = config.PROCESSED_TICKETS_PATH
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed tickets file not found at: {filepath}\n"
            f"Please run the preprocessing notebook (00_eda_and_cleaning.ipynb) first."
        )
    
    df = pd.read_csv(filepath, low_memory=False)
    
    print(f"✓ Loaded {len(df):,} processed tickets from {filepath.name}")
    return df
