# =============================================================================
# config.py - Central Configuration for RAG Pipeline
# =============================================================================
"""
This module contains all configuration settings for the RAG pipeline.
Centralizing config makes it easy to change paths, model names, and parameters.

KEY CONCEPTS FOR BEGINNERS:
- Configuration files help avoid "magic numbers" scattered in code
- Using Path objects makes code work across Windows/Mac/Linux
- Environment variables allow runtime customization without code changes
"""

import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Get the project root directory (parent of src/)
# __file__ is the path to this config.py file
# .parent gets the src/ folder, .parent again gets the project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Input file: raw tickets from Kaggle
RAW_TICKETS_PATH = RAW_DATA_DIR / "tickets.csv"

# Output file: cleaned tickets (PII removed)
PROCESSED_TICKETS_PATH = PROCESSED_DATA_DIR / "tickets_clean.csv"

# Vector store path (FAISS index will persist here)
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store" / "faiss"

# Model cache directory (HuggingFace models will be downloaded here)
MODELS_DIR = PROJECT_ROOT / "models" / "hf"
VECTOR_STORE_PATH = "models/faiss_store"
# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Embedding model: sentence-transformers/all-MiniLM-L6-v2
# - Small and fast (80MB)
# - Good quality for semantic search
# - Produces 384-dimensional vectors
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM model: google/flan-t5-small
# - Small and CPU-friendly (~300MB)
# - Good for instruction-following tasks
# - Produces reasonable quality for demos
LLM_MODEL_NAME = "google/flan-t5-small"

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================

# Chunk size: how many characters per chunk
# - 500 is a good balance between context and specificity
# - Too small = loses context, too large = dilutes relevance
CHUNK_SIZE = 500

# Chunk overlap: how many characters overlap between chunks
# - Overlap helps preserve context at chunk boundaries
# - 50 chars is ~10% overlap, which is reasonable
CHUNK_OVERLAP = 50

# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================

# Number of documents to retrieve for each query
# - More docs = more context but slower and may include noise
# - 5 is a good starting point
RETRIEVAL_K = 5

# =============================================================================
# HUGGINGFACE CACHE SETUP
# =============================================================================

def setup_hf_cache():
    """
    Configure HuggingFace to cache models in our project's models/ folder.
    
    WHY THIS MATTERS:
    - By default, HF downloads models to ~/.cache/huggingface/
    - Setting HF_HOME keeps everything in our project folder
    - This makes the project more portable and self-contained
    
    CALL THIS EARLY in your code (before importing transformers/sentence_transformers)
    """
    # Create the models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set the environment variable
    # HF_HOME is the master setting that controls all HF caching
    os.environ["HF_HOME"] = str(MODELS_DIR)
    
    # Also set these for older versions of the libraries
    os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODELS_DIR)
    
    print(f"✓ HuggingFace cache set to: {MODELS_DIR}")


# =============================================================================
# PII (Personally Identifiable Information) COLUMNS
# =============================================================================

# Columns that contain PII and MUST be removed before embedding
# NEVER include these in the text that gets embedded!
PII_COLUMNS_TO_DROP = ["Customer Name", "Customer Email"]

# Columns to keep as metadata (not embedded, but stored for reference)
# Age and Gender are kept as metadata but NOT included in embedded text
METADATA_COLUMNS = [
    "Ticket ID",
    "Customer Age", 
    "Customer Gender",
    "Product Purchased",
    "Date of Purchase",
    "Ticket Type",
    "Ticket Status",
    "Resolution",
    "Ticket Priority",
    "Ticket Channel",
    "First Response Time",
    "Time to Resolution",
    "Customer Satisfaction Rating"
]

# Columns that will be combined into the document text for embedding
TEXT_COLUMNS_FOR_EMBEDDING = ["Ticket Subject", "Ticket Description", "Resolution"]
