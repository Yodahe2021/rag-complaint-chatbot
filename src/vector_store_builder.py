from typing import List, Dict
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

def build_vector_store(
    df: pd.DataFrame,
    chunk_size: int = 300,
    overlap: int = 50,
    model_name: str = "all-MiniLM-L6-v2",
    vector_dir: str = "../data/vector_store"
):
    import os
    os.makedirs(vector_dir, exist_ok=True)

    # Chunking
    all_chunks, metadata = [], []
    for idx, row in df.iterrows():
        text = row.get("clean_narrative", "")
        if not text.strip():
            continue
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i+chunk_size])
            all_chunks.append(chunk)
            metadata.append({
                "Complaint ID": row["Complaint ID"],
                "Product": row["Product"]
            })

    # Embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)

    # FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Persist
    faiss.write_index(index, os.path.join(vector_dir, "complaints_faiss.index"))
    pd.DataFrame(metadata).to_csv(os.path.join(vector_dir, "complaints_metadata.csv"), index=False)
    
    print(f"FAISS index saved. Total vectors: {index.ntotal}")
    return index, all_chunks, metadata
