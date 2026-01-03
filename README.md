# CFPB Consumer Complaint RAG Vector Store

This repository contains the pipeline to process, embed, and store CFPB consumer complaint narratives for **Retrieval-Augmented Generation (RAG)** tasks. It includes text chunking, dense vector embeddings with Sentence Transformers, and FAISS-based semantic search.

---

## 📂 Project Structure

project_root/
│
├─ README.md
├─ requirements.txt
├─ .gitignore
│
├─ data/
│ ├─ raw/ # Original raw complaint datasets
│ ├─ processed/ # Cleaned and preprocessed complaint data
│ └─ vector_store/ # FAISS index and metadata
│
├─ src/ # Source code modules
│ └─ vector_store_builder.py
│
├─ notebooks/ # Exploratory notebooks
│ └─ 02_build_vector_store.ipynb
│
├─ tests/ # Unit tests
│ └─ test_vector_store.py
│
└─ .github/workflows/ # Optional CI/CD pipelines

---

## 🛠 Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/cfpb-rag-vectorstore.git
cd cfpb-rag-vectorstore
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
Key dependencies include:

pandas, numpy – data manipulation

sentence-transformers – generating embeddings

faiss – vector store

tqdm – progress bars

scikit-learn – stratified sampling
data/processed/filtered_complaints.csv
