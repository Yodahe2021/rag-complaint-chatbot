from typing import List, Dict
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS, Chroma # Added Chroma here
# --- Corrected Imports below ---
from langchain_core.documents import Document 
# BaseRetriever is not used in the class definition, so we can remove it for now.
from transformers import pipeline

from src.prompts import RAG_PROMPT_TEMPLATE


class RAGPipeline:
    # Change 'top_k' to 'retrieval_k' in the definition
    def __init__(
        self,
        vector_store_path: str,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        retrieval_k: int = 5, # Renamed from top_k
    ):
        # Update the internal variable name reference as well
        self.top_k = retrieval_k 
        # ... rest of the code is the same ...


        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Load vector store
        self.vector_store = Chroma(
            persist_directory=vector_store_path,
            embedding_function=self.embedding_model
        )

        # Load LLM
        self.llm = pipeline(
            "text-generation",
            model=llm_model_name,
            tokenizer=llm_model_name,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=True,
        )

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve top-k relevant complaint chunks"""
        return self.vector_store.similarity_search(query, k=self.top_k)

    def build_prompt(self, query: str, docs: List[Document]) -> str:
        """Construct prompt from retrieved documents"""
        context = "\n\n".join(
            f"- {doc.page_content}" for doc in docs
        )

        return RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=query
        )

    def generate_answer(self, prompt: str) -> str:
        """Generate answer using LLM"""
        output = self.llm(prompt)[0]["generated_text"]
        return output.split("Answer:")[-1].strip()

    def ask(self, query: str) -> Dict:
        """End-to-end RAG pipeline"""
        docs = self.retrieve(query)
        prompt = self.build_prompt(query, docs)
        answer = self.generate_answer(prompt)

        return {
            "question": query,
            "answer": answer,
            "sources": docs,
        }
