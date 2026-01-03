from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

# ------------------------------
# PDF Setup
# ------------------------------
pdf_file = "CFPB_RAG_Interim_Report.pdf"
doc = SimpleDocTemplate(pdf_file, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
styles = getSampleStyleSheet()
Story = []

# Custom styles
title_style = styles["Title"]
heading_style = styles["Heading2"]
subheading_style = styles["Heading3"]
normal_style = styles["BodyText"]

# ------------------------------
# Title
# ------------------------------
Story.append(Paragraph("Interim Report: CFPB Consumer Complaint Vector Store for RAG", title_style))
Story.append(Spacer(1, 12))
Story.append(Paragraph("Author: Yodahe Tsegaye", normal_style))
Story.append(Paragraph("Date: January 3, 2026", normal_style))
Story.append(Paragraph("GitHub Repository: https://github.com/Yodahe2021/cfpb-rag-vectorstore", normal_style))
Story.append(Spacer(1, 20))

# ------------------------------
# Executive Summary
# ------------------------------
Story.append(Paragraph("1. Executive Summary & Introduction", heading_style))
summary_text = """
This interim report summarizes the development of a text processing and semantic retrieval pipeline 
for CFPB consumer complaints. The objective is to prepare a Retrieval-Augmented Generation (RAG) system 
by converting complaint narratives into a vectorized format suitable for semantic search.
Key accomplishments include:
<ul>
<li>Data Preparation & Sampling: Filtered and cleaned 10k–15k complaints stratified across major product categories.</li>
<li>Text Chunking: Split long narratives into overlapping word-based segments to improve embedding quality.</li>
<li>Embedding Generation: Created dense vector embeddings using the 'all-MiniLM-L6-v2' model.</li>
<li>FAISS Vector Store: Built and persisted a FAISS vector index for efficient semantic similarity search.</li>
<li>Metadata Alignment: Stored chunk-level metadata including Complaint ID and Product for retrieval context.</li>
</ul>
"""
Story.append(Paragraph(summary_text, normal_style))
Story.append(Spacer(1, 12))

# ------------------------------
# Business Understanding
# ------------------------------
Story.append(Paragraph("2. Business Understanding & Problem Statement", heading_style))
context_text = """
Consumer complaints provide insights into customer issues, fraud risk, and product weaknesses. 
A RAG system enables organizations to quickly retrieve relevant complaints, analyze trends, 
and support automated report generation and decision-making.
"""
Story.append(Paragraph(context_text, normal_style))
Story.append(Spacer(1, 12))

challenge_text = """
Key Challenges:
<ul>
<li>Variable Text Lengths: Complaint narratives vary from a few words to several thousand words.</li>
<li>Preserving Context: Splitting text without losing semantic meaning requires overlapping chunks.</li>
<li>Efficient Search: Millions of chunks need to be embedded and searchable with low latency.</li>
<li>Stratified Sampling: Ensure balanced representation of complaints across financial products.</li>
</ul>
"""
Story.append(Paragraph(challenge_text, normal_style))
Story.append(Spacer(1, 12))

# ------------------------------
# Data Preparation
# ------------------------------
Story.append(Paragraph("3. Data Preparation & Exploration", heading_style))
data_text = """
Dataset Overview:
- Source: CFPB consumer complaint dataset.
- Filtered columns: Complaint ID, Product, clean_narrative.
- Data size: Stratified sample of 10k–15k complaints.
- Missing values: Removed any complaints without narratives.
"""
Story.append(Paragraph(data_text, normal_style))
Story.append(Spacer(1, 12))

chunking_text = """
Text Chunking:
- Rationale: Embedding models perform optimally on shorter segments (~300 words).
- Method: Overlapping chunks of 300 words with 50-word overlap.
- Outcome: Each complaint split into multiple chunks while preserving context.
"""
Story.append(Paragraph(chunking_text, normal_style))
Story.append(Spacer(1, 12))

# Example Table
Story.append(Paragraph("Example Chunk Metadata:", subheading_style))
data_table = [
    ["Complaint ID", "Product", "Chunk Snippet"],
    ["12345", "Credit card", "I was charged multiple fees I did not authorize..."],
    ["12345", "Credit card", "...contacted the bank but received no response for weeks..."]
]
t = Table(data_table, colWidths=[80, 80, 300])
t.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
    ('GRID', (0,0), (-1,-1), 0.5, colors.black),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
]))
Story.append(t)
Story.append(Spacer(1, 12))

# ------------------------------
# Embedding & Vector Store
# ------------------------------
Story.append(Paragraph("4. Embedding & Vector Store Construction", heading_style))
embedding_text = """
- Model: 'all-MiniLM-L6-v2' (Sentence Transformers)
- Each text chunk converted into a 384-dimensional dense vector.
- FAISS Index: IndexFlatL2, stored at 'data/vector_store/complaints_faiss.index'.
- Metadata: Stored at 'data/vector_store/complaints_metadata.csv'.
"""
Story.append(Paragraph(embedding_text, normal_style))
Story.append(Spacer(1, 12))

# ------------------------------
# Sanity Test & Retrieval
# ------------------------------
Story.append(Paragraph("5. Sanity Test & Preliminary Retrieval", heading_style))
retrieval_text = """
A preliminary semantic search was conducted:

Query: 'credit card charged fees I did not authorize'

Top Retrieved Chunks:
1. I was charged unexpected fees on my credit card and the bank did not respond...
2. After contacting customer support about fraudulent charges, I received no help...
3. The bank continued billing me despite multiple complaints about unauthorized fees...
"""
Story.append(Paragraph(retrieval_text, normal_style))
Story.append(Spacer(1, 12))

# ------------------------------
# Next Steps & Conclusion
# ------------------------------
Story.append(Paragraph("6. Next Steps", heading_style))
next_steps_text = """
1. Full Dataset Integration: Extend from 15k sampled complaints to the full dataset.
2. Vector Store Optimization: Explore GPU-based FAISS indices for large-scale retrieval.
3. Integration with LLMs: Connect FAISS store to RAG pipelines.
4. Evaluation Metrics: Recall@k and relevance scoring.
5. Automation: Convert pipeline into a reusable Python module.
"""
Story.append(Paragraph(next_steps_text, normal_style))
Story.append(Spacer(1, 12))

Story.append(Paragraph("7. Conclusion", heading_style))
conclusion_text = """
The pipeline demonstrates effective text preprocessing, chunking, high-quality embeddings, 
a fully functional FAISS vector store with metadata, and promising preliminary retrieval results. 
The next phase will focus on full-scale deployment and integration with RAG models.
"""
Story.append(Paragraph(conclusion_text, normal_style))

# ------------------------------
# Build PDF
# ------------------------------
doc.build(Story)
print(f"PDF generated successfully: {pdf_file}")
