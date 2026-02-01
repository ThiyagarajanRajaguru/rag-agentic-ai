\## 📄 Document Ingestion Details



\- Source PDF: Ebook-Agentic-AI.pdf

\- Total chunks created: 148

\- Chunk size: 700 characters

\- Chunk overlap: 100 characters



This configuration was chosen to balance semantic completeness

and retrieval accuracy in the RAG pipeline.



\## Vector Store (FAISS)



\- Embedding Model: sentence-transformers/all-MiniLM-L6-v2

\- Vector Database: FAISS

\- Chunk size: 700

\- Chunk overlap: 100

\- Total chunks: 148



Embeddings are generated offline and stored locally

to enable fast semantic retrieval.



\## Demo



Example Question:

What is Agentic AI?



Example Answer:

Agentic AI goes beyond traditional AI by acting autonomously

to achieve goals in a proactive, adaptive, and impact-focused manner.



The answer is generated using retrieved context from the source

document, ensuring grounded and reliable responses.



\## How to Run

uvicorn api:app --reload

Open http://127.0.0.1:8000/docs



