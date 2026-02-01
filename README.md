# Agentic RAG System using LangChain, FAISS, Ollama & LangGraph

This project demonstrates an end-to-end **Agentic Retrieval-Augmented Generation (RAG)** system.
It processes a PDF document, creates semantic embeddings, stores them in a vector database,
and answers user questions using a locally hosted LLM with agent-based orchestration.

The project is designed with **simple architecture and clear separation of steps** so it can be
easily explained in interviews.

---

## 📄 Document Ingestion Details

- **Source PDF:** Ebook-Agentic-AI.pdf  
- **Total chunks created:** 148  
- **Chunk size:** 700 characters  
- **Chunk overlap:** 100 characters  

This configuration was chosen to balance **semantic completeness** and
**retrieval accuracy** in the RAG pipeline.

---

## 🧠 Vector Store (FAISS)

- **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2  
- **Vector Database:** FAISS  
- **Chunk size:** 700  
- **Chunk overlap:** 100  
- **Total chunks:** 148  

Embeddings are generated offline and stored locally to enable
**fast and efficient semantic retrieval**.

---

## 🤖 Agentic AI & LangGraph

The system uses **LangGraph** to model an agent-style workflow where:
- User input is processed
- Relevant context is retrieved from FAISS
- The LLM generates grounded answers using retrieved context

This demonstrates **Agentic AI behavior** by structuring decision flow
instead of a single linear LLM call.

---

*Question:**
