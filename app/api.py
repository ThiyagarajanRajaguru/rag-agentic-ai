from fastapi import FastAPI
from pydantic import BaseModel
from rag_graph import rag_app

app = FastAPI(
    title="Agentic RAG API",
    description="Agentic AI using LangGraph + FAISS + Ollama",
    version="1.0"
)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_agent(query: Query):
    result = rag_app.invoke({
        "question": query.question
    })

    return {
        "question": query.question,
        "answer": result["answer"],
        "confidence": result["confidence"],
        "contexts": result["contexts"]
    }
