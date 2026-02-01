from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# -----------------------
# State definition
# -----------------------
class RAGState(TypedDict):
    question: str
    contexts: List[str]
    answer: str
    confidence: float

# -----------------------
# Load components
# -----------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "../data/faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

llm = Ollama(model="llama3")

# -----------------------
# Graph Nodes
# -----------------------
def retrieve_node(state: RAGState):
    docs = vectorstore.similarity_search(state["question"], k=3)
    contexts = [doc.page_content for doc in docs]
    return {"contexts": contexts}

def generate_node(state: RAGState):
    context_text = "\n\n".join(state["contexts"])

    prompt = f"""
Use ONLY the context below to answer.
If the answer is not in the context, say "I don't know".

Context:
{context_text}

Question:
{state['question']}

Answer:
"""
    answer = llm.invoke(prompt)
    return {"answer": answer}

def confidence_node(state: RAGState):
    # Simple, explainable confidence logic
    confidence = min(0.95, 0.6 + (len(state["contexts"]) * 0.1))
    return {"confidence": confidence}

# -----------------------
# Build Graph
# -----------------------
graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("confidence", confidence_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "confidence")
graph.add_edge("confidence", END)

rag_app = graph.compile()
