from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS index
vectorstore = FAISS.load_local(
    "../data/faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

# Load LLM (Ollama)
llm = Ollama(model="llama3")

def ask_question(query):
    # Retrieve relevant chunks
    docs = vectorstore.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Use the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""
    return llm.invoke(prompt)

# CLI loop
if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        answer = ask_question(question)
        print("\nAnswer:\n", answer)
