from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
import os

# Load chunks created during ingestion
with open("../data/chunks.pkl", "rb") as f:
    documents = pickle.load(f)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector store
vectorstore = FAISS.from_documents(documents, embedding_model)

# Save FAISS index
os.makedirs("../data/faiss_index", exist_ok=True)
vectorstore.save_local("../data/faiss_index")

print("FAISS vector store created successfully")
