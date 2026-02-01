from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle
import os

# Load PDF
loader = PyPDFLoader("../data/Ebook-Agentic-AI.pdf")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

# Save chunks
os.makedirs("../data", exist_ok=True)
with open("../data/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Chunks saved successfully to data/chunks.pkl")
