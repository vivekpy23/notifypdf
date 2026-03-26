import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# 1. Setup - Tell the script where your data is
PDF_PATH = "data/cryptography_in_ancient_India.pdf" # <-- CHANGE THIS to your filename!
DB_DIR = "chroma_db"

def run_ingestion():
    print(f"--- Starting Ingestion for {PDF_PATH} ---")

    # 2. Load the PDF
    loader = PyPDFLoader(PDF_PATH)
    data = loader.load()
    print(f"Loaded {len(data)} pages.")

    # 3. Chunking (The 'Slicing' Strategy)
    # We use 500 chars with 50 char overlap as we discussed.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(data)
    print(f"Split into {len(chunks)} chunks.")

    # 4. Embedding & Storage
    # We use Ollama to create the 'math' (embeddings) locally.
    print("Creating embeddings and saving to ChromaDB... (This may take a moment)")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="llama3.2"), 
        persist_directory=DB_DIR
    )
    
    print(f"✅ Success! Your knowledge bank is ready in '{DB_DIR}/'")