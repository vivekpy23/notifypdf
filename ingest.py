import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from db_manager import is_file_ingested, mark_as_ingested, get_file_id, truncate_table, truncate_facts

# 1. Setup - Tell the script where your data is
DATA_DIR = "data" # <-- CHANGE THIS to your filename!
DB_DIR = "chroma_db"

def run_ingestion():
    print(f"--- Starting Ingestion for {DATA_DIR} ---")

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]

    new_files = []
    for pdf in pdf_files:
        if not is_file_ingested(pdf):
            new_files.append(pdf)
        else:
            print(f"Skipping {pdf} (Already in Vector DB)")


    if not new_files:
        print("No new PDFs found. Database is up to date.")
        return

    print(f"Found {len(new_files)} new PDFs to ingest...")


    for pdf in new_files:
        pdf_path = os.path.join(DATA_DIR, pdf)
        try:
            # 2. Load the PDF
            loader = PyPDFLoader(pdf_path)
            data = loader.load()
            #print(f"Loaded {len(data)} pages.")

            # 3. Chunking (The 'Slicing' Strategy)
            # We use 500 chars with 50 char overlap as we discussed.
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400, 
                chunk_overlap=50
            )
            chunks = text_splitter.split_documents(data)
            #print(f"Split into {len(chunks)} chunks.")

            # 4. Embedding & Storage
            # We use Ollama to create the 'math' (embeddings) locally.
            print("Creating embeddings and saving to ChromaDB... (This may take a moment)")
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=OllamaEmbeddings(model="nomic-embed-text:latest"), 
                persist_directory=DB_DIR
            )

            mark_as_ingested(pdf, total_chunks=len(chunks))
            file_id = get_file_id(pdf)
            
            print(f"Successfully ingested: {pdf} with file_id: {file_id}")

    
        except Exception as e:
            print(f"❌ Failed to ingest {pdf}: {e}")


    print(f"✅ Success! Your knowledge bank is ready in '{DB_DIR}/'")
    


if __name__ == "__main__":
    run_ingestion()