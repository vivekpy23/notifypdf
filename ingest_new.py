import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from db_manager import is_file_ingested, mark_as_ingested, get_file_id

# Config
DATA_DIR = "data"
DB_DIR = "chroma_db"
BATCH_SIZE = 64  # 🔥 tune between 32–128 based on your CPU

# 🔥 Initialize ONCE (important)
embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
vector_db = Chroma(persist_directory=DB_DIR)

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

    print(f"Found {len(new_files)} new PDFs to ingest...\n")

    for pdf in new_files:
        pdf_path = os.path.join(DATA_DIR, pdf)

        try:
            print(f"\n📄 Processing: {pdf}")

            # 1. Load PDF
            loader = PyPDFLoader(pdf_path)
            data = loader.load()

            # 🔥 Merge all pages into one text (faster chunking)
            full_text = "\n".join([doc.page_content for doc in data])

            # 2. Chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=50
            )

            chunks = text_splitter.split_text(full_text)
            print(f"✂️ Split into {len(chunks)} chunks")

            # 3. Batch Embedding + Insert
            print("🧠 Creating embeddings (batched)...")

            total_chunks = len(chunks)

            for i in range(0, total_chunks, BATCH_SIZE):
                batch_chunks = chunks[i:i+BATCH_SIZE]

                # 🔥 Batch embedding
                embeddings = embedding_model.embed_documents(batch_chunks)

                # Generate unique IDs
                batch_ids = [f"{pdf}_{i+j}" for j in range(len(batch_chunks))]

                # 🔥 Batch insert into Chroma
                vector_db.add_texts(
                    texts=batch_chunks,
                    embeddings=embeddings,
                    ids=batch_ids
                )

                print(f"   ➤ Processed {min(i+BATCH_SIZE, total_chunks)}/{total_chunks}")

            # 🔥 Persist once per file (or move outside loop if many files)
            #vector_db.persist()

            # DB tracking
            mark_as_ingested(pdf, total_chunks=total_chunks)
            file_id = get_file_id(pdf)

            print(f"✅ Successfully ingested: {pdf} (file_id: {file_id})")

        except Exception as e:
            print(f"❌ Failed to ingest {pdf}: {e}")

    print(f"\n🎉 All done! Vector DB ready at '{DB_DIR}/'")


if __name__ == "__main__":
    run_ingestion()