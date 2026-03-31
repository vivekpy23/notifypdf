import hashlib
import os

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import random
from db_manager import DB_PATH, get_existing_fact_id, save_fact, get_file_id

DB_DIR = "chroma_db"
MODEL_NAME = "llama3.2:latest"


# --- PROMPT TEMPLATES (PLACEHOLDERS) ---
FACT_EXTRACTION_PROMPT = """
You are an intelligent fact extraction assistant. Your task is to examine the retrieved context from a PDF and extract exactly ONE fact that is:

- specific
- interesting
- non-trivial
- supported directly by the text
- understandable on its own

Rules:
- Do not summarize the whole document.
- Do not answer like a chatbot.
- Do not give multiple facts.
- Do not invent or assume details not present in the text.
- Avoid generic statements.
- Prefer a fact that is surprising, memorable, rare, or useful.
- If the context does not contain a strong standalone fact, say so clearly.

Text:
{context}

Return output in exactly this format:

Fact: <one fact in 2-3 sentences>
"""

SELECTOR_PROMPT = """
You are a strict evaluator of facts. You understand the value of interesting and unique information.

From the list below, select the ONE best fact.

Rules:
- Must be specific
- Must be interesting and meaningful
- Must be complete and understandable
- Avoid generic or dry statements

Facts:
{facts}

Return output in exactly this format:

<The exact winning fact in 1-3 sentences>
"""

NOTIFICATION_PROMPT = """
You are a concise notification writer.

Convert the selected fact into a small, clear sentence for a mobile notification.

Requirements:
- there should be no title
- body should be sharp and interesting
- do not distort the fact
- keep body under 220 characters

Fact:
{final_fact}

Return output in exactly below format:
<final fact>
"""

# 1. Your Precision Extraction Prompt
# template = """
# You are an intelligent fact extraction assistant.

# Your task is to examine the retrieved context from a PDF and extract exactly ONE fact that is:
# - specific
# - interesting
# - non-trivial
# - supported directly by the text
# - understandable on its own

# Rules:
# - Do not summarize the whole document.
# - Do not answer like a chatbot.
# - Do not give multiple facts.
# - Do not invent or assume details not present in the text.
# - Avoid generic statements.
# - Prefer a fact that is surprising, memorable, rare, or useful.
# - If the context does not contain a strong standalone fact, say so clearly.

# Retrieved context:
# {context}

# Return output in exactly this format:

# Fact: <one fact in 1-3 sentences>
# """


def get_fact_hash(text):
    """Creates a unique MD5 fingerprint for the text."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# --- Helper: Extract filename from metadata ---
def get_source(metadata):
    if not metadata:
        return "Unknown"
    full_path = metadata.get('source', 'Unknown')
    # Returns 'Book_Name.pdf' from 'D:/data/Book_Name.pdf'
    return os.path.basename(full_path) if full_path != 'Unknown' else "Unknown"


def discover_fact():
    print("--- Phase 1: Loading Vector DB & Sampling ---")
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    llm = Ollama(model=MODEL_NAME)

    # 1. Get all IDs to pick random samples
    all_ids = vector_db.get(include=[])['ids']
    if not all_ids:
        print("Vector DB is empty. Please run ingest.py first.")
        return
    
    # Pick 5 random chunks to evaluate
    k = min(5, len(all_ids))
    random_ids = random.sample(all_ids, k)

    # 2. Pull both text and metadata for these specific IDs
    all_data = vector_db.get(ids=random_ids, include=['documents', 'metadatas'])
    chunks = all_data['documents']
    metadatas = all_data['metadatas']

    # --- Phase 2: Fact Extraction ---
    fact_packages = [] 
    print(f"--- Phase 2: Extracting from {len(chunks)} individual chunks ---")
    
    for i in range(len(chunks)):
        source_name = get_source(metadatas[i])
        
        # Extract the raw fact using the LLM
        prompt = FACT_EXTRACTION_PROMPT.format(context=chunks[i])
        raw_fact = llm.invoke(prompt).strip()

        if len(raw_fact) > 20:
            # Package the fact WITH its specific source metadata
            fact_packages.append({
                "fact": raw_fact,
                "source": source_name
            })
            print(f"   [+] Extracted fact from: {source_name}")

    if not fact_packages:
        print("No valid facts found in this run.")
        return

    # --- Phase 3: Selecting the 'Golden' Fact ---
    print("--- Phase 3: Selecting the 'Golden' Fact ---")
    
    # Create a numbered list for the Selector LLM
    numbered_facts = ""
    for idx, pkg in enumerate(fact_packages):
        numbered_facts += f"{idx}: {pkg['fact']}\n"

    selector_input = SELECTOR_PROMPT.format(facts=numbered_facts)
    selection_raw = llm.invoke(selector_input).strip()

    # Attempt to find the index number in the AI's response
    try:
        # Extract only digits (e.g., "The winner is 2" becomes "2")
        winner_idx = int(''.join(filter(str.isdigit, selection_raw)))
        winner_package = fact_packages[winner_idx]
    except (ValueError, IndexError):
        # Fallback to the first fact if the LLM output is messy
        winner_package = fact_packages[0]

    winner_text = winner_package['fact']
    winner_source = winner_package['source']
    print(f"Winner Selected: {winner_text[:100]}... (Source: {winner_source})")

    # --- Phase 4: Database Storage ---
    # 1. Get the SQL File ID for the winning source
    current_file_id = get_file_id(winner_source)
    
    # 2. Deduplication check
    fact_hash = hashlib.md5(winner_text.encode('utf-8')).hexdigest()
    if get_existing_fact_id(fact_hash):
        print("This fact is already in your Vault. Skipping save.")
        return

    # 3. Save to the facts table (linked to the file_id)
    new_fact_id = save_fact(winner_text, fact_hash, file_id=current_file_id)
    
    if new_fact_id:
        print(f"Successfully stored Fact ID: {new_fact_id} in your research database.")
    
        return winner_text, new_fact_id


