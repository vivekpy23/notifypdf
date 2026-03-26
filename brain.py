from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import ingest, random, time

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

Output: <best fact in 1-3 sentences>
"""

NOTIFICATION_PROMPT = """
You are a concise notification writer.

Convert the selected fact into a small, clear sentence for a mobile notification.

Requirements:
- title should be short and clear
- body should be sharp and interesting
- do not become clickbait
- do not distort the fact
- keep body under 220 characters

Fact:
{final_fact}

Output:
<Clear sentence>
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

def discover_fact():
    print("--- Phase 1: Loading Vector DB ---")
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    llm = Ollama(model=MODEL_NAME)

    # 1. Pick 4-8 random chunks
    all_ids = vector_db.get(include=[])['ids']
    k = 3
    if len(all_ids) > k:
        random_ids = random.sample(all_ids, k)
    else:
        random_ids = all_ids

    all_data = vector_db.get(ids=random_ids)
    all_docs = all_data['documents']
    k = min(5, len(all_docs)) # Picking 5 as a middle ground
    random_chunks = random.sample(all_docs, k)
    

    # 2. Extract facts separately for each chunk
    extracted_facts = []
    print(f"--- Phase 2: Extracting from {k} individual chunks ---")
    for i, chunk in enumerate(random_chunks):
        # Fill the extraction prompt with the specific chunk
        prompt = FACT_EXTRACTION_PROMPT.format(context=chunk)
        raw_fact = llm.invoke(prompt)

        # 3. Simple filter: remove empty or extremely short/bad responses
        if len(raw_fact.strip()) > 20: 
            extracted_facts.append(raw_fact.strip())
            print(f"   [+] Fact {i+1} extracted.")

    if not extracted_facts:
        return "No valid facts found in this run."
    
    

    # 4. Pass all gathered facts to the Selector Prompt
    print("--- Phase 3: Selecting the 'Golden' Fact ---")
    all_facts_combined = "\n---\n".join(extracted_facts)
    selector_input = SELECTOR_PROMPT.format(facts=all_facts_combined)
    winner = llm.invoke(selector_input)
    print("Winner FACT selected:", winner)

    # # 5. Final polish for the Notification format
    # print("--- Phase 4: Formatting for Notification ---")
    # final_output = llm.invoke(NOTIFICATION_PROMPT.format(final_fact=winner))
    
    return winner