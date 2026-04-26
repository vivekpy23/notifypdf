import os
import json
import random
from langchain_ollama import ChatOllama
from AgentState import AgentState # <--- Import from your existing file
from db_manager import get_book_weights, get_recent_file_ids, get_file_id, vector_db

# Only the logic goes here now
llm = ChatOllama(model="llama3.2:latest", format="json", temperature=0.1)

LIBRARIAN_PROMPT = """
You are the Librarian Agent.

Your job is to retrieve the strongest possible evidence from the available knowledge sources.

Goal:
Find context that can support one interesting, accurate, source-backed fact.

Instructions:
- Search the vector database and available metadata.
- Do not settle for the first few chunks if the context is weak.
- Prefer chunks that are specific, factual, and self-contained.
- Avoid vague, repetitive, or generic context.
- If results are weak, reformulate the search query and try again.
- Return only evidence-backed context.
- Do not generate the final fact.

Output format:
{
  "search_query_used": "",
  "selected_context": [
    {
      "text": "",
      "source": "",
      "page": "",
      "reason_selected": ""
    }
  ],
  "confidence": "low | medium | high",
  "notes": ""
}
"""

def librarian_node(state: AgentState):
    print("\n--- LIBRARIAN: Formulating Search Strategy ---")
    
    # Generate the search strategy via LLM
    # We pass the current iteration count to encourage variety if looping
    input_msg = f"Provide a research query for discovery iteration {state.get('iteration_count', 1)}."
    response = llm.invoke([
        ("system", LIBRARIAN_PROMPT),
        ("human", input_msg)
    ])
    
    strategy = json.loads(response.content)
    search_query = strategy.get("search_query_used", "mythology")

    # EXECUTE WEIGHTED SAMPLING LOGIC (The "No Shortcuts" Path)
    # Get all IDs and Metadatas from your 500 PDF index
    all_info = vector_db.get(include=['metadatas'])
    all_ids = all_info['ids']
    all_metas = all_info['metadatas']

    # Apply Weights (Likes + Recency Penalty)
    book_weights = get_book_weights()
    recent_files = get_recent_file_ids(hours=48)
    
    calculated_weights = []
    for meta in all_metas:
        source_name = os.path.basename(meta.get('source', ''))
        f_id = get_file_id(source_name)
        
        # Base weight + Like boost
        weight = book_weights.get(f_id, 1.0)
        
        # 48-hour Recency Penalty
        if f_id in recent_files:
            weight *= 0.1 
            
        calculated_weights.append(weight)

    # Perform Weighted Selection
    chosen_ids = random.choices(all_ids, weights=calculated_weights, k=5)
    results = vector_db.get(ids=chosen_ids, include=['documents', 'metadatas'])

    # Format findings for the State
    selected_context = []
    for i in range(len(results['documents'])):
        selected_context.append({
            "text": results['documents'][i],
            "source": results['metadatas'][i].get('source'),
            "page": results['metadatas'][i].get('page'),
            "reason_selected": "High relevance score and source weight"
        })

    return {
        "retrieved_chunks": selected_context,
        "search_query": search_query,
        "iteration_count": state.get("iteration_count", 0) + 1
    }





HISTORIAN_PROMPT = """
You are the Extraction Agent.

Your job is to convert retrieved context into a clean, interesting, self-contained fact.

Goal:
Create one candidate fact that is accurate, specific, and worth sharing.

Instructions:
- Use only the provided context.
- Do not add unsupported details.
- Make the fact understandable without requiring the reader to know the source.
- Prefer surprising, specific, or counter-intuitive details.
- Avoid generic summaries.
- Keep it concise but meaningful.
- Include source references from the provided context.

Output format:
{
  "candidate_fact": "",
  "why_it_is_interesting": "",
  "supporting_sources": [
    {
      "source": "",
      "page": "",
      "supporting_text": ""
    }
  ],
  "confidence": "low | medium | high"
}
"""

def historian_node(state: AgentState):
    print("--- ✍️ HISTORIAN: Synthesizing Fact ---")
    
    # 1. Prepare context for the model
    # We take the chunks the Librarian just put into the state
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {"error_log": state.get("error_log", []) + ["Historian found no context to work with."]}

    context_text = json.dumps(chunks, indent=2)
    
    # 2. Call the LLM
    # We provide the system prompt and the raw evidence as the "human" input
    response = llm.invoke([
        ("system", HISTORIAN_PROMPT),
        ("human", f"Evidence provided by Librarian:\n{context_text}")
    ])
    
    # 3. Parse and update state
    try:
        fact_data = json.loads(response.content)
        return {"candidate_fact": fact_data}
    except Exception as e:
        error_msg = f"Historian JSON Parsing Error: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error_log": state.get("error_log", []) + [error_msg]}
    




CRITIC_PROMPT = """
You are the Critic Agent.

Your job is to evaluate whether the candidate fact is good enough to publish.

Check the fact against four criteria:

1. Source Support:
Is every claim supported by the retrieved context?

2. Interestingness:
Is this fact surprising, specific, or worth sending?

3. Clarity:
Can a normal reader understand it without extra background?

4. Novelty:
Is it different from previously published facts?

Important:
Do not fail a fact only because one term may be unfamiliar.
If the issue can be fixed by adding a short explanation in parentheses or replacing the term with simpler wording, pass it and rewrite the approved_fact.

Instructions:

- Reject facts that are vague, boring, repetitive, exaggerated, or weakly supported.
- Do not rewrite the fact unless it only needs a small improvement.
- If rejected, clearly explain what needs to change.

Output format:
{
  "verdict": "pass | fail",
  "quality_score": 0,
  "issues": [
    ""
  ],
  "revision_instruction": "",
  "approved_fact": ""
}
"""

def critic_node(state: AgentState):
    print("--- ⚖️ CRITIC: Evaluating Quality ---")
    
    # 1. Gather the fact and the context it came from
    fact = state.get("candidate_fact")
    context = state.get("retrieved_chunks")
    
    if not fact:
        return {"critic_verdict": {"verdict": "fail", "issues": ["No fact provided to evaluate."]}}

    # 2. Call the LLM for evaluation
    # We provide both the fact and the context so the Critic can check for hallucinations
    evaluation_input = {
        "candidate_fact": fact,
        "original_context": context
    }
    
    response = llm.invoke([
        ("system", CRITIC_PROMPT),
        ("human", f"Evaluate this research output:\n{json.dumps(evaluation_input, indent=2)}")
    ])
    
    # 3. Parse verdict
    try:
        verdict_data = json.loads(response.content)
        return {"critic_verdict": verdict_data}
    except Exception as e:
        error_msg = f"Critic JSON Parsing Error: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "critic_verdict": {"verdict": "fail", "issues": [error_msg]},
            "error_log": state.get("error_log", []) + [error_msg]
        }
    



PUBLISHER_PROMPT = """
You are the Publisher Agent.

Your job is to format the approved fact for delivery.

Goal:
Turn the approved fact into a polished notification-ready message.

Instructions:
- Keep the message short and engaging.
- Do not add unsupported claims.
- Preserve factual accuracy.
- Make the first line attention-grabbing.
- Include source name/page if available.
- Also create optional social versions if requested.

Output format:
{
  "ntfy_title": "",
  "ntfy_message": "",
  "short_social_post": "",
  "reel_hook": "",
  "source_note": "",
  "feedback_options": ["like", "dislike"]
}
"""

def publisher_node(state: AgentState):
    print("--- 📢 PUBLISHER: Formatting Final Output ---")
    
    # 1. Get the approved fact (either from Critic or Historian depending on state)
    # The Critic usually passes the 'approved_fact' in its JSON if it passes
    approved_fact = state.get("critic_verdict", {}).get("approved_fact")
    
    # Fallback to the candidate fact if the critic didn't modify it
    if not approved_fact:
        approved_fact = state.get("candidate_fact", {}).get("candidate_fact")

    if not approved_fact:
        return {"error_log": state.get("error_log", []) + ["Publisher found no approved fact."]}

    # 2. Format the fact using the LLM
    response = llm.invoke([
        ("system", PUBLISHER_PROMPT),
        ("human", f"Format this approved research fact for notification:\n{approved_fact}")
    ])
    
    # 3. Parse and update state
    try:
        notification_data = json.loads(response.content)
        return {"final_notification": notification_data}
    except Exception as e:
        error_msg = f"Publisher JSON Parsing Error: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error_log": state.get("error_log", []) + [error_msg]}
    


ARCHIVIST_PROMPT = """
You are the Archivist Agent.

Your job is to prevent repetition and improve future recommendations.

Instructions:
- Compare the new fact with previously published, liked, disliked, and rejected facts.
- Detect duplicates or near-duplicates.
- Track topic, entity, source, and reason for acceptance/rejection.
- Suggest whether this fact should be published now or saved for later.

Output format:
{
  "is_duplicate": false,
  "similar_previous_facts": [],
  "topic_tags": [],
  "publish_recommendation": "publish | save_for_later | reject",
  "reason": ""
}
"""



import hashlib
from db_manager import is_already_sent # Import your existing function

def archivist_node(state: AgentState):
    print("--- 📂 ARCHIVIST: Checking Long-term Memory ---")
    
    fact_text = state.get("candidate_fact", {}).get("candidate_fact", "")
    if not fact_text:
        return {"archivist_report": {"publish_recommendation": "reject"}}

    # 1. HARD CHECK: The Hash (Your existing logic)
    # We create a hash of the clean fact text
    fact_hash = hashlib.md5(fact_text.encode()).hexdigest()
    
    if is_already_sent(fact_hash):
        print("🛑 Exact hash match found in DB. Rejecting.")
        return {
            "archivist_report": {
                "is_duplicate": True,
                "publish_recommendation": "reject",
                "reason": "Exact hash match in database."
            }
        }

    # 2. SOFT CHECK: Semantic Similarity (The Agent logic)
    # We only send the last 10-20 facts so the LLM can check for "Concept Overlap"
    # This prevents you from getting two facts about the same king in two days.
    from db_manager import get_recent_facts
    past_facts = get_recent_facts(limit=15) 
    
    response = llm.invoke([
        ("system", ARCHIVIST_PROMPT),
        ("human", f"New Fact: {fact_text}\n\nRecently Sent Facts: {json.dumps(past_facts)}")
    ])
    
    try:
        archivist_data = json.loads(response.content)
        # Store the hash in the report so the Publisher can save it to DB later
        archivist_data["fact_hash"] = fact_hash 
        return {"archivist_report": archivist_data}
    except Exception as e:
        return {"error_log": state.get("error_log", []) + [f"Archivist Error: {e}"]}
    


from db_manager import save_fact, get_file_id
import os

def saver_node(state: AgentState):
    print("--- 💾 SAVER: Committing Discovery to Database ---")
    
    # 1. Pull the data from the previous agents
    final_fact = state['candidate_fact']['candidate_fact']
    fact_hash = state['archivist_report']['fact_hash']
    
    # We grab the source of the first chunk to link it to a file_id
    # (Since Historian might use multiple chunks, we pick the primary one)
    source_path = state['retrieved_chunks'][0].get('source', 'Unknown')
    file_id = get_file_id(os.path.basename(source_path))

    # 2. Call your EXISTING function
    fact_id = save_fact(
        fact_text=final_fact,
        fact_hash=fact_hash,
        file_id=file_id
    )
    
    if fact_id:
        print(f"✅ Fact saved with ID: {fact_id}")
    
    # We pass the ID forward so the Notification script knows which URL to build
    return {"final_notification": {**state['final_notification'], "fact_id": fact_id}}