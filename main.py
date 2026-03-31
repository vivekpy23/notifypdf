import requests, random
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import brain
import json

# --- CONFIGURATION ---
DB_DIR = "chroma_db"
MODEL_NAME = "llama3.2:latest"
NTFY_TOPIC = "vivek_facts"  # Your existing topic
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"


# --- THE DELIVERY SYSTEM ---
def send_to_mobile(text, fact_id):
    print(f"--- Sending Fact {fact_id} to Mobile ---")
    
    # Replace 'YOUR_SERVER_IP' with your actual computer's IP or a webhook URL
    # Example: http://192.168.1.10:5000/feedback?id=123&score=1
    base_url = "http://192.168.1.15:5000/feedback" 


        # Actions format: label, url, method
    actions = [
    {
        "action": "http",
        "label": "Like",
        "url": base_url,
        "method": "POST",
        "headers": {"Content-Type": "application/json"}, # <--- Add this
        "body": json.dumps({"id": fact_id, "score": 1}), # Use 'body' instead of 'params'
        "clear": True
    },
    {
        "action": "http",
        "label": "Dislike",
        "url": base_url,
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"id": fact_id, "score": -1}),
        "clear": True
    }
]

    try:
        
        requests.post(
            NTFY_URL,
            data=text.encode('utf-8'),
            headers={
                "Title": "NEW DISCOVERY",
                "Tags": "scroll,brain",
                "Priority": "high",
                "Actions": json.dumps(actions) # Convert list to JSON string
            }
        )
        print(f"Silent notification sent for Fact ID: {fact_id}")
    except Exception as e:
        print(f"Delivery failed: {e}")



if __name__ == "__main__":
    found_fact, fact_id = brain.discover_fact()
    send_to_mobile(found_fact, fact_id)