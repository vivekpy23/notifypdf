import requests, random
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import brain

# --- CONFIGURATION ---
DB_DIR = "chroma_db"
MODEL_NAME = "llama3.2:latest"
NTFY_TOPIC = "vivek_facts"  # Your existing topic
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"


# --- THE DELIVERY SYSTEM ---
def send_to_mobile(text):
    print(f"--- Formatting & Sending to {NTFY_TOPIC} ---")
    
    # We split the text to make it look cleaner in the notification
    # ntfy uses the 'Title' header for the bold top line
    # and the 'Tags' header for the small icons
    
    try:
        requests.post(
            NTFY_URL,
            data=text.encode('utf-8'),
            headers={
                "Title": "TODAY's FACT",   # Bold header
                "Tags": "classical_building,brain,sparkles", # Icons: 🏛️ 🧠 ✨
                "Priority": "high"                           # Makes it pop up
            }
        )
        print("✅ Polished notification sent!")
    except Exception as e:
        print(f"❌ Delivery failed: {e}")

# --- EXECUTION ---
if __name__ == "__main__":
    found_fact = brain.discover_fact()
    send_to_mobile(found_fact)