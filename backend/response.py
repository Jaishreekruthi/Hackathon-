import os
import requests
from dotenv import load_dotenv

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

chat_history = []

# Simple keywords to detect health-related queries
health_keywords = [
    "health", "fitness", "diet", "medicine", "doctor", "hospital",
    "symptom", "treatment", "disease", "mental health", "wellness",
    "nutrition", "exercise", "yoga", "diabetes", "blood pressure"
]

def is_health_related(user_input: str) -> bool:
    user_input = user_input.lower()
    return any(keyword in user_input for keyword in health_keywords)

def get_bot_response(user_input):
    # ✅ Check if health-related
    if not is_health_related(user_input):
        return "Sorry, I don’t have access to other resources, I’m a Health care bot."

    chat_history.append(user_input)

    data = query({
        "inputs": {
            "past_user_inputs": chat_history[:-1],
            "generated_responses": [],
            "text": user_input
        }
    })

    try:
        bot_reply = data['generated_text']
    except:
        bot_reply = "Sorry, I’m having trouble right now."

    chat_history.append(bot_reply)
    return bot_reply
