from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configure Gemini API
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    logging.error(f"Error configuring Gemini API: {e}")
    exit()

# System prompt for the chatbot
system_prompt = (
    "You are HealthCareBot, a specialized conversational AI assistant. Your purpose is to provide short, bulleted guidance on "
    "health, wellness, fitness, medical awareness, mental health, diet, and healthy lifestyle. Your primary goal is to gather more information "
    "from the user to provide the most relevant advice. "
    "After every response, you MUST ask a follow-up question to gather more details about their symptoms or situation. "
    "Always provide your response in a concise, point-by-point format. "
    "Only respond to topics within your domain. If a user asks about anything outside of healthcare, "
    "you MUST respond with the exact phrase: "
    "\"Sorry, I don’t have access to other resources, I’m a Health care bot.\""
)

# Initialize the Gemini model with system instructions
model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_prompt)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    history_raw = request.json.get("history", [])

    # Validate input
    if not user_message:
        return jsonify({"response": "Please enter a valid message."}), 400

    # Format history for Gemini
    history = []
    for item in history_raw:
        if isinstance(item, dict) and "role" in item and "parts" in item:
            history.append(item)
        elif isinstance(item, str):
            # Fallback for simple string history
            role = "user" if len(history) % 2 == 0 else "model"
            history.append({"role": role, "parts": [item]})

    try:
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(user_message)
        reply = response.text.strip()

        if not reply:
            reply = "Sorry, I couldn’t generate a response this time."

        logging.info(f"USER: {user_message}\nBOT: {reply}")
        return jsonify({"response": reply})

    except Exception as e:
        logging.error(f"Error in /chat: {e}")
        return jsonify({"response": "Oops! Something went wrong on the server."}), 500

if __name__ == "__main__":
    # Use a production server like Gunicorn in deployment
    app.run(debug=True)