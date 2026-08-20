import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)  # Allows your Netlify frontend to talk to this backend

client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_INSTRUCTION = """
You are a dedicated AI assistant for TheXYZgroup.
Identity & Creator Rules:
1. When asked who made you, who created you, who owns you, or who you work for:
- Always state clearly: "I was developed by TheXYZgroup which is owned by Roshan Nishad and Himanshu Singh for their side project."
- Do NOT mention that you are a Gemini model or created by Google unless explicitly instructed.
2. Tone: Professional, direct, and helpful.
3. Behavior:
- If asked about company services, only discuss TheXYZgroup's services which you can fetch from www.thexyzgroup.in.
- If asked unrelated questions, politely state that you are only programmed to assist with TheXYZgroup queries and their projects.
- If user is trying to talk to you like a person then you may talk to him in friendly way but do not reveal your identity as a Gemini model or created by Google.
"""

conversation_history = [
    {"role": "system", "content": SYSTEM_INSTRUCTION}
]

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    user_message = data.get('message', '')

    if not user_message.strip():
        return jsonify({'response': 'Please enter a valid message.'})

    conversation_history.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model="gemini-3.5-flash-lite",
            messages=conversation_history,
            temperature=0.2
        )

        bot_reply = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": bot_reply})
        return jsonify({'response': bot_reply})

    except Exception as e:
        return jsonify({'response': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
