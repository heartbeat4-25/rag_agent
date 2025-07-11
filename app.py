from flask_cors import CORS
from flask import Flask, request, jsonify
from query_utils import generate_answer
import os

app = Flask(__name__)
CORS(app)

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json
        query = data.get("question", "")
        answer = generate_answer(query)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Flask RAG 系统运行中，请向 POST /ask 提问。"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
