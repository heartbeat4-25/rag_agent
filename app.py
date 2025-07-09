from flask import Flask, request, jsonify
from query_utils import generate_answer

app = Flask(__name__)

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
    app.run(debug=True)
