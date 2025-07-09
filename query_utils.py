import os
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

model = SentenceTransformer("BAAI/bge-base-zh")
index = faiss.read_index("vector.index")
with open("texts.txt", "r", encoding="utf-8") as f:
    texts = f.readlines()

def search_relevant_text(query, top_k=3):
    q_vec = model.encode([query], normalize_embeddings=True)
    D, I = index.search(q_vec, k=top_k)
    return [texts[i].strip() for i in I[0]]

def generate_answer(query):
    context = search_relevant_text(query)
    prompt = (
        "你是一个知识助手，请基于以下内容回答用户问题：\n\n"
        + "\n\n".join(context)
        + f"\n\n用户问题：{query}\n回答："
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content
