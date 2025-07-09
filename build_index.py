import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss

from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# 如果你没有配置环境变量，可以取消注释并写上路径：
pytesseract.pytesseract.tesseract_cmd = r"D:\warehouse\Tesseract-OCR\tesseract.exe"

def extract_text_from_pdf(pdf_path):
    text = ""

    # 尝试使用 PyMuPDF 提取文本
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print(f"PyMuPDF 提取失败：{e}")

    # 如果提取到的文字过少，则启用 OCR
    if len(text.strip()) < 30:
        print(f"内容过少，切换 OCR 识别：{pdf_path}")
        try:
            images = convert_from_path(pdf_path, poppler_path=r"D:\warehouse\popper\poppler-24.08.0\Library\bin")
            for img in images:
                text += pytesseract.image_to_string(img, lang="chi_sim")
        except Exception as e:
            print(f"OCR 识别失败：{e}")

    return text

# 主流程
docs_dir = "docs"
all_paragraphs = []

for fname in os.listdir(docs_dir):
    if fname.endswith(".pdf"):
        print(f"正在处理：{fname}")
        content = extract_text_from_pdf(os.path.join(docs_dir, fname))
        paragraphs = [p.strip() for p in content.split("\n") if len(p.strip()) > 20]
        all_paragraphs.extend(paragraphs)

if not all_paragraphs:
    print("没有成功提取任何段落，请检查 PDF 是否为有效文本格式。")
    exit(1)

model = SentenceTransformer("BAAI/bge-base-zh")
vectors = model.encode(all_paragraphs, normalize_embeddings=True)

index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)
faiss.write_index(index, "vector.index")

with open("texts.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(all_paragraphs))

print("向量索引构建完成，共提取段落数：", len(all_paragraphs))
