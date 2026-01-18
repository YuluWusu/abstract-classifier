import streamlit as st
import joblib
import re
from sentence_transformers import SentenceTransformer

class EmbeddingVectorizer:
    def __init__(self, model):
        self.model = model

    def transform(self, texts):
        return self.model.encode(texts)

# ===== Load model =====
kmeans = joblib.load("kmeans_model.pkl")
cluster_to_label = joblib.load("cluster_to_label.pkl")
embedding_vectorizer = joblib.load("embedding_model.pkl")
id_to_label = joblib.load("id_to_label.pkl")

# ===== Preprocess =====
def preprocess_text(text: str) -> str:
    text = text.strip().replace("\n", " ")
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# ===== Predict =====
def predict_category(abstract: str):
    abstract = preprocess_text(abstract)
    embedding = embedding_vectorizer.transform([abstract])
    cluster_id = kmeans.predict(embedding)[0]
    label_id = cluster_to_label[cluster_id]
    return id_to_label[label_id]
# ===== Label ====
def translate_label_auto(label):
    mapping = {
        "cs": "Computer Science (Khoa học máy tính)",
        "cond-mat": "Condensed Matter (Vật lý vật chất ngưng tụ)",
        "astro-ph": "Astrophysics (Vật lý thiên văn)",
        "math": "Mathematics (Toán học chung)",
        "physics": "Physics (Vật Lý Học)"
    }
    
    parts = label.split('.')
    prefix = parts[0]

    category = mapping.get(prefix, prefix)

    if len(parts) > 1 and parts[1]:
        sub_category = parts[1].upper()
        return f"{category} ({sub_category})"

    return category
# ===== UI =====
st.set_page_config(page_title="Abstract Classification", layout="centered")

st.title("📄 Ứng dụng học máy không giám sát trong việc phân loại đề tài nghiên cứu.")
st.write("Nhập abstract bài báo khoa học để AI dự đoán lĩnh vực")

abstract_input = st.text_area(
    "Nhập abstract:",
    height=250,
    placeholder="Paste abstract tiếng Anh vào đây..."
)

if st.button("🔍 Phân loại"):
    if abstract_input.strip() == "":
        st.warning("⚠️ Vui lòng nhập abstract!")
    else:
        result = predict_category(abstract_input)
        st.success(f"✅ Lĩnh vực dự đoán: **{translate_label_auto(result)}**")
