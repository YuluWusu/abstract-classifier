import streamlit as st

import joblib
import re
from sentence_transformers import SentenceTransformer

class EmbeddingVectorizer:
    def __init__(self, model):
        self.model = model

    def transform(self, texts):
        return self.model.encode(texts)

kmeans = joblib.load("kmeans_model.pkl")
cluster_to_label = joblib.load("cluster_to_label.pkl")
embedding_vectorizer = joblib.load("embedding_model.pkl")
id_to_label = joblib.load("id_to_label.pkl")

def preprocess_text(text: str) -> str:
    text = text.strip().replace("\n", " ")
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def predict_category(abstract: str):
    abstract = preprocess_text(abstract)

    embedding = embedding_vectorizer.transform([abstract])
    cluster_id = kmeans.predict(embedding)[0]

    label_id = cluster_to_label[cluster_id]
    return id_to_label[label_id]

st.title("ArXiv Abstract Classifier")

abstract_input = st.text_area("Nhập abstract")

if st.button("Phân loại"):
    result = predict_category(abstract_input)
    st.success(f"Chủ đề dự đoán: {result}")
