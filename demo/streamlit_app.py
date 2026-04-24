import sys
from pathlib import Path

import joblib
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import clean_text

RESULTS_DIR = PROJECT_ROOT / "results"

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("Fake News Detection Demo")
st.caption("TF-IDF / LSTM-style / BERT-style 模型对比演示")

model_option = st.selectbox(
    "选择模型",
    options=["tfidf", "lstm", "bert"],
    index=0,
)

text = st.text_area("Input news text", height=180)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please input some text.")
    else:
        processed = clean_text(text)
        if model_option == "tfidf":
            model = joblib.load(RESULTS_DIR / "tfidf_model.joblib")
            vectorizer = joblib.load(RESULTS_DIR / "tfidf_vectorizer.joblib")
            x = vectorizer.transform([processed])
            pred = int(model.predict(x)[0])
        elif model_option == "lstm":
            model = joblib.load(RESULTS_DIR / "lstm_model.joblib")
            pred = int(model.predict([processed])[0])
        else:
            model = joblib.load(RESULTS_DIR / "bert_model.joblib")
            pred = int(model.predict([processed])[0])

        label = "Real" if pred == 1 else "Fake"
        st.success(f"Model: {model_option} | Prediction: {label}")
