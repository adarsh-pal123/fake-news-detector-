# app.py
import gradio as gr
import joblib
from utils import clean_text
import os

MODEL_PATH = "model.pkl"
VEC_PATH = "vectorizer.pkl"

def load_models():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    vec = joblib.load(VEC_PATH)
    return model, vec

model, vectorizer = load_models()

def predict(text):
    if model is None or vectorizer is None:
        return "Model not found. Please upload model.pkl and vectorizer.pkl to the repo or train the model."
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(vec)[0].max())
    label = "REAL" if int(pred) == 1 else "FAKE"
    if proba:
        return f"{label} (confidence {proba*100:.1f}%)"
    return label

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=6, placeholder="Paste news text here..."),
    outputs="text",
    title="Fake News Detector",
    description="Detect whether a news text is REAL or FAKE."
)

if __name__ == "__main__":
    demo.launch()
