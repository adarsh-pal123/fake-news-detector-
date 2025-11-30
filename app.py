import gradio as gr
import joblib
from utils import clean_text

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_news(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    
    if pred == 0:
        return "🚨 FAKE NEWS"
    return "✔ REAL NEWS"

interface = gr.Interface(
    fn=predict_news,
    inputs="text",
    outputs="text",
    title="Fake News Detector",
    description="Enter news content to classify it as Fake or Real."
)

interface.launch()
