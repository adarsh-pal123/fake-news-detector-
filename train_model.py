# train_model.py
# Run this in Google Colab to train a model and save model.pkl and vectorizer.pkl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from utils import clean_text

# If you upload your own CSV, set data_path to the uploaded filename in Colab.
# The CSV must have columns: 'text' and 'label' where label is 0 for FAKE and 1 for REAL.
data_path = "train.csv"

def load_and_prepare(path):
    df = pd.read_csv(path)
    # If dataset uses title + text columns, combine:
    if 'title' in df.columns and 'text' in df.columns:
        df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df = df.dropna(subset=['text', 'label'])
    df['clean_text'] = df['text'].astype(str).apply(clean_text)
    return df

def train_and_save(path="train.csv"):
    df = load_and_prepare(path)
    X = df['clean_text']
    y = df['label'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, digits=4))

    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("Saved model.pkl and vectorizer.pkl to current folder.")

if __name__ == "__main__":
    print("This script is meant to be run in Google Colab. Call train_and_save() after uploading data.")
