import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from utils import clean_text

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasets/fake-news/master/data/fake.csv")
df["label"] = 0  # Fake news

df2 = pd.read_csv("https://raw.githubusercontent.com/datasets/fake-news/master/data/real.csv")
df2["label"] = 1  # Real news

data = pd.concat([df, df2], ignore_index=True)
data = data[["text", "label"]].dropna()

# Clean text
data["text"] = data["text"].apply(clean_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Accuracy
pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Model saved!")
