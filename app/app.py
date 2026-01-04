from flask import Flask, request, jsonify
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (safe if already downloaded)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

app = Flask(__name__)

# Load artifacts
with open("data/processed/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("data/processed/logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    words = text.split()
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    cleaned_text = clean_text(data["text"])
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]

    result = "Real News" if prediction == 1 else "Fake News"

    return jsonify({"prediction": result})


@app.route("/")
def home():
    return "Fake News Detection API is running 🚀"


if __name__ == "__main__":
    app.run(debug=True)
