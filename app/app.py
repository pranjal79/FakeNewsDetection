from flask import Flask, request, jsonify, render_template
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (safe if already present)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

app = Flask(__name__)

# ---------------- LOAD MODEL & VECTORIZER ---------------- #

with open("data/processed/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("data/processed/logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ---------------- TEXT CLEANING FUNCTION ---------------- #

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


# ---------------- BROWSER UI ROUTE ---------------- #

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# ---------------- FORM ANALYSIS ROUTE ---------------- #

@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form["news_text"]

    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text).max()

    result = "Real News" if prediction == 1 else "Fake News"

    return render_template(
        "index.html",
        prediction=result,
        confidence=round(probability * 100, 2),
        cleaned_text=cleaned_text
    )


# ---------------- API ROUTE (JSON RESPONSE) ---------------- #

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    input_text = data["text"]

    cleaned_text = clean_text(input_text)
    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text).max()

    result = "Real News" if prediction == 1 else "Fake News"

    return jsonify({
        "input_text": input_text,
        "cleaned_text": cleaned_text,
        "prediction": result,
        "confidence": round(float(probability), 4)
    })


# ---------------- RUN FLASK APP ---------------- #

if __name__ == "__main__":
    app.run(debug=True)
