import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords only once
nltk.download("stopwords", quiet=True)

app = Flask(__name__)

# Load model & tokenizer once at startup
model = load_model("lstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# ------------------ TEXT CLEANING ------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# ------------------ PREDICTION ------------------
def predict_sentiment(text):
    text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100)

    prediction = model.predict(padded, verbose=0)[0][0]

    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = round(float(prediction) * 100, 2)

    return sentiment, confidence

# ------------------ ROUTE ------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None

    if request.method == "POST":
        text = request.form.get("text")

        if text and text.strip() != "":
            result, confidence = predict_sentiment(text)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence
    )

# ------------------ MAIN ------------------
if __name__ == "__main__":
    app.run(debug=True)
