#Predict
from flask import Flask, render_template, request, redirect, url_for
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import json
from transformers import AutoTokenizer, AutoModel
from pyvi import ViTokenizer
import re
import joblib

app = Flask(__name__)

best_model = joblib.load("RF.pkl") 

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
phobert_model = AutoModel.from_pretrained("vinai/phobert-base")


with open("vietnamesestopwords.txt", "r", encoding="utf-8") as f:
    stopwords = set(f.read().splitlines())

with open("tudonham.json", "r", encoding="utf-8") as f:
    tudonham = json.load(f)
with open("tudonspam.json", "r", encoding="utf-8") as f:
    tudonspam = json.load(f)
with open("tughepham.json", "r", encoding="utf-8") as f:
    tughepham = json.load(f)
with open("tughepspam.json", "r", encoding="utf-8") as f:
    tughepspam = json.load(f)

def preprocess_text(text, update_vocab=False, label=None):
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = text.strip()
    tokens = ViTokenizer.tokenize(text).split()
    tokens = [word for word in tokens if word not in stopwords]
    return " ".join(tokens)

def extract_frequency_features(text, tudonham, tudonspam, tughepham, tughepspam):
    text = text.lower()

    # Đếm tần suất từ đơn
    tudonhamfreq = sum([tudonham.get(token, 0) for token in text.split()])
    tudonspamfreq = sum([tudonspam.get(token, 0) for token in text.split()])

    # Đếm tần suất từ ghép
    tughephamfreq = sum([freq for token, freq in tughepham.items() if token in text])
    tughepspamfreq = sum([freq for token, freq in tughepspam.items() if token in text])

    return [tudonhamfreq, tudonspamfreq, tughephamfreq, tughepspamfreq]


def predict_email(email, stopwords=[], tudonham=None, tudonspam=None, tughepham=None, tughepspam=None, tokenizer=None, phobert_model=None, best_model=None, update_vocab=False, label=None):
    # Tiền xử lý email
    processed_email = preprocess_text(email, stopwords)
    
    # Trích xuất đặc trưng tần suất
    frequency_features = extract_frequency_features(
        processed_email, tudonham, tudonspam, tughepham, tughepspam)
    
    # Trích xuất đặc trưng từ PhoBERT
    inputs = tokenizer(processed_email, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        phobert_features = phobert_model(**inputs).last_hidden_state.mean(dim=1).numpy().flatten()
    
    # Kết hợp đặc trưng PhoBERT và tần suất
    features_combined = np.hstack((phobert_features, frequency_features)).reshape(1, -1)
    
    prediction = best_model.predict(features_combined)
    label = "Spam" if prediction[0] == 1 else "Not Spam"
    return label

# Test dự đoán


# API dự đoán
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        email = request.form.get("email")
        if not email.strip():
            return render_template("index.html", error="Email content cannot be empty!")

        try:
            processed_email = preprocess_text(email)
            frequency_features = extract_frequency_features(
                processed_email, tudonham, tudonspam, tughepham, tughepspam
            )
            inputs = tokenizer(processed_email, padding=True, truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                phobert_features = phobert_model(**inputs).last_hidden_state.mean(dim=1).numpy().flatten()

            features_combined = np.hstack((phobert_features, frequency_features)).reshape(1, -1)
            prediction = best_model.predict(features_combined)
            result = "Spam" if prediction[0] == 1 else "Not Spam"
            return redirect(url_for("show", prediction=result))
        except Exception as e:
            return render_template("index.html", error=f"An error occurred: {str(e)}")

    return render_template("index.html")
@app.route("/show")
def show():
    prediction = request.args.get("prediction")  # Get the prediction result from query params
    return render_template("show.html", prediction=prediction)

def predict_email(email):
    # Placeholder for your prediction logic
    return 1 if "spam" in email.lower() else 0  



# Chạy ứng dụng
if __name__ == "__main__":
    app.run(debug=True)