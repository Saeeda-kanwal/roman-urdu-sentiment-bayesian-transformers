import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

model, tokenizer = load_model()

# Predict function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).numpy()[0]
    labels = ["Negative", "Positive"]
    return labels[np.argmax(probs)], probs

# UI
st.title("📊 Roman Urdu Sentiment Analyzer")
user_input = st.text_area("Enter Roman Urdu text (e.g., ye movie zabardast thi):")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment, probabilities = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: **{sentiment}**")
        st.write(f"Confidence Scores: {probabilities}")
    else:
        st.warning("Please enter some text.")
