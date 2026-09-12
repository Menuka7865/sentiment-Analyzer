# app.py
import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return labels[pred], proba

# --- UI ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")

st.title("💬 Real-World Sentiment Analyzer")
st.write("Trained on real app store reviews. Enter any text below to analyze its sentiment.")

user_input = st.text_area("Enter text (review, comment, feedback, etc.)", height=120)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        label, proba = predict_sentiment(user_input)

        emoji = {"Negative": "😠", "Neutral": "😐", "Positive": "😀"}
        color = {"Negative": "error", "Neutral": "warning", "Positive": "success"}

        getattr(st, color[label])(f"{emoji[label]} **{label}**")

        st.write("Confidence breakdown:")
        st.progress(float(proba[0]), text=f"Negative: {proba[0]*100:.1f}%")
        st.progress(float(proba[1]), text=f"Neutral: {proba[1]*100:.1f}%")
        st.progress(float(proba[2]), text=f"Positive: {proba[2]*100:.1f}%")

st.markdown("---")
st.caption("Model: Logistic Regression + TF-IDF | Trained on real Google Play Store reviews")