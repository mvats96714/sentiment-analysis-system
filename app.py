import streamlit as st
import pickle
from src.preprocess import clean_text
import re
import time
import nltk

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("words")

model = pickle.load(open("model/model.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬")

st.title("🎬 Movie Sentiment Analyzer")
st.markdown("Analyze whether a movie review is Positive or Negative.")

st.divider()

review = st.text_area("✍ Enter your movie review:", height=150)

# ---------- Language Warning ----------
from nltk.corpus import words
english_words = set(words.words())

def is_probably_english(text):
    tokens = text.lower().split()
    if len(tokens) == 0:
        return True
    english_count = sum(1 for word in tokens if word in english_words)
    return english_count / len(tokens) > 0.5

if review:
    if not is_probably_english(review):
        st.warning("⚠ This review may not be proper English. Model works best on English reviews.")

# ---------- Prediction ----------
if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.error("Please enter a review first.")
    else:
        cleaned = clean_text(review)
        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0]

        confidence = max(proba) * 100

        st.divider()
        st.subheader("📊 Prediction Result")

        if prediction == "positive":
            st.success("😊 Positive Review")
        else:
            st.error("😡 Negative Review")

        st.metric("Confidence", f"{confidence:.2f}%")

        # ---------- Animated Progress ----------
        st.subheader("🔎 Confidence Meter")
        progress_bar = st.progress(0)

        progress_value = int(confidence)
        for i in range(progress_value):
            time.sleep(0.01)
            progress_bar.progress(i + 1)

st.divider()
st.markdown(
    """
    <div style='text-align: center; padding: 10px;'>
        Built by <b>Mohit</b>  <br>
        <a href='https://github.com/mvats96714' target='_blank'>
            🔗 View on GitHub
        </a>
    </div>
    """,
    unsafe_allow_html=True
)