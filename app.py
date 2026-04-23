import streamlit as st
import pickle
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

@st.cache_resource
def load_models():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('best_ml_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    return svm_model, vectorizer

svm_model, vectorizer = load_models()

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

def predict(text):
    cleaned = clean_text(text)
    labels = {0: '😠 Negative', 1: '😐 Neutral', 2: '😊 Positive'}
    colors = {0: 'red', 1: 'orange', 2: 'green'}
    vec  = vectorizer.transform([cleaned])
    pred = svm_model.predict(vec)[0]
    return labels[pred], colors[pred]

# UI
st.set_page_config(page_title="Sentiment Analyzer", page_icon="✈️")
st.title("✈️ Airline Tweet Sentiment Analyzer")
st.markdown("Analyze the sentiment of airline-related tweets using an SVM model trained on TF-IDF features.")

user_input = st.text_area("Enter a tweet:", placeholder="e.g. @united my flight was delayed again!")

if st.button("Analyze ↗"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label, color = predict(user_input)
        st.markdown(f"### Prediction: :{color}[{label}]")
