import streamlit as st
import pickle
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load models
@st.cache_resource
def load_models():
    from sklearn.svm import LinearSVC
    from tensorflow.keras.models import load_model

    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('best_ml_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    lstm_model = load_model('lstm_model.h5')
    return svm_model, vectorizer, lstm_model, tokenizer

svm_model, vectorizer, lstm_model, tokenizer = load_models()

# Clean text
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

# Predict
def predict(text, model_choice):
    cleaned = clean_text(text)
    labels  = {0: '😠 Negative', 1: '😐 Neutral', 2: '😊 Positive'}
    colors  = {0: 'red', 1: 'orange', 2: 'green'}

    if model_choice == "SVM (TF-IDF)":
        from scipy.sparse import csr_matrix
        vec  = vectorizer.transform([cleaned])
        pred = svm_model.predict(vec)[0]
        return labels[pred], colors[pred], None
    else:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq  = pad_sequences(tokenizer.texts_to_sequences([cleaned]), maxlen=50)
        prob = lstm_model.predict(seq)[0]
        pred = np.argmax(prob)
        return labels[pred], colors[pred], prob

# UI
st.set_page_config(page_title="Sentiment Analyzer", page_icon="✈️")
st.title("✈️ Airline Tweet Sentiment Analyzer")
st.markdown("Analyze the sentiment of airline-related tweets using ML and DL models.")

model_choice = st.selectbox("Choose a model:", ["SVM (TF-IDF)", "LSTM (Deep Learning)"])
user_input   = st.text_area("Enter a tweet:", placeholder="e.g. @united my flight was delayed again!")

if st.button("Analyze ↗"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label, color, probs = predict(user_input, model_choice)
        st.markdown(f"### Prediction: :{color}[{label}]")

        if probs is not None
