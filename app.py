import streamlit as st
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

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

@st.cache_resource
def train_model():
    df = pd.read_csv('Tweets.csv')
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['label'] = df['airline_sentiment'].map(label_map)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']
    model = LinearSVC(class_weight='balanced', max_iter=2000)
    model.fit(X, y)
    return model, vectorizer

# UI
st.set_page_config(page_title="Sentiment Analyzer", page_icon="✈️")
st.title("✈️ Airline Tweet Sentiment Analyzer")
st.markdown("Powered by SVM + TF-IDF trained on real airline tweets.")

with st.spinner("Loading model... (first time takes ~30 seconds)"):
    model, vectorizer = train_model()

st.success("✅ Model ready!")

user_input = st.text_area("Enter a tweet:", placeholder="e.g. @united my flight was delayed again!")

if st.button("Analyze ↗"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        labels = {0: '😠 Negative', 1: '😐 Neutral', 2: '😊 Positive'}
        colors = {0: 'red', 1: 'orange', 2: 'green'}
        st.markdown(f"### Prediction: :{colors[pred]}[{labels[pred]}]")
