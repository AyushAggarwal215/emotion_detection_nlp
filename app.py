import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# Load model, vectorizer and emotion mapping
model = pickle.load(open("model.pkl", "rb"))
tfidf_vectorizer = pickle.load(open("tfidf.pkl", "rb"))
emotion_number = pickle.load(open("emotion_mapping.pkl", "rb"))


# Reverse mapping
number_emotion = {v: k for k, v in emotion_number.items()}


# Stopwords
stop_words = set(stopwords.words("english"))


def remove_punc(txt):
    return txt.translate(str.maketrans("", "", string.punctuation))


def remove_num(txt):
    new = ""

    for i in txt:
        if not i.isdigit():
            new = new + i

    return new


def remove_emoji(txt):
    new = ""

    for i in txt:
        if i.isascii():
            new = new + i

    return new


def remove_stopwords(txt):
    words = word_tokenize(txt)

    cleaned = []

    for i in words:
        if i not in stop_words:
            cleaned.append(i)

    return " ".join(cleaned)


def preprocess(text):

    text = text.lower()
    text = remove_punc(text)
    text = remove_num(text)
    text = remove_emoji(text)
    text = remove_stopwords(text)

    return text


st.title("😊 Emotion Detection using NLP")

st.write("Enter a sentence and the model will predict the emotion.")

text = st.text_area("Enter your text:")


if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        cleaned_text = preprocess(text)

        text_tfidf = tfidf_vectorizer.transform([cleaned_text])

        prediction = model.predict(text_tfidf)[0]

        emotion = number_emotion[prediction]

        st.success(f"Predicted Emotion: {emotion}")
