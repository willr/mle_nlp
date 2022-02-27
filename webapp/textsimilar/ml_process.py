import re

from flask import current_app as app

from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from webapp.textsimilar.models import SimilarityTest

from . import get_model

MAX_SEQUENCE_LENGTH = 60  
MAX_NUM_WORDS = 200

tokenizer = None

def tokenize_text(q1: str, q2: str):
    global tokenizer
    if tokenizer == None:
        tokenizer_json_path = app.config['TOKENIZER_PATH']
        with open(tokenizer_json_path, "r") as f:
            token_json = f.read()
        tokenizer = tokenizer_from_json(token_json)

    seq_q1 = tokenizer.texts_to_sequences([q1])
    seq_q2 = tokenizer.texts_to_sequences([q2])

    q1_data = pad_sequences(seq_q1, maxlen=MAX_SEQUENCE_LENGTH)
    q2_data = pad_sequences(seq_q2, maxlen=MAX_SEQUENCE_LENGTH)

    word_index = tokenizer.word_index

    return (q1_data, q2_data, word_index)


def clean_text(text):
    text = str(text).split()
    text = " ".join(text)

    # Use re to clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text, re.IGNORECASE)
    text = re.sub(r"what's", "what is ", text, re.IGNORECASE)
    text = re.sub(r"\’s", " ", text, re.IGNORECASE)
    text = re.sub(r"\'s", " ", text, re.IGNORECASE)
    text = re.sub(r"\'ve", " have ", text, re.IGNORECASE)
    text = re.sub(r"can't", "cannot ", text, re.IGNORECASE)
    text = re.sub(r"n't", " not ", text, re.IGNORECASE)
    text = re.sub(r"i'm", "i am ", text, re.IGNORECASE)
    text = re.sub(r"\'re", " are ", text, re.IGNORECASE)
    text = re.sub(r"\'d", " would ", text, re.IGNORECASE)
    text = re.sub(r"\'ll", " will ", text, re.IGNORECASE)
    text = re.sub(r"\‘", " ", text, re.IGNORECASE)
    text = re.sub(r"\’", " ", text, re.IGNORECASE)
    text = re.sub(r"\"", " ", text, re.IGNORECASE)
    text = re.sub(r"\“", " ", text, re.IGNORECASE)
    text = re.sub(r"\”", " ", text, re.IGNORECASE)
    text = re.sub(r",", " ", text, re.IGNORECASE)
    text = re.sub(r"\.", " ", text, re.IGNORECASE)
    text = re.sub(r"!", " ! ", text, re.IGNORECASE)
    text = re.sub(r"\/", " ", text, re.IGNORECASE)
    text = re.sub(r"\^", " ^ ", text, re.IGNORECASE)
    text = re.sub(r"\+", " + ", text, re.IGNORECASE)
    text = re.sub(r"\-", " - ", text, re.IGNORECASE)
    text = re.sub(r"\=", " = ", text, re.IGNORECASE)
    text = re.sub(r"'", " ", text, re.IGNORECASE)
    text = re.sub(r":", " : ", text, re.IGNORECASE)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text, re.IGNORECASE)
    text = re.sub(r" e g ", " eg ", text, re.IGNORECASE)
    text = re.sub(r" b g ", " bg ", text, re.IGNORECASE)
    text = re.sub(r" u s ", " american ", text, re.IGNORECASE)
    text = re.sub(r" 9 11 ", "911", text, re.IGNORECASE)
    text = re.sub(r"e - mail", "email", text, re.IGNORECASE)
    text = re.sub(r"j k", "jk", text, re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text, re.IGNORECASE)
    text = re.sub(r"\？", " ", text, re.IGNORECASE)

    return text

def predict(q1: str, q2: str):
    q1 = clean_text(q1)
    q2 = clean_text(q2)

    q1_data, q2_data, word_index = tokenize_text(q1, q2)

    model = get_model()
    predictions = model([q1_data, q2_data])
    actual_result = float(predictions[0][0])
    rounded_result = round(actual_result  * 100, 4)

    sm = SimilarityTest(q1=q1, q2=q2, probability=actual_result, rounded=rounded_result)
    return sm
