
import numpy as np
import re
import tensorflow as tf

from typing import Any, Dict

from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 60  
MAX_NUM_WORDS = 200

_trained_model = None
tokenizer = None

base_model = 'bilstm5'
tokenizer_path = f'./data_ignore/tokenizer.{base_model}.json'
ml_model_path = f'./data_ignore/{base_model}'

def tokenize_text(q1: str, q2: str):
    # tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    # tokenizer.fit_on_texts([q1, q2])  # Load the Tokenizer from json.. current fit is wrong... 
    global tokenizer

    if tokenizer == None:
        with open(tokenizer_path, "r") as f:
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

def get_model():
    global _trained_model
    if _trained_model is None:
        _trained_model=tf.keras.models.load_model(ml_model_path) 

    return _trained_model

def predict(q1: str, q2: str):
    q1 = clean_text(q1)
    q2 = clean_text(q2)

    q1_data, q2_data, word_index = tokenize_text(q1, q2)

    model = get_model()
    predictions = model([q1_data, q2_data])
    print(f'q1_data: {q1_data}')
    print(f'q2_data: {q2_data}')
    print('-' * 10)
    print(f'q1: {q1}')
    print(f'q2: {q2}')
    # print(f'info: {type(predictions)} value: {predictions[0][0]}')
    result = float(predictions[0][0])
    print(f'same probabilty percentage: {round(result  * 100, 4)} %  actual: {predictions[0][0]}\n' + '-' * 20)


# main driver function
if __name__ == '__main__':
    q1 = 'What can make Physics easy to learn?'
    q2 = 'How can you make physics easy to learn?'
    predict(q1, q2)
    # embed = get_word_embeddings()

    q1 = 'What should I do to be a great geologist?'
    q2 = 'How can I be a good geologist?'
    predict(q1, q2)

    # Do you believe there is life after death? 	Is it true that there is life after death? 	1
    q1 = 'Do you believe there is life after death?'
    q2 = 'Is it true that there is life after death?'
    predict(q1, q2)

    # How do I read and find my YouTube comments? How can I see all my Youtube comments? 1
    q1 = 'How do I read and find my YouTube comments?'
    q2 = 'How can I see all my Youtube comments?'
    predict(q1, q2)
    
    q1 = 'How do I see the color blue?'
    q2 = 'How do I see the color blue?'
    predict(q1, q2)

    # What is one coin? 	What's this coin? 	0
    q1 = 'What is one coin?'
    q2 = 'What\'s this coin?'
    predict(q1, q2)

    # Why do girls want to be friends with the guy they reject? How do guys feel after rejecting a girl? 0
    q1 = 'Why do girls want to be friends with the guy they reject?'
    q2 = 'How do guys feel after rejecting a girl?'
    predict(q1, q2)