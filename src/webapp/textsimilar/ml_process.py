from flask import current_app as app

from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

import constants as c
from text_preprocessing import normalize_text
from webapp.textsimilar.models import SimilarityTest

from . import get_model

tokenizer = None

def tokenize_text(q1: str, q2: str):
    global tokenizer
    if tokenizer == None:
        tokenizer_json_path = app.config['TOKENIZER_PATH']
        print(f'loading tokenizer from path: {tokenizer_json_path}')
        with open(tokenizer_json_path, "r") as f:
            token_json = f.read()
        tokenizer = tokenizer_from_json(token_json)

    seq_q1 = tokenizer.texts_to_sequences([q1])
    seq_q2 = tokenizer.texts_to_sequences([q2])

    q1_data = pad_sequences(seq_q1, maxlen=c.MAX_SEQUENCE_LENGTH)
    q2_data = pad_sequences(seq_q2, maxlen=c.MAX_SEQUENCE_LENGTH)

    word_index = tokenizer.word_index

    return (q1_data, q2_data, word_index)


def predict(q1: str, q2: str):
    q1 = normalize_text(q1)
    q2 = normalize_text(q2)

    q1_data, q2_data, word_index = tokenize_text(q1, q2)

    model = get_model()
    predictions = model([q1_data, q2_data])
    actual_result = float(predictions[0][0])
    rounded_result = round(actual_result  * 100, 4)

    sm = SimilarityTest(q1=q1, q2=q2, probability=actual_result, rounded=rounded_result)
    return sm
