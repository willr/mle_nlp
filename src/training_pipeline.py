import re
from typing import Any, Dict, Sequence, Tuple
import numpy as np
import os
import pandas as pd
import tensorflow as tf

import constants as c

from sklearn.model_selection import train_test_split

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, Dropout, Activation, LSTM, Lambda, Bidirectional, Dot, concatenate, BatchNormalization, GlobalAveragePooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K


def create_word_embedding():
    embeddings_index = {}
    with open(c.PATH_TO_GLOVE_FILE) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    return embeddings_index

def normalize_text(text):
    
    # split words
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


def generate_train_data() -> Tuple[Sequence, Sequence, Sequence]:
    df = pd.read_csv(c.PATH_TO_QUESTIONS)
    y = df['is_duplicate']
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=c.TEST_SIZE, random_state=c.RANDOM_STATE)

    train_q1 = X_train['question1'].values
    train_q2 = X_train['question2'].values
    train_labels = X_train['is_duplicate'].values

    train_text_q1 = [] # preprocessed text of q1
    train_text_q2 = [] # preprocessed text of q2

    text_set = set() # complete set of words for building embeddings

    for text in train_q1:
        tt = normalize_text(text)
        text_set.add(tt)
        train_text_q1.append(tt)
    for text in train_q2:
        tt = normalize_text(text)
        text_set.add(tt)
        train_text_q2.append(tt)

    return train_text_q1, train_text_q2, train_labels

def tokenize_test_data(train_text_q1: str, train_text_q2: str) -> Tuple[Sequence, Sequence, Sequence, Tokenizer]:
    tokenizer = Tokenizer(num_words=c.MAX_NUM_WORDS)
    tokenizer.fit_on_texts(train_text_q1 + train_text_q2)  # generate a token dictionary, 

    train_sequences_1 = tokenizer.texts_to_sequences(train_text_q1)  # sequence of q1
    train_sequences_2 = tokenizer.texts_to_sequences(train_text_q2)  # sequence of q2

    word_index = tokenizer.word_index
    
    # Pad all train with Max_Sequence_Length: 60
    train_data_1 = pad_sequences(train_sequences_1, maxlen=c.MAX_SEQUENCE_LENGTH)  # padded_sequence of q1 as train_data
    train_data_2 = pad_sequences(train_sequences_2, maxlen=c.MAX_SEQUENCE_LENGTH)  # padded_sequence of q2 as train_data

    return train_data_1, train_data_2, word_index, tokenizer

def define_model(word_index: Dict[Any, int], embeddings_index: Dict[str, int]) -> Tuple[Model, str]:
    num_tokens = len(word_index) + 2

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, c.EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(
        num_tokens,
        c.EMBEDDING_DIM,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,
    )

    # BiLSTM layer
    from tensorflow.keras.layers import Bidirectional, LSTM
    lstm_layer = Bidirectional(LSTM(c.N_HIDDEN, dropout=c.DROPOUT_RATE_LSTM, recurrent_dropout=c.DROPOUT_RATE_LSTM))

    # Define inputs
    seq1 = Input(shape=(c.MAX_SEQUENCE_LENGTH,), dtype='int32')
    seq2 = Input(shape=(c.MAX_SEQUENCE_LENGTH,), dtype='int32')

    # Run inputs through embedding
    emb1 = embedding_layer(seq1)
    emb2 = embedding_layer(seq2)

    # Run through LSTM layers
    lstm_a = lstm_layer(emb1)
    lstm_b = lstm_layer(emb2)

    # cosin_sim_func = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([lstm_a, lstm_b])
    dotted = Dot(axes=-1, normalize=True)([lstm_a, lstm_b])

    l1_norm = lambda x: 1 - K.abs(x[0] - x[1])
    l1_dist = Lambda(function=l1_norm, output_shape=lambda x: x[0], name='L1_distance')([lstm_a, lstm_b])

    merged = concatenate([lstm_a, lstm_b, l1_dist, dotted])
    merged = BatchNormalization()(merged)
    merged = Dropout(c.DROPOUT_RATE_DENSE)(merged)

    merged = Dense(c.N_DENSE, activation=c.ACTIVE_FUNC)(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(c.DROPOUT_RATE_DENSE)(merged)

    merged = Dense(c.N_DENSE, activation=c.ACTIVE_FUNC)(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(c.DROPOUT_RATE_DENSE)(merged)

    merged = Dense(c.N_DENSE, activation=c.ACTIVE_FUNC)(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(c.DROPOUT_RATE_DENSE)(merged)

    merged = Dense(c.N_DENSE, activation=c.ACTIVE_FUNC)(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(c.DROPOUT_RATE_DENSE)(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    bst_model_path = c.VERSION + '.h5' 

    model = Model(inputs=[seq1, seq2], outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model, bst_model_path
    
def run_model_training(model: Model, bst_model_path: str, train_data_1: Sequence, train_data_2: Sequence, train_labels: Sequence):
    print('Starting the model training')
    # Set early stopping (large patience should be useful)
    early_stopping =EarlyStopping(monitor='val_loss', patience=c.MODEL_TRAINING_PATIENCE)
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    hist = model.fit([train_data_1, train_data_2], train_labels, \
            validation_split=.1, \
            epochs=1, batch_size=128, shuffle=True, \
            callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path) # sotre model parameters in .h5 file
    bst_val_score = min(hist.history['val_loss'])

def save_model(model: Model, tokenizer: Tokenizer):
    # save the model
    local_model_path = f'data_ignore/{c.VERSION}'
    cwd = os.getcwd()
    print(f'save model to: {os.path.join(cwd, local_model_path)}')
    model.save(local_model_path)
    tokenizer_json = tokenizer.to_json()
    print(f'size of tokenizer json: {len(tokenizer_json)}')
    local_tokenizer_path = f'./data_ignore/tokenizer.{c.VERSION}.json'
    print(f'saving tokenizer to: {os.path.join(cwd, local_tokenizer_path)}')
    with open(local_tokenizer_path, 'w') as token_json:
        token_json.write(tokenizer_json)

if __name__ == "__main__":

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    print('Loading Word Embeddings')
    embeddings_index = create_word_embedding()

    print('Split train data')
    train_text_q1, train_text_q2, train_labels = generate_train_data()

    print('tokenize train data')
    train_data_1, train_data_2, word_index, tokenizer = tokenize_test_data(train_text_q1=train_text_q1, train_text_q2=train_text_q2)

    print('define ml model')
    model, bst_model_path = define_model(word_index=word_index, embeddings_index=embeddings_index)

    print('run ml model training')
    run_model_training(model=model, bst_model_path=bst_model_path, train_data_1=train_data_1, train_data_2=train_data_2, train_labels=train_labels)

    print('save ml model')
    save_model(model=model, tokenizer=tokenizer)

