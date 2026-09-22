from typing import Any, Dict, Sequence, Tuple
import numpy as np
import os
import pandas as pd
import tensorflow as tf

import constants as c
from text_preprocessing import normalize_text

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


def generate_train_data() -> Tuple[Sequence, Sequence, Sequence]:
    df = pd.read_csv(c.PATH_TO_QUESTIONS)

    q1 = df['question1'].values
    q2 = df['question2'].values
    labels = df['is_duplicate'].values

    text_q1 = [normalize_text(text) for text in q1]
    text_q2 = [normalize_text(text) for text in q2]

    return text_q1, text_q2, labels


def split_train_test(text_q1: Sequence, text_q2: Sequence, labels: Sequence) -> Tuple[Sequence, ...]:
    (train_text_q1, test_text_q1,
     train_text_q2, test_text_q2,
     train_labels, test_labels) = train_test_split(
        text_q1, text_q2, labels,
        test_size=c.TEST_SIZE,
        random_state=c.RANDOM_STATE,
        stratify=labels,
    )
    return train_text_q1, train_text_q2, train_labels, test_text_q1, test_text_q2, test_labels


def tokenize_data(train_text_q1: Sequence, train_text_q2: Sequence,
                   test_text_q1: Sequence, test_text_q2: Sequence) -> Tuple[Any, ...]:
    tokenizer = Tokenizer(num_words=c.MAX_NUM_WORDS)
    # Fit on the training split only - fitting on test text would leak test
    # vocabulary into the tokenizer and inflate held-out metrics.
    tokenizer.fit_on_texts(train_text_q1 + train_text_q2)

    train_sequences_1 = tokenizer.texts_to_sequences(train_text_q1)
    train_sequences_2 = tokenizer.texts_to_sequences(train_text_q2)
    test_sequences_1 = tokenizer.texts_to_sequences(test_text_q1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_text_q2)

    word_index = tokenizer.word_index

    train_data_1 = pad_sequences(train_sequences_1, maxlen=c.MAX_SEQUENCE_LENGTH)
    train_data_2 = pad_sequences(train_sequences_2, maxlen=c.MAX_SEQUENCE_LENGTH)
    test_data_1 = pad_sequences(test_sequences_1, maxlen=c.MAX_SEQUENCE_LENGTH)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=c.MAX_SEQUENCE_LENGTH)

    return train_data_1, train_data_2, test_data_1, test_data_2, word_index, tokenizer


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
            epochs=25, batch_size=128, shuffle=True, \
            callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path) # sotre model parameters in .h5 file

    return hist

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
    import mlflow

    from evaluate import evaluate_model, print_report, save_report

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    with mlflow.start_run():
        mlflow.log_params({
            'max_sequence_length': c.MAX_SEQUENCE_LENGTH,
            'max_num_words': c.MAX_NUM_WORDS,
            'embedding_dim': c.EMBEDDING_DIM,
            'n_hidden': c.N_HIDDEN,
            'n_dense': c.N_DENSE,
            'dropout_rate_lstm': c.DROPOUT_RATE_LSTM,
            'dropout_rate_dense': c.DROPOUT_RATE_DENSE,
            'active_func': c.ACTIVE_FUNC,
            'model_training_patience': c.MODEL_TRAINING_PATIENCE,
            'test_size': c.TEST_SIZE,
            'random_state': c.RANDOM_STATE,
            'version': c.VERSION,
        })

        print('Loading Word Embeddings')
        embeddings_index = create_word_embedding()

        print('Generating normalized text data')
        text_q1, text_q2, labels = generate_train_data()

        print('Splitting train/test data')
        (train_text_q1, train_text_q2, train_labels,
         test_text_q1, test_text_q2, test_labels) = split_train_test(text_q1, text_q2, labels)

        print('Tokenizing train/test data (tokenizer fit on train split only)')
        (train_data_1, train_data_2, test_data_1, test_data_2,
         word_index, tokenizer) = tokenize_data(
            train_text_q1=train_text_q1, train_text_q2=train_text_q2,
            test_text_q1=test_text_q1, test_text_q2=test_text_q2,
        )

        print('define ml model')
        model, bst_model_path = define_model(word_index=word_index, embeddings_index=embeddings_index)

        print('run ml model training')
        hist = run_model_training(
            model=model, bst_model_path=bst_model_path,
            train_data_1=train_data_1, train_data_2=train_data_2, train_labels=train_labels,
        )

        val_acc_key = 'val_acc' if 'val_acc' in hist.history else 'val_accuracy'
        mlflow.log_metrics({
            'best_val_loss': min(hist.history['val_loss']),
            'best_val_acc': max(hist.history[val_acc_key]),
        })

        print('save ml model')
        save_model(model=model, tokenizer=tokenizer)

        print('evaluate ml model on held-out test set')
        report = evaluate_model(model, test_data_1, test_data_2, test_labels)
        print_report(report)
        mlflow.log_metrics({
            'test_accuracy': report['accuracy'],
            'test_precision': report['precision'],
            'test_recall': report['recall'],
            'test_f1': report['f1'],
            'test_roc_auc': report['roc_auc'],
        })

        report_path = f'data_ignore/eval_report.{c.VERSION}.json'
        save_report(report, report_path)

        mlflow.log_artifacts(f'data_ignore/{c.VERSION}', artifact_path='model')
        mlflow.log_artifact(f'./data_ignore/tokenizer.{c.VERSION}.json')
        mlflow.log_artifact(report_path)
