"""Text normalization shared by training and serving.

This used to be hand-duplicated in training_pipeline.py (normalize_text) and
webapp/textsimilar/ml_process.py (clean_text) - identical regex logic,
copy-pasted. That's a training/serving skew hazard: a model is only valid for
the exact normalization it was trained against, so any edit made to one copy
and not the other silently changes what the served model sees at inference
time versus what it saw at training time.

If you change the logic here, the currently saved model + tokenizer are
invalidated and must be retrained.
"""
import re


def normalize_text(text: str) -> str:
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
