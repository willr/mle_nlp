from os import environ, path
import os

basedir = path.abspath(path.dirname(__file__))

class Config:
    SECRET_KEY = os.urandom(32)
    STATIC_FOLDER = 'static'
    TEMPLATES_FOLDER = 'templates'

class ProdConfig(Config):
    FLASK_ENV = 'production'
    DEBUG = False
    TESTING = False
    ML_MODEL_PATH = environ.get('ML_MODEL_PATH')
    TOKENIZER_PATH = environ.get('TOKENIZER_JSON_PATH')

class TestConfig(Config):
    FLASK_ENV = 'test'
    DEBUG = False
    TESTING = True
    ML_MODEL_PATH = './data_ignore/bilstm5'
    TOKENIZER_PATH = './tokenizer.bilstm5.json'

class DevConfig(Config):
    FLASK_ENV = 'development'
    DEBUG = True
    TESTING = False
    ML_MODEL_PATH = './data_ignore/bilstm5'
    TOKENIZER_PATH = './tokenizer.bilstm5.json'

