from enum import Enum
from flask import Flask, current_app as app

import os

from keras.models import load_model
from .config import ProdConfig, TestConfig, DevConfig

PastTS = []
Nav = [
        {'name': 'Home', 'url': '/'},
        {'name': 'Test-If-Text-Similar', 'url': '/is_similar'},
    ]
ML_Model = None

class Environment(Enum):
    PRODUCTION = 1,
    TEST = 2,
    DEVELOPMENT = 3

def create_app(env: Environment):
    app = Flask(
        __name__,
        instance_relative_config=False,
    )

    if env == Environment.PRODUCTION:
        app.config.from_object('config.ProdConfig')
    elif env == Environment.TEST:
        app.config.from_object('config.TestConfig')
    elif env == Environment.DEVELOPMENT:
        app.config.from_object(DevConfig)
    else:
        raise ValueError(f'Failed to handle passed in Environment: {env}')

    with app.app_context():
        from . import routes  # Import routes

        SECRET_KEY = os.urandom(32)
        app.config['SECRET_KEY'] = SECRET_KEY
        
        # loading model.. this will take a while..
        # get_model()

        return app

def get_model():
    global ML_Model
    if ML_Model is None:
        model_path = app.config['ML_MODEL_PATH']
        print(f'Loading ML Model: {model_path} this will take a while....')
        ML_Model = load_model(model_path)
        print(f'Finished loading ML Model: {model_path}')

    return ML_Model
