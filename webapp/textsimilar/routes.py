from typing import Any, Dict
from flask import current_app, url_for, render_template, redirect, request, g, jsonify
from .forms import SimilarityForm
from .models import SimilarityTest
from . import PastTS, Nav
from .ml_process import predict as  ml_predict

app = current_app

@app.route('/hello')
def hello_world():
    return 'Hello World\n'

@app.route('/is_similar', methods=['GET', 'POST'])
def is_similar():
    form = SimilarityForm()
    if form.is_submitted():
        q1 = request.form.get('q1')
        q2 = request.form.get('q2')

        predict(q1, q2)

        return redirect(url_for("home"))

    return render_template(
        "similarity.jinja2",
        nav = Nav,
        form=form,
        template="form-template"
    )

@app.route('/api/submit', methods=['POST'])
def json_submit():
    questions = request.json
    if not isinstance(questions, list):
        # just treat it as a single query
        _, result = process_json_predict(questions)
        return jsonify(result)

    for q in questions:
        result = []
        errored, result = process_json_predict(q)
        if errored:
            return jsonify(result)

    return jsonify(result)

@app.route('/')
def home():

    return render_template(
        'home.html',
        nav = Nav,
        results = PastTS,
        title="Text Similarity Demo site",
        description="simple form to test the ml backend"
    )

def sm_to_json(sm: SimilarityTest):
    json = {}
    json['q1'] = sm.q1
    json['q2'] = sm.q1
    json['perct'] = sm.rounded
    json['result'] = sm.probability

    return json

def predict(q1: str, q2: str):
    # call the model...
    sm = ml_predict(q1, q2)
    # sm = SimilarityTest(q1=q1, q2=q2)
    PastTS.insert(0, sm)

    return sm_to_json(sm)

def predict_json(json: Dict[str, Any]):
    if 'q1' in json:
        q1 = json['q1']
    else:
        raise ValueError(f'missing json key: q1')
    if 'q2' in json:
        q2 = json['q2']
    else:
        raise ValueError(f'missing json key: q2')

    return predict(q1, q2)

def process_json_predict(json: Dict[str, Any]):
    errored = False
    try:
        result = predict_json(json)
    except Exception as ex:
        result = json
        if hasattr(ex, 'message'):
            result['error'] = f'{ex.message}'
        else:
            result['error'] = f'{ex}'
        
        errored = True
    return errored, result
