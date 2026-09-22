import webapp.textsimilar.routes as routes_module
from webapp.textsimilar.models import SimilarityTest


def fake_ml_predict(q1: str, q2: str) -> SimilarityTest:
    """Stands in for the real model call so these API-layer tests don't
    need to load TF/Keras or have a model file on disk. Real model
    correctness is the eval report's job (see src/evaluate.py), not
    these tests'."""
    return SimilarityTest(q1=q1, q2=q2, probability=0.42, rounded=42.0)


def test_api_submit_single_object_echoes_q1_and_q2(client, monkeypatch):
    monkeypatch.setattr(routes_module, 'ml_predict', fake_ml_predict)

    resp = client.post('/api/submit', json={'q1': 'hello world', 'q2': 'goodbye world'})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body['q1'] == 'hello world'
    # Regression test: sm_to_json() used to set json['q2'] = sm.q1 (copy-paste
    # bug), so the API always echoed q1 back twice.
    assert body['q2'] == 'goodbye world'


def test_api_submit_list_of_objects(client, monkeypatch):
    monkeypatch.setattr(routes_module, 'ml_predict', fake_ml_predict)

    resp = client.post('/api/submit', json=[{'q1': 'a', 'q2': 'b'}, {'q1': 'c', 'q2': 'd'}])

    assert resp.status_code == 200
    body = resp.get_json()
    assert body['q1'] == 'c'
    assert body['q2'] == 'd'


def test_api_submit_missing_q2_returns_error(client, monkeypatch):
    monkeypatch.setattr(routes_module, 'ml_predict', fake_ml_predict)

    resp = client.post('/api/submit', json={'q1': 'hello world'})

    assert resp.status_code == 200
    body = resp.get_json()
    assert 'error' in body
