import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from evaluate import evaluate_model


class FakeModel:
    """Stub standing in for the real Keras model - returns fixed
    probabilities regardless of input, so this test doesn't need TF/Keras
    or a trained model file on disk."""

    def __init__(self, probs):
        self._probs = np.array(probs).reshape(-1, 1)

    def predict(self, inputs):
        return self._probs


LABELS = [0, 1, 1, 0, 1, 1, 0, 0]
PROBS = [0.1, 0.9, 0.4, 0.2, 0.8, 0.6, 0.3, 0.7]
# preds @ threshold 0.5: [0, 1, 0, 0, 1, 1, 0, 1]
# TP=3 (idx 2,5,6), TN=3 (idx 1,4,7), FP=1 (idx 8), FN=1 (idx 3)


def test_evaluate_model_metrics():
    report = evaluate_model(FakeModel(PROBS), test_data_1=None, test_data_2=None, test_labels=LABELS)

    assert report['n_test_examples'] == 8
    assert report['threshold'] == 0.5
    assert report['accuracy'] == pytest.approx(0.75)
    assert report['precision'] == pytest.approx(0.75)
    assert report['recall'] == pytest.approx(0.75)
    assert report['f1'] == pytest.approx(0.75)
    assert report['roc_auc'] == pytest.approx(roc_auc_score(LABELS, PROBS))
    assert report['confusion_matrix'] == [[3, 1], [1, 3]]


def test_evaluate_model_respects_custom_threshold():
    report = evaluate_model(
        FakeModel(PROBS), test_data_1=None, test_data_2=None, test_labels=LABELS, threshold=0.75
    )

    assert report['threshold'] == 0.75
    # only idx 2 (0.9) and idx 5 (0.8) clear a 0.75 threshold
    assert report['confusion_matrix'] == [[4, 0], [2, 2]]
