"""Held-out evaluation for the Siamese BiLSTM duplicate-question model.

The original training pipeline never scored the model against a held-out
test set - only Keras' internal validation loss from `validation_split`
was ever computed, and it wasn't even logged anywhere. This module is the
first real held-out evaluation this project has had.
"""
import json
import os
from typing import Any, Dict, Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model,
    test_data_1: Sequence,
    test_data_2: Sequence,
    test_labels: Sequence,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    probs = model.predict([test_data_1, test_data_2]).ravel()
    preds = (probs >= threshold).astype(int)

    return {
        'n_test_examples': len(test_labels),
        'threshold': threshold,
        'accuracy': accuracy_score(test_labels, preds),
        'precision': precision_score(test_labels, preds),
        'recall': recall_score(test_labels, preds),
        'f1': f1_score(test_labels, preds),
        'roc_auc': roc_auc_score(test_labels, probs),
        'confusion_matrix': confusion_matrix(test_labels, preds).tolist(),
    }


def save_report(report: Dict[str, Any], path: str) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)


def print_report(report: Dict[str, Any]) -> None:
    print('Evaluation report')
    print(f"  n_test_examples: {report['n_test_examples']}")
    print(f"  threshold:       {report['threshold']}")
    print(f"  accuracy:        {report['accuracy']:.4f}")
    print(f"  precision:       {report['precision']:.4f}")
    print(f"  recall:          {report['recall']:.4f}")
    print(f"  f1:              {report['f1']:.4f}")
    print(f"  roc_auc:         {report['roc_auc']:.4f}")
    print(f"  confusion_matrix: {report['confusion_matrix']}")
