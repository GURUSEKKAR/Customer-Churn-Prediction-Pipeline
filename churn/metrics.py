"""Metrics shared by holdout evaluation and live monitoring."""

import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score

SCORE_BINS = np.linspace(0, 1, 11)


def score_summary(probabilities, predictions) -> dict:
    probabilities = np.asarray(probabilities, dtype=float)
    count = probabilities.size
    counts, _ = np.histogram(probabilities, bins=SCORE_BINS)
    return {
        "count": count,
        "mean_probability": float(probabilities.mean()) if count else None,
        "predicted_churn_rate": float(np.mean(predictions)) if count else None,
        "score_histogram": (counts / count).tolist() if count else None,
    }


def classification_metrics(actual, probabilities, predictions) -> dict:
    return {
        "precision": float(precision_score(actual, predictions, zero_division=0)),
        "recall": float(recall_score(actual, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probabilities)),
    }
