"""Metrics shared by holdout evaluation and live monitoring."""

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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


def cutoff_metrics(actual, predictions, fp_cost: float, fn_cost: float) -> dict:
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(
        actual, predictions, labels=[0, 1]
    ).ravel())
    total_cost = fp * fp_cost + fn * fn_cost
    return {
        "customers": tn + fp + fn + tp,
        "churners": tp + fn,
        "flagged": tp + fp,
        "precision": float(precision_score(actual, predictions, zero_division=0)),
        "recall": float(recall_score(actual, predictions, zero_division=0)) if tp + fn else None,
        "f1": float(f1_score(actual, predictions, zero_division=0)),
        "confusion_matrix": {
            "true_negatives": tn, "false_positives": fp,
            "false_negatives": fn, "true_positives": tp,
        },
        "business_cost": {"total": total_cost, "per_customer": total_cost / (tn + fp + fn + tp)},
    }


def ranking_metrics(actual, probabilities) -> dict:
    labels = np.asarray(actual, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    top_count = math.ceil(0.1 * len(labels))
    top = np.argsort(-scores, kind="stable")[:top_count]
    # For equal scores exactly at a bucket boundary, match numpy.histogram.
    bucket = np.minimum(np.searchsorted(SCORE_BINS, scores, side="right") - 1, 9)
    calibration = []
    for index, (lower, upper) in enumerate(zip(SCORE_BINS, SCORE_BINS[1:])):
        rows = bucket == index
        calibration.append({
            "lower_probability": float(lower),
            "upper_probability": float(upper),
            "count": int(rows.sum()),
            "mean_probability": float(scores[rows].mean()) if rows.any() else None,
            "observed_churn_rate": float(labels[rows].mean()) if rows.any() else None,
        })
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "top_decile_lift": {
            "customers": top_count,
            "churn_rate": float(labels[top].mean()),
            "lift": float(labels[top].mean() / labels.mean()),
        },
        "calibration_bins": calibration,
    }
