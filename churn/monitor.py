"""SQLite prediction log and monitoring summaries for the active model."""

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from churn.metrics import classification_metrics, score_summary

MIN_MONITORING_ROWS = 30


def init_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                model_version TEXT NOT NULL,
                predicted_at TEXT NOT NULL,
                probability REAL NOT NULL CHECK (probability BETWEEN 0 AND 1),
                predicted_churn INTEGER NOT NULL CHECK (predicted_churn IN (0, 1)),
                actual_churn INTEGER CHECK (actual_churn IN (0, 1))
            );
            CREATE INDEX IF NOT EXISTS predictions_by_version_and_time
                ON predictions (model_version, predicted_at);
            """
        )


def log_prediction(
    path: Path, prediction_id: str, version: str, probability: float, prediction: int
) -> None:
    predicted_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            """
            INSERT INTO predictions (id, model_version, predicted_at, probability, predicted_churn)
            VALUES (?, ?, ?, ?, ?)
            """,
            (prediction_id, version, predicted_at, probability, prediction),
        )


def record_outcome(path: Path, prediction_id: str, actual: int) -> str:
    with closing(sqlite3.connect(path)) as connection, connection:
        result = connection.execute(
            """
            UPDATE predictions
            SET actual_churn = ?
            WHERE id = ? AND (actual_churn IS NULL OR actual_churn = ?)
            """,
            (actual, prediction_id, actual),
        )
        if result.rowcount:
            return "saved"
        exists = connection.execute(
            "SELECT 1 FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
        return "conflict" if exists else "missing"


def monitoring_report(path: Path, report: dict, days: int) -> dict:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    version = report["model_version"]
    baseline = report["monitoring_baseline"]
    with closing(sqlite3.connect(path)) as connection:
        rows = connection.execute(
            """
            SELECT probability, predicted_churn, actual_churn
            FROM predictions
            WHERE model_version = ? AND predicted_at >= ?
            """,
            (version, since.isoformat(timespec="microseconds")),
        ).fetchall()

    current = score_summary([row[0] for row in rows], [row[1] for row in rows])
    score_drift = {"status": "insufficient_predictions", "total_variation": None}
    if current["count"] >= MIN_MONITORING_ROWS:
        histogram_pairs = zip(current["score_histogram"], baseline["score_histogram"])
        score_drift = {
            "status": "available",
            "total_variation": 0.5 * sum(abs(recent - past) for recent, past in histogram_pairs),
        }

    labeled = [row for row in rows if row[2] is not None]
    performance = {
        "status": "insufficient_outcomes",
        "labeled_count": len(labeled),
        "metrics": None,
        "delta_from_holdout": None,
    }
    if len(labeled) >= MIN_MONITORING_ROWS:
        probabilities, predictions, actual = zip(*labeled)
        if len(set(actual)) == 2:
            metrics = classification_metrics(actual, probabilities, predictions)
            performance.update(
                status="available",
                metrics=metrics,
                delta_from_holdout={
                    metric: value - baseline["performance"][metric]
                    for metric, value in metrics.items()
                },
            )
        else:
            performance["status"] = "needs_both_outcome_classes"

    return {
        "model_version": version,
        "window_days": days,
        "window_start_utc": since.isoformat(timespec="microseconds"),
        "minimum_sample_size": MIN_MONITORING_ROWS,
        "holdout_baseline": baseline,
        "recent_predictions": current,
        "score_drift": score_drift,
        "observed_performance": performance,
    }
