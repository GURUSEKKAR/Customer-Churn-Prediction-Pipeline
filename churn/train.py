"""Tune three classifiers and export the winning pipeline and holdout report."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split

from churn.features import RAW_FEATURES, build_pipeline
from churn.metrics import SCORE_BINS, classification_metrics, score_summary

SEED = 42
TEST_SIZE = 0.2
FOLDS = 5
THRESHOLD = 0.5
SCORING = {
    "roc_auc": "roc_auc",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
}


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    records = pd.read_csv(path)
    missing = set((*RAW_FEATURES, "Churn")) - set(records.columns)
    if missing:
        raise ValueError(f"CSV is missing fields: {', '.join(sorted(missing))}.")
    records = records.drop_duplicates()
    if "customerID" in records and records["customerID"].dropna().duplicated().any():
        raise ValueError("CSV must contain at most one row per customerID.")
    targets = records["Churn"].astype("string").str.strip().str.lower().map(
        {"no": 0, "yes": 1}
    )
    if targets.isna().any():
        raise ValueError("Every Churn label in the CSV must be Yes or No.")
    if targets.nunique() != 2 or targets.value_counts().min() < FOLDS + 2:
        raise ValueError(f"The CSV needs at least {FOLDS + 2} examples of each churn class.")
    return records.loc[:, list(RAW_FEATURES)], targets.astype(int)


def model_candidates() -> dict:
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000),
            {"classifier__C": [0.01, 0.1, 1, 10]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=SEED),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [6, None],
                "classifier__min_samples_leaf": [2, 8],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=SEED),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [2, 3],
            },
        ),
    }


def train(data_path: Path, output_dir: Path, jobs: int, repeats: int) -> dict:
    features, targets = load_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=TEST_SIZE, stratify=targets, random_state=SEED
    )
    validation = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=repeats, random_state=SEED)
    comparison = {}
    fitted_models = {}
    for name, (classifier, grid) in model_candidates().items():
        search = GridSearchCV(
            build_pipeline(classifier),
            grid,
            cv=validation,
            scoring=SCORING,
            refit="roc_auc",
            n_jobs=jobs,
            error_score="raise",
        )
        print(f"Tuning {name}...", flush=True)
        search.fit(x_train, y_train)
        fitted_models[name] = search.best_estimator_
        comparison[name] = {
            "best_parameters": {
                key.removeprefix("classifier__"): value
                for key, value in search.best_params_.items()
            },
            "cross_validation": {
                metric: {
                    "mean": float(search.cv_results_[f"mean_test_{metric}"][search.best_index_]),
                    "std": float(search.cv_results_[f"std_test_{metric}"][search.best_index_]),
                }
                for metric in SCORING
            },
        }

    selected = max(comparison, key=lambda name: comparison[name]["cross_validation"]["roc_auc"]["mean"])
    for name, model in fitted_models.items():
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= THRESHOLD).astype(int)
        comparison[name]["holdout"] = classification_metrics(y_test, probabilities, predictions)
        if name == selected:
            baseline = score_summary(probabilities, predictions)

    report = {
        "selected_model": selected,
        "model_version": f"{selected}-{uuid4().hex[:12]}",
        "selection_metric": "mean_cross_validation_roc_auc",
        "decision_threshold": THRESHOLD,
        "dataset": {
            "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "rows": len(targets),
            "churn_rate": float(targets.mean()),
            "train_rows": len(y_train),
            "holdout_rows": len(y_test),
        },
        "random_seed": SEED,
        "cross_validation_folds": FOLDS,
        "cross_validation_repeats": repeats,
        "models": comparison,
        "monitoring_baseline": {
            **baseline,
            "score_bin_edges": SCORE_BINS.tolist(),
            "performance": comparison[selected]["holdout"],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as staging:
        staged_model = Path(staging) / "churn_model.joblib"
        staged_report = Path(staging) / "metrics.json"
        joblib.dump({"pipeline": fitted_models[selected], "report": report}, staged_model)
        staged_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staged_model.replace(output_dir / staged_model.name)
        staged_report.replace(output_dir / staged_report.name)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/telco_churn.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--jobs", type=int, default=1, help="Parallel grid-search workers")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats of five-fold model CV")
    args = parser.parse_args()
    if args.jobs == 0:
        parser.error("--jobs must be nonzero.")
    if args.repeats < 2:
        parser.error("--repeats must be at least two.")
    try:
        report = train(args.data, args.output_dir, args.jobs, args.repeats)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Training failed: {error}\n")
    print(f"Selected {report['selected_model']} by mean cross-validation ROC-AUC.")
    print(f"Saved the model and metrics to {args.output_dir}/.")


if __name__ == "__main__":
    main()
