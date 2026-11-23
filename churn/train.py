"""Compare three classifiers and export the Gradient Boosting pipeline."""

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from churn.features import RAW_FEATURES, build_pipeline
from churn.metrics import SCORE_BINS, cutoff_metrics, ranking_metrics, score_summary
from churn.threshold import choose_threshold

SEED = 42
TEST_SIZE = 0.2
FOLDS = 5
DEFAULT_MODEL = "gradient_boosting"
THRESHOLD_POLICIES = ("fixed", "f1", "recall", "cost")
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


def train(
    data_path: Path,
    output_dir: Path,
    jobs: int,
    *,
    repeats: int,
    policy: str,
    fixed_threshold: float,
    target_recall: float,
    fp_cost: float,
    fn_cost: float,
) -> dict:
    features, targets = load_data(data_path)
    if not math.isfinite(max(fp_cost, fn_cost) * len(targets)):
        raise ValueError("Choose smaller cost units for this dataset.")
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

    cv_leader = max(
        comparison, key=lambda name: comparison[name]["cross_validation"]["roc_auc"]["mean"]
    )
    development_model = fitted_models[DEFAULT_MODEL]
    threshold_cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + 1)
    oof_scores = cross_val_predict(
        development_model, x_train, y_train, cv=threshold_cv, n_jobs=jobs, method="predict_proba"
    )[:, 1]
    cutoffs = {"fixed": fixed_threshold}
    for candidate in THRESHOLD_POLICIES[1:]:
        cutoffs[candidate] = choose_threshold(
            y_train, oof_scores, candidate, target_recall, fp_cost, fn_cost
        )
    threshold = cutoffs[policy]

    boosting_scores = development_model.predict_proba(x_test)[:, 1]
    threshold_comparison = {}
    for name, cutoff in cutoffs.items():
        predictions = (boosting_scores >= cutoff).astype(int)
        threshold_comparison[name] = {
            "cutoff": cutoff,
            "out_of_fold": cutoff_metrics(y_train, oof_scores >= cutoff, fp_cost, fn_cost),
            "holdout": cutoff_metrics(y_test, predictions, fp_cost, fn_cost),
        }

    for name, model in fitted_models.items():
        # Compare classifiers at a common fixed cutoff. The policy table above
        # evaluates the tuned scenarios only for the deployment model.
        probabilities = (
            boosting_scores if name == DEFAULT_MODEL else model.predict_proba(x_test)[:, 1]
        )
        comparison[name]["holdout"] = {
            "cutoff": fixed_threshold,
            **cutoff_metrics(y_test, probabilities >= fixed_threshold, fp_cost, fn_cost),
            **ranking_metrics(y_test, probabilities),
        }

    chosen_predictions = boosting_scores >= threshold
    chosen_holdout = threshold_comparison[policy]["holdout"]
    baseline = score_summary(boosting_scores, chosen_predictions)

    report = {
        "selected_model": DEFAULT_MODEL,
        "cross_validation_leader": cv_leader,
        "model_version": f"{DEFAULT_MODEL}-{uuid4().hex[:12]}",
        "selection_metric": "mean_cross_validation_roc_auc",
        "decision_threshold": threshold,
        "decision_policy": policy,
        "policy_inputs": {
            "target_recall": target_recall,
            "false_positive_cost": fp_cost,
            "false_negative_cost": fn_cost,
        },
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
        "threshold_oof_seed": SEED + 1,
        "models": comparison,
        "threshold_comparison": threshold_comparison,
        "monitoring_baseline": {
            **baseline,
            "score_bin_edges": SCORE_BINS.tolist(),
            "performance": {
                "precision": chosen_holdout["precision"],
                "recall": chosen_holdout["recall"],
                "roc_auc": comparison[DEFAULT_MODEL]["holdout"]["roc_auc"],
            },
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as staging:
        staged_model = Path(staging) / "churn_model.joblib"
        staged_report = Path(staging) / "metrics.json"
        joblib.dump({"pipeline": development_model, "report": report}, staged_model)
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
    parser.add_argument("--threshold-policy", choices=THRESHOLD_POLICIES, default="fixed")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fixed cutoff for comparison")
    parser.add_argument("--target-recall", type=float, default=0.70)
    parser.add_argument("--false-positive-cost", type=float, default=1.0)
    parser.add_argument("--false-negative-cost", type=float, default=5.0)
    args = parser.parse_args()
    if args.jobs == 0:
        parser.error("--jobs must be nonzero.")
    if args.repeats < 2:
        parser.error("--repeats must be at least two.")
    if not 0 <= args.threshold <= 1 or not 0 < args.target_recall <= 1:
        parser.error("--threshold must be in [0, 1] and --target-recall in (0, 1].")
    costs = (args.false_positive_cost, args.false_negative_cost)
    if any(not math.isfinite(cost) or cost < 0 for cost in costs) or not any(costs):
        parser.error("Costs must be finite, nonnegative, and at least one must be positive.")
    try:
        report = train(
            args.data, args.output_dir, args.jobs,
            repeats=args.repeats,
            policy=args.threshold_policy,
            fixed_threshold=args.threshold,
            target_recall=args.target_recall,
            fp_cost=args.false_positive_cost,
            fn_cost=args.false_negative_cost,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Training failed: {error}\n")
    print(f"Cross-validation ROC-AUC leader: {report['cross_validation_leader']}.")
    print(f"Packaged {report['selected_model']} at cutoff {report['decision_threshold']:.5f}.")
    print(f"Saved the model and metrics to {args.output_dir}/.")


if __name__ == "__main__":
    main()
