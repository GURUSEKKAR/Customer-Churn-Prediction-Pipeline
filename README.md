# Customer Churn Prediction Pipeline

Train a churn classifier with pandas and scikit-learn, serve churn probabilities through Flask, and track changes in model scores and observed performance.

## Quick start

Use Python 3.11–3.13. The commands below run from the repository root in a macOS or Linux shell.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

mkdir -p data
curl -fL 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv' -o data/telco_churn.csv
python -m churn.train

python -m flask --app churn.api run --port 8000
```

Training writes the fitted pipeline to `artifacts/churn_model.joblib` and the evaluation to `artifacts/metrics.json`. Use `python -m churn.train --help` to select a CSV, output directory, or number of parallel training workers.

## Model training and results

The [IBM Telco Customer Churn sample](https://github.com/IBM/telco-customer-churn-on-icp4d#flow) contains 7,043 customers, of whom 26.5% churned. The training script removes duplicate rows and excludes the customer ID from model inputs. It converts blank charges to missing values, derives average charges to date and an Internet add-on count, and fits imputation, scaling, and one-hot encoding inside each cross-validation fold.

The script uses a stratified 80/20 split and tunes every classifier on the training partition with five-fold stratified cross-validation. It selects the model with the highest mean cross-validation ROC-AUC, then reports precision, recall, and ROC-AUC on the 1,409-customer holdout. Precision and recall use a churn threshold of 0.50.

| Model | CV ROC-AUC | Holdout ROC-AUC | Holdout precision | Holdout recall |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8458 | 0.8411 | 0.6531 | 0.5588 |
| Random Forest | 0.8470 | 0.8417 | 0.6794 | 0.4759 |
| **Gradient Boosting** | **0.8499** | **0.8452** | **0.6763** | **0.5027** |

[Full metrics and best hyperparameters](artifacts/metrics.json)

## REST API

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/health` | Check that the model is loaded and see its version. |
| POST | `/predict` | Score one customer and return the churn probability, binary prediction, and prediction ID. |
| PUT | `/predictions/<prediction_id>/outcome` | Record a confirmed churn outcome. Use `1` for churn and `0` for retained. |
| GET | `/monitoring?days=30` | Compare recent scores and available outcome-based metrics with the holdout baseline. |

With the Flask server running, open a second terminal in the repository root and send the [example customer](examples/customer.json).

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @examples/customer.json
```

Use the returned `prediction_id` when the actual outcome is known.

```bash
prediction_id='paste-the-returned-prediction-id'
curl -fsS -X PUT "http://127.0.0.1:8000/predictions/${prediction_id}/outcome" \
  -H 'Content-Type: application/json' \
  -d '{"actual_churn": 1}'
curl -fsS 'http://127.0.0.1:8000/monitoring?days=30'
```

Requests use the 19 fields shown in the example with their original IBM column names. `SeniorCitizen` is an integer 0 or 1. `tenure` is an integer number of months. The two charge fields are numbers, and `TotalCharges` also accepts `null`. Tenure and charge values range from 0 through 1,000,000,000. Categorical values use the [IBM vocabulary](churn/features.py). Resubmitting the same confirmed outcome succeeds. A conflicting outcome returns HTTP 409.

## Monitoring

SQLite stores each prediction ID, timestamp, model version, score, class, and any reported outcome. The monitoring window uses prediction timestamps and accepts 1–365 days. The report includes the mean score and predicted churn rate for the active model version.

- **Score drift** measures how different the recent probability distribution is from the holdout distribution across ten score bins. Total variation ranges from 0 for identical distributions to 1 for disjoint distributions. It becomes available after 30 predictions.
- **Observed performance** reports precision, recall, ROC-AUC, and their differences from the holdout once 30 predictions in the window have confirmed outcomes and both classes are present. The report includes the labeled count so outcome coverage can be checked.

Set `CHURN_MODEL_PATH` or `CHURN_DB_PATH` to override the default model file or `instance/predictions.sqlite3` log. Restart Flask after retraining to load the new model version.
