# Customer Churn Prediction Pipeline

Compare churn classifiers, set an outreach cutoff from training predictions, and serve the refitted Gradient Boosting model with optional SHAP explanations.

## Run the packaged model

Use Python 3.11–3.13. Run these commands from the extracted project directory in a macOS or Linux shell.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m flask --app churn.api run --port 8000
```

The package includes the fitted model in `artifacts/churn_model.joblib` and its evaluation in [metrics.json](artifacts/metrics.json). The artifact records the library versions and the preprocessing source checksum.

Open [the dashboard](http://127.0.0.1:8000/) to edit the example profile, run a prediction, and see which features contributed to the score. The Monitoring view displays recent prediction totals and confirmed outcomes. On Render, open the service's base URL to use the same dashboard.

## Reproduce training and choose a cutoff

```bash
mkdir -p data
curl -fL 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv' -o data/telco_churn.csv
python -m churn.train
```

The report records the source CSV's SHA-256 hash. The [IBM Telco Customer Churn sample](https://github.com/IBM/telco-customer-churn-on-icp4d#flow) has 7,043 customers. Customer ID and gender are excluded from modeling. For zero-tenure accounts, a blank `TotalCharges` becomes zero and the current monthly charge is used as the average charge. For customers with tenure, missing charges are imputed using medians learned within the training fold.

Training uses a stratified 80/20 split. The three classifiers are tuned on 5,634 customers using three repeats of five-fold stratified cross-validation. The report gives each metric's mean and standard deviation across those folds. After tuning, five-fold out-of-fold predictions on the training partition supply the outreach cutoff. All holdout results use 1,409 customers and models fitted on the training partition. The deployment model is then refitted on all 7,043 customers with the chosen hyperparameters and cutoff.

| Policy | Example command |
| --- | --- |
| Fixed cutoff (packaged default) | `python -m churn.train --threshold 0.50` |
| Maximize F1 | `python -m churn.train --threshold-policy f1` |
| Meet an out-of-fold recall target | `python -m churn.train --threshold-policy recall --target-recall 0.70` |
| Minimize cost from false alerts and missed churners | `python -m churn.train --threshold-policy cost --false-positive-cost 1 --false-negative-cost 5` |

The 1:5 cost ratio is an example in relative units. For each policy, the report shows customers flagged, churners found, false alerts, and cost on both training out-of-fold and holdout predictions. Contract groups in the holdout have their own counts and recall. The model comparison includes ROC-AUC, average precision (stepwise area under the precision-recall curve), Brier score, calibration bins, F1, confusion matrices, and lift among the top-scored 10%.

## Results from the packaged run

| Model | CV ROC-AUC mean ± fold SD | Holdout ROC-AUC | Holdout average precision |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.84647 ± 0.01159 | 0.84127 | 0.62784 |
| Random Forest | 0.84738 ± 0.01146 | 0.84247 | 0.65681 |
| Gradient Boosting | 0.84953 ± 0.01233 | 0.84565 | 0.66518 |

| Outreach policy | Cutoff | Customers flagged | Churners found | False alerts | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed | 0.50000 | 289 | 189 | 100 | 50.5% |
| F1 | 0.32910 | 493 | 277 | 216 | 74.1% |
| 70% training recall target | 0.35468 | 467 | 268 | 199 | 71.7% |
| Example 1:5 cost ratio | 0.13858 | 775 | 336 | 439 | 89.8% |

The 70% recall policy found 79 more churners for 178 more contacts than the fixed cutoff. It identified 2 of the 45 holdout churners on one- and two-year contracts.

## REST API

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/health` | Check the loaded model and version. |
| POST | `/predict` | Score one customer and log the prediction. |
| POST | `/predict?explain=true` | Include SHAP contributions in the scoring response. |
| PUT | `/predictions/<prediction_id>/outcome` | Record a confirmed outcome using `actual_churn` of `1` or `0`. |
| GET | `/monitoring?days=30` | Compare recent scores and available performance with the development holdout reference. |

Send the [example customer](examples/customer.json) from a second terminal.

```bash
curl -fsS 'http://127.0.0.1:8000/predict?explain=true' \
  -H 'Content-Type: application/json' \
  --data-binary @examples/customer.json
```

The request uses the 18 fields in the example and the original IBM column names. Legacy payloads may include `gender`, which is discarded. `SeniorCitizen` is an integer 0 or 1 and `tenure` is an integer number of months. The charge fields are numbers, with `null` accepted for `TotalCharges`. Tenure and charge values range from 0 through 1,000,000,000. See the [accepted categorical values](churn/features.py).

TreeSHAP adds positive and negative contributions to a baseline in log-odds. Positive values raise the model's churn score. The API groups one-hot contributions under their originating fields and sorts them by magnitude. Applying the logistic function to the sum recovers the returned churn probability. The report also includes mean absolute SHAP contributions over the holdout for the development model.

## Record the outcome and inspect monitoring

```bash
prediction_id='paste-the-returned-prediction-id'
curl -fsS -X PUT "http://127.0.0.1:8000/predictions/${prediction_id}/outcome" \
  -H 'Content-Type: application/json' \
  -d '{"actual_churn": 1}'
curl -fsS 'http://127.0.0.1:8000/monitoring?days=30'
```

SQLite stores prediction IDs, timestamps, model version, probability, decision, and actual outcome when available. Score drift uses ten probability bins and requires 30 predictions. Precision, recall, and ROC-AUC require 30 confirmed outcomes in the window and examples of both classes. The evaluation reference comes from the model fitted on the 80% development split. Refitting on all rows can shift the score distribution.

Set `CHURN_MODEL_PATH` or `CHURN_DB_PATH` to override the packaged model path or default database at `instance/predictions.sqlite3`. Restart Flask after retraining to load the updated model and cutoff.
