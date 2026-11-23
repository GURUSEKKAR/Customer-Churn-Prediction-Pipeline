import json
import math
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import joblib
import pandas as pd

from churn.api import create_app
from churn.features import engineer_features, validate_customer
from churn.metrics import cutoff_metrics, ranking_metrics
from churn.threshold import choose_threshold

CUSTOMER = json.loads(Path("examples/customer.json").read_text())


class ThresholdTests(unittest.TestCase):
    def test_ties_and_business_objectives(self):
        actual = [0, 1, 0, 1, 0]
        scores = [0.9, 0.8, 0.8, 0.4, 0.3]
        self.assertEqual(choose_threshold(actual, scores, "f1", 0.7, 1, 5), 0.4)
        self.assertEqual(choose_threshold(actual, scores, "recall", 0.5, 1, 5), 0.8)
        self.assertEqual(choose_threshold(actual, scores, "cost", 0.7, 1, 5), 0.4)
        self.assertEqual(choose_threshold(actual, scores, "cost", 0.7, 10, 1), 1.0)

    def test_cutoff_cost_calibration_and_lift(self):
        metrics = cutoff_metrics([0, 0, 1, 1], [0, 1, 0, 1], 1, 5)
        self.assertEqual(metrics["business_cost"], {"total": 6, "per_customer": 1.5})
        self.assertEqual(metrics["confusion_matrix"], {
            "true_negatives": 1, "false_positives": 1, "false_negatives": 1, "true_positives": 1
        })
        ranking = ranking_metrics([0, 1, 0, 1], [0.0, 0.3, 0.6, 1.0])
        self.assertEqual(sum(item["count"] for item in ranking["calibration_bins"]), 4)
        self.assertEqual(ranking["top_decile_lift"]["lift"], 2.0)


class FeatureTests(unittest.TestCase):
    def test_billing_history_and_legacy_gender(self):
        new = {**CUSTOMER, "tenure": 0, "TotalCharges": " ", "gender": "Female"}
        missing = {**CUSTOMER, "tenure": 4, "TotalCharges": None}
        features = engineer_features(pd.DataFrame([new, missing, CUSTOMER]))
        self.assertEqual(features.loc[0, "TotalCharges"], 0)
        self.assertEqual(features.loc[0, "AverageChargesToDate"], CUSTOMER["MonthlyCharges"])
        self.assertTrue(pd.isna(features.loc[1, "TotalCharges"]))
        self.assertTrue(pd.isna(features.loc[1, "AverageChargesToDate"]))
        self.assertAlmostEqual(features.loc[2, "AverageChargesToDate"], 79.85)
        self.assertNotIn("gender", features)
        self.assertEqual(validate_customer({**CUSTOMER, "gender": "Male"}), CUSTOMER)


class PackagedPipelineTests(unittest.TestCase):
    def test_report_tracks_evaluation_and_full_data_refit(self):
        bundle = joblib.load("artifacts/churn_model.joblib")
        report = bundle["report"]
        self.assertEqual(report, json.loads(Path("artifacts/metrics.json").read_text()))
        data = report["dataset"]
        self.assertEqual(data["train_rows"] + data["holdout_rows"], data["rows"])
        self.assertEqual(data["final_fit_rows"], data["rows"])
        classifier = bundle["pipeline"].named_steps["classifier"]
        self.assertAlmostEqual(classifier.init_.class_prior_[1], data["churn_rate"])
        self.assertGreaterEqual(report["cross_validation_repeats"], 2)
        chosen = report["threshold_comparison"][report["decision_policy"]]
        self.assertEqual(chosen["cutoff"], report["decision_threshold"])
        self.assertEqual(
            sum(group["customers"] for group in chosen["holdout_by_contract"].values()),
            data["holdout_rows"],
        )

    def test_prediction_shap_and_monitoring(self):
        with TemporaryDirectory() as folder:
            with patch.dict(os.environ, {"CHURN_DB_PATH": str(Path(folder) / "test.sqlite3")}):
                app = create_app()
            app.testing = True
            client = app.test_client()
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.post("/predict?explain=yes", json=CUSTOMER).status_code, 400)
            plain = client.post("/predict", json=CUSTOMER).get_json()
            explained = client.post(
                "/predict?explain=true", json={**CUSTOMER, "gender": "Female"}
            ).get_json()
            self.assertNotIn("explanation", plain)
            self.assertEqual(plain["churn_probability"], explained["churn_probability"])
            self.assertEqual(
                explained["predicted_churn"],
                int(explained["churn_probability"] >= explained["decision_threshold"]),
            )
            explanation = explained["explanation"]
            self.assertEqual(explanation["units"], "log_odds")
            self.assertNotIn("gender", {row["feature"] for row in explanation["contributions"]})
            raw_score = explanation["baseline"] + sum(
                row["value"] for row in explanation["contributions"]
            )
            self.assertAlmostEqual(raw_score, explanation["model_output"])
            self.assertAlmostEqual(1 / (1 + math.exp(-raw_score)), explained["churn_probability"])
            outcome = client.put(
                f'/predictions/{explained["prediction_id"]}/outcome', json={"actual_churn": 1}
            )
            self.assertEqual(outcome.status_code, 200)
            monitoring = client.get("/monitoring").get_json()
            self.assertEqual(monitoring["recent_predictions"]["count"], 2)
            self.assertEqual(monitoring["observed_performance"]["labeled_count"], 1)
