"""Flask REST API for churn predictions, actual outcomes, and monitoring."""

import json
import os
from pathlib import Path
from uuid import UUID, uuid4

import joblib
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from churn.explain import ChurnExplainer
from churn.features import (
    CATEGORICAL_VALUES,
    MAX_NUMERIC_VALUE,
    RAW_FEATURES,
    feature_source_hash,
    validate_customer,
)
from churn.monitor import init_log, log_prediction, monitoring_report, record_outcome

EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "customer.json"
FORM_GROUPS = (
    ("Account", (
        ("tenure", "Tenure in months"),
        ("Contract", "Contract"),
        ("SeniorCitizen", "Senior citizen"),
        ("Partner", "Partner"),
        ("Dependents", "Dependents"),
    )),
    ("Services", (
        ("PhoneService", "Phone service"),
        ("MultipleLines", "Multiple lines"),
        ("InternetService", "Internet service"),
        ("OnlineSecurity", "Online security"),
        ("OnlineBackup", "Online backup"),
        ("DeviceProtection", "Device protection"),
        ("TechSupport", "Tech support"),
        ("StreamingTV", "Streaming TV"),
        ("StreamingMovies", "Streaming movies"),
    )),
    ("Billing", (
        ("PaymentMethod", "Payment method"),
        ("PaperlessBilling", "Paperless billing"),
        ("MonthlyCharges", "Monthly charges"),
        ("TotalCharges", "Total charges"),
    )),
)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    model_path = Path(os.environ.get("CHURN_MODEL_PATH", "artifacts/churn_model.joblib"))
    log_path = Path(os.environ.get("CHURN_DB_PATH", "instance/predictions.sqlite3"))
    if not model_path.is_file():
        raise RuntimeError(f"Model not found at {model_path}. Run python -m churn.train first.")

    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    report = artifact["report"]
    if report.get("artifact_metadata", {}).get("features_sha256") != feature_source_hash():
        raise RuntimeError("Model preprocessing differs from this code. Run python -m churn.train.")
    model_version = report["model_version"]
    threshold = report["decision_threshold"]
    explainer = ChurnExplainer(pipeline)
    init_log(log_path)

    @app.errorhandler(HTTPException)
    def json_error(error):
        response = error.get_response()
        response.set_data(app.json.dumps({"error": error.description}))
        response.content_type = "application/json"
        return response

    @app.get("/")
    def dashboard():
        return render_template(
            "dashboard.html",
            groups=FORM_GROUPS,
            choices={**CATEGORICAL_VALUES, "SeniorCitizen": {0: "No", 1: "Yes"}},
            example=json.loads(EXAMPLE_PATH.read_text(encoding="utf-8")),
            max_numeric_value=MAX_NUMERIC_VALUE,
            model_name=report["selected_model"].replace("_", " ").title(),
            cutoff=f"{threshold:.1%}",
        )

    @app.get("/health")
    def health():
        return jsonify(
            status="ok",
            model=report["selected_model"],
            model_version=model_version,
        )

    @app.post("/predict")
    def predict():
        explain = request.args.get("explain", "false")
        if explain not in {"true", "false"}:
            abort(400, description="explain must be true or false.")
        try:
            customer = validate_customer(request.get_json())
        except ValueError as error:
            abort(400, description=str(error))

        record = pd.DataFrame([customer], columns=RAW_FEATURES)
        probability = float(pipeline.predict_proba(record)[0, 1])
        prediction = int(probability >= threshold)
        prediction_id = str(uuid4())
        result = {
            "prediction_id": prediction_id,
            "model_version": model_version,
            "churn_probability": probability,
            "predicted_churn": prediction,
            "decision_threshold": threshold,
        }
        if explain == "true":
            result["explanation"] = explainer.explain_customer(record)
        log_prediction(log_path, prediction_id, model_version, probability, prediction)
        return jsonify(result)

    @app.put("/predictions/<uuid:prediction_id>/outcome")
    def save_outcome(prediction_id: UUID):
        payload = request.get_json()
        if (
            not isinstance(payload, dict)
            or payload.keys() != {"actual_churn"}
            or type(payload["actual_churn"]) is not int
            or payload["actual_churn"] not in (0, 1)
        ):
            abort(400, description="Send an actual_churn field with the integer 0 or 1.")

        outcome = record_outcome(log_path, str(prediction_id), payload["actual_churn"])
        if outcome == "missing":
            abort(404, description="Unknown prediction ID.")
        if outcome == "conflict":
            abort(409, description="A different outcome was already recorded.")
        return jsonify(prediction_id=str(prediction_id), actual_churn=payload["actual_churn"])

    @app.get("/monitoring")
    def monitoring():
        days = request.args.get("days", default="30")
        try:
            window = int(days)
        except ValueError:
            abort(400, description="days must be an integer from 1 to 365.")
        if not 1 <= window <= 365:
            abort(400, description="days must be an integer from 1 to 365.")
        return jsonify(monitoring_report(log_path, report, window))

    return app
