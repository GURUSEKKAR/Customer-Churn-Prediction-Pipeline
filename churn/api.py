"""Flask REST API for churn predictions, actual outcomes, and monitoring."""

import os
from pathlib import Path
from uuid import UUID, uuid4

import joblib
import pandas as pd
from flask import Flask, abort, jsonify, request
from werkzeug.exceptions import HTTPException

from churn.features import RAW_FEATURES, validate_customer
from churn.monitor import init_log, log_prediction, monitoring_report, record_outcome


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    model_path = Path(os.environ.get("CHURN_MODEL_PATH", "artifacts/churn_model.joblib"))
    log_path = Path(os.environ.get("CHURN_DB_PATH", "instance/predictions.sqlite3"))
    if not model_path.is_file():
        raise RuntimeError(f"Model not found at {model_path}. Run python -m churn.train first.")

    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    report = artifact["report"]
    model_version = report["model_version"]
    threshold = report["decision_threshold"]
    init_log(log_path)

    @app.errorhandler(HTTPException)
    def json_error(error):
        response = error.get_response()
        response.set_data(app.json.dumps({"error": error.description}))
        response.content_type = "application/json"
        return response

    @app.get("/health")
    def health():
        return jsonify(
            status="ok",
            model=report["selected_model"],
            model_version=model_version,
        )

    @app.post("/predict")
    def predict():
        try:
            customer = validate_customer(request.get_json())
        except ValueError as error:
            abort(400, description=str(error))

        record = pd.DataFrame([customer], columns=RAW_FEATURES)
        probability = float(pipeline.predict_proba(record)[0, 1])
        prediction = int(probability >= threshold)
        prediction_id = str(uuid4())
        log_prediction(log_path, prediction_id, model_version, probability, prediction)
        return jsonify(
            prediction_id=prediction_id,
            model_version=model_version,
            churn_probability=probability,
            predicted_churn=prediction,
            decision_threshold=threshold,
        )

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
