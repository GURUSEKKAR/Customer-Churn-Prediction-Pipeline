"""Shared input schema and feature engineering for training and inference."""

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

YES_NO = ("Yes", "No")
INTERNET_ADD_ONS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)
CATEGORICAL_VALUES = {
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": (*YES_NO, "No phone service"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    **{field: (*YES_NO, "No internet service") for field in INTERNET_ADD_ONS},
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": YES_NO,
    "PaymentMethod": (
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ),
}
NUMERIC_INPUTS = ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")
RAW_FEATURES = (*NUMERIC_INPUTS, *CATEGORICAL_VALUES)
NUMERIC_FEATURES = (*NUMERIC_INPUTS, "AverageChargesToDate", "InternetAddOnCount")
MAX_NUMERIC_VALUE = 1_000_000_000


def feature_source_hash() -> str:
    return sha256(Path(__file__).read_bytes()).hexdigest()


def validate_customer(payload: object) -> dict:
    """Validate a JSON customer record and return its normalized fields."""
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object containing one customer.")

    missing = sorted(set(RAW_FEATURES) - payload.keys())
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}.")
    extra = sorted(payload.keys() - set(RAW_FEATURES) - {"gender"})
    if extra:
        raise ValueError(f"Unexpected fields: {', '.join(extra)}.")

    customer = {}
    for field in NUMERIC_INPUTS:
        value = payload[field]
        if field == "TotalCharges" and value is None:
            customer[field] = None
            continue
        if field == "SeniorCitizen":
            if type(value) is not int or value not in (0, 1):
                raise ValueError("SeniorCitizen must be the integer 0 or 1.")
        elif (
            type(value) not in (int, float)
            or not 0 <= value <= MAX_NUMERIC_VALUE
            or (field == "tenure" and type(value) is not int)
        ):
            kind = "an integer" if field == "tenure" else "a number"
            raise ValueError(f"{field} must be {kind} from 0 to {MAX_NUMERIC_VALUE}.")
        customer[field] = value

    for field, allowed in CATEGORICAL_VALUES.items():
        value = payload[field]
        if not isinstance(value, str) or value.strip() not in allowed:
            raise ValueError(f"{field} must be one of: {', '.join(allowed)}.")
        customer[field] = value.strip()
    return customer


def engineer_features(records: pd.DataFrame) -> pd.DataFrame:
    features = records.loc[:, list(RAW_FEATURES)].copy()
    for field in NUMERIC_INPUTS:
        numeric = pd.to_numeric(features[field], errors="coerce")
        features[field] = numeric.where(numeric.between(0, MAX_NUMERIC_VALUE))
    for field in CATEGORICAL_VALUES:
        text = features[field].astype("string").str.strip().replace("", pd.NA)
        features[field] = text.astype(object).where(text.notna(), np.nan)

    zero_tenure = features["tenure"].eq(0)
    features.loc[zero_tenure & features["TotalCharges"].isna(), "TotalCharges"] = 0
    historical_average = features["TotalCharges"] / features["tenure"].where(
        features["tenure"] > 0
    )
    features["AverageChargesToDate"] = historical_average.where(
        ~zero_tenure, features["MonthlyCharges"]
    )
    add_ons = features.loc[:, list(INTERNET_ADD_ONS)]
    features["InternetAddOnCount"] = add_ons.eq("Yes").sum(axis=1).where(
        add_ons.notna().all(axis=1)
    )
    return features


def build_pipeline(classifier) -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_VALUES)),
        ]
    )
    return Pipeline(
        [
            ("features", FunctionTransformer(engineer_features)),
            ("preprocess", preprocessing),
            ("classifier", classifier),
        ]
    )
