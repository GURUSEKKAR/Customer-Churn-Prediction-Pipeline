"""Explain the selected Gradient Boosting model in additive log-odds."""

import numpy as np
import shap


class ChurnExplainer:
    def __init__(self, pipeline):
        self.preprocess = pipeline[:-1]
        self.explainer = shap.TreeExplainer(
            pipeline.named_steps["classifier"],
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )

        columns = pipeline.named_steps["preprocess"]
        numeric = columns.transformers_[0][2]
        categorical = columns.transformers_[1][2]
        encoder = columns.named_transformers_["categorical"].named_steps["encode"]
        self.fields = [*numeric, *categorical]
        widths = [1] * len(numeric) + [len(values) for values in encoder.categories_]
        self.group_starts = np.cumsum([0, *widths[:-1]])

    def contributions(self, records):
        processed = self.preprocess.transform(records)
        explanation = self.explainer(processed)
        grouped = np.add.reduceat(explanation.values, self.group_starts, axis=1)
        return explanation.base_values, grouped

    def explain_customer(self, record) -> dict:
        baselines, grouped = self.contributions(record)
        contributions = grouped[0]
        baseline = float(baselines[0])
        return {
            "method": "tree_shap",
            "units": "log_odds",
            "baseline": baseline,
            "model_output": baseline + float(contributions.sum()),
            "contributions": [
                {"feature": self.fields[index], "value": float(contributions[index])}
                for index in np.argsort(-np.abs(contributions), kind="stable")
            ],
        }

    def importance(self, records) -> list[dict]:
        _, contributions = self.contributions(records)
        importance = np.abs(contributions).mean(axis=0)
        return [
            {"feature": self.fields[index], "mean_absolute_log_odds": float(importance[index])}
            for index in np.argsort(-importance, kind="stable")
        ]
