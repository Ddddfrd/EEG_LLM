"""Causal feature construction and compact supervised temporal controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import ProbabilityTimeline
from .features import causal_probability_histories, compute_enrollment_statistics
from .rules import enrollment_probabilities

SUPERVISED_FEATURE_NAMES = (
    "p_t_minus_7",
    "p_t_minus_6",
    "p_t_minus_5",
    "p_t_minus_4",
    "p_t_minus_3",
    "p_t_minus_2",
    "p_t_minus_1",
    "p_t",
    "enrollment_median",
    "enrollment_scaled_mad",
    "enrollment_q95",
    "history_slope",
    "history_variance",
    "record_start",
)


def build_supervised_features(timeline: ProbabilityTimeline) -> np.ndarray:
    """Build 14 causal features with histories reset at EDF boundaries."""
    enrollment = compute_enrollment_statistics(enrollment_probabilities(timeline))
    histories = causal_probability_histories(timeline, enrollment)
    slope = (histories[:, -1] - histories[:, 0]) / max(1, histories.shape[1] - 1)
    variance = np.var(histories, axis=1, dtype=np.float64).astype(np.float32)
    record_start = np.ones(timeline.row_count, dtype=np.float32)
    record_start[1:] = (
        timeline.record_indices[1:] != timeline.record_indices[:-1]
    ).astype(np.float32)
    static = np.tile(
        np.asarray(
            [enrollment.median, enrollment.scaled_mad, enrollment.quantile_95],
            dtype=np.float32,
        ),
        (timeline.row_count, 1),
    )
    features = np.column_stack(
        (histories, static, slope, variance, record_start)
    ).astype(np.float32, copy=False)
    if features.shape != (timeline.row_count, len(SUPERVISED_FEATURE_NAMES)):
        raise RuntimeError("Supervised feature contract is invalid")
    if not np.isfinite(features).all():
        raise RuntimeError("Supervised features contain non-finite values")
    return features


@dataclass(frozen=True)
class SupervisedControl:
    name: str
    estimator: Pipeline
    parameter_count: int

    def predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        values = self.estimator.predict_proba(features)[:, 1]
        return np.asarray(values, dtype=np.float32)

    def contract(self) -> dict[str, Any]:
        model = self.estimator.named_steps["model"]
        return {
            "name": self.name,
            "features": list(SUPERVISED_FEATURE_NAMES),
            "feature_count": len(SUPERVISED_FEATURE_NAMES),
            "parameter_count": self.parameter_count,
            "estimator": type(model).__name__,
            "standardization": "StandardScaler fit on chb20 only",
        }


def _balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.uint8)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if not positives or not negatives:
        raise ValueError("Supervised fitting requires both classes")
    weights = np.empty(labels.size, dtype=np.float64)
    weights[labels == 0] = labels.size / (2.0 * negatives)
    weights[labels == 1] = labels.size / (2.0 * positives)
    return weights


def fit_logistic_control(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 42,
) -> SupervisedControl:
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    estimator.fit(features, labels)
    model = estimator.named_steps["model"]
    count = int(model.coef_.size + model.intercept_.size)
    return SupervisedControl("logistic_regression", estimator, count)


def fit_mlp_control(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 42,
) -> SupervisedControl:
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(32, 32),
                    activation="relu",
                    solver="adam",
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=200,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=15,
                    random_state=seed,
                ),
            ),
        ]
    )
    estimator.fit(
        features,
        labels,
        model__sample_weight=_balanced_sample_weights(labels),
    )
    model = estimator.named_steps["model"]
    count = int(
        sum(values.size for values in model.coefs_)
        + sum(values.size for values in model.intercepts_)
    )
    if count > 10_000:
        raise RuntimeError(f"MLP control exceeds the 10K parameter budget: {count}")
    return SupervisedControl("mlp_32x32", estimator, count)
