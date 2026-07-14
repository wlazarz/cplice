"""Tests for standalone model-based pseudo-labeling."""

from __future__ import annotations

import numpy as np
import pytest

from algorithms.competitive.model_based_labeling import ModelBasedLabeling


class ProbabilityModel:
    """Small classifier exposing predict_proba."""

    def fit(self, features, labels):
        self.threshold = float(np.mean(features[:, 0].astype(float)))
        return self

    def predict(self, features):
        values = features[:, 0].astype(float)
        return (values >= self.threshold).astype(int)

    def predict_proba(self, features):
        values = features[:, 0].astype(float)
        positive = 1.0 / (1.0 + np.exp(-(values - self.threshold)))
        return np.column_stack([1.0 - positive, positive])


class DecisionModel:
    """Small classifier exposing decision_function only."""

    def fit(self, features, labels):
        self.threshold = float(np.mean(features[:, 0].astype(float)))
        return self

    def predict(self, features):
        values = features[:, 0].astype(float)
        return (values >= self.threshold).astype(int)

    def decision_function(self, features):
        return features[:, 0].astype(float) - self.threshold


class PredictOnlyModel:
    """Small estimator without a confidence interface."""

    def fit(self, features, labels):
        self.label = labels[0]
        return self

    def predict(self, features):
        return np.repeat(self.label, len(features))


@pytest.fixture
def unlabeled_data():
    return np.array([[0.0], [0.1], [0.9], [1.0]])


@pytest.fixture
def labeled_data():
    return {
        0: np.array([[-0.2], [0.0]]),
        1: np.array([[1.0], [1.2]]),
    }


def test_predict_proba_path(unlabeled_data, labeled_data):
    labels, confidences = ModelBasedLabeling(unlabeled_data).label_data(
        labeled_data,
        ProbabilityModel(),
        confidence_cutoff=0.6,
        max_iterations=3,
    )

    assert labels.shape == (4,)
    assert confidences.shape == (4,)
    assert np.all(np.isfinite(confidences))


def test_decision_function_path(unlabeled_data, labeled_data):
    labels, confidences = ModelBasedLabeling(unlabeled_data).label_data(
        labeled_data,
        DecisionModel(),
        confidence_cutoff=0.6,
        max_iterations=3,
    )

    assert labels.shape == (4,)
    assert np.all((confidences >= 0.5) & (confidences <= 1.0))


def test_predict_only_model_supports_single_pass(
    unlabeled_data,
    labeled_data,
):
    labels, confidences = ModelBasedLabeling(unlabeled_data).label_data(
        labeled_data,
        PredictOnlyModel(),
        confidence_cutoff=None,
    )

    assert labels.shape == (4,)
    assert np.all(np.isnan(confidences))


def test_predict_only_model_requires_custom_confidence_for_iteration(
    unlabeled_data,
    labeled_data,
):
    with pytest.raises(TypeError):
        ModelBasedLabeling(unlabeled_data).label_data(
            labeled_data,
            PredictOnlyModel(),
            confidence_cutoff=0.9,
        )


def test_custom_confidence_getter(unlabeled_data, labeled_data):
    def confidence_getter(model, features, predictions):
        return np.full(len(features), 0.95)

    labels, confidences = ModelBasedLabeling(unlabeled_data).label_data(
        labeled_data,
        PredictOnlyModel(),
        confidence_cutoff=0.9,
        confidence_getter=confidence_getter,
    )

    assert labels.shape == (4,)
    assert np.allclose(confidences, 0.95)


def test_empty_class_groups_are_rejected_when_all_are_empty(
    unlabeled_data,
):
    with pytest.raises(ValueError):
        ModelBasedLabeling(unlabeled_data).label_data(
            {0: [], 1: []},
            ProbabilityModel(),
        )