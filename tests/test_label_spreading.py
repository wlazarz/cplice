"""Tests for LabelSpreadingLabeling."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from algorithms.competitive.label_spreading import LabelSpreadingLabeling


def test_numeric_label_spreading_preserves_row_order() -> None:
    unlabeled = np.array(
        [
            [0.1, 0.0],
            [9.9, 10.0],
            [0.2, 0.1],
            [10.1, 9.9],
        ]
    )
    labeled = {
        "left": np.array([[0.0, 0.0], [0.0, 0.2]]),
        "right": np.array([[10.0, 10.0], [10.0, 9.8]]),
    }

    labeler = LabelSpreadingLabeling(unlabeled)
    predictions, confidences = labeler.label_data(
        labeled,
        kernel="knn",
        n_neighbors=3,
    )

    assert predictions.tolist() == [
        "left",
        "right",
        "left",
        "right",
    ]
    assert confidences.shape == (len(unlabeled),)
    assert labeler.probabilities_.shape == (len(unlabeled), 2)
    assert labeler.classes_.tolist() == ["left", "right"]


def test_categorical_dataframe_works_with_transformer() -> None:
    unlabeled = pd.DataFrame(
        {
            "color": ["red", "blue"],
            "value": [0.1, 9.9],
        }
    )
    labeled = {
        "left": pd.DataFrame(
            {
                "color": ["red", "red"],
                "value": [0.0, 0.2],
            }
        ),
        "right": pd.DataFrame(
            {
                "color": ["blue", "blue"],
                "value": [10.0, 9.8],
            }
        ),
    }
    transformer = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["color"],
            ),
            ("numeric", StandardScaler(), ["value"]),
        ]
    )

    labeler = LabelSpreadingLabeling(
        unlabeled,
        transformer=transformer,
    )
    predictions, confidences = labeler.label_data(
        labeled,
        kernel="knn",
        n_neighbors=3,
    )

    assert predictions.tolist() == ["left", "right"]
    assert np.all((0 <= confidences) & (confidences <= 1))


def test_raw_categorical_data_requires_transformer() -> None:
    unlabeled = pd.DataFrame({"color": ["red", "blue"]})
    labeled = {
        "left": pd.DataFrame({"color": ["red"]}),
        "right": pd.DataFrame({"color": ["blue"]}),
    }

    labeler = LabelSpreadingLabeling(unlabeled)

    with pytest.raises(ValueError, match="requires numeric features"):
        labeler.label_data(labeled)


def test_empty_class_is_ignored() -> None:
    unlabeled = np.array([[0.1], [9.9]])
    labeled = {
        "empty": np.empty((0, 1)),
        "left": np.array([[0.0], [0.2]]),
        "right": np.array([[10.0], [9.8]]),
    }

    labeler = LabelSpreadingLabeling(unlabeled)
    predictions, _ = labeler.label_data(
        labeled,
        n_neighbors=2,
    )

    assert set(predictions) == {"left", "right"}
    assert "empty" not in labeler.classes_


def test_one_nonempty_class_is_rejected() -> None:
    labeler = LabelSpreadingLabeling(np.array([[0.1], [0.2]]))

    with pytest.raises(ValueError, match="at least two"):
        labeler.label_data(
            {
                "left": np.array([[0.0]]),
                "empty": np.empty((0, 1)),
            }
        )


def test_neighbor_count_is_reduced_for_small_dataset() -> None:
    labeler = LabelSpreadingLabeling(np.array([[0.1], [9.9]]))
    labeler.label_data(
        {
            "left": np.array([[0.0]]),
            "right": np.array([[10.0]]),
        },
        n_neighbors=100,
    )

    assert labeler.effective_n_neighbors_ == 3


def test_empty_unlabeled_data_returns_empty_arrays() -> None:
    labeler = LabelSpreadingLabeling(np.empty((0, 2)))
    predictions, confidences = labeler.label_data(
        {
            "left": np.array([[0.0, 0.0]]),
            "right": np.array([[1.0, 1.0]]),
        }
    )

    assert predictions.size == 0
    assert confidences.size == 0
    assert labeler.probabilities_.shape == (0, 2)


def test_invalid_alpha_is_rejected() -> None:
    labeler = LabelSpreadingLabeling(np.array([[0.1], [0.9]]))

    with pytest.raises(ValueError, match="open interval"):
        labeler.label_data(
            {
                "left": np.array([[0.0]]),
                "right": np.array([[1.0]]),
            },
            alpha=1.0,
        )