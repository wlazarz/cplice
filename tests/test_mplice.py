"""Basic tests for the mixed conditional CPLICE prototype."""

import numpy as np
import pandas as pd
import pytest

from mixed_conditional_cplice import MixedConditionalCPLICELabeling


def build_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": [
                "a",
                "a",
                "a",
                "b",
                "b",
                "b",
                "a",
                "b",
            ],
            "value": [
                0.0,
                0.2,
                0.4,
                9.8,
                10.0,
                10.2,
                0.3,
                9.9,
            ],
        }
    )


def test_labels_all_rows() -> None:
    data = build_dataset()
    labeler = MixedConditionalCPLICELabeling(
        data,
        categorical_columns=["group"],
        numerical_columns=["value"],
        conditional_pairs=[("group", "value")],
    )

    labels = labeler.label_data(
        initial_clusters={
            "left": [0, 1],
            "right": [3, 4],
        },
        expansion_rate=0.25,
    )

    assert labels.shape == (len(data),)
    assert not any(label is None for label in labels)
    assert np.all(labels[[0, 1]] == "left")
    assert np.all(labels[[3, 4]] == "right")


def test_empty_initial_cluster_is_rejected() -> None:
    data = build_dataset()
    labeler = MixedConditionalCPLICELabeling(
        data,
        categorical_columns=["group"],
        numerical_columns=["value"],
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        labeler.label_data(
            initial_clusters={
                "left": [],
                "right": [3, 4],
            },
            expansion_rate=0.25,
        )


def test_overlapping_initial_indices_are_rejected() -> None:
    data = build_dataset()
    labeler = MixedConditionalCPLICELabeling(
        data,
        categorical_columns=["group"],
        numerical_columns=["value"],
    )

    with pytest.raises(ValueError, match="multiple initial clusters"):
        labeler.label_data(
            initial_clusters={
                "left": [0, 1],
                "right": [1, 4],
            },
            expansion_rate=0.25,
        )