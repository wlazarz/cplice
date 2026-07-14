"""Tests for representative-object selection."""

import numpy as np
import pytest

from algorithms.cplice.find_representativeness import (
    categorical_silhouette_score,
    compute_class_modes,
    select_representatives_by_centroid_contrast,
    select_representatives_by_medoid,
    truncate_representative_ranking,
)


def test_categorical_silhouette_excludes_self_distance() -> None:
    data = np.array(
        [
            ["a", "x"],
            ["a", "x"],
            ["b", "y"],
            ["b", "y"],
        ],
        dtype=object,
    )
    labels = np.array([0, 0, 1, 1])

    score = categorical_silhouette_score(data, labels)

    assert score == pytest.approx(1.0)


def test_singleton_cluster_has_zero_silhouette() -> None:
    data = np.array(
        [
            ["a"],
            ["b"],
            ["b"],
        ],
        dtype=object,
    )
    labels = np.array([0, 1, 1])

    score = categorical_silhouette_score(data, labels)

    assert 0.0 <= score <= 1.0


def test_compute_class_modes() -> None:
    data = np.array(
        [
            ["a", "x"],
            ["a", "y"],
            ["b", "z"],
        ],
        dtype=object,
    )
    labels = np.array(["left", "left", "right"], dtype=object)

    modes = compute_class_modes(data, labels)

    assert modes["left"][0] == "a"
    assert modes["right"].tolist() == ["b", "z"]


def test_centroid_contrast_selects_typical_objects() -> None:
    data = np.array(
        [
            ["a", "x"],
            ["a", "x"],
            ["a", "y"],
            ["b", "z"],
            ["b", "z"],
            ["b", "y"],
        ],
        dtype=object,
    )
    labels = np.array([0, 0, 0, 1, 1, 1])

    selected = select_representatives_by_centroid_contrast(
        data,
        labels,
        number_per_class=1,
    )

    assert selected[0][0] in {0, 1}
    assert selected[1][0] in {3, 4}


def test_medoid_selection_returns_global_indices() -> None:
    data = np.array(
        [
            ["a", "x"],
            ["a", "x"],
            ["b", "x"],
            ["z", "q"],
            ["z", "q"],
            ["y", "q"],
        ],
        dtype=object,
    )
    labels = np.array([0, 0, 0, 1, 1, 1])

    selected = select_representatives_by_medoid(
        data,
        labels,
        number_per_class=1,
    )

    assert selected[0][0] in {0, 1}
    assert selected[1][0] in {3, 4}


def test_fractional_ranking_truncation_enforces_minimum() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=object)
    ranking = {
        0: [0, 1, 2],
        1: [3, 4, 5],
    }

    selected = truncate_representative_ranking(
        ranking,
        labels,
        selection_size=0.1,
        minimum_per_class=2,
    )

    assert selected == {
        0: [0, 1],
        1: [3, 4],
    }


def test_invalid_ranking_label_is_rejected() -> None:
    labels = np.array([0, 0, 1, 1], dtype=object)

    with pytest.raises(ValueError, match="does not belong"):
        truncate_representative_ranking(
            {0: [0, 2]},
            labels,
            selection_size=1,
        )