"""Tests for CPLICE-aligned clustering metrics."""

import math

import numpy as np
import pytest

from evaluation.metrics import (
    CPLICEClusteringEvaluator,
    calculate_mode,
    contrastive_outlier_score,
    pairwise_categorical_hamming,
)


def build_separated_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=object)
    distances = pairwise_categorical_hamming(data)
    return data, labels, distances


def test_mode_is_stable_and_column_wise() -> None:
    data = np.array(
        [
            ["a", 1],
            ["b", 2],
            ["a", 2],
            ["b", 1],
        ],
        dtype=object,
    )

    mode = calculate_mode(data)

    assert mode.tolist() == ["a", 1]


def test_silhouette_uses_singleton_zero_convention() -> None:
    data = np.array(
        [
            ["a"],
            ["b"],
            ["b"],
        ],
        dtype=object,
    )
    labels = np.array([0, 1, 1], dtype=object)
    distances = pairwise_categorical_hamming(data)

    evaluator = CPLICEClusteringEvaluator(
        labels,
        distances,
        categorical_data=data,
    )

    score = evaluator.silhouette()

    assert score is not None
    assert 0.0 <= score <= 1.0


def test_recommended_metrics_are_finite_for_normal_partition() -> None:
    data, labels, distances = build_separated_data()
    evaluator = CPLICEClusteringEvaluator(
        labels,
        distances,
        categorical_data=data,
    )

    report = evaluator.evaluate()
    values = report.values()

    assert values["silhouette"] is not None
    assert values["dunn"] is not None
    assert values["distance_pseudo_f"] is not None
    assert values["medoid_davies_bouldin"] is not None
    assert values["normalized_entropy"] is not None


def test_dunn_is_infinite_for_zero_intra_and_positive_inter() -> None:
    data = np.array(
        [
            ["a"],
            ["a"],
            ["b"],
            ["b"],
        ],
        dtype=object,
    )
    labels = np.array([0, 0, 1, 1], dtype=object)
    distances = pairwise_categorical_hamming(data)

    evaluator = CPLICEClusteringEvaluator(labels, distances)

    assert math.isinf(evaluator.dunn_index())


def test_excluded_outlier_is_removed_from_cluster_metrics() -> None:
    data = np.array(
        [
            ["a"],
            ["a"],
            ["b"],
            ["b"],
            ["z"],
        ],
        dtype=object,
    )
    labels = np.array([0, 0, 1, 1, -1], dtype=object)
    distances = pairwise_categorical_hamming(data)

    evaluator = CPLICEClusteringEvaluator(
        labels,
        distances,
        categorical_data=data,
        excluded_label=-1,
    )

    assert evaluator.sample_count == 4
    assert evaluator.excluded_count == 1


def test_outlier_score_uses_precomputed_distances() -> None:
    data = np.array(
        [
            ["a"],
            ["a"],
            ["b"],
            ["b"],
            ["z"],
        ],
        dtype=object,
    )
    labels = np.array([0, 0, 1, 1, -1], dtype=object)
    distances = pairwise_categorical_hamming(data)

    score = contrastive_outlier_score(
        labels,
        distances,
        outlier_label=-1,
        strategy="k_closest_per_cluster",
        number_of_neighbors=1,
    )

    assert score == pytest.approx(1.0)


def test_experimental_metrics_handle_single_category_cluster() -> None:
    data = np.array(
        [
            ["a", "x"],
            ["a", "x"],
            ["b", "y"],
            ["b", "y"],
        ],
        dtype=object,
    )
    labels = np.array([0, 0, 1, 1], dtype=object)
    distances = pairwise_categorical_hamming(data)

    evaluator = CPLICEClusteringEvaluator(
        labels,
        distances,
        categorical_data=data,
    )

    m3, m4 = evaluator.experimental_frequency_cutoff_scores()

    assert m3 > 0
    assert math.isinf(m4)