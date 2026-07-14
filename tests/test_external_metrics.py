"""Tests for external CPLICE evaluation metrics."""

import math

import numpy as np
import pytest

from external_metrics import (
    align_cluster_labels,
    conditional_cluster_entropy,
    evaluate_external_labels,
    jaccard_index,
    multiclass_roc_auc,
    shannon_entropy,
    variation_of_information,
)


def test_shannon_entropy_for_balanced_binary_sequence() -> None:
    value = shannon_entropy([0, 0, 1, 1], base=2)

    assert value == pytest.approx(1.0)


def test_weighted_conditional_entropy_is_zero_for_pure_clusters() -> None:
    predicted = np.array(["a", "a", "b", "b"], dtype=object)
    true = np.array([0, 0, 1, 1], dtype=object)

    value = conditional_cluster_entropy(
        predicted,
        true,
        average="weighted",
    )

    assert value == pytest.approx(0.0)


def test_variation_of_information_is_permutation_invariant() -> None:
    first = np.array([0, 0, 1, 1])
    second = np.array(["b", "b", "a", "a"], dtype=object)

    assert variation_of_information(first, second) == pytest.approx(0.0)


def test_jaccard_is_not_accuracy_for_multiclass_case() -> None:
    true = np.array([0, 0, 1, 1])
    predicted = np.array([0, 1, 1, 1])

    accuracy = np.mean(true == predicted)
    jaccard = jaccard_index(true, predicted, average="macro")

    assert jaccard != pytest.approx(accuracy)


def test_binary_auc_uses_continuous_scores() -> None:
    true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    auc = multiclass_roc_auc(
        true,
        scores,
        score_labels=[0, 1],
    )

    assert auc == pytest.approx(1.0)


def test_multiclass_auc_uses_score_columns() -> None:
    true = np.array(["a", "b", "c", "a", "b", "c"], dtype=object)
    scores = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )

    auc = multiclass_roc_auc(
        true,
        scores,
        score_labels=["a", "b", "c"],
    )

    assert auc == pytest.approx(1.0)


def test_hungarian_alignment_recovers_permuted_labels() -> None:
    true = np.array(["left", "left", "right", "right"], dtype=object)
    predicted = np.array([1, 1, 0, 0], dtype=object)

    aligned, mapping = align_cluster_labels(true, predicted)

    assert aligned.tolist() == true.tolist()
    assert mapping == {0: "right", 1: "left"}


def test_report_excludes_outliers_and_missing_labels() -> None:
    true = np.array([0, 0, 1, 1, -1, None], dtype=object)
    predicted = np.array([0, 0, 1, 0, -1, 1], dtype=object)

    report = evaluate_external_labels(
        true,
        predicted,
        ignored_labels=[-1],
    )

    assert report.sample_count == 4
    assert report.excluded_count == 2
    assert report.classification["accuracy"] == pytest.approx(0.75)


def test_auc_is_none_when_scores_are_not_supplied() -> None:
    true = np.array([0, 0, 1, 1])
    predicted = np.array([0, 1, 1, 1])

    report = evaluate_external_labels(true, predicted)

    assert report.classification["roc_auc"] is None


def test_identical_partitions_have_perfect_clustering_metrics() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2])

    report = evaluate_external_labels(labels, labels)

    assert report.clustering["adjusted_rand"] == pytest.approx(1.0)
    assert report.clustering[
        "normalized_mutual_information"
    ] == pytest.approx(1.0)
    assert report.clustering[
        "variation_of_information"
    ] == pytest.approx(0.0)
    assert math.isclose(
        report.clustering["conditional_cluster_entropy"],
        0.0,
    )