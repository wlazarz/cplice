"""Tests for fast categorical distance matrices."""

import numpy as np
import pandas as pd
import pytest

from algorithms.cplice.object_distances import (
    CategoricalDistanceCalculator,
    compute_distance_matrix,
)


@pytest.fixture
def categorical_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "color": ["red", "red", "blue", "green", "blue"],
            "shape": ["round", "square", "round", "round", "square"],
            "size": ["small", "small", "large", "large", "small"],
        }
    )


@pytest.mark.parametrize(
    "metric",
    [
        "dice",
        "eskin",
        "hamming",
        "iof",
        "jaccard",
        "lin",
        "overlap",
        "s2",
    ],
)
def test_matrix_matches_scalar_distance(
    categorical_data: pd.DataFrame,
    metric: str,
) -> None:
    calculator = CategoricalDistanceCalculator(
        categorical_data,
        metric,
    )
    matrix = calculator.pairwise(block_size=2)

    assert matrix.shape == (
        len(categorical_data),
        len(categorical_data),
    )
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix >= 0)

    values = categorical_data.to_numpy(dtype=object)
    for first_index in range(len(values)):
        for second_index in range(len(values)):
            expected = calculator.distance(
                values[first_index],
                values[second_index],
            )
            assert matrix[first_index, second_index] == pytest.approx(
                expected
            )


def test_overlap_and_hamming_have_expected_scale(
    categorical_data: pd.DataFrame,
) -> None:
    overlap = compute_distance_matrix(
        categorical_data,
        "overlap",
        block_size=2,
    )
    hamming = compute_distance_matrix(
        categorical_data,
        "hamming",
        block_size=2,
    )

    assert hamming[0, 2] == 2
    assert overlap[0, 2] == pytest.approx(2 / 3)


def test_jaccard_uses_categorical_match_definition(
    categorical_data: pd.DataFrame,
) -> None:
    matrix = compute_distance_matrix(
        categorical_data,
        "jaccard",
        block_size=2,
    )

    mismatches = 2
    feature_count = 3
    expected = 2 * mismatches / (feature_count + mismatches)
    assert matrix[0, 2] == pytest.approx(expected)


def test_feature_statistics_are_column_specific() -> None:
    data = pd.DataFrame(
        {
            "first": ["x", "x", "y", "y"],
            "second": ["x", "z", "z", "z"],
        }
    )
    calculator = CategoricalDistanceCalculator(data, "iof")

    assert calculator.features[0].counts.tolist() == [2, 2]
    assert calculator.features[1].counts.tolist() == [1, 3]


def test_s2_does_not_require_external_one_hot_encoding(
    categorical_data: pd.DataFrame,
) -> None:
    matrix = compute_distance_matrix(
        categorical_data,
        "s2",
        block_size=2,
    )

    assert matrix[0, 0] == 0
    assert 0 <= matrix[0, 3] <= 1


def test_memmap_output(
    categorical_data: pd.DataFrame,
    tmp_path,
) -> None:
    output_path = tmp_path / "distances.npy"
    matrix = compute_distance_matrix(
        categorical_data,
        "lin",
        block_size=2,
        dtype=np.float32,
        output_path=output_path,
    )

    assert isinstance(matrix, np.memmap)
    assert output_path.exists()
    assert matrix.dtype == np.float32
    assert np.allclose(matrix, matrix.T)


def test_unknown_metric_is_rejected(
    categorical_data: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="Unsupported metric"):
        compute_distance_matrix(categorical_data, "unknown")