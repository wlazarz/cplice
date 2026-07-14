"""One-nearest-neighbor pseudo-labeling for categorical data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from algorithms.labeling_template import ClusterLabel, LabelingTemplate

DistanceMatrix: TypeAlias = NDArray[np.float64]
LabeledIndices: TypeAlias = dict[ClusterLabel, NDArray[np.int_]]


class KNNLabeling(LabelingTemplate):
    """Assign labels using the nearest initially labeled object.

    Parameters
    ----------
    df
        Categorical input data. Rows represent objects and columns represent
        features.
    metric
        Name of the qualitative distance measure. See
        :class:`algorithms.labeling_template.LabelingTemplate` for supported
        values.
    distance_matrix
        Optional precomputed pairwise distance matrix. When omitted, the
        matrix is calculated automatically from ``metric``. Supplying a
        previously calculated matrix is useful for computationally expensive
        metrics such as Lin or S2.

    Notes
    -----
    This class implements one-nearest-neighbor labeling. Every object,
    including an initially labeled object, is assigned the label of its
    nearest labeled example. An initially labeled object normally retains its
    label because its distance to itself is zero.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        metric: str,
        distance_matrix: NDArray[np.floating[Any]] | None = None,
    ) -> None:
        super().__init__(df, metric)

        if distance_matrix is None:
            self.distance_matrix = self.compute_distance_matrix()
        else:
            self.distance_matrix = self._validate_distance_matrix(
                distance_matrix
            )

    def label_data(
        self,
        labeled_data: Mapping[
            ClusterLabel,
            Sequence[int] | NDArray[np.integer],
        ],
    ) -> list[ClusterLabel]:
        """Assign a label to every object using one-nearest neighbor.

        Parameters
        ----------
        labeled_data
            Mapping from class labels to indices of initially labeled
            objects. Classes with empty index sequences are ignored.

        Returns
        -------
        list
            Predicted label for every row in the input data.

        Raises
        ------
        ValueError
            If no labeled examples are provided, an index lies outside the
            input data, or the same object is assigned to multiple labels.
        """
        normalized_labeled_data = self._normalize_labeled_data(labeled_data)
        labels: list[ClusterLabel] = []

        for sample_index in range(len(self.df)):
            best_label: ClusterLabel | None = None
            smallest_distance = float("inf")

            for label, example_indices in normalized_labeled_data.items():
                distances = self.distance_matrix[
                    sample_index,
                    example_indices,
                ]
                nearest_distance = float(np.min(distances))

                if nearest_distance < smallest_distance:
                    smallest_distance = nearest_distance
                    best_label = label

            if best_label is None:
                raise RuntimeError(
                    "No label could be assigned despite valid labeled data."
                )

            labels.append(best_label)

        return labels

    def _validate_distance_matrix(
        self,
        distance_matrix: NDArray[np.floating[Any]],
    ) -> DistanceMatrix:
        """Validate and normalize a precomputed distance matrix."""
        matrix = np.asarray(distance_matrix, dtype=np.float64)
        expected_shape = (len(self.df), len(self.df))

        if matrix.shape != expected_shape:
            raise ValueError(
                "'distance_matrix' must have shape "
                f"{expected_shape}, received {matrix.shape}."
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                "'distance_matrix' must contain only finite values."
            )

        return matrix

    def _normalize_labeled_data(
        self,
        labeled_data: Mapping[
            ClusterLabel,
            Sequence[int] | NDArray[np.integer],
        ],
    ) -> LabeledIndices:
        """Validate labeled indices and discard classes without examples."""
        if not labeled_data:
            raise ValueError(
                "'labeled_data' must contain at least one labeled object."
            )

        normalized: LabeledIndices = {}
        seen_indices: set[int] = set()

        for label, example_indices in labeled_data.items():
            index_array = np.asarray(example_indices, dtype=int).reshape(-1)

            if index_array.size == 0:
                continue
            if np.any(index_array < 0) or np.any(index_array >= len(self.df)):
                raise ValueError(
                    f"Label {label!r} contains an index outside the input "
                    "data."
                )

            duplicate_indices = seen_indices.intersection(
                int(index) for index in index_array
            )
            if duplicate_indices:
                duplicates = ", ".join(
                    str(index) for index in sorted(duplicate_indices)
                )
                raise ValueError(
                    "An object cannot belong to multiple labels. "
                    f"Duplicated indices: {duplicates}."
                )

            seen_indices.update(int(index) for index in index_array)
            normalized[label] = index_array

        if not normalized:
            raise ValueError(
                "'labeled_data' must contain at least one labeled object; "
                "all provided index sequences are empty."
            )

        return normalized