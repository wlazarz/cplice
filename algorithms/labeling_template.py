"""Shared categorical-distance utilities for CPLICE-style algorithms."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from evaluation.metrics import calculate_mode
from algorithms.cplice.object_distances import (
    CategoricalDistanceCalculator,
)

ClusterLabel: TypeAlias = Hashable
FeatureArray: TypeAlias = NDArray[Any]
CentroidDictionary: TypeAlias = dict[ClusterLabel, FeatureArray]


class LabelingTemplate:
    """Provide distances and mode-based centroids for categorical data.

    Parameters
    ----------
    df
        Original categorical data. S2 encoding is handled internally by the
        distance calculator; ``df`` must not be one-hot encoded beforehand.
    metric
        Categorical distance name accepted by
        :class:`CategoricalDistanceCalculator`.
    feature_weights
        Optional non-negative feature weights.
    distance_working_memory_mb
        Temporary-memory budget for automatic matrix block sizing.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        metric: str,
        *,
        feature_weights: Sequence[float] | Mapping[str, float] | None = None,
        distance_working_memory_mb: float = 256.0,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("'df' must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("'df' cannot be empty.")

        self.metric = metric
        self.df: FeatureArray = df.to_numpy(dtype=object, copy=True)
        self.distance_calculator = CategoricalDistanceCalculator(
            df,
            metric,
            feature_weights=feature_weights,
            working_memory_mb=distance_working_memory_mb,
        )

    def distance(self, first: ArrayLike, second: ArrayLike) -> float:
        """Calculate the configured distance between two objects."""
        return self.distance_calculator.distance(first, second)

    def compute_centroids(
        self,
        cluster_dict: Mapping[
            ClusterLabel,
            Sequence[int] | NDArray[np.integer[Any]],
        ],
    ) -> CentroidDictionary:
        """Compute a mode-based centroid for every cluster."""
        centroids: CentroidDictionary = {}

        for label, indices in cluster_dict.items():
            index_array = np.asarray(indices, dtype=int).reshape(-1)
            if index_array.size == 0:
                raise ValueError(
                    f"Cluster {label!r} must contain at least one object."
                )

            centroids[label] = calculate_mode(self.df[index_array])

        return centroids

    def compute_distance_matrix(
        self,
        *,
        block_size: int | None = None,
        dtype: Any = np.float64,
        output_path: str | Path | None = None,
    ) -> NDArray[np.floating[Any]] | np.memmap:
        """Calculate a matrix ready to pass to CPLICE or KNN."""
        return self.distance_calculator.pairwise(
            block_size=block_size,
            dtype=dtype,
            output_path=output_path,
        )

    @staticmethod
    def build_labeled_dataset(
        labeled_data: Mapping[
            ClusterLabel,
            Sequence[ArrayLike],
        ],
    ) -> tuple[FeatureArray, list[ClusterLabel]]:
        """Convert a label-to-objects mapping into features and labels."""
        feature_rows: list[ArrayLike] = []
        labels: list[ClusterLabel] = []

        for label, rows in labeled_data.items():
            row_list = list(rows)
            feature_rows.extend(row_list)
            labels.extend([label] * len(row_list))

        if not feature_rows:
            raise ValueError(
                "'labeled_data' must contain at least one object."
            )

        return np.vstack(feature_rows), labels