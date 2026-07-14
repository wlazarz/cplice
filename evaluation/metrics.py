"""Fast clustering metrics aligned with CPLICE distance geometry.

The primary metrics in this module operate on the same precomputed distance
matrix that CPLICE uses during labeling. This avoids mixing incompatible
geometries such as one-hot Euclidean distance, normalized Hamming distance,
and a custom categorical distance.

The module also contains categorical prototype metrics and explicitly marked
experimental distribution metrics for research comparisons.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

ClusterLabel: TypeAlias = Hashable
DistanceMatrix: TypeAlias = NDArray[np.float64]
AverageMode: TypeAlias = Literal["micro", "macro"]
OutlierStrategy: TypeAlias = Literal[
    "average",
    "median",
    "k_closest_per_cluster",
]


@dataclass(frozen=True, slots=True)
class MetricValue:
    """A clustering metric value with its optimization direction."""

    value: float | None
    higher_is_better: bool
    description: str


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Collection of CPLICE-aligned clustering metrics."""

    metrics: Mapping[str, MetricValue]
    sample_count: int
    cluster_count: int
    excluded_count: int

    def values(self) -> dict[str, float | None]:
        """Return a plain name-to-value mapping."""
        return {
            metric_name: metric.value
            for metric_name, metric in self.metrics.items()
        }


class CPLICEClusteringEvaluator:
    """Evaluate CPLICE labels using a shared pairwise distance matrix.

    Parameters
    ----------
    labels
        One predicted cluster label per row.
    distance_matrix
        Pairwise distance matrix produced with the same metric used by CPLICE.
        Reusing this matrix is strongly recommended for expensive metrics.
    categorical_data
        Optional original categorical feature matrix. It is required only for
        entropy, mode-based, and experimental distribution metrics.
    excluded_label
        Optional label representing outliers or rows excluded from ordinary
        cluster-quality metrics. Common examples are ``-1`` and ``None``.
    exclude_missing_labels
        Whether rows labeled ``None`` or ``NaN`` should be excluded.
    validate_matrix
        Whether to perform complete distance-matrix validation. Disable only
        when the matrix has already been validated in a trusted pipeline.
    symmetry_tolerance
        Absolute tolerance used when checking matrix symmetry and its diagonal.

    Notes
    -----
    Distance-based Calinski-Harabasz is replaced by a distance-based pseudo-F
    statistic. Standard Calinski-Harabasz assumes Euclidean feature geometry
    and should not be applied directly to arbitrary CPLICE distances.
    """

    def __init__(
        self,
        labels: ArrayLike,
        distance_matrix: NDArray[np.floating[Any]],
        *,
        categorical_data: ArrayLike | None = None,
        excluded_label: ClusterLabel | None = None,
        exclude_missing_labels: bool = True,
        validate_matrix: bool = True,
        symmetry_tolerance: float = 1e-8,
    ) -> None:
        raw_labels = np.asarray(labels, dtype=object)
        if raw_labels.ndim != 1:
            raise ValueError("'labels' must be one-dimensional.")

        raw_matrix = np.asarray(distance_matrix, dtype=np.float64)
        expected_shape = (len(raw_labels), len(raw_labels))
        if raw_matrix.shape != expected_shape:
            raise ValueError(
                "'distance_matrix' must have shape "
                f"{expected_shape}, received {raw_matrix.shape}."
            )

        if validate_matrix:
            validate_distance_matrix(
                raw_matrix,
                symmetry_tolerance=symmetry_tolerance,
            )

        included_mask = np.ones(len(raw_labels), dtype=bool)

        if exclude_missing_labels:
            included_mask &= np.fromiter(
                (not _is_missing(label) for label in raw_labels),
                dtype=bool,
                count=len(raw_labels),
            )

        if excluded_label is not None:
            included_mask &= np.fromiter(
                (
                    not _labels_equal(label, excluded_label)
                    for label in raw_labels
                ),
                dtype=bool,
                count=len(raw_labels),
            )

        self._original_size = len(raw_labels)
        self._included_indices = np.where(included_mask)[0]
        self.labels = raw_labels[included_mask]
        self.distance_matrix = raw_matrix[
            np.ix_(self._included_indices, self._included_indices)
        ]
        self.excluded_count = int((~included_mask).sum())

        if categorical_data is None:
            self.categorical_data: NDArray[Any] | None = None
        else:
            raw_data = np.asarray(categorical_data, dtype=object)
            if raw_data.ndim != 2:
                raise ValueError(
                    "'categorical_data' must be two-dimensional."
                )
            if len(raw_data) != self._original_size:
                raise ValueError(
                    "'categorical_data' must contain one row per label."
                )
            self.categorical_data = raw_data[included_mask]

        if len(self.labels) == 0:
            raise ValueError("No rows remain after label exclusion.")

        self.cluster_labels = tuple(pd.unique(self.labels).tolist())
        self.cluster_indices: dict[ClusterLabel, NDArray[np.int_]] = {
            label: np.where(
                np.fromiter(
                    (
                        _labels_equal(value, label)
                        for value in self.labels
                    ),
                    dtype=bool,
                    count=len(self.labels),
                )
            )[0]
            for label in self.cluster_labels
        }

    @property
    def sample_count(self) -> int:
        """Number of rows included in ordinary cluster metrics."""
        return len(self.labels)

    @property
    def cluster_count(self) -> int:
        """Number of represented clusters."""
        return len(self.cluster_labels)

    @cached_property
    def medoid_indices(self) -> dict[ClusterLabel, int]:
        """Return global-within-evaluator medoid indices for every cluster."""
        medoids: dict[ClusterLabel, int] = {}

        for label, indices in self.cluster_indices.items():
            if len(indices) == 1:
                medoids[label] = int(indices[0])
                continue

            within = self.distance_matrix[np.ix_(indices, indices)]
            local_medoid = int(np.argmin(within.sum(axis=1)))
            medoids[label] = int(indices[local_medoid])

        return medoids

    @cached_property
    def cluster_modes(self) -> dict[ClusterLabel, NDArray[Any]]:
        """Return stable mode-based prototypes for categorical data."""
        data = self._require_categorical_data()
        return {
            label: calculate_mode(data[indices])
            for label, indices in self.cluster_indices.items()
        }

    def silhouette(self) -> float | None:
        """Calculate the exact silhouette using the CPLICE distance matrix.

        Singleton clusters receive a silhouette value of zero.
        """
        if not self._has_valid_multi_cluster_partition():
            return None

        scores = np.zeros(self.sample_count, dtype=float)

        for label, own_indices in self.cluster_indices.items():
            if len(own_indices) == 1:
                scores[own_indices[0]] = 0.0
                continue

            within = self.distance_matrix[np.ix_(own_indices, own_indices)]
            mean_within = within.sum(axis=1) / (len(own_indices) - 1)

            nearest_other = np.full(
                len(own_indices),
                np.inf,
                dtype=float,
            )

            for other_label, other_indices in self.cluster_indices.items():
                if _labels_equal(other_label, label):
                    continue

                mean_to_other = self.distance_matrix[
                    np.ix_(own_indices, other_indices)
                ].mean(axis=1)
                nearest_other = np.minimum(
                    nearest_other,
                    mean_to_other,
                )

            denominator = np.maximum(mean_within, nearest_other)
            scores[own_indices] = np.divide(
                nearest_other - mean_within,
                denominator,
                out=np.zeros_like(mean_within),
                where=denominator > 0,
            )

        return float(scores.mean())

    def dunn_index(self) -> float | None:
        """Calculate the exact Dunn index using complete linkage internally.

        The numerator is the minimum inter-cluster object distance. The
        denominator is the maximum intra-cluster object distance.
        """
        if self.cluster_count < 2:
            return None

        maximum_intra = 0.0
        for indices in self.cluster_indices.values():
            if len(indices) < 2:
                continue
            within = self.distance_matrix[np.ix_(indices, indices)]
            maximum_intra = max(maximum_intra, float(within.max()))

        minimum_inter = math.inf
        labels = self.cluster_labels

        for first_position, first_label in enumerate(labels[:-1]):
            first_indices = self.cluster_indices[first_label]

            for second_label in labels[first_position + 1 :]:
                second_indices = self.cluster_indices[second_label]
                between = self.distance_matrix[
                    np.ix_(first_indices, second_indices)
                ]
                minimum_inter = min(
                    minimum_inter,
                    float(between.min()),
                )

        if not math.isfinite(minimum_inter):
            return None
        if maximum_intra == 0:
            return math.inf if minimum_inter > 0 else None

        return float(minimum_inter / maximum_intra)

    def distance_based_pseudo_f(self) -> float | None:
        """Calculate a distance-based pseudo-F statistic.

        This statistic is the distance-matrix analogue commonly used in
        PERMANOVA-style analyses:

        total_ss = sum(d_ij^2) / n
        within_ss = sum_g sum(d_ij^2 within g) / n_g

        Higher values indicate stronger separation relative to dispersion.
        """
        if not self._has_valid_multi_cluster_partition():
            return None

        total_ss = _upper_triangle_sum_of_squares(
            self.distance_matrix
        ) / self.sample_count

        within_ss = 0.0
        for indices in self.cluster_indices.values():
            if len(indices) < 2:
                continue

            within = self.distance_matrix[np.ix_(indices, indices)]
            within_ss += (
                _upper_triangle_sum_of_squares(within) / len(indices)
            )

        between_ss = total_ss - within_ss
        within_degrees = self.sample_count - self.cluster_count
        between_degrees = self.cluster_count - 1

        if within_degrees <= 0 or between_degrees <= 0:
            return None
        if within_ss == 0:
            return math.inf if between_ss > 0 else None

        return float(
            (between_ss / between_degrees)
            / (within_ss / within_degrees)
        )

    def medoid_davies_bouldin(self) -> float | None:
        """Calculate a Davies-Bouldin generalization based on medoids.

        Cluster scatter is the mean distance to its medoid. Separation is the
        distance between cluster medoids. Lower values are better.
        """
        if self.cluster_count < 2:
            return None

        scatters: dict[ClusterLabel, float] = {}
        for label, indices in self.cluster_indices.items():
            medoid = self.medoid_indices[label]
            scatters[label] = float(
                self.distance_matrix[indices, medoid].mean()
            )

        ratios = np.zeros(self.cluster_count, dtype=float)

        for first_position, first_label in enumerate(self.cluster_labels):
            first_medoid = self.medoid_indices[first_label]
            worst_ratio = -math.inf

            for second_label in self.cluster_labels:
                if _labels_equal(second_label, first_label):
                    continue

                second_medoid = self.medoid_indices[second_label]
                separation = float(
                    self.distance_matrix[first_medoid, second_medoid]
                )
                scatter_sum = (
                    scatters[first_label] + scatters[second_label]
                )

                if separation == 0:
                    ratio = math.inf if scatter_sum > 0 else 0.0
                else:
                    ratio = scatter_sum / separation

                worst_ratio = max(worst_ratio, ratio)

            ratios[first_position] = worst_ratio

        return float(ratios.mean())

    def within_cluster_dispersion(
        self,
        *,
        average: AverageMode = "micro",
    ) -> float:
        """Calculate mean pairwise distance inside clusters.

        ``micro`` weights every object pair equally. ``macro`` weights every
        non-singleton cluster equally.
        """
        _validate_average_mode(average)

        cluster_means: list[float] = []
        pair_counts: list[int] = []

        for indices in self.cluster_indices.values():
            pair_count = len(indices) * (len(indices) - 1) // 2
            if pair_count == 0:
                continue

            within = self.distance_matrix[np.ix_(indices, indices)]
            upper_values = within[np.triu_indices(len(indices), k=1)]
            cluster_means.append(float(upper_values.mean()))
            pair_counts.append(pair_count)

        if not cluster_means:
            return 0.0

        if average == "macro":
            return float(np.mean(cluster_means))

        return float(np.average(cluster_means, weights=pair_counts))

    def between_medoid_separation(self) -> float | None:
        """Calculate the mean distance between cluster medoids."""
        if self.cluster_count < 2:
            return None

        medoids = np.asarray(
            [
                self.medoid_indices[label]
                for label in self.cluster_labels
            ],
            dtype=int,
        )
        medoid_distances = self.distance_matrix[
            np.ix_(medoids, medoids)
        ]
        upper_values = medoid_distances[
            np.triu_indices(len(medoids), k=1)
        ]
        return float(upper_values.mean())

    def dispersion_separation_ratio(
        self,
        *,
        average: AverageMode = "micro",
    ) -> float | None:
        """Return within-cluster dispersion divided by medoid separation."""
        separation = self.between_medoid_separation()
        if separation is None:
            return None

        dispersion = self.within_cluster_dispersion(average=average)
        if separation == 0:
            return math.inf if dispersion > 0 else None

        return float(dispersion / separation)

    def normalized_cluster_entropy(self) -> float:
        """Calculate size-weighted, feature-normalized cluster entropy.

        Entropy is calculated separately for every column, preventing identical
        raw values in different columns from being treated as one category.
        Each feature entropy is divided by its global maximum entropy
        ``log(number_of_categories)``. Lower values are better.
        """
        data = self._require_categorical_data()
        normalizers = _feature_entropy_normalizers(data)

        total = 0.0
        for indices in self.cluster_indices.values():
            cluster_data = data[indices]
            feature_entropies = _column_entropies(
                cluster_data,
                normalizers=normalizers,
            )
            cluster_entropy = float(feature_entropies.mean())
            total += (
                len(indices) / self.sample_count
            ) * cluster_entropy

        return float(total)

    def mode_mismatch_rate(self) -> float:
        """Calculate the proportion of feature values differing from modes."""
        data = self._require_categorical_data()
        mismatch_count = 0
        value_count = 0

        for label, indices in self.cluster_indices.items():
            cluster_data = data[indices]
            mode = self.cluster_modes[label]
            mismatch_count += int((cluster_data != mode).sum())
            value_count += int(cluster_data.size)

        if value_count == 0:
            return 0.0
        return float(mismatch_count / value_count)

    def mode_separation(self) -> float | None:
        """Calculate mean normalized Hamming distance between class modes."""
        if self.cluster_count < 2:
            return None

        modes = np.asarray(
            [
                self.cluster_modes[label]
                for label in self.cluster_labels
            ],
            dtype=object,
        )
        distances = pairwise_categorical_hamming(modes)
        upper_values = distances[
            np.triu_indices(len(modes), k=1)
        ]
        return float(upper_values.mean())

    def mode_mismatch_separation_ratio(self) -> float | None:
        """Return mode mismatch rate divided by between-mode separation."""
        separation = self.mode_separation()
        if separation is None:
            return None

        mismatch = self.mode_mismatch_rate()
        if separation == 0:
            return math.inf if mismatch > 0 else None

        return float(mismatch / separation)

    def experimental_global_frequency_score(self) -> float:
        """Calculate a corrected, column-aware global-frequency score.

        This is a replacement for the original M1 implementation. For every
        cell, the score uses the global frequency of that value in the same
        feature, normalized by the number of rows. Higher values mean that
        clusters contain globally common feature values.

        This remains an experimental descriptive metric, not a standard
        clustering-validity index.
        """
        data = self._require_categorical_data()
        row_count, column_count = data.shape
        frequency_tables = _column_frequency_tables(data)

        per_cluster_scores: list[float] = []

        for indices in self.cluster_indices.values():
            cluster_data = data[indices]
            total = 0.0

            for column_index in range(column_count):
                frequencies = frequency_tables[column_index]
                total += sum(
                    frequencies[_category_key(value)] / row_count
                    for value in cluster_data[:, column_index]
                )

            per_cluster_scores.append(
                total / cluster_data.size
            )

        return float(np.mean(per_cluster_scores))

    def experimental_distribution_divergence(self) -> float:
        """Calculate mean feature-wise KL divergence from global distributions.

        This is a corrected, interpretable replacement for the original M2.
        Categories are column-aware and a small data-dependent smoothing term
        prevents division by zero. Higher values indicate clusters whose
        feature distributions differ more strongly from the full dataset.
        """
        data = self._require_categorical_data()
        global_distributions = _column_probability_tables(data)
        cluster_scores: list[float] = []

        for indices in self.cluster_indices.values():
            cluster_data = data[indices]
            feature_scores: list[float] = []

            for column_index in range(data.shape[1]):
                global_distribution = global_distributions[column_index]
                cluster_distribution = _probability_table(
                    cluster_data[:, column_index]
                )
                category_keys = tuple(global_distribution)
                epsilon = 1.0 / (
                    max(len(cluster_data), 1)
                    * max(len(category_keys), 1)
                    * 1000.0
                )

                divergence = 0.0
                for key in category_keys:
                    cluster_probability = cluster_distribution.get(
                        key,
                        0.0,
                    )
                    if cluster_probability <= 0:
                        continue

                    global_probability = max(
                        global_distribution[key],
                        epsilon,
                    )
                    divergence += cluster_probability * math.log(
                        cluster_probability / global_probability
                    )

                feature_scores.append(divergence)

            cluster_scores.append(float(np.mean(feature_scores)))

        return float(np.mean(cluster_scores))

    def experimental_frequency_cutoff_scores(
        self,
    ) -> tuple[float, float]:
        """Calculate corrected versions of the original M3 and M4.

        Values are treated as ``(column, category)`` pairs. The cutoff is the
        largest decrease between adjacent sorted frequencies. Degenerate
        one-category clusters are handled without calling ``argmax`` on an
        empty array.
        """
        data = self._require_categorical_data()
        concentration_scores: list[float] = []
        ratio_scores: list[float] = []

        for indices in self.cluster_indices.values():
            cluster_data = data[indices]
            counts = _column_aware_value_counts(cluster_data)
            sorted_counts = np.sort(counts)[::-1]

            if len(sorted_counts) == 1:
                cutoff_position = 0
            else:
                drops = sorted_counts[:-1] - sorted_counts[1:]
                if float(drops.max()) <= 0:
                    cutoff_position = len(sorted_counts) - 1
                else:
                    cutoff_position = int(np.argmax(drops))

            above = float(
                sorted_counts[: cutoff_position + 1].sum()
            )
            below = float(
                sorted_counts[cutoff_position + 1 :].sum()
            )

            concentration_scores.append(
                above / (len(sorted_counts) * cluster_data.size)
            )
            ratio_scores.append(
                above / below if below > 0 else math.inf
            )

        concentration = float(np.mean(concentration_scores))
        finite_ratios = [
            value for value in ratio_scores if math.isfinite(value)
        ]
        ratio = (
            float(np.mean(finite_ratios))
            if finite_ratios
            else math.inf
        )
        return concentration, ratio

    def evaluate(
        self,
        *,
        include_categorical_metrics: bool = True,
        include_experimental_metrics: bool = False,
    ) -> EvaluationReport:
        """Calculate a recommended CPLICE evaluation bundle."""
        metrics: dict[str, MetricValue] = {
            "silhouette": MetricValue(
                self.silhouette(),
                True,
                "Exact silhouette using the CPLICE distance matrix.",
            ),
            "dunn": MetricValue(
                self.dunn_index(),
                True,
                "Minimum inter-cluster distance divided by maximum "
                "intra-cluster distance.",
            ),
            "distance_pseudo_f": MetricValue(
                self.distance_based_pseudo_f(),
                True,
                "Distance-based pseudo-F; replaces Euclidean "
                "Calinski-Harabasz.",
            ),
            "medoid_davies_bouldin": MetricValue(
                self.medoid_davies_bouldin(),
                False,
                "Davies-Bouldin generalization using cluster medoids.",
            ),
            "within_dispersion": MetricValue(
                self.within_cluster_dispersion(),
                False,
                "Micro-averaged within-cluster pairwise distance.",
            ),
            "between_medoid_separation": MetricValue(
                self.between_medoid_separation(),
                True,
                "Mean pairwise distance between cluster medoids.",
            ),
            "dispersion_separation_ratio": MetricValue(
                self.dispersion_separation_ratio(),
                False,
                "Within dispersion divided by medoid separation.",
            ),
        }

        if include_categorical_metrics:
            self._require_categorical_data()
            metrics.update(
                {
                    "normalized_entropy": MetricValue(
                        self.normalized_cluster_entropy(),
                        False,
                        "Size-weighted and feature-normalized entropy.",
                    ),
                    "mode_mismatch_rate": MetricValue(
                        self.mode_mismatch_rate(),
                        False,
                        "Fraction of feature values differing from "
                        "cluster modes.",
                    ),
                    "mode_separation": MetricValue(
                        self.mode_separation(),
                        True,
                        "Mean normalized Hamming distance between modes.",
                    ),
                    "mode_mismatch_separation_ratio": MetricValue(
                        self.mode_mismatch_separation_ratio(),
                        False,
                        "Mode mismatch divided by mode separation.",
                    ),
                }
            )

        if include_experimental_metrics:
            self._require_categorical_data()
            cutoff_m3, cutoff_m4 = (
                self.experimental_frequency_cutoff_scores()
            )
            metrics.update(
                {
                    "experimental_global_frequency": MetricValue(
                        self.experimental_global_frequency_score(),
                        True,
                        "Column-aware corrected replacement for M1.",
                    ),
                    "experimental_distribution_divergence": MetricValue(
                        self.experimental_distribution_divergence(),
                        True,
                        "Mean feature-wise KL divergence; corrected M2.",
                    ),
                    "experimental_frequency_concentration": MetricValue(
                        cutoff_m3,
                        True,
                        "Corrected frequency-cutoff concentration; M3.",
                    ),
                    "experimental_frequency_ratio": MetricValue(
                        cutoff_m4,
                        True,
                        "Corrected above/below cutoff ratio; M4.",
                    ),
                }
            )

        return EvaluationReport(
            metrics=metrics,
            sample_count=self.sample_count,
            cluster_count=self.cluster_count,
            excluded_count=self.excluded_count,
        )

    def _has_valid_multi_cluster_partition(self) -> bool:
        return (
            self.cluster_count >= 2
            and self.cluster_count < self.sample_count
        )

    def _require_categorical_data(self) -> NDArray[Any]:
        if self.categorical_data is None:
            raise ValueError(
                "This metric requires 'categorical_data'."
            )
        return self.categorical_data


def calculate_mode(data: ArrayLike) -> NDArray[Any]:
    """Calculate a deterministic column-wise mode.

    Ties are resolved by the first occurrence in the column. Missing values are
    treated as valid categorical values rather than dropped.
    """
    matrix = np.asarray(data, dtype=object)
    if matrix.ndim != 2:
        raise ValueError("'data' must be two-dimensional.")
    if len(matrix) == 0:
        raise ValueError("Cannot calculate a mode for an empty array.")

    modes: list[Any] = []

    for column_index in range(matrix.shape[1]):
        column = matrix[:, column_index]
        codes, uniques = pd.factorize(
            pd.Series(column, dtype="object"),
            sort=False,
            use_na_sentinel=False,
        )
        counts = np.bincount(codes)
        modes.append(uniques[int(np.argmax(counts))])

    return np.asarray(modes, dtype=object)


def pairwise_categorical_hamming(
    data: ArrayLike,
    *,
    chunk_size: int = 256,
) -> DistanceMatrix:
    """Build a normalized Hamming distance matrix in row chunks."""
    matrix = np.asarray(data, dtype=object)
    if matrix.ndim != 2:
        raise ValueError("'data' must be two-dimensional.")
    if matrix.shape[1] == 0:
        raise ValueError("'data' must contain at least one feature.")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("'chunk_size' must be a positive integer.")

    sample_count = len(matrix)
    distances = np.empty(
        (sample_count, sample_count),
        dtype=np.float64,
    )

    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        distances[start:stop] = np.mean(
            matrix[start:stop, None, :] != matrix[None, :, :],
            axis=2,
            dtype=np.float64,
        )

    np.fill_diagonal(distances, 0.0)
    return distances


def validate_distance_matrix(
    distance_matrix: NDArray[np.floating[Any]],
    *,
    symmetry_tolerance: float = 1e-8,
) -> None:
    """Validate a finite, non-negative, symmetric distance matrix."""
    matrix = np.asarray(distance_matrix, dtype=np.float64)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("'distance_matrix' must be square.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "'distance_matrix' must contain only finite values."
        )
    if np.any(matrix < -symmetry_tolerance):
        raise ValueError(
            "'distance_matrix' cannot contain negative values."
        )
    if not np.allclose(
        matrix,
        matrix.T,
        atol=symmetry_tolerance,
        rtol=0.0,
    ):
        raise ValueError("'distance_matrix' must be symmetric.")
    if not np.allclose(
        np.diag(matrix),
        0.0,
        atol=symmetry_tolerance,
        rtol=0.0,
    ):
        raise ValueError(
            "The diagonal of 'distance_matrix' must contain zeros."
        )


def contrastive_outlier_score(
    labels: ArrayLike,
    distance_matrix: NDArray[np.floating[Any]],
    *,
    outlier_label: ClusterLabel = -1,
    strategy: OutlierStrategy = "average",
    number_of_neighbors: int = 5,
) -> float | None:
    """Evaluate how strongly outliers are separated from labeled inliers.

    Parameters
    ----------
    labels
        Cluster labels containing a dedicated outlier label.
    distance_matrix
        Pairwise matrix aligned with ``labels``.
    outlier_label
        Label identifying outliers.
    strategy
        ``average`` and ``median`` aggregate all outlier-to-inlier distances.
        ``k_closest_per_cluster`` averages the nearest distances separately in
        every inlier cluster.
    number_of_neighbors
        Number of nearest objects per inlier cluster for the third strategy.

    Returns
    -------
    float or None
        Mean contrastive outlier score, or ``None`` when no valid outlier to
        inlier comparison exists.
    """
    label_array = np.asarray(labels, dtype=object)
    matrix = np.asarray(distance_matrix, dtype=np.float64)

    if label_array.ndim != 1:
        raise ValueError("'labels' must be one-dimensional.")
    if matrix.shape != (len(label_array), len(label_array)):
        raise ValueError(
            "'distance_matrix' must align with 'labels'."
        )
    if strategy not in {
        "average",
        "median",
        "k_closest_per_cluster",
    }:
        raise ValueError(f"Unsupported strategy: {strategy!r}.")
    if number_of_neighbors <= 0:
        raise ValueError(
            "'number_of_neighbors' must be greater than zero."
        )

    outlier_mask = np.fromiter(
        (
            _labels_equal(label, outlier_label)
            for label in label_array
        ),
        dtype=bool,
        count=len(label_array),
    )
    inlier_mask = ~outlier_mask

    outlier_indices = np.where(outlier_mask)[0]
    inlier_indices = np.where(inlier_mask)[0]

    if len(outlier_indices) == 0 or len(inlier_indices) == 0:
        return None

    outlier_to_inlier = matrix[
        np.ix_(outlier_indices, inlier_indices)
    ]

    if strategy == "average":
        return float(outlier_to_inlier.mean())
    if strategy == "median":
        return float(np.median(outlier_to_inlier))

    inlier_labels = label_array[inlier_indices]
    cluster_labels = pd.unique(inlier_labels)
    per_outlier_scores: list[float] = []

    for outlier_position in range(len(outlier_indices)):
        nearest_values: list[float] = []

        for cluster_label in cluster_labels:
            cluster_mask = np.fromiter(
                (
                    _labels_equal(label, cluster_label)
                    for label in inlier_labels
                ),
                dtype=bool,
                count=len(inlier_labels),
            )
            cluster_distances = outlier_to_inlier[
                outlier_position,
                cluster_mask,
            ]
            neighbor_count = min(
                number_of_neighbors,
                len(cluster_distances),
            )
            if neighbor_count == 0:
                continue

            partitioned = np.partition(
                cluster_distances,
                neighbor_count - 1,
            )
            nearest_values.extend(
                partitioned[:neighbor_count].tolist()
            )

        if nearest_values:
            per_outlier_scores.append(
                float(np.mean(nearest_values))
            )

    if not per_outlier_scores:
        return None
    return float(np.mean(per_outlier_scores))


def _upper_triangle_sum_of_squares(
    matrix: DistanceMatrix,
) -> float:
    values = matrix[np.triu_indices(len(matrix), k=1)]
    return float(np.dot(values, values))


def _validate_average_mode(average: str) -> None:
    if average not in {"micro", "macro"}:
        raise ValueError("'average' must be 'micro' or 'macro'.")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True

    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False

    return bool(result) if np.isscalar(result) else False


def _labels_equal(first: Any, second: Any) -> bool:
    if _is_missing(first) and _is_missing(second):
        return True

    try:
        result = first == second
    except (TypeError, ValueError):
        return False

    return bool(result) if np.isscalar(result) else False


def _category_key(value: Any) -> tuple[str, Any]:
    if _is_missing(value):
        return ("missing", None)

    try:
        hash(value)
        return ("value", value)
    except TypeError:
        return ("repr", repr(value))


def _column_frequency_tables(
    data: NDArray[Any],
) -> list[dict[tuple[str, Any], int]]:
    tables: list[dict[tuple[str, Any], int]] = []

    for column_index in range(data.shape[1]):
        frequencies: dict[tuple[str, Any], int] = {}
        for value in data[:, column_index]:
            key = _category_key(value)
            frequencies[key] = frequencies.get(key, 0) + 1
        tables.append(frequencies)

    return tables


def _probability_table(
    values: Sequence[Any] | NDArray[Any],
) -> dict[tuple[str, Any], float]:
    counts: dict[tuple[str, Any], int] = {}

    for value in values:
        key = _category_key(value)
        counts[key] = counts.get(key, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return {}

    return {
        key: count / total
        for key, count in counts.items()
    }


def _column_probability_tables(
    data: NDArray[Any],
) -> list[dict[tuple[str, Any], float]]:
    return [
        _probability_table(data[:, column_index])
        for column_index in range(data.shape[1])
    ]


def _column_aware_value_counts(
    data: NDArray[Any],
) -> NDArray[np.int_]:
    counts: list[int] = []

    for column_index in range(data.shape[1]):
        column_counts: dict[tuple[str, Any], int] = {}
        for value in data[:, column_index]:
            key = _category_key(value)
            column_counts[key] = column_counts.get(key, 0) + 1
        counts.extend(column_counts.values())

    return np.asarray(counts, dtype=int)


def _feature_entropy_normalizers(
    data: NDArray[Any],
) -> NDArray[np.float64]:
    normalizers = np.ones(data.shape[1], dtype=float)

    for column_index in range(data.shape[1]):
        category_count = len(
            _probability_table(data[:, column_index])
        )
        if category_count > 1:
            normalizers[column_index] = math.log(category_count)

    return normalizers


def _column_entropies(
    data: NDArray[Any],
    *,
    normalizers: NDArray[np.float64],
) -> NDArray[np.float64]:
    entropies = np.zeros(data.shape[1], dtype=float)

    for column_index in range(data.shape[1]):
        probabilities = np.asarray(
            list(
                _probability_table(
                    data[:, column_index]
                ).values()
            ),
            dtype=float,
        )
        positive = probabilities[probabilities > 0]
        raw_entropy = float(
            -np.sum(positive * np.log(positive))
        )
        entropies[column_index] = (
            raw_entropy / normalizers[column_index]
        )

    return entropies