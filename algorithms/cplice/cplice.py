"""CPLICE pseudo-labeling for categorical data."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Final, Literal, TypeAlias, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from algorithms.labeling_template import (
    CentroidDictionary,
    ClusterLabel,
    LabelingTemplate,
)
from evaluation.metrics import calculate_mode

StrategyName: TypeAlias = Literal[
    "centroid",
    "nearest",
    "farthest",
    "mean",
    "outside_mean",
]
ClusterAssignments: TypeAlias = list[ClusterLabel | None]
DistanceValues: TypeAlias = list[float]
DistanceMatrix: TypeAlias = NDArray[np.float64]

SUPPORTED_STRATEGIES: Final[frozenset[str]] = frozenset(
    {
        "centroid",
        "nearest",
        "farthest",
        "mean",
        "outside_mean",
    }
)
MATRIX_BASED_STRATEGIES: Final[frozenset[str]] = frozenset(
    {
        "nearest",
        "farthest",
        "mean",
        "outside_mean",
    }
)


class CPLICELabeling(LabelingTemplate):
    """Label categorical data using iterative cluster expansion.

    CPLICE progressively expands clusters initialized with labeled objects.
    During each iteration, unlabeled objects are provisionally assigned to
    their best cluster. Only a gradually increasing fraction of the
    best-scoring candidates is retained.

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
        Optional precomputed pairwise distance matrix. Supplying a matrix is
        useful for computationally expensive metrics such as Lin or S2. When
        a matrix-based strategy is selected and this argument is omitted, the
        matrix is calculated automatically from ``metric``.
    strategy
        Cluster scoring strategy:

        - ``"centroid"``: distance from the current cluster centroid;
        - ``"nearest"``: minimum distance to an object in the cluster;
        - ``"farthest"``: maximum distance to an object in the cluster;
        - ``"mean"``: mean distance to objects in the cluster;
        - ``"outside_mean"``: negative mean distance to assigned objects
          outside the cluster.

        Lower scores are always preferred.

    Notes
    -----
    The original implementation used the names ``"max"`` for minimum
    in-cluster distance and ``"min"`` for maximum in-cluster distance. The
    names used here describe the actual calculations while preserving the
    underlying algorithm.

    References
    ----------
    W. Łazarz and A. Nowak-Brzezińska, "Categorical Pseudo-Labeling with
    Iterative Cluster Expansion," Procedia Computer Science, vol. 270,
    pp. 937-946, 2025. doi:10.1016/j.procs.2025.09.214
    """

    def __init__(
        self,
        df: pd.DataFrame,
        metric: str,
        distance_matrix: NDArray[np.floating[Any]] | None = None,
        strategy: StrategyName = "centroid",
    ) -> None:
        normalized_strategy = self._validate_strategy(strategy)
        super().__init__(df, metric)

        self.strategy: StrategyName = normalized_strategy
        self.distance_matrix: DistanceMatrix | None

        if distance_matrix is not None:
            self.distance_matrix = self._validate_distance_matrix(
                distance_matrix
            )
        elif self.strategy in MATRIX_BASED_STRATEGIES:
            self.distance_matrix = self.compute_distance_matrix()
        else:
            self.distance_matrix = None

    def calculate_assignment_scores(
        self,
        cluster_assignments: Sequence[ClusterLabel],
    ) -> DistanceValues:
        """Calculate each object's score within its assigned cluster.

        Parameters
        ----------
        cluster_assignments
            Cluster label assigned to every input object.

        Returns
        -------
        list of float
            Score for every object under the configured strategy.

        Raises
        ------
        ValueError
            If assignments are missing, contain ``None``, or have an invalid
            length.
        """
        assignment_array = np.asarray(cluster_assignments, dtype=object)
        self._validate_assignment_length(assignment_array)

        if any(label is None for label in assignment_array):
            raise ValueError(
                "'cluster_assignments' must not contain unassigned objects."
            )

        centroids: CentroidDictionary = {}
        for cluster_label in dict.fromkeys(assignment_array.tolist()):
            cluster_data = self.df[assignment_array == cluster_label]
            centroids[cluster_label] = calculate_mode(cluster_data)

        scores: DistanceValues = []
        for sample_index, cluster_label in enumerate(assignment_array):
            scores.append(
                self.calculate_cluster_score(
                    sample_index=sample_index,
                    cluster_assignments=assignment_array,
                    centroid=centroids[cluster_label],
                    cluster_label=cluster_label,
                )
            )

        return scores

    def calculate_cluster_score(
        self,
        sample_index: int,
        cluster_assignments: Sequence[ClusterLabel | None],
        centroid: NDArray[Any],
        cluster_label: ClusterLabel,
    ) -> float:
        """Calculate an object's score relative to a cluster.

        Parameters
        ----------
        sample_index
            Row index of the evaluated object.
        cluster_assignments
            Current cluster assignments. Objects not selected in the current
            iteration may be represented by ``None``.
        centroid
            Current centroid of the evaluated cluster.
        cluster_label
            Label of the evaluated cluster.

        Returns
        -------
        float
            CPLICE score. Lower values indicate a better assignment.

        Notes
        -----
        A matrix-based strategy can encounter an empty distance vector, for
        example when scoring the only object in a singleton cluster. In that
        case, the method falls back to the object's distance from the cluster
        centroid. This keeps the algorithm operational without changing its
        cluster-expansion principle.
        """
        assignment_array = np.asarray(cluster_assignments, dtype=object)
        self._validate_assignment_length(assignment_array)
        self._validate_sample_index(sample_index)

        if self.strategy == "centroid":
            return self.distance(self.df[sample_index], centroid)

        distance_matrix = self._get_distance_matrix()

        if self.strategy in {"nearest", "farthest", "mean"}:
            cluster_mask = assignment_array == cluster_label
            cluster_mask[sample_index] = False
            distances = distance_matrix[sample_index, cluster_mask]

            if distances.size == 0:
                return self.distance(self.df[sample_index], centroid)
            if self.strategy == "nearest":
                return float(np.min(distances))
            if self.strategy == "farthest":
                return float(np.max(distances))
            return float(np.mean(distances))

        assigned_mask = np.fromiter(
            (label is not None for label in assignment_array),
            dtype=bool,
            count=len(assignment_array),
        )
        outside_cluster_mask = (
            (assignment_array != cluster_label) & assigned_mask
        )
        outside_distances = distance_matrix[
            sample_index,
            outside_cluster_mask,
        ]

        if outside_distances.size == 0:
            return self.distance(self.df[sample_index], centroid)

        return -float(np.mean(outside_distances))

    def select_best_candidates(
        self,
        proposed_assignments: Sequence[ClusterLabel | None],
        scores: Sequence[float],
        expansion_fraction: float,
        initial_cluster_sizes: Mapping[ClusterLabel, int],
    ) -> ClusterAssignments:
        """Retain the lowest-scoring candidates proposed for each class.

        The number retained for a class is the greater of its initial size and
        ``ceil(number_of_candidates * expansion_fraction)``.

        Parameters
        ----------
        proposed_assignments
            Provisional cluster assignment for every object.
        scores
            Score corresponding to every provisional assignment.
        expansion_fraction
            Current fraction controlling cluster expansion.
        initial_cluster_sizes
            Number of initially labeled objects in every cluster.

        Returns
        -------
        list
            Partial cluster assignments. Candidates not retained in the
            current iteration are represented by ``None``.

        Raises
        ------
        ValueError
            If the provided sequences have different lengths or if a proposed
            label is absent from ``initial_cluster_sizes``.
        """
        assignment_array = np.asarray(proposed_assignments, dtype=object)
        score_array = np.asarray(scores, dtype=float)

        self._validate_assignment_length(assignment_array)
        if len(score_array) != len(assignment_array):
            raise ValueError(
                "'proposed_assignments' and 'scores' must have the same "
                "length."
            )

        selected_assignments: ClusterAssignments = [None] * len(self.df)
        valid_labels = [
            label
            for label in dict.fromkeys(assignment_array.tolist())
            if label is not None
        ]

        for cluster_label in valid_labels:
            if cluster_label not in initial_cluster_sizes:
                raise ValueError(
                    f"Missing initial size for cluster {cluster_label!r}."
                )

            candidate_indices = np.where(
                assignment_array == cluster_label
            )[0]
            number_to_select = max(
                initial_cluster_sizes[cluster_label],
                math.ceil(len(candidate_indices) * expansion_fraction),
            )
            number_to_select = min(
                number_to_select,
                len(candidate_indices),
            )

            if number_to_select == 0:
                continue

            candidate_scores = score_array[candidate_indices]
            selected_indices = candidate_indices[
                np.argsort(candidate_scores)[:number_to_select]
            ]
            for sample_index in selected_indices:
                selected_assignments[int(sample_index)] = cluster_label

        return selected_assignments

    def assign_to_best_clusters(
        self,
        centroids: CentroidDictionary,
        current_assignments: Sequence[ClusterLabel | None],
        fixed_indices: Collection[int],
    ) -> tuple[ClusterAssignments, DistanceValues, bool]:
        """Provisionally assign each non-fixed object to its best cluster.

        Parameters
        ----------
        centroids
            Current mode-based centroid for every cluster.
        current_assignments
            Current partial cluster assignments.
        fixed_indices
            Row indices whose initial labels must remain unchanged.

        Returns
        -------
        tuple
            Provisional assignments, their scores, and a flag indicating
            whether at least one non-fixed object was evaluated.

        Raises
        ------
        ValueError
            If no centroids are available.
        RuntimeError
            If no valid score can be calculated for an object.
        """
        if not centroids:
            raise ValueError("At least one cluster centroid is required.")

        assignments: ClusterAssignments = [None] * len(self.df)
        scores: DistanceValues = [float("inf")] * len(self.df)
        fixed_index_set = set(fixed_indices)
        evaluated_candidate = False

        for sample_index in range(len(self.df)):
            if sample_index in fixed_index_set:
                assignments[sample_index] = current_assignments[sample_index]
                scores[sample_index] = float("-inf")
                continue

            best_cluster: ClusterLabel | None = None
            best_score = float("inf")

            for cluster_label, centroid in centroids.items():
                cluster_score = self.calculate_cluster_score(
                    sample_index=sample_index,
                    cluster_assignments=current_assignments,
                    centroid=centroid,
                    cluster_label=cluster_label,
                )
                if cluster_score < best_score:
                    best_score = cluster_score
                    best_cluster = cluster_label

            if best_cluster is None:
                raise RuntimeError(
                    "No valid cluster score could be calculated for object "
                    f"{sample_index}."
                )

            assignments[sample_index] = best_cluster
            scores[sample_index] = best_score
            evaluated_candidate = True

        return assignments, scores, evaluated_candidate

    def label_data(
        self,
        initial_clusters: Mapping[
            ClusterLabel,
            Sequence[int] | NDArray[np.integer],
        ],
        expansion_rate: float,
    ) -> NDArray[Any]:
        """Run CPLICE and return a cluster label for every input object.

        Parameters
        ----------
        initial_clusters
            Mapping from cluster labels to indices of initially labeled
            objects. Initial labels remain fixed throughout the algorithm.
        expansion_rate
            Positive initial expansion fraction. The same value is added after
            every iteration, preserving the schedule of the original
            implementation.

        Returns
        -------
        numpy.ndarray
            Final cluster assignment for every row.

        Raises
        ------
        ValueError
            If ``expansion_rate`` is not positive, no initial clusters are
            supplied, an initial cluster is empty, an index is invalid, or one
            object is assigned to more than one initial cluster.
        """
        if expansion_rate <= 0:
            raise ValueError("'expansion_rate' must be greater than zero.")
        if not initial_clusters:
            raise ValueError(
                "'initial_clusters' must contain at least one cluster."
            )

        normalized_clusters = self._normalize_initial_clusters(
            initial_clusters
        )
        centroids = self.compute_centroids(normalized_clusters)

        cluster_assignments: ClusterAssignments = [None] * len(self.df)
        initial_cluster_sizes: dict[ClusterLabel, int] = {}

        for cluster_label, sample_indices in normalized_clusters.items():
            initial_cluster_sizes[cluster_label] = len(sample_indices)
            for sample_index in sample_indices:
                cluster_assignments[int(sample_index)] = cluster_label

        fixed_indices = {
            int(sample_index)
            for sample_indices in normalized_clusters.values()
            for sample_index in sample_indices
        }

        current_expansion_fraction = expansion_rate

        while any(
            assignment is None for assignment in cluster_assignments
        ):
            (
                proposed_assignments,
                scores,
                evaluated_candidate,
            ) = self.assign_to_best_clusters(
                centroids=centroids,
                current_assignments=cluster_assignments,
                fixed_indices=fixed_indices,
            )

            cluster_assignments = self.select_best_candidates(
                proposed_assignments=proposed_assignments,
                scores=scores,
                expansion_fraction=current_expansion_fraction,
                initial_cluster_sizes=initial_cluster_sizes,
            )
            current_expansion_fraction += expansion_rate

            new_centroids: CentroidDictionary = {}
            assignment_array = np.asarray(
                cluster_assignments,
                dtype=object,
            )

            for cluster_label in normalized_clusters:
                cluster_data = self.df[
                    assignment_array == cluster_label
                ]
                if cluster_data.size > 0:
                    new_centroids[cluster_label] = calculate_mode(
                        cluster_data
                    )

            if not evaluated_candidate:
                break

            centroids = new_centroids

        return np.asarray(cluster_assignments, dtype=object)

    def _get_distance_matrix(self) -> DistanceMatrix:
        """Return the pairwise matrix, calculating it lazily if necessary."""
        if self.distance_matrix is None:
            self.distance_matrix = self.compute_distance_matrix()
        return self.distance_matrix

    def _validate_distance_matrix(
        self,
        distance_matrix: NDArray[np.floating[Any]],
    ) -> DistanceMatrix:
        """Validate and normalize a user-provided distance matrix."""
        matrix = np.asarray(distance_matrix, dtype=np.float64)
        expected_shape = (len(self.df), len(self.df))

        if matrix.shape != expected_shape:
            raise ValueError(
                "'distance_matrix' must have shape "
                f"{expected_shape}, received {matrix.shape}."
            )

        return matrix

    def _validate_assignment_length(
        self,
        assignments: NDArray[Any],
    ) -> None:
        """Ensure that assignments match the number of input objects."""
        if len(assignments) != len(self.df):
            raise ValueError(
                "Assignments must contain one value for every row in 'df'; "
                f"expected {len(self.df)}, received {len(assignments)}."
            )

    def _validate_sample_index(self, sample_index: int) -> None:
        """Ensure that a sample index points to an existing object."""
        if not 0 <= sample_index < len(self.df):
            raise IndexError(
                f"Sample index {sample_index} is outside the input data."
            )

    def _normalize_initial_clusters(
        self,
        initial_clusters: Mapping[
            ClusterLabel,
            Sequence[int] | NDArray[np.integer],
        ],
    ) -> dict[ClusterLabel, NDArray[np.int_]]:
        """Validate initial cluster indices and return integer arrays."""
        normalized: dict[ClusterLabel, NDArray[np.int_]] = {}
        seen_indices: set[int] = set()

        for cluster_label, sample_indices in initial_clusters.items():
            index_array = np.asarray(sample_indices, dtype=int).reshape(-1)

            if index_array.size == 0:
                raise ValueError(
                    f"Initial cluster {cluster_label!r} cannot be empty."
                )
            if len(set(index_array.tolist())) != len(index_array):
                raise ValueError(
                    f"Initial cluster {cluster_label!r} contains duplicate "
                    "indices."
                )
            if np.any(index_array < 0) or np.any(index_array >= len(self.df)):
                raise ValueError(
                    f"Initial cluster {cluster_label!r} contains an index "
                    "outside the input data."
                )

            duplicate_indices = seen_indices.intersection(
                int(index) for index in index_array
            )
            if duplicate_indices:
                duplicates = ", ".join(
                    str(index) for index in sorted(duplicate_indices)
                )
                raise ValueError(
                    "An object cannot belong to multiple initial clusters. "
                    f"Duplicated indices: {duplicates}."
                )

            seen_indices.update(int(index) for index in index_array)
            normalized[cluster_label] = index_array

        return normalized

    @staticmethod
    def _validate_strategy(strategy: str) -> StrategyName:
        """Validate and return a cluster scoring strategy."""
        if strategy not in SUPPORTED_STRATEGIES:
            supported = ", ".join(sorted(SUPPORTED_STRATEGIES))
            raise ValueError(
                f"Unsupported strategy {strategy!r}. "
                f"Supported strategies: {supported}."
            )

        return cast(StrategyName, strategy)
