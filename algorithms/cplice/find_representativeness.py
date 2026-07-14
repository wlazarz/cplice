"""Representative-object selection for categorical datasets.

The module contains experimental strategies for selecting initially labeled
objects in pseudo-labeling studies. The strategies are intentionally kept
separate because they optimize different notions of representativeness:

- diversity across subclusters,
- similarity to the own class prototype and separation from other classes,
- medoid-like centrality within a class,
- truncation of a previously computed ranking.

Categorical dissimilarity is measured as the proportion of mismatching
features, equivalent to normalized Hamming distance.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

ClassLabel: TypeAlias = Hashable
SelectionDictionary: TypeAlias = dict[ClassLabel, list[int]]
CategoricalArray: TypeAlias = NDArray[Any]
LabelArray: TypeAlias = NDArray[np.object_]


@dataclass(frozen=True, slots=True)
class KModesSelectionResult:
    """Result of selecting the number of K-Modes clusters."""

    labels: NDArray[np.int_]
    centroids: CategoricalArray
    silhouette_score: float
    number_of_clusters: int


def categorical_silhouette_score(
    data: ArrayLike,
    labels: ArrayLike,
    *,
    distance_matrix: NDArray[np.floating[Any]] | None = None,
    chunk_size: int = 256,
) -> float:
    """Calculate a silhouette score for categorical data.

    Unlike the original implementation, the within-cluster distance excludes
    the evaluated object itself. A singleton cluster receives a silhouette of
    zero, following the convention used by common silhouette implementations.

    Parameters
    ----------
    data
        Two-dimensional categorical feature matrix.
    labels
        One class or cluster label per row.
    distance_matrix
        Optional precomputed normalized Hamming distance matrix.
    chunk_size
        Number of rows processed at once when the distance matrix is built.

    Returns
    -------
    float
        Mean categorical silhouette score.

    Raises
    ------
    ValueError
        If fewer than two clusters are present or input dimensions are
        inconsistent.
    """
    feature_matrix = _as_2d_array(data, "data")
    label_array = _as_label_array(labels, len(feature_matrix))

    unique_labels = pd.unique(label_array)
    if len(unique_labels) < 2:
        raise ValueError(
            "The silhouette score requires at least two clusters."
        )
    if len(unique_labels) >= len(feature_matrix):
        raise ValueError(
            "The silhouette score requires fewer clusters than objects."
        )

    if distance_matrix is None:
        distances = pairwise_categorical_hamming(
            feature_matrix,
            chunk_size=chunk_size,
        )
    else:
        distances = _validate_distance_matrix(
            distance_matrix,
            len(feature_matrix),
        )

    sample_scores = np.zeros(len(feature_matrix), dtype=float)

    cluster_indices = {
        label: np.where(label_array == label)[0]
        for label in unique_labels
    }

    for label, own_indices in cluster_indices.items():
        if len(own_indices) == 1:
            sample_scores[own_indices[0]] = 0.0
            continue

        within_distances = distances[np.ix_(own_indices, own_indices)]
        within_means = within_distances.sum(axis=1) / (
            len(own_indices) - 1
        )

        nearest_other_means = np.full(
            len(own_indices),
            np.inf,
            dtype=float,
        )

        for other_label, other_indices in cluster_indices.items():
            if other_label == label:
                continue

            between_means = distances[
                np.ix_(own_indices, other_indices)
            ].mean(axis=1)
            nearest_other_means = np.minimum(
                nearest_other_means,
                between_means,
            )

        denominators = np.maximum(
            within_means,
            nearest_other_means,
        )
        cluster_scores = np.divide(
            nearest_other_means - within_means,
            denominators,
            out=np.zeros_like(within_means),
            where=denominators > 0,
        )
        sample_scores[own_indices] = cluster_scores

    return float(sample_scores.mean())


def find_optimal_kmodes(
    data: ArrayLike,
    *,
    minimum_clusters: int = 2,
    maximum_clusters: int = 10,
    initialization: str = "Huang",
    number_of_initializations: int = 5,
    random_state: int | None = 42,
    chunk_size: int = 256,
) -> KModesSelectionResult:
    """Select the K-Modes solution with the best categorical silhouette.

    The candidate upper bound is restricted by the number of rows and the
    number of distinct categorical objects. A pairwise distance matrix is
    computed only once and reused for every candidate value of ``k``.

    Parameters
    ----------
    data
        Two-dimensional categorical feature matrix.
    minimum_clusters
        Smallest candidate number of clusters.
    maximum_clusters
        Largest candidate number of clusters.
    initialization
        Initialization method passed to ``kmodes.kmodes.KModes``.
    number_of_initializations
        Number of K-Modes initializations for each candidate.
    random_state
        Random state passed to K-Modes.
    chunk_size
        Chunk size used when calculating categorical distances.

    Returns
    -------
    KModesSelectionResult
        Labels, centroids, score, and selected number of clusters.

    Raises
    ------
    ImportError
        If the optional ``kmodes`` dependency is not installed.
    ValueError
        If the data do not support at least two non-degenerate clusters.
    """
    feature_matrix = _as_2d_array(data, "data")
    _validate_positive_integer(
        minimum_clusters,
        "minimum_clusters",
    )
    _validate_positive_integer(
        maximum_clusters,
        "maximum_clusters",
    )
    _validate_positive_integer(
        number_of_initializations,
        "number_of_initializations",
    )

    if minimum_clusters < 2:
        raise ValueError("'minimum_clusters' must be at least 2.")
    if maximum_clusters < minimum_clusters:
        raise ValueError(
            "'maximum_clusters' cannot be smaller than "
            "'minimum_clusters'."
        )

    sample_count = len(feature_matrix)
    distinct_count = len(
        pd.DataFrame(feature_matrix).drop_duplicates()
    )
    maximum_valid_clusters = min(
        maximum_clusters,
        sample_count - 1,
        distinct_count,
    )

    if maximum_valid_clusters < minimum_clusters:
        raise ValueError(
            "The data do not contain enough distinct objects to evaluate "
            "the requested K-Modes range."
        )

    kmodes_class = _load_kmodes()
    distance_matrix = pairwise_categorical_hamming(
        feature_matrix,
        chunk_size=chunk_size,
    )

    best_result: KModesSelectionResult | None = None

    for number_of_clusters in range(
        minimum_clusters,
        maximum_valid_clusters + 1,
    ):
        model = kmodes_class(
            n_clusters=number_of_clusters,
            init=initialization,
            n_init=number_of_initializations,
            verbose=0,
            random_state=random_state,
        )
        cluster_labels = np.asarray(
            model.fit_predict(feature_matrix),
            dtype=int,
        )

        if len(np.unique(cluster_labels)) < 2:
            continue

        score = categorical_silhouette_score(
            feature_matrix,
            cluster_labels,
            distance_matrix=distance_matrix,
        )

        candidate = KModesSelectionResult(
            labels=cluster_labels,
            centroids=np.asarray(model.cluster_centroids_),
            silhouette_score=score,
            number_of_clusters=number_of_clusters,
        )

        if (
            best_result is None
            or candidate.silhouette_score
            > best_result.silhouette_score
        ):
            best_result = candidate

    if best_result is None:
        raise ValueError(
            "K-Modes did not produce a valid clustering solution."
        )

    return best_result


def select_diverse_representatives(
    data: ArrayLike,
    number_of_points: int = 50,
    *,
    maximum_clusters: int = 10,
    random_state: int | None = 42,
    chunk_size: int = 256,
) -> list[int]:
    """Select central objects while balancing discovered subclusters.

    The data are first partitioned with K-Modes. Objects within each discovered
    subcluster are ordered by distance to its centroid. Selection then proceeds
    in round-robin order across subclusters, preserving diversity rather than
    simply choosing the globally most central objects.

    For fewer than three rows, or when all rows are identical, the method falls
    back to medoid-centrality ranking.

    Parameters
    ----------
    data
        Categorical objects belonging to one known class.
    number_of_points
        Maximum number of representatives to select.
    maximum_clusters
        Maximum number of internal K-Modes clusters.
    random_state
        Random state passed to K-Modes.
    chunk_size
        Chunk size for pairwise distance calculations.

    Returns
    -------
    list
        Row indices relative to ``data``.
    """
    feature_matrix = _as_2d_array(data, "data")
    _validate_positive_integer(number_of_points, "number_of_points")

    target_count = min(number_of_points, len(feature_matrix))
    if target_count == 0:
        return []

    distinct_count = len(
        pd.DataFrame(feature_matrix).drop_duplicates()
    )
    if len(feature_matrix) < 3 or distinct_count < 2:
        return rank_by_medoid_centrality(
            feature_matrix,
            number_of_points=target_count,
            chunk_size=chunk_size,
        )

    result = find_optimal_kmodes(
        feature_matrix,
        maximum_clusters=maximum_clusters,
        random_state=random_state,
        chunk_size=chunk_size,
    )

    queues: dict[int, deque[int]] = {}
    cluster_order = sorted(np.unique(result.labels).tolist())

    for cluster_label in cluster_order:
        local_indices = np.where(
            result.labels == cluster_label
        )[0]
        centroid = result.centroids[cluster_label]
        distances = categorical_distance_to_centroid(
            feature_matrix[local_indices],
            centroid,
        )
        ordered_indices = local_indices[
            np.argsort(distances, kind="stable")
        ]
        queues[cluster_label] = deque(
            int(index) for index in ordered_indices
        )

    selected: list[int] = []

    while len(selected) < target_count:
        selected_in_round = False

        for cluster_label in cluster_order:
            queue = queues[cluster_label]
            if not queue:
                continue

            selected.append(queue.popleft())
            selected_in_round = True

            if len(selected) == target_count:
                break

        if not selected_in_round:
            break

    return selected


def select_representatives_by_subclustering(
    data: ArrayLike,
    class_labels: ArrayLike,
    number_per_class: int,
    *,
    maximum_clusters: int = 10,
    random_state: int | None = 42,
    chunk_size: int = 256,
) -> SelectionDictionary:
    """Select diverse representatives independently inside every class."""
    feature_matrix = _as_2d_array(data, "data")
    label_array = _as_label_array(
        class_labels,
        len(feature_matrix),
    )
    _validate_positive_integer(number_per_class, "number_per_class")

    selections: SelectionDictionary = {}

    for class_label in pd.unique(label_array):
        global_indices = np.where(
            label_array == class_label
        )[0]
        local_selection = select_diverse_representatives(
            feature_matrix[global_indices],
            number_of_points=number_per_class,
            maximum_clusters=maximum_clusters,
            random_state=random_state,
            chunk_size=chunk_size,
        )
        selections[class_label] = [
            int(global_indices[local_index])
            for local_index in local_selection
        ]

    return selections


def categorical_similarity_to_centroid(
    data: ArrayLike,
    centroid: ArrayLike,
) -> NDArray[np.float64]:
    """Return the proportion of matching features for every object."""
    feature_matrix = _as_2d_array(data, "data")
    centroid_array = np.asarray(centroid, dtype=object).reshape(-1)

    if feature_matrix.shape[1] != len(centroid_array):
        raise ValueError(
            "'centroid' must contain one value per feature."
        )

    return np.mean(
        feature_matrix == centroid_array,
        axis=1,
        dtype=float,
    )


def compute_class_modes(
    data: ArrayLike,
    class_labels: ArrayLike,
) -> dict[ClassLabel, CategoricalArray]:
    """Compute a mode-based prototype for every class."""
    feature_matrix = _as_2d_array(data, "data")
    label_array = _as_label_array(
        class_labels,
        len(feature_matrix),
    )

    modes: dict[ClassLabel, CategoricalArray] = {}

    for class_label in pd.unique(label_array):
        class_data = feature_matrix[
            label_array == class_label
        ]
        modes[class_label] = _column_modes(class_data)

    return modes


def select_representatives_by_centroid_contrast(
    data: ArrayLike,
    class_labels: ArrayLike,
    number_per_class: int,
    *,
    competing_similarity: Literal["mean", "maximum"] = "mean",
) -> SelectionDictionary:
    """Select objects typical for their class and atypical for other classes.

    The score is the similarity to the object's own class mode minus an
    aggregate similarity to competing class modes.

    Parameters
    ----------
    data
        Two-dimensional categorical feature matrix.
    class_labels
        Known class label for every row.
    number_per_class
        Maximum number of representatives selected per class.
    competing_similarity
        ``"mean"`` subtracts the average similarity to other class modes.
        ``"maximum"`` subtracts the strongest competing similarity and is
        therefore a stricter boundary-avoidance criterion.

    Returns
    -------
    dict
        Class labels mapped to globally indexed representative rows.
    """
    feature_matrix = _as_2d_array(data, "data")
    label_array = _as_label_array(
        class_labels,
        len(feature_matrix),
    )
    _validate_positive_integer(number_per_class, "number_per_class")

    if competing_similarity not in {"mean", "maximum"}:
        raise ValueError(
            "'competing_similarity' must be 'mean' or 'maximum'."
        )

    modes = compute_class_modes(feature_matrix, label_array)
    class_order = list(modes)
    similarity_matrix = np.column_stack(
        [
            categorical_similarity_to_centroid(
                feature_matrix,
                modes[class_label],
            )
            for class_label in class_order
        ]
    )
    class_positions = {
        class_label: position
        for position, class_label in enumerate(class_order)
    }

    selections: SelectionDictionary = {}

    for class_label in class_order:
        class_indices = np.where(
            label_array == class_label
        )[0]
        own_position = class_positions[class_label]
        own_similarity = similarity_matrix[
            class_indices,
            own_position,
        ]

        other_positions = [
            position
            for position in range(len(class_order))
            if position != own_position
        ]

        if not other_positions:
            competing_scores = np.zeros(
                len(class_indices),
                dtype=float,
            )
        elif competing_similarity == "mean":
            competing_scores = similarity_matrix[
                np.ix_(class_indices, other_positions)
            ].mean(axis=1)
        else:
            competing_scores = similarity_matrix[
                np.ix_(class_indices, other_positions)
            ].max(axis=1)

        representativeness = own_similarity - competing_scores
        order = np.argsort(
            -representativeness,
            kind="stable",
        )
        selected_indices = class_indices[
            order[: min(number_per_class, len(class_indices))]
        ]
        selections[class_label] = selected_indices.astype(int).tolist()

    return selections


def rank_by_medoid_centrality(
    data: ArrayLike,
    number_of_points: int,
    *,
    chunk_size: int = 256,
) -> list[int]:
    """Rank objects by their total within-set Hamming distance.

    Lower total distance means greater medoid-like centrality. Distances are
    accumulated in chunks to avoid constructing a full three-dimensional
    comparison tensor.
    """
    feature_matrix = _as_2d_array(data, "data")
    _validate_positive_integer(number_of_points, "number_of_points")
    _validate_positive_integer(chunk_size, "chunk_size")

    target_count = min(number_of_points, len(feature_matrix))
    if target_count == 0:
        return []

    distance_sums = _categorical_distance_row_sums(
        feature_matrix,
        chunk_size=chunk_size,
    )
    order = np.argsort(distance_sums, kind="stable")
    return order[:target_count].astype(int).tolist()


def select_representatives_by_medoid(
    data: ArrayLike,
    class_labels: ArrayLike,
    number_per_class: int = 200,
    *,
    chunk_size: int = 256,
) -> SelectionDictionary:
    """Select medoid-central representatives independently per class."""
    feature_matrix = _as_2d_array(data, "data")
    label_array = _as_label_array(
        class_labels,
        len(feature_matrix),
    )
    _validate_positive_integer(number_per_class, "number_per_class")

    selections: SelectionDictionary = {}

    for class_label in pd.unique(label_array):
        global_indices = np.where(
            label_array == class_label
        )[0]
        local_indices = rank_by_medoid_centrality(
            feature_matrix[global_indices],
            number_of_points=number_per_class,
            chunk_size=chunk_size,
        )
        selections[class_label] = [
            int(global_indices[local_index])
            for local_index in local_indices
        ]

    return selections


def truncate_representative_ranking(
    representative_ranking: Mapping[
        ClassLabel,
        Sequence[int],
    ],
    class_labels: ArrayLike,
    selection_size: int | float,
    *,
    minimum_per_class: int = 2,
) -> SelectionDictionary:
    """Truncate an existing per-class representative ranking.

    Parameters
    ----------
    representative_ranking
        Ordered row indices for every class, from most to least
        representative.
    class_labels
        Class label for every row in the original dataset.
    selection_size
        Positive integer for an absolute count, or a float in ``(0, 1]`` for
        a fraction of each class.
    minimum_per_class
        Minimum selected count when ``selection_size`` is a fraction. The
        result is still capped by class size and available ranking length.

    Returns
    -------
    dict
        Truncated and validated representative rankings.
    """
    label_array = np.asarray(class_labels, dtype=object)
    if label_array.ndim != 1:
        raise ValueError("'class_labels' must be one-dimensional.")

    _validate_non_negative_integer(
        minimum_per_class,
        "minimum_per_class",
    )

    is_absolute = (
        isinstance(selection_size, int)
        and not isinstance(selection_size, bool)
    )
    is_fraction = isinstance(
        selection_size,
        (float, np.floating),
    )

    if is_absolute:
        if selection_size <= 0:
            raise ValueError(
                "An integer 'selection_size' must be positive."
            )
    elif is_fraction:
        if (
            not math.isfinite(float(selection_size))
            or selection_size <= 0
            or selection_size > 1
        ):
            raise ValueError(
                "A fractional 'selection_size' must be in (0, 1]."
            )
    else:
        raise TypeError(
            "'selection_size' must be a positive integer or a fraction."
        )

    result: SelectionDictionary = {}

    for class_label, ranked_indices in representative_ranking.items():
        class_size = int(np.sum(label_array == class_label))
        if class_size == 0:
            raise ValueError(
                f"Class {class_label!r} does not occur in 'class_labels'."
            )

        validated_ranking: list[int] = []
        seen_indices: set[int] = set()

        for raw_index in ranked_indices:
            index = int(raw_index)

            if index < 0 or index >= len(label_array):
                raise ValueError(
                    f"Ranking for class {class_label!r} contains index "
                    f"{index}, which is outside 'class_labels'."
                )
            if label_array[index] != class_label:
                raise ValueError(
                    f"Index {index} does not belong to class "
                    f"{class_label!r}."
                )
            if index not in seen_indices:
                validated_ranking.append(index)
                seen_indices.add(index)

        if is_absolute:
            target_count = int(selection_size)
        else:
            target_count = max(
                minimum_per_class,
                math.ceil(class_size * float(selection_size)),
            )

        target_count = min(
            target_count,
            class_size,
            len(validated_ranking),
        )
        result[class_label] = validated_ranking[:target_count]

    return result


def pairwise_categorical_hamming(
    data: ArrayLike,
    other_data: ArrayLike | None = None,
    *,
    chunk_size: int = 256,
) -> NDArray[np.float64]:
    """Compute normalized categorical Hamming distances in chunks."""
    left = _as_2d_array(data, "data")
    right = (
        left
        if other_data is None
        else _as_2d_array(other_data, "other_data")
    )
    _validate_positive_integer(chunk_size, "chunk_size")

    if left.shape[1] != right.shape[1]:
        raise ValueError(
            "'data' and 'other_data' must have the same number of features."
        )

    distances = np.empty(
        (len(left), len(right)),
        dtype=float,
    )

    for start in range(0, len(left), chunk_size):
        stop = min(start + chunk_size, len(left))
        distances[start:stop] = np.mean(
            left[start:stop, None, :] != right[None, :, :],
            axis=2,
            dtype=float,
        )

    return distances


def categorical_distance_to_centroid(
    data: ArrayLike,
    centroid: ArrayLike,
) -> NDArray[np.float64]:
    """Return normalized Hamming distance to a categorical centroid."""
    return 1.0 - categorical_similarity_to_centroid(
        data,
        centroid,
    )


def _categorical_distance_row_sums(
    data: CategoricalArray,
    *,
    chunk_size: int,
) -> NDArray[np.float64]:
    row_sums = np.empty(len(data), dtype=float)

    for start in range(0, len(data), chunk_size):
        stop = min(start + chunk_size, len(data))
        distances = np.mean(
            data[start:stop, None, :] != data[None, :, :],
            axis=2,
            dtype=float,
        )
        row_sums[start:stop] = distances.sum(axis=1)

    return row_sums


def _column_modes(data: CategoricalArray) -> CategoricalArray:
    if len(data) == 0:
        raise ValueError("Cannot calculate a mode for an empty array.")

    modes: list[Any] = []

    for column_index in range(data.shape[1]):
        values, counts = np.unique(
            data[:, column_index],
            return_counts=True,
        )
        modes.append(values[int(np.argmax(counts))])

    return np.asarray(modes, dtype=object)


def _as_2d_array(data: ArrayLike, name: str) -> CategoricalArray:
    array = np.asarray(data, dtype=object)

    if array.ndim != 2:
        raise ValueError(f"'{name}' must be a two-dimensional array.")
    if array.shape[0] == 0:
        raise ValueError(f"'{name}' must contain at least one row.")
    if array.shape[1] == 0:
        raise ValueError(f"'{name}' must contain at least one feature.")

    return array


def _as_label_array(
    labels: ArrayLike,
    expected_length: int,
) -> LabelArray:
    array = np.asarray(labels, dtype=object)

    if array.ndim != 1:
        raise ValueError("'labels' must be one-dimensional.")
    if len(array) != expected_length:
        raise ValueError(
            "'labels' must contain one value for every row in 'data'."
        )

    return array


def _validate_distance_matrix(
    distance_matrix: NDArray[np.floating[Any]],
    sample_count: int,
) -> NDArray[np.float64]:
    matrix = np.asarray(distance_matrix, dtype=float)
    expected_shape = (sample_count, sample_count)

    if matrix.shape != expected_shape:
        raise ValueError(
            f"'distance_matrix' must have shape {expected_shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "'distance_matrix' must contain only finite values."
        )
    if np.any(matrix < 0):
        raise ValueError(
            "'distance_matrix' cannot contain negative values."
        )

    return matrix


def _validate_positive_integer(value: int, name: str) -> None:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(f"'{name}' must be a positive integer.")


def _validate_non_negative_integer(value: int, name: str) -> None:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, bool)
        or value < 0
    ):
        raise ValueError(
            f"'{name}' must be a non-negative integer."
        )


def _load_kmodes() -> Any:
    try:
        from kmodes.kmodes import KModes
    except ImportError as error:
        raise ImportError(
            "K-Modes strategies require the optional 'kmodes' package. "
            "Install it with: pip install kmodes"
        ) from error

    return KModes


# ---------------------------------------------------------------------------
# Backward-compatible wrappers for the original research scripts.
# New code should use the descriptive function names above.
# ---------------------------------------------------------------------------

def categorical_silhouette(
    X: ArrayLike,
    labels: ArrayLike,
    metric: str = "hamming",
) -> float:
    """Backward-compatible wrapper for categorical_silhouette_score."""
    if metric != "hamming":
        raise ValueError("Only normalized Hamming distance is supported.")
    return categorical_silhouette_score(X, labels)


def optimal_kmodes(
    X: ArrayLike,
    metric: str = "hamming",
) -> tuple[NDArray[np.int_], CategoricalArray]:
    """Backward-compatible wrapper returning labels and centroids."""
    if metric != "hamming":
        raise ValueError("Only normalized Hamming distance is supported.")
    result = find_optimal_kmodes(X)
    return result.labels, result.centroids


def select_representative_for_label(
    X: ArrayLike,
    num_points: int = 50,
) -> list[int]:
    """Backward-compatible wrapper for diverse representative selection."""
    return select_diverse_representatives(
        X,
        number_of_points=num_points,
    )


def compute_representativeness_by_clustering(
    df: ArrayLike,
    labels: ArrayLike,
    top_n: int,
) -> SelectionDictionary:
    """Backward-compatible wrapper for subclustering selection."""
    return select_representatives_by_subclustering(
        df,
        labels,
        number_per_class=top_n,
    )


def similarity_to_centroid(
    data: ArrayLike,
    centroid: ArrayLike,
) -> NDArray[np.float64]:
    """Backward-compatible wrapper for centroid similarity."""
    return categorical_similarity_to_centroid(data, centroid)


def compute_cluster_modes(
    df: ArrayLike,
    labels: ArrayLike,
) -> dict[ClassLabel, CategoricalArray]:
    """Backward-compatible wrapper for class-mode calculation."""
    return compute_class_modes(df, labels)


def compute_representativeness(
    df: ArrayLike,
    labels: ArrayLike,
    top_n: int,
) -> SelectionDictionary:
    """Backward-compatible wrapper for centroid-contrast selection."""
    return select_representatives_by_centroid_contrast(
        df,
        labels,
        number_per_class=top_n,
    )


def get_representative_objects_3(
    df: ArrayLike,
    classes: ArrayLike,
    n: int = 200,
) -> SelectionDictionary:
    """Backward-compatible wrapper for medoid selection."""
    return select_representatives_by_medoid(
        df,
        classes,
        number_per_class=n,
    )


def get_top_representativeness(
    representativeness: Mapping[
        ClassLabel,
        Sequence[int],
    ],
    labels: ArrayLike,
    top_n: int | float,
) -> SelectionDictionary:
    """Backward-compatible wrapper for ranking truncation."""
    return truncate_representative_ranking(
        representativeness,
        labels,
        selection_size=top_n,
    )