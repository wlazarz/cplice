"""External evaluation metrics for CPLICE predictions.

The module separates two different evaluation perspectives:

1. Classification metrics, which assume that predicted labels have the same
   semantic meaning as the reference labels.
2. Clustering metrics, which are invariant to a permutation of cluster names.

For seeded CPLICE, classification metrics are usually meaningful because the
initial labels preserve class identities. For arbitrary clustering methods,
use permutation-invariant metrics or enable optional label alignment.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    cohen_kappa_score,
    f1_score,
    fowlkes_mallows_score,
    hamming_loss as sklearn_hamming_loss,
    homogeneity_completeness_v_measure,
    jaccard_score,
    matthews_corrcoef as sklearn_matthews_corrcoef,
    mutual_info_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

ClassLabel: TypeAlias = Hashable
AverageName: TypeAlias = Literal[
    "binary",
    "micro",
    "macro",
    "weighted",
]
EntropyAverage: TypeAlias = Literal["weighted", "macro"]


@dataclass(frozen=True, slots=True)
class ExternalEvaluationReport:
    """Structured external evaluation of predicted labels."""

    classification: Mapping[str, float | None]
    clustering: Mapping[str, float]
    distribution: Mapping[str, Mapping[str, int | float]]
    sample_count: int
    excluded_count: int
    label_mapping: Mapping[ClassLabel, ClassLabel] | None = None

    def to_flat_dict(self) -> dict[str, Any]:
        """Return a flat dictionary convenient for CSV result tables."""
        result: dict[str, Any] = {
            **self.classification,
            **self.clustering,
            "sample_count": self.sample_count,
            "excluded_count": self.excluded_count,
            "distribution": dict(self.distribution),
        }
        if self.label_mapping is not None:
            result["label_mapping"] = dict(self.label_mapping)
        return result


def evaluate_external_labels(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    y_score: ArrayLike | None = None,
    score_labels: Sequence[ClassLabel] | None = None,
    average: AverageName = "macro",
    zero_division: int | float = 0,
    ignored_labels: Iterable[ClassLabel] | None = None,
    ignore_missing: bool = True,
    align_predicted_labels: bool = False,
    include_classification_metrics: bool = True,
    entropy_average: EntropyAverage = "weighted",
) -> ExternalEvaluationReport:
    """Evaluate CPLICE predictions against known reference labels.

    Parameters
    ----------
    y_true
        Reference class labels.
    y_pred
        Labels predicted by CPLICE.
    y_score
        Optional confidence scores or class probabilities. A one-dimensional
        array is accepted for binary classification. Multiclass evaluation
        requires a two-dimensional array with one column per class.
    score_labels
        Class order corresponding to columns of ``y_score``. Passing
        ``model.classes_`` is recommended.
    average
        Averaging strategy for precision, recall, F1, and Jaccard.
    zero_division
        Value used by scikit-learn when a class has no predicted or true
        samples for a metric denominator.
    ignored_labels
        Labels excluded from evaluation, for example ``[-1]`` for outliers.
        A row is excluded if either its true or predicted label is ignored.
    ignore_missing
        Exclude rows whose true or predicted label is ``None`` or ``NaN``.
    align_predicted_labels
        Align cluster identifiers to reference labels with the Hungarian
        algorithm before classification metrics. Do not enable this for the
        main seeded-CPLICE experiment unless label identities are genuinely
        arbitrary.
    include_classification_metrics
        Disable when predicted labels are pure cluster identifiers with no
        semantic correspondence to reference classes.
    entropy_average
        ``"weighted"`` weights cluster entropies by cluster size.
        ``"macro"`` gives every predicted cluster equal weight.

    Returns
    -------
    ExternalEvaluationReport
        Structured classification, clustering, and distribution metrics.
    """
    prepared = _prepare_inputs(
        y_true,
        y_pred,
        ignored_labels=ignored_labels,
        ignore_missing=ignore_missing,
    )
    true_labels = prepared.true_labels
    predicted_labels = prepared.predicted_labels

    score_array: NDArray[np.float64] | None = None
    if y_score is not None:
        raw_scores = np.asarray(y_score, dtype=np.float64)
        if raw_scores.ndim not in {1, 2}:
            raise ValueError(
                "'y_score' must be one- or two-dimensional."
            )
        if len(raw_scores) != prepared.original_count:
            raise ValueError(
                "'y_score' must contain one row per original label."
            )
        score_array = raw_scores[prepared.included_mask]

    label_mapping: dict[ClassLabel, ClassLabel] | None = None
    labels_for_classification = predicted_labels

    if align_predicted_labels:
        (
            labels_for_classification,
            label_mapping,
        ) = align_cluster_labels(true_labels, predicted_labels)

    classification_metrics: dict[str, float | None] = {}
    if include_classification_metrics:
        classification_metrics = _classification_metrics(
            true_labels,
            labels_for_classification,
            y_score=score_array,
            score_labels=score_labels,
            average=average,
            zero_division=zero_division,
        )

    clustering_metrics = _clustering_metrics(
        true_labels,
        predicted_labels,
        entropy_average=entropy_average,
    )

    distribution = label_distribution(
        predicted_labels,
        true_labels,
        normalize=False,
        stringify_keys=True,
    )

    return ExternalEvaluationReport(
        classification=classification_metrics,
        clustering=clustering_metrics,
        distribution=distribution,
        sample_count=len(true_labels),
        excluded_count=prepared.original_count - len(true_labels),
        label_mapping=label_mapping,
    )


def shannon_entropy(
    sequence: ArrayLike,
    *,
    base: float = math.e,
) -> float:
    """Calculate Shannon entropy of a one-dimensional sequence."""
    values = np.asarray(sequence, dtype=object)
    if values.ndim != 1:
        raise ValueError("'sequence' must be one-dimensional.")
    if len(values) == 0:
        return 0.0
    if not math.isfinite(base) or base <= 0 or base == 1:
        raise ValueError("'base' must be positive and different from 1.")

    counts = np.asarray(
        list(Counter(_label_key(value) for value in values).values()),
        dtype=np.float64,
    )
    probabilities = counts / counts.sum()
    entropy_value = -np.sum(
        probabilities * np.log(probabilities)
    )
    return float(entropy_value / math.log(base))


def conditional_cluster_entropy(
    predicted_clusters: ArrayLike,
    true_labels: ArrayLike,
    *,
    average: EntropyAverage = "weighted",
    base: float = math.e,
) -> float:
    """Calculate reference-label entropy inside predicted clusters.

    Lower values mean that predicted clusters contain more homogeneous
    reference classes.

    ``weighted`` is recommended because it corresponds to conditional entropy
    ``H(Y_true | Y_pred)``. ``macro`` reproduces the equal-cluster weighting
    used by the original project.
    """
    clusters = _as_one_dimensional(
        predicted_clusters,
        "predicted_clusters",
    )
    labels = _as_one_dimensional(true_labels, "true_labels")

    if len(clusters) != len(labels):
        raise ValueError(
            "'predicted_clusters' and 'true_labels' must have equal length."
        )
    if average not in {"weighted", "macro"}:
        raise ValueError("'average' must be 'weighted' or 'macro'.")
    if len(clusters) == 0:
        return 0.0

    cluster_values = pd.unique(clusters)
    entropies: list[float] = []
    weights: list[int] = []

    for cluster in cluster_values:
        mask = _label_mask(clusters, cluster)
        cluster_labels = labels[mask]
        entropies.append(
            shannon_entropy(cluster_labels, base=base)
        )
        weights.append(len(cluster_labels))

    if average == "macro":
        return float(np.mean(entropies))
    return float(np.average(entropies, weights=weights))


def adjusted_rand_index(
    first_labels: ArrayLike,
    second_labels: ArrayLike,
) -> float:
    """Calculate the adjusted Rand index."""
    first, second = _validate_label_pair(
        first_labels,
        second_labels,
    )
    first_codes, second_codes = _encode_label_pair(first, second)
    return float(adjusted_rand_score(first_codes, second_codes))


def fowlkes_mallows_index(
    first_labels: ArrayLike,
    second_labels: ArrayLike,
) -> float:
    """Calculate the Fowlkes-Mallows index."""
    first, second = _validate_label_pair(
        first_labels,
        second_labels,
    )
    first_codes, second_codes = _encode_label_pair(first, second)
    return float(fowlkes_mallows_score(first_codes, second_codes))


def variation_of_information(
    first_labels: ArrayLike,
    second_labels: ArrayLike,
    *,
    base: float = math.e,
) -> float:
    """Calculate variation of information between two partitions.

    The result is label-permutation invariant and non-negative. Natural
    logarithms are used by default; set ``base=2`` for bits.
    """
    first, second = _validate_label_pair(
        first_labels,
        second_labels,
    )
    if not math.isfinite(base) or base <= 0 or base == 1:
        raise ValueError("'base' must be positive and different from 1.")

    first_entropy = shannon_entropy(first, base=math.e)
    second_entropy = shannon_entropy(second, base=math.e)
    first_codes, second_codes = _encode_label_pair(first, second)
    mutual_information = float(
        mutual_info_score(first_codes, second_codes)
    )
    value = max(
        first_entropy + second_entropy - 2.0 * mutual_information,
        0.0,
    )
    return float(value / math.log(base))


def normalized_mutual_information(
    first_labels: ArrayLike,
    second_labels: ArrayLike,
) -> float:
    """Calculate normalized mutual information."""
    first, second = _validate_label_pair(
        first_labels,
        second_labels,
    )
    first_codes, second_codes = _encode_label_pair(first, second)
    return float(
        normalized_mutual_info_score(
            first_codes,
            second_codes,
            average_method="arithmetic",
        )
    )


def label_distribution(
    predicted_labels: ArrayLike,
    true_labels: ArrayLike,
    *,
    normalize: bool = False,
    stringify_keys: bool = True,
) -> dict[Any, dict[Any, int | float]]:
    """Return the reference-label distribution inside each prediction group."""
    predicted, reference = _validate_label_pair(
        predicted_labels,
        true_labels,
    )
    result: dict[Any, dict[Any, int | float]] = {}

    for predicted_label in pd.unique(predicted):
        mask = _label_mask(predicted, predicted_label)
        counter = Counter(
            _label_key(value) for value in reference[mask]
        )
        total = sum(counter.values())

        outer_key: Any = (
            str(predicted_label)
            if stringify_keys
            else predicted_label
        )
        inner: dict[Any, int | float] = {}

        for key, count in counter.items():
            display_key = str(key[1]) if stringify_keys else key[1]
            inner[display_key] = (
                count / total if normalize and total > 0 else int(count)
            )

        result[outer_key] = inner

    return result


def multiclass_roc_auc(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    score_labels: Sequence[ClassLabel] | None = None,
    average: Literal["macro", "weighted", "micro"] = "macro",
) -> float:
    """Calculate ROC AUC from continuous scores or probabilities.

    Hard predicted labels are not accepted as a substitute for ``y_score``.

    For multiclass scores, pass ``score_labels=model.classes_`` so the score
    columns can be matched unambiguously to class labels.
    """
    true_labels = _as_one_dimensional(y_true, "y_true")
    scores = np.asarray(y_score, dtype=np.float64)

    if len(scores) != len(true_labels):
        raise ValueError(
            "'y_score' must contain one row per true label."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("'y_score' must contain only finite values.")

    observed_classes = list(pd.unique(true_labels))
    if len(observed_classes) < 2:
        raise ValueError("ROC AUC requires at least two classes.")

    classes = (
        list(score_labels)
        if score_labels is not None
        else observed_classes
    )
    class_positions = {
        _label_key(label): position
        for position, label in enumerate(classes)
    }

    missing_classes = [
        value
        for value in observed_classes
        if _label_key(value) not in class_positions
    ]
    if missing_classes:
        raise ValueError(
            "'score_labels' does not contain all observed true classes."
        )

    true_codes = np.asarray(
        [class_positions[_label_key(value)] for value in true_labels],
        dtype=int,
    )

    if scores.ndim == 1:
        if len(classes) != 2:
            raise ValueError(
                "One-dimensional scores are valid only for binary "
                "classification."
            )
        binary_true = (true_codes == 1).astype(int)
        return float(roc_auc_score(binary_true, scores))

    if scores.ndim != 2:
        raise ValueError("'y_score' must be one- or two-dimensional.")
    if scores.shape[1] != len(classes):
        raise ValueError(
            "The number of score columns must match 'score_labels'."
        )

    if len(classes) == 2:
        binary_true = (true_codes == 1).astype(int)
        return float(roc_auc_score(binary_true, scores[:, 1]))

    binarized_true = label_binarize(
        true_codes,
        classes=np.arange(len(classes)),
    )
    return float(
        roc_auc_score(
            binarized_true,
            scores,
            average=average,
        )
    )


def align_cluster_labels(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> tuple[NDArray[np.object_], dict[ClassLabel, ClassLabel]]:
    """Align predicted cluster identifiers to reference labels.

    The Hungarian algorithm maximizes the number of matching rows. If there
    are more predicted clusters than reference classes, unmatched clusters are
    mapped to their locally most frequent reference class.
    """
    true_labels, predicted_labels = _validate_label_pair(
        y_true,
        y_pred,
    )

    true_values = list(pd.unique(true_labels))
    predicted_values = list(pd.unique(predicted_labels))

    contingency = np.zeros(
        (len(predicted_values), len(true_values)),
        dtype=int,
    )

    for predicted_index, predicted_value in enumerate(predicted_values):
        predicted_mask = _label_mask(
            predicted_labels,
            predicted_value,
        )
        for true_index, true_value in enumerate(true_values):
            contingency[predicted_index, true_index] = int(
                np.sum(
                    predicted_mask
                    & _label_mask(true_labels, true_value)
                )
            )

    row_indices, column_indices = linear_sum_assignment(-contingency)
    mapping: dict[ClassLabel, ClassLabel] = {
        predicted_values[row]: true_values[column]
        for row, column in zip(
            row_indices,
            column_indices,
            strict=True,
        )
    }

    for predicted_index, predicted_value in enumerate(predicted_values):
        if predicted_value in mapping:
            continue
        best_true_index = int(
            np.argmax(contingency[predicted_index])
        )
        mapping[predicted_value] = true_values[best_true_index]

    aligned = np.asarray(
        [mapping[value] for value in predicted_labels],
        dtype=object,
    )
    return aligned, mapping


def compare_methods(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    y_score: ArrayLike | None = None,
    score_labels: Sequence[ClassLabel] | None = None,
    average: AverageName = "macro",
    zero_division: int | float = 0,
    ignored_labels: Iterable[ClassLabel] | None = None,
    ignore_missing: bool = True,
    align_predicted_labels: bool = False,
) -> dict[str, Any]:
    """Backward-friendly flat evaluation entry point."""
    report = evaluate_external_labels(
        y_true,
        y_pred,
        y_score=y_score,
        score_labels=score_labels,
        average=average,
        zero_division=zero_division,
        ignored_labels=ignored_labels,
        ignore_missing=ignore_missing,
        align_predicted_labels=align_predicted_labels,
    )
    return report.to_flat_dict()


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculate classification accuracy."""
    true_labels, predicted_labels = _validate_label_pair(
        y_true,
        y_pred,
    )
    true_codes, predicted_codes = _encode_label_pair(
        true_labels,
        predicted_labels,
    )
    return float(accuracy_score(true_codes, predicted_codes))


def precision(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    avg: AverageName = "macro",
    *,
    zero_division: int | float = 0,
) -> float:
    """Calculate averaged precision."""
    true_labels, predicted_labels = _validate_label_pair(
        y_true,
        y_pred,
    )
    true_codes, predicted_codes = _encode_label_pair(
        true_labels,
        predicted_labels,
    )
    return float(
        precision_score(
            true_codes,
            predicted_codes,
            average=avg,
            zero_division=zero_division,
        )
    )


def recall(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    avg: AverageName = "macro",
    *,
    zero_division: int | float = 0,
) -> float:
    """Calculate averaged recall."""
    true_labels, predicted_labels = _validate_label_pair(
        y_true,
        y_pred,
    )
    true_codes, predicted_codes = _encode_label_pair(
        true_labels,
        predicted_labels,
    )
    return float(
        recall_score(
            true_codes,
            predicted_codes,
            average=avg,
            zero_division=zero_division,
        )
    )


def f1(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    avg: AverageName = "macro",
    *,
    zero_division: int | float = 0,
) -> float:
    """Calculate averaged F1 score."""
    true_labels, predicted_labels = _validate_label_pair(
        y_true,
        y_pred,
    )
    true_codes, predicted_codes = _encode_label_pair(
        true_labels,
        predicted_labels,
    )
    return float(
        f1_score(
            true_codes,
            predicted_codes,
            average=avg,
            zero_division=zero_division,
        )
    )


def matthews_corrcoef(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    """Calculate binary or multiclass Matthews correlation coefficient."""
    true_labels, predicted_labels = _validate_label_pair(
        y_true,
        y_pred,
    )
    true_codes, predicted_codes = _encode_label_pair(
        true_labels,
        predicted_labels,
    )
    return float(
        sklearn_matthews_corrcoef(
            true_codes,
            predicted_codes,
        )
    )


def cohens_kappa(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    """Calculate Cohen's kappa."""
    true_labels, predicted_labels = _validate_label_pair(
        y_true,
        y_pred,
    )
    true_codes, predicted_codes = _encode_label_pair(
        true_labels,
        predicted_labels,
    )
    return float(
        cohen_kappa_score(
            true_codes,
            predicted_codes,
        )
    )


def jaccard_index(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: AverageName = "macro",
    zero_division: int | float = 0,
) -> float:
    """Calculate the actual averaged Jaccard score."""
    true_labels, predicted_labels = _validate_label_pair(
        y_true,
        y_pred,
    )
    true_codes, predicted_codes = _encode_label_pair(
        true_labels,
        predicted_labels,
    )
    return float(
        jaccard_score(
            true_codes,
            predicted_codes,
            average=average,
            zero_division=zero_division,
        )
    )


def hamming_loss(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    """Calculate the fraction of incorrectly predicted labels."""
    true_labels, predicted_labels = _validate_label_pair(
        y_true,
        y_pred,
    )
    true_codes, predicted_codes = _encode_label_pair(
        true_labels,
        predicted_labels,
    )
    return float(
        sklearn_hamming_loss(
            true_codes,
            predicted_codes,
        )
    )


def auc_roc(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    score_labels: Sequence[ClassLabel] | None = None,
) -> float:
    """Backward-compatible wrapper requiring continuous scores."""
    return multiclass_roc_auc(
        y_true,
        y_score,
        score_labels=score_labels,
    )


def rand_index(
    clustering1: ArrayLike,
    clustering2: ArrayLike,
) -> float:
    """Backward-compatible alias for the adjusted Rand index."""
    return adjusted_rand_index(clustering1, clustering2)


def vi_index(
    clustering1: ArrayLike,
    clustering2: ArrayLike,
) -> float:
    """Backward-compatible variation-of-information wrapper."""
    return variation_of_information(clustering1, clustering2)


def nmi_index(
    clustering1: ArrayLike,
    clustering2: ArrayLike,
) -> float:
    """Backward-compatible NMI wrapper."""
    return normalized_mutual_information(
        clustering1,
        clustering2,
    )


def shannon_for_full_set(
    labels: ArrayLike,
    class_values: ArrayLike,
) -> float:
    """Backward-compatible macro cluster-entropy wrapper."""
    return conditional_cluster_entropy(
        labels,
        class_values,
        average="macro",
    )


@dataclass(frozen=True, slots=True)
class _PreparedInputs:
    true_labels: NDArray[np.object_]
    predicted_labels: NDArray[np.object_]
    included_mask: NDArray[np.bool_]
    original_count: int


def _prepare_inputs(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    ignored_labels: Iterable[ClassLabel] | None,
    ignore_missing: bool,
) -> _PreparedInputs:
    true_labels, predicted_labels = _validate_label_pair(
        y_true,
        y_pred,
    )
    included_mask = np.ones(len(true_labels), dtype=bool)

    if ignore_missing:
        included_mask &= np.fromiter(
            (
                not _is_missing(true_value)
                and not _is_missing(predicted_value)
                for true_value, predicted_value in zip(
                    true_labels,
                    predicted_labels,
                    strict=True,
                )
            ),
            dtype=bool,
            count=len(true_labels),
        )

    ignored = tuple(ignored_labels or ())
    if ignored:
        included_mask &= np.fromiter(
            (
                not any(
                    _labels_equal(true_value, ignored_label)
                    or _labels_equal(
                        predicted_value,
                        ignored_label,
                    )
                    for ignored_label in ignored
                )
                for true_value, predicted_value in zip(
                    true_labels,
                    predicted_labels,
                    strict=True,
                )
            ),
            dtype=bool,
            count=len(true_labels),
        )

    filtered_true = true_labels[included_mask]
    filtered_predicted = predicted_labels[included_mask]

    if len(filtered_true) == 0:
        raise ValueError(
            "No rows remain after applying evaluation exclusions."
        )

    return _PreparedInputs(
        true_labels=filtered_true,
        predicted_labels=filtered_predicted,
        included_mask=included_mask,
        original_count=len(true_labels),
    )


def _classification_metrics(
    y_true: NDArray[np.object_],
    y_pred: NDArray[np.object_],
    *,
    y_score: NDArray[np.float64] | None,
    score_labels: Sequence[ClassLabel] | None,
    average: AverageName,
    zero_division: int | float,
) -> dict[str, float | None]:
    true_codes, predicted_codes = _encode_label_pair(y_true, y_pred)

    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(true_codes, predicted_codes)),
        f"precision_{average}": float(
            precision_score(
                true_codes,
                predicted_codes,
                average=average,
                zero_division=zero_division,
            )
        ),
        f"recall_{average}": float(
            recall_score(
                true_codes,
                predicted_codes,
                average=average,
                zero_division=zero_division,
            )
        ),
        f"f1_{average}": float(
            f1_score(
                true_codes,
                predicted_codes,
                average=average,
                zero_division=zero_division,
            )
        ),
        f"jaccard_{average}": float(
            jaccard_score(
                true_codes,
                predicted_codes,
                average=average,
                zero_division=zero_division,
            )
        ),
        "hamming_loss": float(
            sklearn_hamming_loss(true_codes, predicted_codes)
        ),
        "matthews_corrcoef": float(
            sklearn_matthews_corrcoef(true_codes, predicted_codes)
        ),
        "cohens_kappa": float(
            cohen_kappa_score(true_codes, predicted_codes)
        ),
        "roc_auc": None,
    }

    if y_score is not None:
        metrics["roc_auc"] = multiclass_roc_auc(
            y_true,
            y_score,
            score_labels=score_labels,
        )

    return metrics


def _clustering_metrics(
    y_true: NDArray[np.object_],
    y_pred: NDArray[np.object_],
    *,
    entropy_average: EntropyAverage,
) -> dict[str, float]:
    true_codes, predicted_codes = _encode_label_pair(y_true, y_pred)
    homogeneity, completeness, v_measure = (
        homogeneity_completeness_v_measure(
            true_codes,
            predicted_codes,
        )
    )

    return {
        "adjusted_rand": float(
            adjusted_rand_score(true_codes, predicted_codes)
        ),
        "fowlkes_mallows": float(
            fowlkes_mallows_score(true_codes, predicted_codes)
        ),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(
                true_codes,
                predicted_codes,
                average_method="arithmetic",
            )
        ),
        "variation_of_information": variation_of_information(
            y_true,
            y_pred,
        ),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "conditional_cluster_entropy": (
            conditional_cluster_entropy(
                y_pred,
                y_true,
                average=entropy_average,
            )
        ),
    }


def _validate_label_pair(
    first_labels: ArrayLike,
    second_labels: ArrayLike,
) -> tuple[NDArray[np.object_], NDArray[np.object_]]:
    first = _as_one_dimensional(first_labels, "first_labels")
    second = _as_one_dimensional(second_labels, "second_labels")

    if len(first) != len(second):
        raise ValueError(
            "Both label arrays must have the same length."
        )
    if len(first) == 0:
        raise ValueError("Label arrays cannot be empty.")

    return first, second


def _as_one_dimensional(
    values: ArrayLike,
    name: str,
) -> NDArray[np.object_]:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"'{name}' must be one-dimensional.")
    return array


def _encode_label_pair(
    first: NDArray[np.object_],
    second: NDArray[np.object_],
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Encode two label arrays with one shared semantic mapping."""
    code_by_key: dict[tuple[str, Any], int] = {}

    def encode(values: NDArray[np.object_]) -> NDArray[np.int_]:
        result = np.empty(len(values), dtype=int)
        for index, value in enumerate(values):
            key = _label_key(value)
            if key not in code_by_key:
                code_by_key[key] = len(code_by_key)
            result[index] = code_by_key[key]
        return result

    return encode(first), encode(second)


def _label_mask(
    labels: NDArray[np.object_],
    target: Any,
) -> NDArray[np.bool_]:
    return np.fromiter(
        (
            _labels_equal(value, target)
            for value in labels
        ),
        dtype=bool,
        count=len(labels),
    )


def _labels_equal(first: Any, second: Any) -> bool:
    if _is_missing(first) and _is_missing(second):
        return True

    try:
        result = first == second
    except (TypeError, ValueError):
        return False

    return bool(result) if np.isscalar(result) else False


def _is_missing(value: Any) -> bool:
    if value is None:
        return True

    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False

    return bool(result) if np.isscalar(result) else False


def _label_key(value: Any) -> tuple[str, Any]:
    if _is_missing(value):
        return ("missing", None)

    try:
        hash(value)
        return ("value", value)
    except TypeError:
        return ("representation", repr(value))