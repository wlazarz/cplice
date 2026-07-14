"""Semi-supervised pseudo-labeling with scikit-learn LabelSpreading."""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import issparse, spmatrix
from sklearn.base import clone
from sklearn.semi_supervised import LabelSpreading
from algorithms.cplice.object_distances import _validate_positive_integer

ClassLabel: TypeAlias = Hashable
FeatureInput: TypeAlias = pd.DataFrame | ArrayLike
KernelType: TypeAlias = str | Callable[
    [Any, Any],
    NDArray[np.floating[Any]],
]


class LabelSpreadingLabeling:
    """Assign pseudo-labels with graph-based label spreading.

    Parameters
    ----------
    unlabeled_data
        Objects that should receive pseudo-labels. Rows are objects and
        columns are features.
    transformer
        Optional scikit-learn-compatible transformer implementing
        ``fit_transform``. It is fitted once on the combined labeled and
        unlabeled feature matrix. Use a ``OneHotEncoder`` or
        ``ColumnTransformer`` when the input contains categorical columns.
        When omitted, the combined feature matrix must already be numeric.
    copy
        Whether to copy the unlabeled input data.

    Notes
    -----
    This class is independent of ``LabelingTemplate`` because it does not use
    the CPLICE categorical distance interface. LabelSpreading constructs its
    own affinity graph using the selected kernel.

    After :meth:`label_data`, the following fitted attributes are available:

    ``model_``
        Fitted scikit-learn ``LabelSpreading`` instance.
    ``transformer_``
        Fitted cloned transformer, or ``None`` when no transformer was used.
    ``classes_``
        Original user-provided class labels in probability-column order.
    ``probabilities_``
        Class-distribution matrix for the originally unlabeled rows.
    ``confidences_``
        Maximum class probability for every originally unlabeled row.
    ``predictions_``
        Predicted labels in the original row order.
    ``n_iter_``
        Number of label-spreading iterations performed.
    """

    def __init__(
        self,
        unlabeled_data: FeatureInput,
        *,
        transformer: Any | None = None,
        copy: bool = True,
    ) -> None:
        self._unlabeled_is_dataframe = isinstance(
            unlabeled_data,
            pd.DataFrame,
        )
        self.unlabeled_data = self._prepare_unlabeled_input(
            unlabeled_data,
            copy=copy,
        )
        self.transformer = transformer

        self.model_: LabelSpreading | None = None
        self.transformer_: Any | None = None
        self.classes_: NDArray[np.object_] | None = None
        self.probabilities_: NDArray[np.float64] | None = None
        self.confidences_: NDArray[np.float64] | None = None
        self.predictions_: NDArray[np.object_] | None = None
        self.n_iter_: int | None = None
        self.effective_n_neighbors_: int | None = None

    def label_data(
        self,
        labeled_data: Mapping[
            ClassLabel,
            FeatureInput | Sequence[ArrayLike],
        ],
        *,
        kernel: KernelType = "knn",
        gamma: float = 20.0,
        n_neighbors: int = 7,
        alpha: float = 0.2,
        max_iterations: int = 30,
        tol: float = 1e-3,
        n_jobs: int | None = -1,
    ) -> tuple[NDArray[np.object_], NDArray[np.float64]]:
        """Fit LabelSpreading and label the stored unlabeled objects.

        Parameters
        ----------
        labeled_data
            Mapping from original class labels to labeled feature rows.
            Empty class groups are ignored. At least two non-empty classes are
            required.
        kernel
            ``"knn"``, ``"rbf"``, or a callable affinity function accepted by
            scikit-learn ``LabelSpreading``.
        gamma
            RBF-kernel parameter. Must be positive.
        n_neighbors
            Number of neighbors for the KNN graph. If it exceeds the number
            of available objects minus one, it is reduced to the largest valid
            value and stored in ``effective_n_neighbors_``.
        alpha
            Soft-clamping factor in the open interval ``(0, 1)``.
        max_iterations
            Maximum number of propagation iterations.
        tol
            Convergence tolerance.
        n_jobs
            Number of parallel jobs used by the KNN kernel. ``None`` follows
            scikit-learn defaults and ``-1`` requests all available workers.

        Returns
        -------
        tuple
            Predicted labels and maximum class probabilities for the
            originally unlabeled objects. Both arrays preserve input row
            order.

        Raises
        ------
        ValueError
            If data dimensions, hyperparameters, labels, or numeric values are
            invalid.
        TypeError
            If the transformer or kernel has an unsupported interface.
        """
        self._validate_hyperparameters(
            kernel=kernel,
            gamma=gamma,
            n_neighbors=n_neighbors,
            alpha=alpha,
            max_iterations=max_iterations,
            tol=tol,
            n_jobs=n_jobs,
        )

        labeled_features, original_labels = (
            self._build_labeled_dataset(labeled_data)
        )
        unlabeled_count = len(self.unlabeled_data)

        if unlabeled_count == 0:
            empty_labels = np.asarray([], dtype=object)
            empty_confidences = np.asarray([], dtype=np.float64)
            self.model_ = None
            self.transformer_ = None
            self.classes_ = np.asarray(
                _unique_labels_in_order(original_labels),
                dtype=object,
            )
            self.probabilities_ = np.empty(
                (0, len(self.classes_)),
                dtype=np.float64,
            )
            self.confidences_ = empty_confidences
            self.predictions_ = empty_labels
            self.n_iter_ = 0
            self.effective_n_neighbors_ = None
            return empty_labels, empty_confidences

        combined_features = self._combine_features(
            labeled_features,
            self.unlabeled_data,
        )
        transformed_features = self._fit_transform_features(
            combined_features
        )

        class_labels = _unique_labels_in_order(original_labels)
        if len(class_labels) < 2:
            raise ValueError(
                "'labeled_data' must contain at least two non-empty classes."
            )

        label_to_code = {
            label: code
            for code, label in enumerate(class_labels)
        }
        encoded_labeled = np.asarray(
            [label_to_code[label] for label in original_labels],
            dtype=int,
        )
        labeled_count = len(encoded_labeled)

        training_labels = np.full(
            labeled_count + unlabeled_count,
            -1,
            dtype=int,
        )
        training_labels[:labeled_count] = encoded_labeled

        effective_n_neighbors = min(
            n_neighbors,
            len(training_labels) - 1,
        )
        if effective_n_neighbors < 1:
            raise ValueError(
                "LabelSpreading requires at least two total objects."
            )

        model = LabelSpreading(
            kernel=kernel,
            gamma=float(gamma),
            n_neighbors=effective_n_neighbors,
            alpha=float(alpha),
            max_iter=max_iterations,
            tol=float(tol),
            n_jobs=n_jobs,
        )
        model.fit(transformed_features, training_labels)

        unlabeled_slice = slice(labeled_count, None)
        encoded_predictions = np.asarray(
            model.transduction_[unlabeled_slice],
            dtype=int,
        )
        probabilities = np.asarray(
            model.label_distributions_[unlabeled_slice],
            dtype=np.float64,
        )

        class_array = np.asarray(class_labels, dtype=object)
        predictions = class_array[encoded_predictions]
        confidences = probabilities.max(axis=1)

        self.model_ = model
        self.classes_ = class_array
        self.probabilities_ = probabilities
        self.confidences_ = confidences
        self.predictions_ = predictions
        self.n_iter_ = int(model.n_iter_)
        self.effective_n_neighbors_ = effective_n_neighbors

        return predictions.copy(), confidences.copy()

    def _prepare_unlabeled_input(
        self,
        data: FeatureInput,
        *,
        copy: bool,
    ) -> pd.DataFrame | NDArray[Any]:
        if isinstance(data, pd.DataFrame):
            if data.shape[1] == 0:
                raise ValueError(
                    "'unlabeled_data' must contain at least one feature."
                )
            return data.reset_index(drop=True).copy(deep=copy)

        array = np.asarray(data)
        if array.ndim == 1:
            if array.size == 0:
                raise ValueError(
                    "A one-dimensional empty input does not define the "
                    "number of features."
                )
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError(
                "'unlabeled_data' must be two-dimensional."
            )
        if array.shape[1] == 0:
            raise ValueError(
                "'unlabeled_data' must contain at least one feature."
            )
        return array.copy() if copy else array

    def _build_labeled_dataset(
        self,
        labeled_data: Mapping[
            ClassLabel,
            FeatureInput | Sequence[ArrayLike],
        ],
    ) -> tuple[pd.DataFrame | NDArray[Any], list[ClassLabel]]:
        if not isinstance(labeled_data, Mapping):
            raise TypeError("'labeled_data' must be a mapping.")
        if not labeled_data:
            raise ValueError("'labeled_data' cannot be empty.")

        feature_groups: list[pd.DataFrame | NDArray[Any]] = []
        labels: list[ClassLabel] = []

        for label, examples in labeled_data.items():
            _validate_hashable_label(label)
            group = self._normalize_labeled_group(
                examples,
                class_label=label,
            )
            if len(group) == 0:
                continue

            feature_groups.append(group)
            labels.extend([label] * len(group))

        if not feature_groups:
            raise ValueError(
                "'labeled_data' must contain at least one labeled object."
            )

        unique_labels = _unique_labels_in_order(labels)
        if len(unique_labels) < 2:
            raise ValueError(
                "'labeled_data' must contain at least two non-empty classes."
            )

        if self._unlabeled_is_dataframe:
            combined = pd.concat(
                [
                    group
                    if isinstance(group, pd.DataFrame)
                    else pd.DataFrame(
                        group,
                        columns=self.unlabeled_data.columns,
                    )
                    for group in feature_groups
                ],
                ignore_index=True,
            )
        else:
            combined = np.vstack(
                [
                    group.to_numpy()
                    if isinstance(group, pd.DataFrame)
                    else group
                    for group in feature_groups
                ]
            )

        return combined, labels

    def _normalize_labeled_group(
        self,
        examples: FeatureInput | Sequence[ArrayLike],
        *,
        class_label: ClassLabel,
    ) -> pd.DataFrame | NDArray[Any]:
        if self._unlabeled_is_dataframe:
            expected_columns = self.unlabeled_data.columns

            if isinstance(examples, pd.DataFrame):
                missing_columns = set(expected_columns) - set(
                    examples.columns
                )
                extra_columns = set(examples.columns) - set(
                    expected_columns
                )
                if missing_columns or extra_columns:
                    raise ValueError(
                        f"Labeled rows for class {class_label!r} must have "
                        "the same columns as 'unlabeled_data'."
                    )
                return examples.loc[:, expected_columns].reset_index(
                    drop=True
                )

            array = np.asarray(examples, dtype=object)
            if array.ndim == 1:
                if array.size == 0:
                    return pd.DataFrame(columns=expected_columns)
                array = array.reshape(1, -1)
            if array.ndim != 2:
                raise ValueError(
                    f"Labeled rows for class {class_label!r} must be "
                    "two-dimensional."
                )
            if array.shape[1] != len(expected_columns):
                raise ValueError(
                    f"Labeled rows for class {class_label!r} contain "
                    f"{array.shape[1]} features; expected "
                    f"{len(expected_columns)}."
                )
            return pd.DataFrame(array, columns=expected_columns)

        array = np.asarray(examples)
        if array.ndim == 1:
            if array.size == 0:
                return np.empty(
                    (0, self.unlabeled_data.shape[1]),
                    dtype=self.unlabeled_data.dtype,
                )
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError(
                f"Labeled rows for class {class_label!r} must be "
                "two-dimensional."
            )
        if array.shape[1] != self.unlabeled_data.shape[1]:
            raise ValueError(
                f"Labeled rows for class {class_label!r} contain "
                f"{array.shape[1]} features; expected "
                f"{self.unlabeled_data.shape[1]}."
            )
        return array

    def _combine_features(
        self,
        labeled_features: pd.DataFrame | NDArray[Any],
        unlabeled_features: pd.DataFrame | NDArray[Any],
    ) -> pd.DataFrame | NDArray[Any]:
        if isinstance(unlabeled_features, pd.DataFrame):
            if not isinstance(labeled_features, pd.DataFrame):
                labeled_features = pd.DataFrame(
                    labeled_features,
                    columns=unlabeled_features.columns,
                )
            return pd.concat(
                [labeled_features, unlabeled_features],
                ignore_index=True,
            )

        labeled_array = (
            labeled_features.to_numpy()
            if isinstance(labeled_features, pd.DataFrame)
            else np.asarray(labeled_features)
        )
        return np.vstack(
            [labeled_array, np.asarray(unlabeled_features)]
        )

    def _fit_transform_features(
        self,
        combined_features: pd.DataFrame | NDArray[Any],
    ) -> NDArray[np.float64] | spmatrix:
        if self.transformer is not None:
            if not hasattr(self.transformer, "fit_transform"):
                raise TypeError(
                    "'transformer' must implement 'fit_transform'."
                )

            try:
                fitted_transformer = clone(self.transformer)
            except (TypeError, RuntimeError):
                fitted_transformer = self.transformer

            transformed = fitted_transformer.fit_transform(
                combined_features
            )
            self.transformer_ = fitted_transformer
        else:
            transformed = combined_features
            self.transformer_ = None

        if issparse(transformed):
            sparse_matrix = transformed.astype(
                np.float64,
                copy=False,
            )
            if sparse_matrix.ndim != 2:
                raise ValueError(
                    "The transformed feature matrix must be "
                    "two-dimensional."
                )
            if not np.all(np.isfinite(sparse_matrix.data)):
                raise ValueError(
                    "The transformed feature matrix contains NaN or "
                    "infinite values."
                )
            return sparse_matrix

        try:
            numeric_matrix = np.asarray(
                transformed,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "LabelSpreading requires numeric features. Pass a fitted-"
                "compatible transformer such as OneHotEncoder or "
                "ColumnTransformer for categorical data."
            ) from error

        if numeric_matrix.ndim != 2:
            raise ValueError(
                "The transformed feature matrix must be two-dimensional."
            )
        if not np.all(np.isfinite(numeric_matrix)):
            raise ValueError(
                "The transformed feature matrix contains NaN or infinite "
                "values. Impute missing values before LabelSpreading."
            )

        return numeric_matrix

    @staticmethod
    def _validate_hyperparameters(
        *,
        kernel: KernelType,
        gamma: float,
        n_neighbors: int,
        alpha: float,
        max_iterations: int,
        tol: float,
        n_jobs: int | None,
    ) -> None:
        if not callable(kernel) and kernel not in {"knn", "rbf"}:
            raise ValueError(
                "'kernel' must be 'knn', 'rbf', or a callable."
            )

        _validate_positive_finite(gamma, "gamma")
        _validate_positive_integer(n_neighbors, "n_neighbors")
        _validate_positive_integer(
            max_iterations,
            "max_iterations",
        )
        _validate_positive_finite(tol, "tol")

        numeric_alpha = float(alpha)
        if (
            not math.isfinite(numeric_alpha)
            or numeric_alpha <= 0
            or numeric_alpha >= 1
        ):
            raise ValueError("'alpha' must be in the open interval (0, 1).")

        if n_jobs is not None and (
            not isinstance(n_jobs, (int, np.integer))
            or isinstance(n_jobs, bool)
            or n_jobs == 0
        ):
            raise ValueError(
                "'n_jobs' must be None or a non-zero integer."
            )


def _unique_labels_in_order(
    labels: Sequence[ClassLabel],
) -> list[ClassLabel]:
    unique: list[ClassLabel] = []
    seen: set[ClassLabel] = set()

    for label in labels:
        if label not in seen:
            unique.append(label)
            seen.add(label)

    return unique


def _validate_hashable_label(label: ClassLabel) -> None:
    try:
        hash(label)
    except TypeError as error:
        raise TypeError(
            f"Class label {label!r} must be hashable."
        ) from error


def _validate_positive_finite(value: float, name: str) -> float:
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(
            f"'{name}' must be finite and greater than zero."
        )
    return numeric_value