"""Iterative model-based pseudo-labeling."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any, Protocol, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

ClusterLabel: TypeAlias = Hashable
FeatureMatrix: TypeAlias = NDArray[Any]
LabelArray: TypeAlias = NDArray[Any]
ConfidenceArray: TypeAlias = NDArray[np.float64]
ConfidenceGetter: TypeAlias = Callable[
    [Any, FeatureMatrix, LabelArray],
    ArrayLike,
]
PredictionTransform: TypeAlias = Callable[[ArrayLike], ArrayLike]


class PredictiveModel(Protocol):
    """Minimal interface required from a predictive model."""

    def fit(
        self,
        features: FeatureMatrix,
        labels: LabelArray,
    ) -> Any:
        """Fit the model."""

    def predict(self, features: FeatureMatrix) -> ArrayLike:
        """Predict one output for every input row."""


class ModelBasedLabeling:
    """Iteratively assign pseudo-labels using a user-provided model.

    Parameters
    ----------
    df
        Feature matrix containing objects to label. Rows represent objects and
        columns represent model features. The data must already use the
        representation expected by the selected model.

    Notes
    -----
    The model must implement ``fit`` and ``predict``. Confidence is obtained,
    in order of precedence, from a user-provided ``confidence_getter``, the
    model's ``predict_proba`` method, or its ``decision_function`` method.

    This makes the class compatible with common classifiers from scikit-learn,
    XGBoost, and CatBoost. Models without a confidence interface can still be
    used with ``confidence_cutoff=None`` or a custom ``confidence_getter``.
    """

    def __init__(self, df: pd.DataFrame | ArrayLike) -> None:
        self.df = self._prepare_feature_matrix(df)
        self.model_: PredictiveModel | None = None

    def label_data(
        self,
        labeled_data: Mapping[
            ClusterLabel,
            pd.DataFrame | ArrayLike | Sequence[ArrayLike],
        ],
        model: PredictiveModel,
        confidence_cutoff: float | None = 0.9,
        max_iterations: int = 15,
        confidence_getter: ConfidenceGetter | None = None,
        prediction_transform: PredictionTransform | None = None,
    ) -> tuple[LabelArray, ConfidenceArray]:
        """Assign labels through iterative confidence-based self-training.

        The model is first trained on the objects in ``labeled_data``. During
        each iteration, sufficiently confident predictions are added to the
        training set as pseudo-labels. Remaining objects are labeled by the
        final fitted model.

        Parameters
        ----------
        labeled_data
            Mapping from class labels to initial training objects. Empty groups
            are ignored. Every non-empty group must have the same number of
            features as ``df``.
        model
            Any estimator implementing ``fit`` and ``predict``. For iterative
            confidence filtering, it must additionally expose
            ``predict_proba`` or ``decision_function``, unless
            ``confidence_getter`` is provided.
        confidence_cutoff
            Minimum confidence required to add a prediction to the training
            set. Set to ``None`` to disable iterative confidence filtering and
            perform a single fit-and-predict pass.
        max_iterations
            Maximum number of pseudo-label expansion iterations. A value of
            zero skips expansion and labels all objects after the initial fit.
        confidence_getter
            Optional function called as ``confidence_getter(model, features,
            predictions)``. It must return one confidence value in the range
            [0, 1] for every input row. It overrides ``predict_proba`` and
            ``decision_function``.
        prediction_transform
            Optional function applied to raw values returned by
            ``model.predict``. It can normalize model-specific output shapes or
            convert regression-style numeric outputs into class labels.

        Returns
        -------
        tuple
            Predicted labels and confidence scores, both aligned with the
            original row order of ``df``. When no confidence source exists and
            ``confidence_cutoff`` is ``None``, confidence scores are ``NaN``.

        Raises
        ------
        TypeError
            If the model does not implement ``fit`` and ``predict``, or if
            confidence filtering is requested but no confidence source exists.
        ValueError
            If the training data, model output, confidence values, or control
            parameters are invalid.
        """
        self._validate_parameters(confidence_cutoff, max_iterations)
        self._validate_model(model)

        training_features, training_labels = self._prepare_labeled_data(
            labeled_data
        )

        sample_count = len(self.df)
        if sample_count == 0:
            self.model_ = model
            return (
                np.empty(0, dtype=object),
                np.empty(0, dtype=np.float64),
            )

        predicted_labels = np.empty(sample_count, dtype=object)
        confidence_scores = np.full(
            sample_count,
            np.nan,
            dtype=np.float64,
        )
        remaining_indices = np.arange(sample_count, dtype=int)

        if confidence_cutoff is not None:
            self._validate_confidence_source(model, confidence_getter)

            for _ in range(max_iterations):
                if remaining_indices.size == 0:
                    break

                model.fit(training_features, training_labels)
                remaining_features = self.df[remaining_indices]
                predictions, confidences = self._predict_with_confidence(
                    model=model,
                    features=remaining_features,
                    confidence_getter=confidence_getter,
                    prediction_transform=prediction_transform,
                    require_confidence=True,
                )

                confident_mask = confidences >= confidence_cutoff
                if not np.any(confident_mask):
                    break

                confident_indices = remaining_indices[confident_mask]
                confident_predictions = predictions[confident_mask]

                predicted_labels[confident_indices] = confident_predictions
                confidence_scores[confident_indices] = confidences[
                    confident_mask
                ]

                training_features = np.vstack(
                    [training_features, remaining_features[confident_mask]]
                )
                training_labels = np.concatenate(
                    [training_labels, confident_predictions]
                )
                remaining_indices = remaining_indices[~confident_mask]

        model.fit(training_features, training_labels)
        self.model_ = model

        if remaining_indices.size > 0:
            remaining_predictions, remaining_confidences = (
                self._predict_with_confidence(
                    model=model,
                    features=self.df[remaining_indices],
                    confidence_getter=confidence_getter,
                    prediction_transform=prediction_transform,
                    require_confidence=False,
                )
            )
            predicted_labels[remaining_indices] = remaining_predictions
            confidence_scores[remaining_indices] = remaining_confidences

        return np.asarray(predicted_labels.tolist()), confidence_scores

    def _prepare_labeled_data(
        self,
        labeled_data: Mapping[
            ClusterLabel,
            pd.DataFrame | ArrayLike | Sequence[ArrayLike],
        ],
    ) -> tuple[FeatureMatrix, LabelArray]:
        """Validate and stack initial training objects and labels."""
        if not labeled_data:
            raise ValueError(
                "'labeled_data' must contain at least one training object."
            )

        feature_groups: list[FeatureMatrix] = []
        labels: list[ClusterLabel] = []
        expected_feature_count = self.df.shape[1]

        for label, examples in labeled_data.items():
            if isinstance(examples, pd.DataFrame):
                example_array = examples.to_numpy(copy=True)
            else:
                example_array = np.asarray(examples)

            if example_array.size == 0:
                continue

            if example_array.ndim == 1:
                if example_array.shape[0] != expected_feature_count:
                    raise ValueError(
                        f"Training data for label {label!r} must contain "
                        f"{expected_feature_count} features per object."
                    )
                example_array = example_array.reshape(1, -1)

            if example_array.ndim != 2:
                raise ValueError(
                    f"Training data for label {label!r} must be a 2D array."
                )
            if example_array.shape[1] != expected_feature_count:
                raise ValueError(
                    f"Training data for label {label!r} must contain "
                    f"{expected_feature_count} features per object; received "
                    f"{example_array.shape[1]}."
                )

            feature_groups.append(example_array)
            labels.extend([label] * len(example_array))

        if not feature_groups:
            raise ValueError(
                "'labeled_data' must contain at least one training object; "
                "all provided groups are empty."
            )

        return np.vstack(feature_groups), np.asarray(labels)

    @staticmethod
    def _prepare_feature_matrix(
        data: pd.DataFrame | ArrayLike,
    ) -> FeatureMatrix:
        """Convert input data into a two-dimensional NumPy array."""
        if isinstance(data, pd.DataFrame):
            feature_matrix = data.to_numpy(copy=True)
        else:
            feature_matrix = np.asarray(data)

        if feature_matrix.ndim != 2:
            raise ValueError("'df' must be a two-dimensional feature matrix.")
        if feature_matrix.shape[1] == 0:
            raise ValueError("'df' must contain at least one feature column.")

        return feature_matrix

    @staticmethod
    def _validate_parameters(
        confidence_cutoff: float | None,
        max_iterations: int,
    ) -> None:
        """Validate confidence and iteration controls."""
        if confidence_cutoff is not None:
            if isinstance(confidence_cutoff, bool) or not isinstance(
                confidence_cutoff,
                (int, float, np.integer, np.floating),
            ):
                raise TypeError(
                    "'confidence_cutoff' must be a number or None."
                )
            if not 0.0 <= float(confidence_cutoff) <= 1.0:
                raise ValueError(
                    "'confidence_cutoff' must be between 0 and 1."
                )

        if isinstance(max_iterations, bool) or not isinstance(
            max_iterations,
            (int, np.integer),
        ):
            raise TypeError("'max_iterations' must be an integer.")
        if max_iterations < 0:
            raise ValueError("'max_iterations' cannot be negative.")

    @staticmethod
    def _validate_model(model: PredictiveModel) -> None:
        """Ensure that the model exposes the minimal estimator interface."""
        missing_methods = [
            method_name
            for method_name in ("fit", "predict")
            if not callable(getattr(model, method_name, None))
        ]

        if missing_methods:
            missing = ", ".join(missing_methods)
            raise TypeError(
                "'model' must implement fit and predict. "
                f"Missing methods: {missing}."
            )

    @staticmethod
    def _validate_confidence_source(
        model: PredictiveModel,
        confidence_getter: ConfidenceGetter | None,
    ) -> None:
        """Ensure that iterative filtering can obtain confidence scores."""
        has_probability = callable(getattr(model, "predict_proba", None))
        has_decision = callable(getattr(model, "decision_function", None))

        if confidence_getter is None and not (
            has_probability or has_decision
        ):
            raise TypeError(
                "Confidence-based iteration requires 'predict_proba', "
                "'decision_function', or a custom 'confidence_getter'. "
                "Set confidence_cutoff=None for a single prediction pass."
            )

    def _predict_with_confidence(
        self,
        model: PredictiveModel,
        features: FeatureMatrix,
        confidence_getter: ConfidenceGetter | None,
        prediction_transform: PredictionTransform | None,
        require_confidence: bool,
    ) -> tuple[LabelArray, ConfidenceArray]:
        """Predict labels and obtain normalized confidence scores."""
        raw_predictions = model.predict(features)
        transformed_predictions = (
            prediction_transform(raw_predictions)
            if prediction_transform is not None
            else raw_predictions
        )
        predictions = self._normalize_predictions(
            transformed_predictions,
            len(features),
        )

        if confidence_getter is not None:
            raw_confidences = confidence_getter(
                model,
                features,
                predictions,
            )
            confidences = self._normalize_confidences(
                raw_confidences,
                len(features),
                source="confidence_getter",
            )
            return predictions, confidences

        predict_proba = getattr(model, "predict_proba", None)
        if callable(predict_proba):
            probabilities = predict_proba(features)
            confidences = self._confidence_from_probabilities(
                probabilities,
                len(features),
            )
            return predictions, confidences

        decision_function = getattr(model, "decision_function", None)
        if callable(decision_function):
            decision_scores = decision_function(features)
            confidences = self._confidence_from_decision_scores(
                decision_scores,
                len(features),
            )
            return predictions, confidences

        if require_confidence:
            raise TypeError(
                "The selected model does not provide confidence scores."
            )

        return predictions, np.full(len(features), np.nan, dtype=np.float64)

    @staticmethod
    def _normalize_predictions(
        predictions: ArrayLike,
        expected_count: int,
    ) -> LabelArray:
        """Normalize model predictions to one label per input row."""
        prediction_array = np.asarray(predictions)

        if prediction_array.ndim == 2 and prediction_array.shape[1] == 1:
            prediction_array = prediction_array[:, 0]
        if prediction_array.ndim != 1:
            raise ValueError(
                "'model.predict' must return a 1D array or a single-column "
                "2D array."
            )
        if len(prediction_array) != expected_count:
            raise ValueError(
                "'model.predict' must return one value for every input row."
            )

        return prediction_array

    def _confidence_from_probabilities(
        self,
        probabilities: ArrayLike,
        expected_count: int,
    ) -> ConfidenceArray:
        """Extract maximum class confidence from probability output."""
        probability_array = np.asarray(probabilities, dtype=np.float64)

        if probability_array.ndim == 1:
            if len(probability_array) != expected_count:
                raise ValueError(
                    "'predict_proba' must return one row per input object."
                )
            probability_array = np.column_stack(
                [1.0 - probability_array, probability_array]
            )

        if probability_array.ndim != 2 or probability_array.shape[1] == 0:
            raise ValueError(
                "'predict_proba' must return a non-empty 2D array."
            )
        if probability_array.shape[0] != expected_count:
            raise ValueError(
                "'predict_proba' must return one row per input object."
            )
        if not np.all(np.isfinite(probability_array)):
            raise ValueError(
                "'predict_proba' returned non-finite probability values."
            )
        if np.any(probability_array < -1e-8) or np.any(
            probability_array > 1.0 + 1e-8
        ):
            raise ValueError(
                "'predict_proba' returned values outside the range [0, 1]."
            )

        clipped_probabilities = np.clip(probability_array, 0.0, 1.0)
        return clipped_probabilities.max(axis=1)

    def _confidence_from_decision_scores(
        self,
        decision_scores: ArrayLike,
        expected_count: int,
    ) -> ConfidenceArray:
        """Convert decision margins into confidence-like values in [0, 1]."""
        score_array = np.asarray(decision_scores, dtype=np.float64)

        if score_array.ndim == 2 and score_array.shape[1] == 1:
            score_array = score_array[:, 0]

        if score_array.shape[0] != expected_count:
            raise ValueError(
                "'decision_function' must return one row per input object."
            )
        if not np.all(np.isfinite(score_array)):
            raise ValueError(
                "'decision_function' returned non-finite values."
            )

        if score_array.ndim == 1:
            absolute_margin = np.abs(score_array)
            return self._sigmoid(absolute_margin)

        if score_array.ndim != 2 or score_array.shape[1] == 0:
            raise ValueError(
                "'decision_function' must return a 1D or non-empty 2D array."
            )

        shifted_scores = score_array - np.max(
            score_array,
            axis=1,
            keepdims=True,
        )
        exp_scores = np.exp(shifted_scores)
        probabilities = exp_scores / np.sum(
            exp_scores,
            axis=1,
            keepdims=True,
        )
        return probabilities.max(axis=1)

    @staticmethod
    def _normalize_confidences(
        confidences: ArrayLike,
        expected_count: int,
        source: str,
    ) -> ConfidenceArray:
        """Validate custom confidence values."""
        confidence_array = np.asarray(confidences, dtype=np.float64)

        if confidence_array.ndim == 2 and confidence_array.shape[1] == 1:
            confidence_array = confidence_array[:, 0]
        if confidence_array.ndim != 1:
            raise ValueError(f"'{source}' must return a 1D array.")
        if len(confidence_array) != expected_count:
            raise ValueError(
                f"'{source}' must return one value for every input row."
            )
        if not np.all(np.isfinite(confidence_array)):
            raise ValueError(f"'{source}' returned non-finite values.")
        if np.any(confidence_array < 0.0) or np.any(
            confidence_array > 1.0
        ):
            raise ValueError(
                f"'{source}' must return values in the range [0, 1]."
            )

        return confidence_array

    @staticmethod
    def _sigmoid(values: NDArray[np.float64]) -> ConfidenceArray:
        """Calculate a numerically stable logistic transformation."""
        clipped_values = np.clip(values, -709.0, 709.0)
        return 1.0 / (1.0 + np.exp(-clipped_values))