"""Mixed-data CPLICE with robust conditional cluster prototypes.

This module implements an experimental pseudo-labeling algorithm for mixed
numerical and nominal data. It is intentionally designed as a research
prototype rather than an implementation of a standard, established method.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

ClusterLabel: TypeAlias = Hashable
ConditionalPair: TypeAlias = tuple[str, str]
IndexArray: TypeAlias = NDArray[np.int_]
LabelArray: TypeAlias = NDArray[np.object_]


@dataclass(frozen=True, slots=True)
class NumericProfile:
    """Robust location and scale estimates for one numerical feature."""

    center: float
    scale: float
    count: int


@dataclass(frozen=True, slots=True)
class CategoricalProfile:
    """Smoothed category probabilities for one nominal feature."""

    probabilities: dict[Any, float]
    default_probability: float
    normalizer: float


@dataclass(frozen=True, slots=True)
class ConditionalProfile:
    """A numerical profile conditioned on one nominal value."""

    center: float
    scale: float
    count: int
    reliability: float


@dataclass(frozen=True, slots=True)
class ClusterPrototype:
    """Complete mixed-data prototype for one cluster."""

    numerical: dict[str, NumericProfile]
    categorical: dict[str, CategoricalProfile]
    conditional: dict[ConditionalPair, dict[Any, ConditionalProfile]]


class MixedConditionalCPLICELabeling:
    """Pseudo-label mixed data using robust conditional cluster prototypes.

    The algorithm extends the iterative cluster-expansion idea of CPLICE to
    datasets containing numerical and nominal features. Each cluster is
    represented by:

    - robust numerical centers and scales,
    - smoothed nominal category distributions,
    - selected numerical profiles conditioned on nominal values.

    Candidate pseudo-labels are ranked by a combination of their assignment
    cost and the margin between their best and second-best clusters.

    Parameters
    ----------
    df
        Input data. Rows are objects and columns are features.
    categorical_columns
        Nominal feature names.
    numerical_columns
        Numerical feature names.
    conditional_pairs
        Domain-selected ``(categorical_column, numerical_column)`` pairs.
        The default is an empty sequence because automatically using the full
        Cartesian product may overemphasize the conditional component.
    categorical_prior_strength
        Strength of shrinkage from cluster category frequencies toward global
        category probabilities.
    scale_prior_strength
        Strength of shrinkage from local numerical scales toward global robust
        scales.
    conditional_prior_strength
        Strength of shrinkage from category-conditional profiles toward the
        unconditional numerical profile of the same cluster.
    huber_delta
        Transition point of the Huber loss used for standardized numerical
        deviations.
    minimum_scale
        Smallest allowed numerical scale.
    numeric_family_weight
        Weight of the numerical feature family.
    categorical_family_weight
        Weight of the nominal feature family.
    conditional_family_weight
        Weight of the conditional feature family.
    numeric_feature_weights
        Optional per-feature weights for numerical columns.
    categorical_feature_weights
        Optional per-feature weights for nominal columns.
    conditional_pair_weights
        Optional per-pair weights for conditional profiles.
    margin_weight
        Strength of the assignment-margin term used for selecting confident
        pseudo-labels. Set to zero to rank candidates only by assignment cost.
    max_iterations
        Maximum number of cluster-expansion iterations.

    Notes
    -----
    This is an experimental method. The defaults are defensible starting
    points, not universally optimal hyperparameters. They should be evaluated
    through repeated experiments and ablation studies.
    """

    _MISSING_TOKEN = "__MISSING__"

    def __init__(
        self,
        df: pd.DataFrame,
        categorical_columns: Sequence[str],
        numerical_columns: Sequence[str],
        conditional_pairs: Sequence[ConditionalPair] | None = None,
        *,
        categorical_prior_strength: float = 1.0,
        scale_prior_strength: float = 5.0,
        conditional_prior_strength: float = 5.0,
        huber_delta: float = 1.5,
        minimum_scale: float = 1e-8,
        numeric_family_weight: float = 1.0,
        categorical_family_weight: float = 1.0,
        conditional_family_weight: float = 0.5,
        numeric_feature_weights: Mapping[str, float] | None = None,
        categorical_feature_weights: Mapping[str, float] | None = None,
        conditional_pair_weights: Mapping[ConditionalPair, float] | None = None,
        margin_weight: float = 0.25,
        max_iterations: int = 100,
    ) -> None:
        self._validate_dataframe(df)

        self.df = df.reset_index(drop=True).copy()
        self.categorical_columns = tuple(categorical_columns)
        self.numerical_columns = tuple(numerical_columns)
        self.conditional_pairs = tuple(conditional_pairs or ())

        self.categorical_prior_strength = self._validate_positive_float(
            categorical_prior_strength,
            "categorical_prior_strength",
        )
        self.scale_prior_strength = self._validate_non_negative_float(
            scale_prior_strength,
            "scale_prior_strength",
        )
        self.conditional_prior_strength = self._validate_non_negative_float(
            conditional_prior_strength,
            "conditional_prior_strength",
        )
        self.huber_delta = self._validate_positive_float(
            huber_delta,
            "huber_delta",
        )
        self.minimum_scale = self._validate_positive_float(
            minimum_scale,
            "minimum_scale",
        )
        self.numeric_family_weight = self._validate_non_negative_float(
            numeric_family_weight,
            "numeric_family_weight",
        )
        self.categorical_family_weight = self._validate_non_negative_float(
            categorical_family_weight,
            "categorical_family_weight",
        )
        self.conditional_family_weight = self._validate_non_negative_float(
            conditional_family_weight,
            "conditional_family_weight",
        )
        self.margin_weight = self._validate_non_negative_float(
            margin_weight,
            "margin_weight",
        )

        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError("'max_iterations' must be a positive integer.")
        self.max_iterations = max_iterations

        self._validate_columns()
        self._validate_conditional_pairs()

        self.numeric_feature_weights = self._build_weight_mapping(
            self.numerical_columns,
            numeric_feature_weights,
            "numeric_feature_weights",
        )
        self.categorical_feature_weights = self._build_weight_mapping(
            self.categorical_columns,
            categorical_feature_weights,
            "categorical_feature_weights",
        )
        self.conditional_pair_weights = self._build_weight_mapping(
            self.conditional_pairs,
            conditional_pair_weights,
            "conditional_pair_weights",
        )

        self._numerical_data = {
            column: pd.to_numeric(
                self.df[column],
                errors="coerce",
            ).to_numpy(dtype=float)
            for column in self.numerical_columns
        }
        self._categorical_data = {
            column: self.df[column]
            .astype("object")
            .where(pd.notna(self.df[column]), self._MISSING_TOKEN)
            .to_numpy(dtype=object)
            for column in self.categorical_columns
        }

        (
            self._global_numeric_profiles,
            self._global_category_probabilities,
            self._global_category_values,
        ) = self._build_global_statistics()

    def label_data(
        self,
        initial_clusters: Mapping[
            ClusterLabel,
            Sequence[int] | NDArray[np.integer[Any]],
        ],
        expansion_rate: float,
        *,
        expansion_step: float | None = None,
    ) -> LabelArray:
        """Assign a cluster label to every row.

        Parameters
        ----------
        initial_clusters
            Mapping from cluster labels to indices of representative,
            initially labeled objects.
        expansion_rate
            Initial fraction of provisional candidates retained in each
            cluster. Must be in ``(0, 1]``.
        expansion_step
            Increase in the retained fraction after each iteration. When
            omitted, ``expansion_rate`` is used.

        Returns
        -------
        numpy.ndarray
            One cluster label per input row, in the original row order.

        Raises
        ------
        ValueError
            If initial clusters or expansion parameters are invalid.
        """
        normalized_clusters = self._validate_initial_clusters(
            initial_clusters
        )
        current_fraction = self._validate_fraction(
            expansion_rate,
            "expansion_rate",
        )
        fraction_step = (
            current_fraction
            if expansion_step is None
            else self._validate_fraction(expansion_step, "expansion_step")
        )

        initial_labels = self._build_initial_assignments(normalized_clusters)
        current_assignments = initial_labels.copy()
        initial_index_to_label = {
            int(index): label
            for label, indices in normalized_clusters.items()
            for index in indices
        }

        for _ in range(self.max_iterations):
            prototypes = self.compute_prototypes(current_assignments)
            (
                provisional_labels,
                assignment_costs,
                selection_scores,
            ) = self._assign_to_best_clusters(
                prototypes,
                initial_index_to_label,
            )

            next_assignments = self._select_candidates(
                provisional_labels=provisional_labels,
                selection_scores=selection_scores,
                initial_clusters=normalized_clusters,
                retained_fraction=current_fraction,
            )

            all_assigned = all(label is not None for label in next_assignments)
            assignments_unchanged = np.array_equal(
                current_assignments,
                next_assignments,
            )

            current_assignments = next_assignments

            if all_assigned:
                break

            if current_fraction < 1.0:
                current_fraction = min(
                    1.0,
                    current_fraction + fraction_step,
                )
                continue

            if assignments_unchanged:
                break

        final_prototypes = self.compute_prototypes(current_assignments)
        final_labels, _, _ = self._assign_to_best_clusters(
            final_prototypes,
            initial_index_to_label,
        )
        return final_labels

    def compute_prototypes(
        self,
        cluster_assignments: Sequence[ClusterLabel | None],
    ) -> dict[ClusterLabel, ClusterPrototype]:
        """Build one robust prototype for every currently represented class."""
        assignments = np.asarray(cluster_assignments, dtype=object)
        if len(assignments) != len(self.df):
            raise ValueError(
                "'cluster_assignments' must contain one value per input row."
            )

        labels = [
            label
            for label in pd.unique(assignments)
            if label is not None
        ]
        if not labels:
            raise ValueError("At least one cluster must contain labeled rows.")

        prototypes: dict[ClusterLabel, ClusterPrototype] = {}
        for label in labels:
            indices = np.where(assignments == label)[0]
            prototypes[label] = self._build_cluster_prototype(indices)

        return prototypes

    def calculate_cluster_cost(
        self,
        row_index: int,
        prototype: ClusterPrototype,
    ) -> float:
        """Calculate the normalized assignment cost for one row and cluster."""
        if row_index < 0 or row_index >= len(self.df):
            raise IndexError("'row_index' is outside the input data.")

        family_costs: list[tuple[float, float]] = []

        numerical_cost = self._calculate_numerical_family_cost(
            row_index,
            prototype,
        )
        if numerical_cost is not None and self.numeric_family_weight > 0:
            family_costs.append(
                (self.numeric_family_weight, numerical_cost)
            )

        categorical_cost = self._calculate_categorical_family_cost(
            row_index,
            prototype,
        )
        if (
            categorical_cost is not None
            and self.categorical_family_weight > 0
        ):
            family_costs.append(
                (self.categorical_family_weight, categorical_cost)
            )

        conditional_cost = self._calculate_conditional_family_cost(
            row_index,
            prototype,
        )
        if (
            conditional_cost is not None
            and self.conditional_family_weight > 0
        ):
            family_costs.append(
                (self.conditional_family_weight, conditional_cost)
            )

        if not family_costs:
            raise ValueError(
                "No active feature family is available for this row."
            )

        return self._weighted_average(family_costs)

    def _build_cluster_prototype(
        self,
        indices: IndexArray,
    ) -> ClusterPrototype:
        numerical_profiles: dict[str, NumericProfile] = {}

        for column in self.numerical_columns:
            values = self._numerical_data[column][indices]
            observed = values[np.isfinite(values)]
            global_profile = self._global_numeric_profiles[column]

            if observed.size == 0:
                numerical_profiles[column] = NumericProfile(
                    center=global_profile.center,
                    scale=global_profile.scale,
                    count=0,
                )
                continue

            center = float(np.median(observed))
            scale = self._estimate_shrunk_scale(
                observed,
                fallback=global_profile.scale,
                prior_strength=self.scale_prior_strength,
            )
            numerical_profiles[column] = NumericProfile(
                center=center,
                scale=scale,
                count=int(observed.size),
            )

        categorical_profiles: dict[str, CategoricalProfile] = {}

        for column in self.categorical_columns:
            values = self._categorical_data[column][indices]
            categories = self._global_category_values[column]
            global_probabilities = self._global_category_probabilities[column]

            unique_values, counts = np.unique(
                values,
                return_counts=True,
            )
            count_lookup = dict(zip(unique_values, counts, strict=True))
            sample_count = len(values)
            denominator = (
                sample_count + self.categorical_prior_strength
            )

            probabilities = {
                category: (
                    count_lookup.get(category, 0)
                    + self.categorical_prior_strength
                    * global_probabilities[category]
                )
                / denominator
                for category in categories
            }

            category_count = max(len(categories), 2)
            categorical_profiles[column] = CategoricalProfile(
                probabilities=probabilities,
                default_probability=np.finfo(float).tiny,
                normalizer=max(math.log(category_count), 1.0),
            )

        conditional_profiles: dict[
            ConditionalPair,
            dict[Any, ConditionalProfile],
        ] = {}

        for categorical_column, numerical_column in self.conditional_pairs:
            categories = self._categorical_data[categorical_column][indices]
            numerical_values = self._numerical_data[numerical_column][indices]
            base_profile = numerical_profiles[numerical_column]
            pair_profiles: dict[Any, ConditionalProfile] = {}

            for category in np.unique(categories):
                category_mask = categories == category
                values = numerical_values[category_mask]
                observed = values[np.isfinite(values)]
                if observed.size == 0:
                    continue

                reliability = self._reliability(
                    int(observed.size),
                    self.conditional_prior_strength,
                )
                local_center = float(np.median(observed))
                center = (
                    reliability * local_center
                    + (1.0 - reliability) * base_profile.center
                )
                scale = self._estimate_shrunk_scale(
                    observed,
                    fallback=base_profile.scale,
                    prior_strength=self.conditional_prior_strength,
                )

                pair_profiles[category] = ConditionalProfile(
                    center=center,
                    scale=scale,
                    count=int(observed.size),
                    reliability=reliability,
                )

            conditional_profiles[
                (categorical_column, numerical_column)
            ] = pair_profiles

        return ClusterPrototype(
            numerical=numerical_profiles,
            categorical=categorical_profiles,
            conditional=conditional_profiles,
        )

    def _calculate_numerical_family_cost(
        self,
        row_index: int,
        prototype: ClusterPrototype,
    ) -> float | None:
        weighted_costs: list[tuple[float, float]] = []

        for column in self.numerical_columns:
            value = self._numerical_data[column][row_index]
            if not np.isfinite(value):
                continue

            profile = prototype.numerical[column]
            standardized = abs(value - profile.center) / profile.scale
            cost = self._huber(standardized)
            weighted_costs.append(
                (self.numeric_feature_weights[column], cost)
            )

        if not weighted_costs:
            return None
        return self._weighted_average(weighted_costs)

    def _calculate_categorical_family_cost(
        self,
        row_index: int,
        prototype: ClusterPrototype,
    ) -> float | None:
        weighted_costs: list[tuple[float, float]] = []

        for column in self.categorical_columns:
            value = self._categorical_data[column][row_index]
            profile = prototype.categorical[column]
            probability = profile.probabilities.get(
                value,
                profile.default_probability,
            )
            probability = max(
                float(probability),
                np.finfo(float).tiny,
            )
            cost = -math.log(probability) / profile.normalizer
            weighted_costs.append(
                (self.categorical_feature_weights[column], cost)
            )

        if not weighted_costs:
            return None
        return self._weighted_average(weighted_costs)

    def _calculate_conditional_family_cost(
        self,
        row_index: int,
        prototype: ClusterPrototype,
    ) -> float | None:
        if not self.conditional_pairs:
            return None

        weighted_costs: list[tuple[float, float]] = []

        for pair in self.conditional_pairs:
            categorical_column, numerical_column = pair
            numerical_value = self._numerical_data[numerical_column][
                row_index
            ]
            if not np.isfinite(numerical_value):
                continue

            category = self._categorical_data[categorical_column][row_index]
            base_profile = prototype.numerical[numerical_column]
            base_deviation = (
                abs(numerical_value - base_profile.center)
                / base_profile.scale
            )
            base_cost = self._huber(base_deviation)

            conditional_profile = prototype.conditional[pair].get(category)
            if conditional_profile is None:
                conditional_cost = base_cost
            else:
                conditional_deviation = (
                    abs(numerical_value - conditional_profile.center)
                    / conditional_profile.scale
                )
                local_cost = self._huber(conditional_deviation)
                reliability = conditional_profile.reliability
                conditional_cost = (
                    reliability * local_cost
                    + (1.0 - reliability) * base_cost
                )

            weighted_costs.append(
                (self.conditional_pair_weights[pair], conditional_cost)
            )

        if not weighted_costs:
            return None
        return self._weighted_average(weighted_costs)

    def _assign_to_best_clusters(
        self,
        prototypes: Mapping[ClusterLabel, ClusterPrototype],
        initial_index_to_label: Mapping[int, ClusterLabel],
    ) -> tuple[LabelArray, NDArray[np.float64], NDArray[np.float64]]:
        if not prototypes:
            raise ValueError("At least one cluster prototype is required.")

        labels = np.empty(len(self.df), dtype=object)
        assignment_costs = np.empty(len(self.df), dtype=float)
        selection_scores = np.empty(len(self.df), dtype=float)

        prototype_items = tuple(prototypes.items())

        for row_index in range(len(self.df)):
            if row_index in initial_index_to_label:
                labels[row_index] = initial_index_to_label[row_index]
                assignment_costs[row_index] = float("-inf")
                selection_scores[row_index] = float("-inf")
                continue

            scored_clusters = sorted(
                (
                    (
                        self.calculate_cluster_cost(
                            row_index,
                            prototype,
                        ),
                        label,
                    )
                    for label, prototype in prototype_items
                ),
                key=lambda item: item[0],
            )

            best_cost, best_label = scored_clusters[0]
            second_best_cost = (
                scored_clusters[1][0]
                if len(scored_clusters) > 1
                else best_cost
            )
            margin = max(second_best_cost - best_cost, 0.0)

            labels[row_index] = best_label
            assignment_costs[row_index] = best_cost
            selection_scores[row_index] = (
                best_cost - self.margin_weight * margin
            )

        return labels, assignment_costs, selection_scores

    def _select_candidates(
        self,
        provisional_labels: LabelArray,
        selection_scores: NDArray[np.float64],
        initial_clusters: Mapping[ClusterLabel, IndexArray],
        retained_fraction: float,
    ) -> LabelArray:
        selected = np.full(len(self.df), None, dtype=object)

        for label, initial_indices in initial_clusters.items():
            candidate_indices = np.where(
                provisional_labels == label
            )[0]
            retain_count = max(
                len(initial_indices),
                math.ceil(len(candidate_indices) * retained_fraction),
            )
            retain_count = min(retain_count, len(candidate_indices))

            candidate_scores = selection_scores[candidate_indices]
            order = np.argsort(candidate_scores, kind="stable")
            retained_indices = candidate_indices[order[:retain_count]]
            selected[retained_indices] = label
            selected[initial_indices] = label

        return selected

    def _build_global_statistics(
        self,
    ) -> tuple[
        dict[str, NumericProfile],
        dict[str, dict[Any, float]],
        dict[str, tuple[Any, ...]],
    ]:
        numeric_profiles: dict[str, NumericProfile] = {}

        for column in self.numerical_columns:
            values = self._numerical_data[column]
            observed = values[np.isfinite(values)]
            if observed.size == 0:
                raise ValueError(
                    f"Numerical column {column!r} contains no valid values."
                )

            center = float(np.median(observed))
            scale = self._raw_robust_scale(observed)
            if scale is None:
                scale = 1.0

            numeric_profiles[column] = NumericProfile(
                center=center,
                scale=max(scale, self.minimum_scale),
                count=int(observed.size),
            )

        category_probabilities: dict[str, dict[Any, float]] = {}
        category_values: dict[str, tuple[Any, ...]] = {}

        for column in self.categorical_columns:
            values = self._categorical_data[column]
            unique_values, counts = np.unique(
                values,
                return_counts=True,
            )
            probabilities = counts / counts.sum()

            category_values[column] = tuple(unique_values.tolist())
            category_probabilities[column] = {
                category: float(probability)
                for category, probability in zip(
                    unique_values,
                    probabilities,
                    strict=True,
                )
            }

        return (
            numeric_profiles,
            category_probabilities,
            category_values,
        )

    def _estimate_shrunk_scale(
        self,
        values: NDArray[np.float64],
        *,
        fallback: float,
        prior_strength: float,
    ) -> float:
        local_scale = self._raw_robust_scale(values)
        if local_scale is None:
            return max(fallback, self.minimum_scale)

        reliability = self._reliability(
            int(values.size),
            prior_strength,
        )
        scale = (
            reliability * local_scale
            + (1.0 - reliability) * fallback
        )
        return max(float(scale), self.minimum_scale)

    def _raw_robust_scale(
        self,
        values: NDArray[np.float64],
    ) -> float | None:
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]

        if finite_values.size < 2:
            return None

        median = float(np.median(finite_values))
        mad = float(np.median(np.abs(finite_values - median)))
        mad_scale = 1.4826 * mad
        if mad_scale > self.minimum_scale:
            return mad_scale

        first_quartile, third_quartile = np.quantile(
            finite_values,
            [0.25, 0.75],
        )
        iqr_scale = float((third_quartile - first_quartile) / 1.349)
        if iqr_scale > self.minimum_scale:
            return iqr_scale

        standard_deviation = float(
            np.std(finite_values, ddof=1)
        )
        if (
            math.isfinite(standard_deviation)
            and standard_deviation > self.minimum_scale
        ):
            return standard_deviation

        return None

    def _huber(self, standardized_deviation: float) -> float:
        deviation = abs(float(standardized_deviation))
        if deviation <= self.huber_delta:
            return 0.5 * deviation**2
        return self.huber_delta * (
            deviation - 0.5 * self.huber_delta
        )

    @staticmethod
    def _weighted_average(
        weighted_values: Sequence[tuple[float, float]],
    ) -> float:
        active_values = [
            (weight, value)
            for weight, value in weighted_values
            if weight > 0 and math.isfinite(value)
        ]
        if not active_values:
            raise ValueError("No finite, positively weighted values exist.")

        total_weight = sum(weight for weight, _ in active_values)
        weighted_sum = sum(
            weight * value for weight, value in active_values
        )
        return float(weighted_sum / total_weight)

    @staticmethod
    def _reliability(sample_count: int, prior_strength: float) -> float:
        if sample_count <= 0:
            return 0.0
        if prior_strength == 0:
            return 1.0
        return float(sample_count / (sample_count + prior_strength))

    def _build_initial_assignments(
        self,
        initial_clusters: Mapping[ClusterLabel, IndexArray],
    ) -> LabelArray:
        assignments = np.full(len(self.df), None, dtype=object)
        for label, indices in initial_clusters.items():
            assignments[indices] = label
        return assignments

    def _validate_initial_clusters(
        self,
        initial_clusters: Mapping[
            ClusterLabel,
            Sequence[int] | NDArray[np.integer[Any]],
        ],
    ) -> dict[ClusterLabel, IndexArray]:
        if not initial_clusters:
            raise ValueError(
                "'initial_clusters' must contain at least one cluster."
            )

        normalized: dict[ClusterLabel, IndexArray] = {}
        seen_indices: set[int] = set()

        for label, indices in initial_clusters.items():
            index_array = np.asarray(indices, dtype=int).reshape(-1)
            if index_array.size == 0:
                raise ValueError(
                    f"Initial cluster {label!r} cannot be empty."
                )
            if (
                np.any(index_array < 0)
                or np.any(index_array >= len(self.df))
            ):
                raise ValueError(
                    f"Initial cluster {label!r} contains an index "
                    "outside the input data."
                )
            if len(np.unique(index_array)) != len(index_array):
                raise ValueError(
                    f"Initial cluster {label!r} contains duplicate indices."
                )

            overlap = seen_indices.intersection(
                int(index) for index in index_array
            )
            if overlap:
                duplicates = ", ".join(
                    str(index) for index in sorted(overlap)
                )
                raise ValueError(
                    "An object cannot belong to multiple initial clusters. "
                    f"Duplicated indices: {duplicates}."
                )

            seen_indices.update(int(index) for index in index_array)
            normalized[label] = index_array

        return normalized

    def _validate_columns(self) -> None:
        if not self.categorical_columns:
            raise ValueError(
                "'categorical_columns' must contain at least one column."
            )
        if not self.numerical_columns:
            raise ValueError(
                "'numerical_columns' must contain at least one column."
            )

        all_columns = set(self.df.columns)
        requested_columns = (
            set(self.categorical_columns)
            | set(self.numerical_columns)
        )
        missing_columns = requested_columns - all_columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"The following columns are missing from 'df': {missing}."
            )

        overlap = set(self.categorical_columns).intersection(
            self.numerical_columns
        )
        if overlap:
            duplicated = ", ".join(sorted(overlap))
            raise ValueError(
                "Categorical and numerical columns must be disjoint. "
                f"Overlapping columns: {duplicated}."
            )

        if len(set(self.categorical_columns)) != len(
            self.categorical_columns
        ):
            raise ValueError(
                "'categorical_columns' contains duplicate names."
            )
        if len(set(self.numerical_columns)) != len(
            self.numerical_columns
        ):
            raise ValueError(
                "'numerical_columns' contains duplicate names."
            )

    def _validate_conditional_pairs(self) -> None:
        if len(set(self.conditional_pairs)) != len(
            self.conditional_pairs
        ):
            raise ValueError("'conditional_pairs' contains duplicates.")

        categorical_set = set(self.categorical_columns)
        numerical_set = set(self.numerical_columns)

        for categorical_column, numerical_column in self.conditional_pairs:
            if categorical_column not in categorical_set:
                raise ValueError(
                    f"{categorical_column!r} is not a categorical column."
                )
            if numerical_column not in numerical_set:
                raise ValueError(
                    f"{numerical_column!r} is not a numerical column."
                )

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("'df' must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("'df' cannot be empty.")

    @staticmethod
    def _validate_positive_float(value: float, name: str) -> float:
        numeric_value = float(value)
        if not math.isfinite(numeric_value) or numeric_value <= 0:
            raise ValueError(f"'{name}' must be finite and greater than zero.")
        return numeric_value

    @staticmethod
    def _validate_non_negative_float(value: float, name: str) -> float:
        numeric_value = float(value)
        if not math.isfinite(numeric_value) or numeric_value < 0:
            raise ValueError(
                f"'{name}' must be finite and non-negative."
            )
        return numeric_value

    @staticmethod
    def _validate_fraction(value: float, name: str) -> float:
        numeric_value = float(value)
        if (
            not math.isfinite(numeric_value)
            or numeric_value <= 0
            or numeric_value > 1
        ):
            raise ValueError(f"'{name}' must be in the interval (0, 1].")
        return numeric_value

    def _build_weight_mapping(
        self,
        keys: Sequence[Any],
        weights: Mapping[Any, float] | None,
        name: str,
    ) -> dict[Any, float]:
        result = {key: 1.0 for key in keys}
        if weights is None:
            return result

        unknown_keys = set(weights) - set(keys)
        if unknown_keys:
            unknown = ", ".join(repr(key) for key in unknown_keys)
            raise ValueError(
                f"'{name}' contains unsupported keys: {unknown}."
            )

        for key, value in weights.items():
            result[key] = self._validate_non_negative_float(
                value,
                f"{name}[{key!r}]",
            )

        return result