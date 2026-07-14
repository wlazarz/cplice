"""Efficient categorical distance matrices for CPLICE.

The public entry point is :func:`compute_distance_matrix`. All measures use
feature-wise category statistics and return symmetric dissimilarity matrices
whose diagonal is zero.

The implementation processes square blocks from the upper triangle and mirrors
them to the lower triangle. This reduces repeated calculations and controls
temporary memory usage. Large matrices may be written directly to a NumPy
memory-mapped file.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, DTypeLike, NDArray

MetricName: TypeAlias = Literal[
    "dice",
    "eskin",
    "hamming",
    "iof",
    "jaccard",
    "lin",
    "overlap",
    "s2",
]
FloatMatrix: TypeAlias = NDArray[np.floating[Any]]

SUPPORTED_METRICS: Final[frozenset[str]] = frozenset(
    {
        "dice",
        "eskin",
        "hamming",
        "iof",
        "jaccard",
        "lin",
        "overlap",
        "s2",
    }
)

_METRIC_ALIASES: Final[dict[str, str]] = {
    "morlini": "s2",
    "morlini_zani": "s2",
    "mz": "s2",
    "simple_matching": "overlap",
}


@dataclass(frozen=True, slots=True)
class FeatureStatistics:
    """Encoded values and frequency statistics for one feature."""

    codes: NDArray[np.int_]
    code_by_value: Mapping[tuple[str, Any], int]
    counts: NDArray[np.int_]
    probabilities: NDArray[np.float64]

    @property
    def category_count(self) -> int:
        """Return the number of observed categories."""
        return len(self.counts)


@dataclass(frozen=True, slots=True)
class DistanceMatrixMetadata:
    """Metadata describing a calculated matrix."""

    metric: str
    sample_count: int
    feature_count: int
    block_size: int
    dtype: str
    output_path: str | None


class CategoricalDistanceCalculator:
    """Prepare categorical statistics and calculate CPLICE distances.

    Parameters
    ----------
    data
        Categorical observations. Rows are objects and columns are features.
    metric
        Distance measure name. Supported values are ``"overlap"``,
        ``"hamming"``, ``"jaccard"``, ``"dice"``, ``"eskin"``, ``"iof"``,
        ``"lin"``, and ``"s2"``.
    feature_weights
        Optional non-negative weights. A sequence follows column order; a
        mapping is supported when ``data`` is a pandas DataFrame.
    working_memory_mb
        Approximate memory budget used to choose the automatic block size.

    Notes
    -----
    ``dice`` preserves the coordinate-wise categorical definition used in the
    original project. Under that definition it is mathematically equivalent to
    normalized Hamming/overlap distance.

    The S2 implementation works directly on categorical codes. It is
    algebraically equivalent to dummy coding but avoids materializing the full
    one-hot matrix.
    """

    def __init__(
        self,
        data: pd.DataFrame | ArrayLike,
        metric: str,
        *,
        feature_weights: Sequence[float] | Mapping[str, float] | None = None,
        working_memory_mb: float = 256.0,
    ) -> None:
        (
            self.data,
            self.column_names,
        ) = _coerce_categorical_data(data)

        self.metric: str = normalize_metric_name(metric)
        self.feature_weights = _prepare_feature_weights(
            feature_weights,
            self.column_names,
        )
        self.total_feature_weight = float(self.feature_weights.sum())
        self.working_memory_mb = _validate_positive_number(
            working_memory_mb,
            "working_memory_mb",
        )

        self.features = tuple(
            _encode_feature(self.data[:, column_index])
            for column_index in range(self.data.shape[1])
        )
        self.codes = np.column_stack(
            [feature.codes for feature in self.features]
        ).astype(np.int32, copy=False)

        self._eskin_mismatch_costs = np.asarray(
            [
                2.0 / (feature.category_count**2 + 2.0)
                for feature in self.features
            ],
            dtype=np.float64,
        )
        self._log_counts = tuple(
            np.log(feature.counts.astype(np.float64))
            for feature in self.features
        )
        self._log_probabilities = tuple(
            np.log(feature.probabilities)
            for feature in self.features
        )
        self._s2_category_weights = tuple(
            -2.0 * np.log(feature.probabilities)
            for feature in self.features
        )

        self.last_metadata: DistanceMatrixMetadata | None = None

    @property
    def sample_count(self) -> int:
        """Return the number of objects."""
        return self.data.shape[0]

    @property
    def feature_count(self) -> int:
        """Return the number of features."""
        return self.data.shape[1]

    def distance(self, first: ArrayLike, second: ArrayLike) -> float:
        """Calculate the configured distance between two categorical objects.

        Both objects must use categories observed in the data supplied to the
        constructor. This condition is naturally satisfied by CPLICE objects
        and mode-based centroids.
        """
        first_codes = self._encode_object(first, "first")
        second_codes = self._encode_object(second, "second")
        result = self._calculate_block(
            first_codes.reshape(1, -1),
            second_codes.reshape(1, -1),
        )
        return float(result[0, 0])

    def pairwise(
        self,
        *,
        block_size: int | None = None,
        dtype: DTypeLike = np.float64,
        output_path: str | Path | None = None,
    ) -> FloatMatrix | np.memmap:
        """Calculate a symmetric pairwise distance matrix.

        Parameters
        ----------
        block_size
            Side length of a temporary square block. When omitted, a value is
            selected from ``working_memory_mb`` and the chosen metric.
        dtype
            Floating output dtype. ``float32`` halves matrix storage but may
            slightly change tie-breaking in highly similar objects.
        output_path
            Optional path for a NumPy memory-mapped matrix. Use this when a
            dense in-memory ``n x n`` matrix would be too large.

        Returns
        -------
        numpy.ndarray or numpy.memmap
            Symmetric square distance matrix.
        """
        output_dtype = np.dtype(dtype)
        if not np.issubdtype(output_dtype, np.floating):
            raise TypeError("'dtype' must be a floating-point dtype.")

        resolved_block_size = (
            self._automatic_block_size()
            if block_size is None
            else _validate_positive_integer(block_size, "block_size")
        )
        resolved_block_size = min(
            resolved_block_size,
            self.sample_count,
        )

        matrix = _allocate_output_matrix(
            self.sample_count,
            output_dtype,
            output_path,
        )

        for row_start in range(
            0,
            self.sample_count,
            resolved_block_size,
        ):
            row_stop = min(
                row_start + resolved_block_size,
                self.sample_count,
            )
            left_codes = self.codes[row_start:row_stop]

            for column_start in range(
                row_start,
                self.sample_count,
                resolved_block_size,
            ):
                column_stop = min(
                    column_start + resolved_block_size,
                    self.sample_count,
                )
                right_codes = self.codes[column_start:column_stop]

                block = self._calculate_block(
                    left_codes,
                    right_codes,
                ).astype(output_dtype, copy=False)

                matrix[
                    row_start:row_stop,
                    column_start:column_stop,
                ] = block

                if column_start != row_start:
                    matrix[
                        column_start:column_stop,
                        row_start:row_stop,
                    ] = block.T

        np.fill_diagonal(matrix, 0.0)

        if isinstance(matrix, np.memmap):
            matrix.flush()

        self.last_metadata = DistanceMatrixMetadata(
            metric=self.metric,
            sample_count=self.sample_count,
            feature_count=self.feature_count,
            block_size=resolved_block_size,
            dtype=output_dtype.name,
            output_path=(
                str(Path(output_path))
                if output_path is not None
                else None
            ),
        )
        return matrix

    def _calculate_block(
        self,
        left_codes: NDArray[np.integer[Any]],
        right_codes: NDArray[np.integer[Any]],
    ) -> NDArray[np.float64]:
        if self.metric in {
            "dice",
            "hamming",
            "jaccard",
            "overlap",
        }:
            return self._matching_distance_block(
                left_codes,
                right_codes,
            )
        if self.metric == "eskin":
            return self._eskin_distance_block(
                left_codes,
                right_codes,
            )
        if self.metric == "iof":
            return self._iof_distance_block(
                left_codes,
                right_codes,
            )
        if self.metric == "lin":
            return self._lin_distance_block(
                left_codes,
                right_codes,
            )
        if self.metric == "s2":
            return self._s2_distance_block(
                left_codes,
                right_codes,
            )

        raise RuntimeError(
            f"No implementation exists for metric {self.metric!r}."
        )

    def _matching_distance_block(
        self,
        left_codes: NDArray[np.integer[Any]],
        right_codes: NDArray[np.integer[Any]],
    ) -> NDArray[np.float64]:
        mismatch_weight = np.zeros(
            (len(left_codes), len(right_codes)),
            dtype=np.float64,
        )

        for feature_index, weight in enumerate(self.feature_weights):
            if weight == 0:
                continue
            mismatch_weight += weight * (
                left_codes[:, feature_index, None]
                != right_codes[None, :, feature_index]
            )

        if self.metric == "hamming":
            return mismatch_weight

        if self.metric in {"dice", "overlap"}:
            return mismatch_weight / self.total_feature_weight

        denominator = self.total_feature_weight + mismatch_weight
        return np.divide(
            2.0 * mismatch_weight,
            denominator,
            out=np.zeros_like(mismatch_weight),
            where=denominator > 0,
        )

    def _eskin_distance_block(
        self,
        left_codes: NDArray[np.integer[Any]],
        right_codes: NDArray[np.integer[Any]],
    ) -> NDArray[np.float64]:
        distance = np.zeros(
            (len(left_codes), len(right_codes)),
            dtype=np.float64,
        )

        for feature_index, weight in enumerate(self.feature_weights):
            if weight == 0:
                continue
            mismatch = (
                left_codes[:, feature_index, None]
                != right_codes[None, :, feature_index]
            )
            distance += (
                weight
                * self._eskin_mismatch_costs[feature_index]
                * mismatch
            )

        return distance / self.total_feature_weight

    def _iof_distance_block(
        self,
        left_codes: NDArray[np.integer[Any]],
        right_codes: NDArray[np.integer[Any]],
    ) -> NDArray[np.float64]:
        distance = np.zeros(
            (len(left_codes), len(right_codes)),
            dtype=np.float64,
        )

        for feature_index, weight in enumerate(self.feature_weights):
            if weight == 0:
                continue

            left_feature_codes = left_codes[:, feature_index]
            right_feature_codes = right_codes[:, feature_index]
            mismatch = (
                left_feature_codes[:, None]
                != right_feature_codes[None, :]
            )

            log_counts = self._log_counts[feature_index]
            frequency_product = (
                log_counts[left_feature_codes][:, None]
                * log_counts[right_feature_codes][None, :]
            )
            mismatch_distance = np.divide(
                frequency_product,
                1.0 + frequency_product,
                out=np.zeros_like(frequency_product),
                where=mismatch,
            )
            distance += weight * mismatch_distance

        return distance / self.total_feature_weight

    def _lin_distance_block(
        self,
        left_codes: NDArray[np.integer[Any]],
        right_codes: NDArray[np.integer[Any]],
    ) -> NDArray[np.float64]:
        numerator = np.zeros(
            (len(left_codes), len(right_codes)),
            dtype=np.float64,
        )
        denominator = np.zeros_like(numerator)

        for feature_index, weight in enumerate(self.feature_weights):
            if weight == 0:
                continue

            left_feature_codes = left_codes[:, feature_index]
            right_feature_codes = right_codes[:, feature_index]
            probabilities = self.features[
                feature_index
            ].probabilities
            log_probabilities = self._log_probabilities[
                feature_index
            ]

            left_logs = log_probabilities[
                left_feature_codes
            ][:, None]
            right_logs = log_probabilities[
                right_feature_codes
            ][None, :]
            denominator += weight * (left_logs + right_logs)

            same = (
                left_feature_codes[:, None]
                == right_feature_codes[None, :]
            )
            match_numerator = 2.0 * left_logs
            mismatch_numerator = 2.0 * np.log(
                probabilities[left_feature_codes][:, None]
                + probabilities[right_feature_codes][None, :]
            )
            numerator += weight * np.where(
                same,
                match_numerator,
                mismatch_numerator,
            )

        similarity = np.divide(
            numerator,
            denominator,
            out=np.ones_like(numerator),
            where=np.abs(denominator) > np.finfo(float).eps,
        )
        similarity = np.clip(similarity, 0.0, 1.0)
        return 1.0 - similarity

    def _s2_distance_block(
        self,
        left_codes: NDArray[np.integer[Any]],
        right_codes: NDArray[np.integer[Any]],
    ) -> NDArray[np.float64]:
        shared_information = np.zeros(
            (len(left_codes), len(right_codes)),
            dtype=np.float64,
        )
        total_information = np.zeros_like(shared_information)

        for feature_index, weight in enumerate(self.feature_weights):
            if weight == 0:
                continue

            left_feature_codes = left_codes[:, feature_index]
            right_feature_codes = right_codes[:, feature_index]
            information_weights = self._s2_category_weights[
                feature_index
            ]

            left_information = information_weights[
                left_feature_codes
            ][:, None]
            right_information = information_weights[
                right_feature_codes
            ][None, :]
            same = (
                left_feature_codes[:, None]
                == right_feature_codes[None, :]
            )

            shared_information += weight * np.where(
                same,
                left_information,
                0.0,
            )
            total_information += weight * np.where(
                same,
                left_information,
                left_information + right_information,
            )

        similarity = np.divide(
            shared_information,
            total_information,
            out=np.ones_like(shared_information),
            where=total_information > np.finfo(float).eps,
        )
        similarity = np.clip(similarity, 0.0, 1.0)
        return 1.0 - similarity

    def _encode_object(
        self,
        obj: ArrayLike,
        argument_name: str,
    ) -> NDArray[np.int_]:
        values = np.asarray(obj, dtype=object).reshape(-1)
        if len(values) != self.feature_count:
            raise ValueError(
                f"'{argument_name}' must contain {self.feature_count} "
                f"features, received {len(values)}."
            )

        codes = np.empty(self.feature_count, dtype=int)

        for feature_index, value in enumerate(values):
            key = _category_key(value)
            code = self.features[feature_index].code_by_value.get(key)
            if code is None:
                column_name = self.column_names[feature_index]
                raise ValueError(
                    f"Unknown category {value!r} in feature "
                    f"{column_name!r}."
                )
            codes[feature_index] = code

        return codes

    def _automatic_block_size(self) -> int:
        temporary_array_factor = {
            "dice": 3,
            "eskin": 3,
            "hamming": 3,
            "iof": 6,
            "jaccard": 4,
            "lin": 8,
            "overlap": 3,
            "s2": 8,
        }[self.metric]

        available_bytes = self.working_memory_mb * 1024**2
        estimated_elements = available_bytes / (
            temporary_array_factor * np.dtype(np.float64).itemsize
        )
        block_size = int(math.sqrt(max(estimated_elements, 1.0)))
        return max(1, min(block_size, 2048))


def compute_distance_matrix(
    data: pd.DataFrame | ArrayLike,
    metric: str,
    *,
    feature_weights: Sequence[float] | Mapping[str, float] | None = None,
    block_size: int | None = None,
    working_memory_mb: float = 256.0,
    dtype: DTypeLike = np.float64,
    output_path: str | Path | None = None,
) -> FloatMatrix | np.memmap:
    """Calculate a CPLICE-ready pairwise distance matrix.

    Examples
    --------
    Compute an in-memory Lin distance matrix:

    >>> matrix = compute_distance_matrix(dataframe, "lin")

    Write a large S2 matrix directly to disk:

    >>> matrix = compute_distance_matrix(
    ...     dataframe,
    ...     "s2",
    ...     dtype=np.float32,
    ...     output_path="s2_distances.dat",
    ... )
    """
    calculator = CategoricalDistanceCalculator(
        data,
        metric,
        feature_weights=feature_weights,
        working_memory_mb=working_memory_mb,
    )
    return calculator.pairwise(
        block_size=block_size,
        dtype=dtype,
        output_path=output_path,
    )


def calculate_distance(
    first: ArrayLike,
    second: ArrayLike,
    data: pd.DataFrame | ArrayLike,
    metric: str,
    *,
    feature_weights: Sequence[float] | Mapping[str, float] | None = None,
) -> float:
    """Calculate one distance using statistics estimated from ``data``."""
    calculator = CategoricalDistanceCalculator(
        data,
        metric,
        feature_weights=feature_weights,
    )
    return calculator.distance(first, second)


def normalize_metric_name(metric: str) -> str:
    """Normalize and validate a distance metric name."""
    if not isinstance(metric, str):
        raise TypeError("'metric' must be a string.")

    normalized = metric.strip().lower().replace("-", "_")
    normalized = _METRIC_ALIASES.get(normalized, normalized)

    if normalized not in SUPPORTED_METRICS:
        supported = ", ".join(sorted(SUPPORTED_METRICS))
        raise ValueError(
            f"Unsupported metric {metric!r}. Supported metrics: "
            f"{supported}."
        )

    return normalized


def _coerce_categorical_data(
    data: pd.DataFrame | ArrayLike,
) -> tuple[NDArray[Any], tuple[str, ...]]:
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError(
                "'data' must contain at least one row and one feature."
            )
        values = data.to_numpy(dtype=object, copy=True)
        column_names = tuple(str(column) for column in data.columns)
    else:
        values = np.asarray(data, dtype=object)
        if values.ndim != 2:
            raise ValueError("'data' must be two-dimensional.")
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError(
                "'data' must contain at least one row and one feature."
            )
        column_names = tuple(
            f"feature_{index}"
            for index in range(values.shape[1])
        )

    if values.ndim != 2:
        raise ValueError("'data' must be two-dimensional.")

    return values, column_names


def _encode_feature(values: NDArray[Any]) -> FeatureStatistics:
    series = pd.Series(values, dtype="object")
    codes, uniques = pd.factorize(
        series,
        sort=False,
        use_na_sentinel=False,
    )
    counts = np.bincount(codes, minlength=len(uniques)).astype(int)
    probabilities = counts.astype(np.float64) / len(values)

    code_by_value = {
        _category_key(value): int(code)
        for code, value in enumerate(uniques.tolist())
    }

    return FeatureStatistics(
        codes=codes.astype(int, copy=False),
        code_by_value=code_by_value,
        counts=counts,
        probabilities=probabilities,
    )


def _prepare_feature_weights(
    feature_weights: Sequence[float] | Mapping[str, float] | None,
    column_names: tuple[str, ...],
) -> NDArray[np.float64]:
    if feature_weights is None:
        weights = np.ones(len(column_names), dtype=np.float64)
    elif isinstance(feature_weights, Mapping):
        unknown_columns = set(feature_weights) - set(column_names)
        if unknown_columns:
            unknown = ", ".join(sorted(unknown_columns))
            raise ValueError(
                "'feature_weights' contains unknown columns: "
                f"{unknown}."
            )
        weights = np.asarray(
            [
                feature_weights.get(column_name, 1.0)
                for column_name in column_names
            ],
            dtype=np.float64,
        )
    else:
        weights = np.asarray(feature_weights, dtype=np.float64)
        if weights.ndim != 1 or len(weights) != len(column_names):
            raise ValueError(
                "'feature_weights' must contain one value per feature."
            )

    if not np.all(np.isfinite(weights)):
        raise ValueError(
            "'feature_weights' must contain only finite values."
        )
    if np.any(weights < 0):
        raise ValueError("'feature_weights' cannot be negative.")
    if float(weights.sum()) <= 0:
        raise ValueError(
            "At least one feature weight must be greater than zero."
        )

    return weights


def _allocate_output_matrix(
    sample_count: int,
    dtype: np.dtype[Any],
    output_path: str | Path | None,
) -> FloatMatrix | np.memmap:
    shape = (sample_count, sample_count)

    if output_path is None:
        return np.empty(shape, dtype=dtype)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )


def _category_key(value: Any) -> tuple[str, Any]:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False

    if np.isscalar(missing) and bool(missing):
        return ("missing", None)

    try:
        hash(value)
        return ("value", value)
    except TypeError:
        return ("representation", repr(value))


def _validate_positive_integer(value: int, name: str) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(f"'{name}' must be a positive integer.")
    return int(value)


def _validate_positive_number(value: float, name: str) -> float:
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(
            f"'{name}' must be finite and greater than zero."
        )
    return numeric_value