"""Shared embedding-output validation."""

import numpy as np
import numpy.typing as npt


def require_positive_dimension(dimension: int | None) -> int:
    """Validate and return an embedding dimension."""
    if dimension is None or dimension <= 0:
        raise ValueError("embedding model must report a positive dimension")

    return dimension


def validate_embedding_matrix(
    values: object,
    *,
    row_count: int,
    dimension: int,
) -> npt.NDArray[np.float64]:
    """Convert and validate a matrix returned by an embedding model."""
    matrix = np.asarray(values, dtype=np.float64)
    expected_shape = (row_count, dimension)

    if matrix.shape != expected_shape:
        raise ValueError(
            f"embedding model returned shape {matrix.shape}; expected {expected_shape}"
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError("embedding model returned non-finite values")

    return matrix
