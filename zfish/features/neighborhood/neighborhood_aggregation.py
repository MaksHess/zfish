"""
Functions to aggregate feature tables across neighborhoods represented by adjacency matrices and unsing custom aggregation functions.
"""
from typing import Callable

import numba as nb
import numpy as np


def aggregate_table_nan(
    adjacency_matrix, feature_table, nan_skipping_aggregaction_function
):
    return np.apply_along_axis(
        nan_skipping_aggregaction_function,
        1,
        (
            np.expand_dims(adjacency_matrix, axis=-1)
            * np.expand_dims(feature_table, axis=0)
        ),
    )


@nb.jit(parallel=True, cache=False)
def aggregate_column_dense_parallel(
    adjacency_matrix: np.ndarray,
    feature_column: np.ndarray,
    aggregation_function: Callable,
) -> np.ndarray:

    neighborhood_features = np.zeros_like(feature_column)
    for observation_index in nb.prange(adjacency_matrix.shape[0]):
        adjacency_array = adjacency_matrix[observation_index, :]
        adjacency_mask = adjacency_array > 0

        adjacent_features = np.atleast_2d(feature_column[adjacency_mask])
        neighborhood_features[observation_index] = aggregation_function(
            adjacent_features
        )
    return neighborhood_features


@nb.njit(parallel=True)
def aggregate_table_dense_parallel(
    adjacency_matrix: np.ndarray,
    feature_array: np.ndarray,
    aggregation_function: Callable,
) -> np.ndarray:
    neighborhood_features = np.zeros_like(feature_array)

    for var_index in nb.prange(feature_array.shape[1]):
        feature_column = feature_array[:, var_index]
        neighborhood_features[:, var_index] = aggregate_column_dense_parallel(
            adjacency_matrix=adjacency_matrix,
            feature_column=feature_column,
            aggregation_function=aggregation_function,
        )
    return neighborhood_features
