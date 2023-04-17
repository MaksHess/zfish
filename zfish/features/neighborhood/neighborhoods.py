"""
Functions to compute radius, knn or touch adjacejcy matrices based on KDTrees or label images in case of touch neighborhood.
"""
from enum import Enum
from functools import partial
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import KDTree


def radius_adjacency_matrix(
    kdtree: KDTree,
    r: float,
    include_self: bool = True,
    not_adjacent_value: float = np.nan,
):
    query = kdtree.query_radius(kdtree.data, r=r, return_distance=False)

    adjacency_matrix = np.empty((kdtree.data.shape[0], kdtree.data.shape[0]))
    adjacency_matrix.fill(not_adjacent_value)
    for i, r in enumerate(query):
        adjacency_matrix[i, r] = 1.0
    if not include_self:
        np.fill_diagonal(adjacency_matrix, not_adjacent_value)
    return adjacency_matrix


def knn_adjacency_matrix(
    kdtree: KDTree,
    k: int,
    include_self: bool = True,
    not_adjacent_value: float = np.nan,
):
    if k > (kdtree.data.shape[0] - 1):
        print(
            f"k = {k} is larger than number of objects - 1 ({kdtree.data.shape[0]-1}), setting k = {kdtree.data.shape[0]-1}"
        )
        k = kdtree.data.shape[0] - 1

    query = kdtree.query(kdtree.data, k=k + 1, return_distance=False)

    adjacency_matrix = np.empty((kdtree.data.shape[0], kdtree.data.shape[0]))
    adjacency_matrix.fill(not_adjacent_value)
    for i, r in enumerate(query):
        adjacency_matrix[i, r] = 1.0
    if not include_self:
        np.fill_diagonal(adjacency_matrix, not_adjacent_value)
    return adjacency_matrix


# TODO: Handle large k values
def touch_adjacency_matrix(
    weighted_touch_matrix: NDArray,
    k_neighbors: int,
    include_self: bool = True,
    threshold_area: float = 0.0,
    not_adjacent_value: float = np.nan,
):
    touching = weighted_touch_matrix > threshold_area
    k_touching = np.linalg.matrix_power(touching, k_neighbors)
    adjacency_matrix = np.where(k_touching, 1.0, not_adjacent_value)
    if not include_self:
        np.fill_diagonal(adjacency_matrix, not_adjacent_value)
    return adjacency_matrix


class Nhd(Enum):
    DIST = partial(radius_adjacency_matrix)
    KNN = partial(knn_adjacency_matrix)

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)
