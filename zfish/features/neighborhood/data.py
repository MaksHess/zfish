"""
Generate feature vectors / tables for testing.
Generate neighborhoods for testing.
"""
import numpy as np
from scipy import sparse
from sklearn.neighbors import KDTree

from neighborhoods import knn_adjacency_matrix, radius_adjacency_matrix


def get_random_features(n_obs, n_var, seed=42):
    rng = np.random.default_rng(seed=seed)
    return rng.uniform(0.0, 1.0, size=(n_obs, n_var))


def get_linear_features(n_obs, n_var, min_value=0.0, max_value=1.0):
    feature = np.linspace(min_value, max_value, n_obs).reshape(-1, 1)
    return np.ones((1, n_var)) * feature


def get_random_kdtree(n_obs, ndim=3, seed=42):
    rng = np.random.default_rng(seed=seed)
    pointcloud = rng.uniform(0, 1, size=(n_obs, ndim))
    return KDTree(pointcloud)


def get_random_radius_adjacency_matrix(
    r: float,
    include_self: bool,
    n_obs: int,
    not_adjacent_value: float = np.nan,
    seed: int | None = 42,
):
    kdtree = get_random_kdtree(n_obs=n_obs, seed=seed)
    return radius_adjacency_matrix(
        kdtree=kdtree,
        r=r,
        include_self=include_self,
        not_adjacent_value=not_adjacent_value,
    )


def get_random_knn_adjacency_matrix(
    k: int,
    include_self: bool,
    n_obs: int,
    not_adjacent_value: float = np.nan,
    seed: int | None = 42,
):
    kdtree = get_random_kdtree(n_obs=n_obs, seed=seed)
    return knn_adjacency_matrix(
        kdtree=kdtree,
        k=k,
        include_self=include_self,
        not_adjacent_value=not_adjacent_value,
    )


def get_empty_adjacency_matrix(n_obs: int, not_adjacent_value: float = np.nan):
    adjacency_matrix = np.empty((n_obs, n_obs))
    adjacency_matrix.fill(not_adjacent_value)
    return adjacency_matrix


def get_self_adjacency_matrix(n_obs: int, not_adjacent_value: float = np.nan):
    adjacency_matrix = np.empty((n_obs, n_obs))
    adjacency_matrix.fill(not_adjacent_value)
    np.fill_diagonal(adjacency_matrix, 1.0)
    return adjacency_matrix


def get_full_adjacency_matrix(n_obs: int, not_adjacent_value: float = np.nan):
    return np.ones((n_obs, n_obs))


def _to_dense(nhd):
    if sparse.issparse(nhd):
        return nhd.toarray()
    return np.nan_to_num(nhd)
