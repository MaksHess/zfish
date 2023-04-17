import numpy as np

from data import (
    get_empty_adjacency_matrix,
    get_full_adjacency_matrix,
    get_random_kdtree,
    get_self_adjacency_matrix,
)
from neighborhoods import knn_adjacency_matrix, radius_adjacency_matrix

N_OBS = 10


def test_zero_radius_wihout_self_is_empty_neighborhood():
    kdtree = get_random_kdtree(n_obs=N_OBS)
    radius_nhd = radius_adjacency_matrix(kdtree=kdtree, r=0.0, include_self=False)
    knn_nhd = knn_adjacency_matrix(kdtree=kdtree, k=0, include_self=False)
    empty_nhd = get_empty_adjacency_matrix(n_obs=N_OBS)
    assert np.allclose(radius_nhd, empty_nhd, equal_nan=True)
    assert np.allclose(knn_nhd, empty_nhd, equal_nan=True)


def test_zero_radius_with_self_is_self_neighborhood():
    kdtree = get_random_kdtree(n_obs=N_OBS)
    radius_nhd = radius_adjacency_matrix(kdtree=kdtree, r=0.0, include_self=True)
    knn_nhd = knn_adjacency_matrix(kdtree=kdtree, k=0, include_self=True)
    self_nhd = get_self_adjacency_matrix(n_obs=N_OBS)
    assert np.allclose(radius_nhd, self_nhd, equal_nan=True)
    assert np.allclose(knn_nhd, self_nhd, equal_nan=True)


def test_inf_radius_with_self_is_full_neighborhood():
    kdtree = get_random_kdtree(n_obs=N_OBS)
    radius_nhd = radius_adjacency_matrix(kdtree=kdtree, r=np.inf, include_self=True)
    knn_nhd = knn_adjacency_matrix(
        kdtree=kdtree, k=kdtree.data.shape[0] - 1, include_self=True
    )
    full_nhd = get_full_adjacency_matrix(n_obs=N_OBS)
    assert np.allclose(radius_nhd, full_nhd, equal_nan=True)
    assert np.allclose(knn_nhd, full_nhd, equal_nan=True)


if __name__ == "__main__":
    test_zero_radius_wihout_self_is_empty_neighborhood()
    test_zero_radius_with_self_is_self_neighborhood()
    test_inf_radius_with_self_is_full_neighborhood()
