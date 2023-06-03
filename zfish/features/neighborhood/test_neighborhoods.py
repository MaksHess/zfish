# %%
import numpy as np
from sklearn.neighbors import KDTree, NearestNeighbors

from zfish.features.neighborhood.data import (
    get_empty_adjacency_matrix,
    get_full_adjacency_matrix,
    get_random_kdtree,
    get_self_adjacency_matrix,
)
from zfish.features.neighborhood.neighborhoods import (
    knn_adjacency_matrix,
    radius_adjacency_matrix,
)

N_OBS = 10

import polars as pl

df_raw = pl.read_parquet(r"C:\Users\hessm\Documents\Programming\Python\zfish\data\features\full.parquet")
# %%
import matplotlib.pyplot as plt

df_roi = df_raw.filter(pl.col('roi')==(pl.col('roi').unique().sort().first())).filter(pl.col('object')=='nucleiRaw3')

dfpd_centroid = df_roi.select('label', 'Centroid').unnest('Centroid').to_pandas().set_index('label')

kd_tree = KDTree(dfpd_centroid)
neighbors = NearestNeighbors().fit(dfpd_centroid)

# df = dfpd_centroid.sample(10, random_state=42)

neighbors_connec = neighbors.radius_neighbors_graph(dfpd_centroid, radius=50, mode='connectivity').toarray()
neighbors_dist = neighbors.radius_neighbors_graph(dfpd_centroid, radius=50, mode='distance').toarray()
# [len(neighbors.radius_neighbors(dfpd_centroid.iloc[10:20], radius=100)[i]) for i in range(9)]

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
plt.sca(axs[0])
plt.imshow(neighbors_connec)
plt.colorbar()
plt.sca(axs[1])
plt.imshow(neighbors_dist)
plt.colorbar()
# %%
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
