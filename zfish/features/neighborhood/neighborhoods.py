"""
Functions to compute radius, knn or touch adjacejcy matrices based on KDTrees or label images in case of touch neighborhood.
"""
# %%
from collections import abc
from functools import singledispatchmethod
from typing import Callable, Iterable, Sequence, TypeAlias, cast

import networkx as nx
import numpy as np
import polars as pl
from numpy.typing import DTypeLike, NDArray
from scipy import spatial
from sklearn.neighbors import KDTree

from zfish.features.label import get_position_and_orientation_features
from zfish.features.neighborhood import aggregation_functions as agg_funcs
from zfish.features.neighborhood.neighborhood_aggregation import (
    aggregate_table_dense_parallel,
)
from zfish.features.polars_utils import unnest_all_structs
from zfish.features.types import LabelImage


def radius_adjacency_matrix(
    kdtree: KDTree,
    r: float,
    include_self: bool = False,
    not_adjacent_value: float = np.nan,
):
    query = kdtree.query_radius(kdtree.data, r=r, return_distance=False)

    adjacency_matrix = np.empty((kdtree.data.shape[0], kdtree.data.shape[0]))
    adjacency_matrix.fill(not_adjacent_value)
    for i, j in enumerate(query):
        adjacency_matrix[i, j] = 1.0
    if not include_self:
        np.fill_diagonal(adjacency_matrix, not_adjacent_value)
    return adjacency_matrix


def knn_adjacency_matrix(
    kdtree: KDTree,
    k: int,
    include_self: bool = False,
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
    for i, j in enumerate(query):
        adjacency_matrix[i, j] = 1.0
    if not include_self:
        np.fill_diagonal(adjacency_matrix, not_adjacent_value)
    return adjacency_matrix

def delaunay_adjacency_matrix(
    kdtree: KDTree,
    k_neighbors: int,
    include_self: bool = False,
    not_adjacent_value: float = np.nan,
):
    delaunay = spatial.Delaunay(kdtree.data)
    G = nx.Graph()
    for path in delaunay.simplices:
        nx.add_path(G, path)
    touching = np.asarray(
        nx.adjacency_matrix(G, nodelist=np.arange(len(kdtree.data))).todense()
    )
    np.fill_diagonal(touching, 1)
    k_touching = np.linalg.matrix_power(touching, k_neighbors)
    adjacency_matrix = np.where(k_touching, 1.0, not_adjacent_value)
    return adjacency_matrix

def length_clipped_delaunay_adjacency_matrix(
    kdtree: KDTree,
    k_neighbors: int,
    include_self: bool = False,
    not_adjacent_value: float = np.nan,
    distance_cutoff_median_multiple: float = 2.5,
):
    delaunay = spatial.Delaunay(kdtree.data)
    G = nx.Graph()
    for path in delaunay.simplices:
        nx.add_path(G, path)
    touching = np.asarray(
        nx.adjacency_matrix(G, nodelist=np.arange(len(kdtree.data))).todense()
    )
    np.fill_diagonal(touching, 1)
    k_touching = np.linalg.matrix_power(touching, k_neighbors)
    adjacency_matrix = np.where(k_touching, 1.0, not_adjacent_value)
    distance_matrix = np.asarray(
        kdtree.sparse_distance_matrix(kdtree, np.inf).todense()
    )
    distances = distance_matrix[np.where(adjacency_matrix)]
    distance_cutoff = np.median(distances) * distance_cutoff_median_multiple
    adjacency_matrix = np.where(
        distance_matrix < distance_cutoff, adjacency_matrix, not_adjacent_value
    )
    if not include_self:
        np.fill_diagonal(adjacency_matrix, not_adjacent_value)
    return adjacency_matrix


def delaunay_neighborhood(points):
    delaunay = spatial.Delaunay(points, furthest_site=False)
    G = nx.Graph()
    for path in delaunay.simplices:
        nx.add_path(G, path)
    adjacency_matrix = nx.adjacency_matrix(G).todense()
    order_points = points[np.array(G.nodes())]
    distance_matrix = spatial.distance_matrix(order_points, order_points)
    return order_points, adjacency_matrix, distance_matrix


# TODO: Handle large k values
def touch_adjacency_matrix(
    weighted_touch_matrix: NDArray,
    k_neighbors: int,
    include_self: bool = False,
    threshold_area: float = 0.0,
    not_adjacent_value: float = np.nan,
):
    touching = weighted_touch_matrix > threshold_area
    k_touching = np.linalg.matrix_power(touching, k_neighbors)
    adjacency_matrix = np.where(k_touching, 1.0, not_adjacent_value)
    if not include_self:
        np.fill_diagonal(adjacency_matrix, not_adjacent_value)
    return adjacency_matrix


AggFn: TypeAlias = Callable[[NDArray], NDArray]


# TODO: Make stuff more memory efficient (i. e. generator for adjacency_matrix?)
class NeighborhoodAggregator:
    def __init__(
        self,
        name: str | Iterable[str],
        adjacency_matrix: NDArray | Iterable[NDArray],
        features: pl.DataFrame | None,
        meta: pl.DataFrame | None,
    ):
        if isinstance(name, list) and isinstance(adjacency_matrix, list):
            self.adjacency_matrices: dict[str, NDArray] = {
                n: adj for n, adj in zip(name, adjacency_matrix)
            }
        else:
            self.adjacency_matrices = {cast(str, name): cast(NDArray, adjacency_matrix)}
        self._df = features
        self._meta = meta

    @property
    def index(self) -> pl.DataFrame | None:
        if self._meta is not None and "label" in self._meta:
            return self._meta.select("label")

    def agg(
        self,
        func: AggFn | Sequence[AggFn],
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
        out_dtype: DTypeLike | None = None,
    ) -> pl.DataFrame:
        if isinstance(feature_selection, pl.DataFrame):
            features = unnest_all_structs(feature_selection)
        else:
            assert self._df is not None, "No dataframe to aggregate."
            features = unnest_all_structs(self._df.select(feature_selection))

        if not isinstance(func, Sequence):
            funcs = [cast(AggFn, func)]
        else:
            funcs = func

        results = []
        for func in funcs:
            for name, nhd in self.adjacency_matrices.items():
                prefix_str = f"{name}_{func.__name__}__"
                suffix_str = ""

                results.append(
                    pl.DataFrame(
                        data=aggregate_table_dense_parallel(
                            nhd, np.asarray(features), func
                        ),
                        schema=features.columns,
                    )
                    .select(pl.all().prefix(prefix_str))
                    .select(pl.all().suffix(suffix_str))
                )
        return pl.concat(results, how="horizontal")

    def quantile(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
        *qs: float,
    ) -> pl.DataFrame:
        return self.agg(
            list(agg_funcs.quantile_factory(*qs)), feature_selection=feature_selection
        )

    def mode(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg_funcs.Mode, feature_selection=feature_selection)

    def mean(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg_funcs.Mean, feature_selection=feature_selection)

    def median(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg_funcs.Median, feature_selection=feature_selection)

    def sum(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg_funcs.Sum, feature_selection=feature_selection)

    def max(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg_funcs.Max, feature_selection=feature_selection)

    def min(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg_funcs.Min, feature_selection=feature_selection)

    def var(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg_funcs.Var, feature_selection=feature_selection)

    def std(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg_funcs.Std, feature_selection=feature_selection)

    def circmean(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg_funcs.CircMean, feature_selection=feature_selection)

    def circvar(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg_funcs.CircVar, feature_selection=feature_selection)

    def circr(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg_funcs.CircR, feature_selection=feature_selection)

    # def neighbors(
    #     self,
    # ) -> pl.DataFrame:
    #     if self.index is None:
    #         raise ValueError("Provide a `meta` dataframe with `label` column.")
    #     res = []
    #     for name, nhd in self.adjacency_matrices.items():
    #         res.append(
    #             pl.Series(
    #                 f"{name}_Neighbors",
    #                 [
    #                     self.index.filter(np.nan_to_num(nhd_idxs).astype(bool))
    #                     .to_series()
    #                     .to_numpy()
    #                     for nhd_idxs in nhd
    #                 ],
    #                 dtype=pl.List(self.index.dtypes[0]),
    #             )
    #         )

    #     return pl.concat(res)

    def neighbors(
        self,
    ) -> pl.DataFrame:
        # print('reloaded')
        if self.index is None:
            raise ValueError("Provide a `meta` dataframe with `label` column.")
        res = []
        for name, nhd in self.adjacency_matrices.items():
            res.append(
                self.index.join(
                    pl.DataFrame(
                        np.vstack(np.where(np.nan_to_num(nhd))).T,
                        schema={
                            "label": self.index.dtypes[0],
                            "neighbors": self.index.dtypes[0],
                        },
                    )
                    .select(pl.all().map_dict(dict(zip(*self.index.with_row_count()))))
                    .groupby("label", maintain_order=True)
                    .agg(pl.col("neighbors").alias(f"{name}_Neighbors")),
                    on="label",
                    how="outer",
                ).drop("label")
            )
        return pl.concat(res)

    def count(self) -> pl.DataFrame:
        return pl.concat(
            [
                pl.DataFrame(
                    np.nansum(nhd, axis=1, keepdims=True),
                    schema=[f"{name}_count"],
                )
                for name, nhd in self.adjacency_matrices.items()
            ],
            how="horizontal",
        )


class WeightedNeighborhoodAggregator:
    def __init__(
        self,
        name: str | Iterable[str],
        adjacency_matrix: NDArray | Iterable[NDArray],
        weights_matrix: NDArray | Iterable[NDArray],
        features: pl.DataFrame | None,
        index: pl.DataFrame | None,
    ):
        pass

class Nhd:
    def __init__(
        self,
        df: pl.DataFrame,
        kd_tree: KDTree | None = None,
        weighed_touch_matrix: NDArray | None = None,
        # TODO: Should this be none?
        meta_columns: tuple[str, ...] = ("label", "^Centroid(-[xyz])?$"),
        feature_columns: str | Iterable[str] | pl.Expr | None = None,
    ):
        self._df = df
        self.kd_tree = kd_tree
        self.weighed_touch_matrix = weighed_touch_matrix
        self.df_meta = df.select(pl.col(e) for e in meta_columns)

    @classmethod
    def from_labelimage(cls, lbl: LabelImage) -> "Nhd":
        df = get_position_and_orientation_features(lbl)
        return cls.from_dataframe(df)

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame,
        index_column: str = "label",
        centroid_column: str | Sequence[str] = "^Centroid(-[xyz])?$",
    ) -> "Nhd":
        if "roi" in df:
            assert len(df.select("roi").unique()) == 1, "`df` contains multiple 'roi'."
        if "object" in df:
            assert (
                len(df.select("object").unique()) == 1
            ), "`df` contains multiple 'object'."

        if isinstance(centroid_column, str):
            df_centroid = df.select(centroid_column).pipe(unnest_all_structs)
        else:
            df_centroid = df.select(pl.col(e) for e in centroid_column)
        kd_tree = KDTree(df_centroid.select(sorted(df_centroid.columns, reverse=True)))
        return Nhd(df, kd_tree=kd_tree)

    @singledispatchmethod
    def radius(self, r, include_self: bool) -> "NeighborhoodAggregator":
        raise NotImplementedError()

    @radius.register(float)
    @radius.register(int)
    def radius_single(
        self, r: float | int, include_self: bool
    ) -> "NeighborhoodAggregator":
        return self.radius_list([r], include_self=include_self)

    @radius.register(abc.Sequence)
    def radius_list(
        self, r: Sequence[float | int], include_self: bool
    ) -> "NeighborhoodAggregator":
        if self.kd_tree is None:
            raise ValueError("No KDTree for radius query found.")

        names = [f"RADIUS-{_r}{'s' if include_self else ''}" for _r in r]
        adjacency_matrices = [
            radius_adjacency_matrix(
                kdtree=self.kd_tree, r=_r, include_self=include_self
            )
            for _r in r
        ]
        # TODO: Use consistent naming
        return NeighborhoodAggregator(
            name=names,
            adjacency_matrix=adjacency_matrices,
            features=self._df,
            meta=self.df_meta,
        )

    @singledispatchmethod
    def knn(self, k, include_self: bool) -> "NeighborhoodAggregator":
        raise NotImplementedError()

    @knn.register(int)
    def knn_single(self, k: int, include_self: bool) -> "NeighborhoodAggregator":
        return self.knn_list([k], include_self=include_self)

    @knn.register(abc.Sequence)
    def knn_list(
        self, k: Sequence[int], include_self: bool
    ) -> "NeighborhoodAggregator":
        if self.kd_tree is None:
            raise ValueError("No KDTree for knn query found.")

        names = [f"KNN-{_k}{'s' if include_self else ''}" for _k in k]
        adjacency_matrices = [
            knn_adjacency_matrix(kdtree=self.kd_tree, k=_k, include_self=include_self)
            for _k in k
        ]
        # return list(zip(names, adjacency_matrices))
        return NeighborhoodAggregator(
            name=names,
            adjacency_matrix=adjacency_matrices,
            features=self._df,
            meta=self.df_meta,
        )

    # TODO: Implement delaunay neighborhood
    # TODO: Implement touch neighborhood
    def touch(self, s: int, include_self: bool, threshold: float = 0.0):
        if self.weighed_touch_matrix is None:
            raise ValueError("No weighted touch matrix for touch query found.")
        return touch_adjacency_matrix(
            weighted_touch_matrix=self.weighed_touch_matrix,
            k_neighbors=s,
            include_self=include_self,
            threshold_area=threshold,
        )


# # %%
import collections
import functools
import inspect
from collections.abc import Iterable
from itertools import chain

from sklearn.neighbors import NearestNeighbors


class Nhd2:
    def __init__(
        self,
        neighbors: NearestNeighbors | None = None,
        distance_matrix: NDArray | None = None,
        weighted_touch_matrix: NDArray | None = None,
        label: Sequence[str] | Sequence[int] | None = None,
        name: str | None = None,
    ):
        self.name = name
        self.neighbors = neighbors
        self.weighted_touch_matrix = weighted_touch_matrix
        self.label = label


    @classmethod
    def from_labelimage(cls, lbl: LabelImage) -> "Nhd2":
        df = get_position_and_orientation_features(lbl)
        return cls.from_dataframe(df)

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame,
        label_column: str = "label",
        centroid_column: str | Sequence[str] = "^Centroid(-[xyz])?$",
        weighted_touch_matrix: NDArray | None = None,
    ) -> "Nhd2":
        if "roi" in df:
            assert len(df.select("roi").unique()) == 1, "`df` contains multiple roi's."
        if "object" in df:
            assert (
                len(df.select("object").unique()) == 1
            ), "`df` contains multiple object's."

        if isinstance(centroid_column, str):
            df_centroid = df.select(centroid_column).pipe(unnest_all_structs)
        else:
            df_centroid = df.select(pl.col(e) for e in centroid_column)
        neighbors = NearestNeighbors(n=5, radius=40, n_jobs=-1).fit(df_centroid.select(sorted(df_centroid.columns, reverse=True)))
        return Nhd2(
            neighbors=neighbors,
            label=df[label_column],
            weighted_touch_matrix=weighted_touch_matrix,
        )


class make_iterable: # type: ignore
    def __init__(self, *decorator_args):
        self.decorator_args = decorator_args
    def __call__(self, func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            adjusted_kwargs = {
                **dict(zip(inspect.signature(func).parameters, args)), #warp positional arguments in a dict and pass as kwarg
                **kwargs,
            }
            for decorator_arg in self.decorator_args:
                # assert decorator_arg in adjusted_kwargs, f"{decorator_arg} not found in {list(adjusted_kwargs)}"
                # adjusted_kwargs[decorator_arg] = [adjusted_kwargs[decorator_arg]] if not isinstance(Iterable, adjusted_kwargs[decorator_arg]) else adjusted_kwargs[decorator_arg]
                if not isinstance(adjusted_kwargs[decorator_arg], Iterable):
                    adjusted_kwargs[decorator_arg] = [adjusted_kwargs[decorator_arg]]
            result = func(**adjusted_kwargs)
            return result
        return wrapped_func


@make_iterable('r')
def radius_adjacency_matrix_gen(
    kdtree: KDTree,
    r: float | Iterable[float],
    include_self: bool = False,
    not_adjacent_value: float = np.nan,
):
    for _r in r:
        query = kdtree.query_radius(kdtree.data, r=_r, return_distance=False)

        adjacency_matrix = np.empty((kdtree.data.shape[0], kdtree.data.shape[0]))
        adjacency_matrix.fill(not_adjacent_value)
        for i, j in enumerate(query):
            adjacency_matrix[i, j] = 1.0
        if not include_self:
            np.fill_diagonal(adjacency_matrix, not_adjacent_value)
        yield adjacency_matrix

@make_iterable('r')
def knn_adjacency_matrix_gen(
    kdtree: KDTree,
    r: float | Iterable[float],
    include_self: bool = False,
    not_adjacent_value: float = np.nan,
):
    for _r in r:
        query = kdtree.query_radius(kdtree.data, r=_r, return_distance=False)

        adjacency_matrix = np.empty((kdtree.data.shape[0], kdtree.data.shape[0]))
        adjacency_matrix.fill(not_adjacent_value)
        for i, j in enumerate(query):
            adjacency_matrix[i, j] = 1.0
        if not include_self:
            np.fill_diagonal(adjacency_matrix, not_adjacent_value)
        yield adjacency_matrix


# df = pl.read_csv(r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\features\ccp\Ccp_cycle10.csv")
# df_one = df.filter(pl.col('roi')=='B07_px+2788_py+0295')
# points = df_one.select(sorted(df_one.select(pl.col('^Centroid-[xyz]$')).columns, reverse=True))
# tree = KDTree(points)

class NhdAgg:
    adjs: list[Iterable[NeighborhoodAggregator]]
    def __init__(self, adjs=None):
        if adjs is None:
            self.adjs = []
        
    def __iter__(self):
        return self
    
    def __next__(self):
        for e in chain.from_iterable(self.adjs):
            yield e


# adj_r50 = radius_adjacency_matrix_gen(tree, 50)


# import polars as pl

# df = pl.read_csv(r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\features\ccp\Ccp_cycle10.csv")

# # %%
# df_one = df.filter(pl.col('roi')=='G03_px+2426_py-0008')
# df_one
# # %%

# # %%
# import napari

# napari.view_points(points, size=1)
# # %%
# import networkx as nx

# # from zfish.visualize.centroids_and_principal_axes import napari_adjacency
# points = df_one[sorted(df_one.select('^Centroid-[xyz]$').columns, reverse=True)].to_numpy()
# import numpy as np
# from scipy import spatial, stats

# from zfish.visualize.napari import (
#     napari_adjacency,
#     napari_weighted_adjacency,
# )

# # points  = np.random.rand(*(1000, 3)) * 100 - 50
# # points = points[np.where(np.sqrt(np.sum(points*points, axis=1))<50)]
# # points = points[np.where(points[:, 0] > 0)]
# # points, adj, dists = delaunay_neighborhood(points)

# adj = delaunay_adjacency_matrix(spatial.KDTree(points), 1)
# edges = napari_adjacency(points, adj, alpha=0.1, highlight_idx=np.argmax(adj.sum(axis=0)))
# # edges_w = napari_weighted_adjacency(points, adj, dists)
# # %%
# import matplotlib.pyplot as plt
# import seaborn as sns

# ax = sns.histplot(dists[np.where(adj)])
# plt.vlines(12, *ax.get_ylim(), linestyle=':', color='k')


# # %%
# neighbors = dists[np.where(adj)]
# ax = sns.kdeplot(neighbors)
# ax = sns.histplot(neighbors, ax=ax, stat='density', bins=np.arange(-0.5, np.max(neighbors) + 1.5))
# plt.vlines(np.median(neighbors), *ax.get_ylim(), linestyle=':', color='k', linewidth=2)
# plt.vlines(np.median(neighbors) * 3, *ax.get_ylim(), linestyle='-', color='k', linewidth=2)

# # %%
# import napari

# viewer = napari.Viewer()
# viewer.add_points(points, size=1)
# viewer.add_vectors(**edges)
# # edges = napari_weighted_adjacency(points, adj, dists)
# # %%

# # %%
# import seaborn as sns

# sns.kdeplot(adj.sum(axis=0))
# # %%
# vectors = (np.expand_dims(points, 0) - np.expand_dims(points, 1))
# # %%
# feature = points[:, 0] + points[:, 1]
# feature = feature / feature.max()


# feature_diff = np.expand_dims(feature, 0) - np.expand_dims(feature, 1)
# # %%
# viewer = napari.view_points(points, features={'feature': feature}, face_color='feature', size=5)
# # %%
# adjacent_vectors = vectors * np.expand_dims(adj, -1)
# # %%
# grad = (adjacent_vectors * np.expand_dims(feature_diff, -1)).sum(axis=0)
# # %%
# # grad = (adjacent_vectors * feature.reshape(1, -1, 1)).sum(axis=0)

# norm = np.linalg.norm(grad, axis=1, keepdims=True)

# grad_n = grad / norm

# grad_vecs = np.stack([points, grad_n], axis=1)
# # %%

# viewer.add_vectors(grad_vecs, features={'norm': norm.squeeze()}, edge_color='norm', edge_colormap='turbo')
# # %%
# import napari

# viewer = napari.Viewer()
# viewer.add_points(points, size=3, opacity=0.6)
# viewer.add_vectors(**napari_adjacency(points, adj, highlight_idx=400, highlight_alpha=1.0, alpha=0.05))
# # %%
# import napari

# points[delaunay.simplices].reshape(-1, 2, 3)
# points[delaunay.simplices]
# # %%
# # napari.view_shapes(points[delaunay.simplices], shape_type='polygon')
# napari.view_vectors(np.unique(points[delaunay.simplices].reshape(-1, 3), axis=0))
# # %%
# import zfish.features.neighborhood.aggregation_functions as agg
# from zfish.features.neighborhood.neighborhood_aggregation import (
#     aggregate_table_dense_parallel,
# )
# from zfish.roi.spatial_roi import Roi, RoiMap

# fn = r"M:\marvwy\20220721_ZE4i2_aligned\imgs\B05_px+0262_py+0108.h5"
# roi = Roi.from_file(fn, level=1)


# # %%
# from typing import Sequence

# import polars as pl
# from numpy.typing import NDArray

# df = roi.tables["nucleiRaw3"]
# selection = pl.col("^.*DAPI\.1_Mean$")

# nhd = Nhd.from_dataframe(df)
# # res = nhd.radius([0, 100], True).mean(df.select(selection))
# # res2 = nhd.radius([0, 10, 100], True).agg([agg.MEAN, agg.MIN, agg.MAX, agg.VAR], df.select(selection))
# # %%
# qfuncs = list(agg.quantile_factory(*[0.1, 0.25, 0.50, 0.75, 0.9]))
# # %%
# res = nhd.radius(list(range(0, 301, 20)), True).agg(qfuncs, df.select(selection))
# # %%
# res2 = nhd.radius(list(range(0, 301, 20)), True).agg(qfuncs, df.select(selection))
# # %%
# from zfish.features.polars_utils import (
#     split_column_name_to_column,
# )

# to_plot = split_column_name_to_column(res, sep='__', index=tuple(), column_name='nhd').to_pandas()

# sns.kdeplot(to_plot, x='DAPI.1_Mean', hue='nhd')
# # %%
# import seaborn as sns

# for r in range(0, 100, 10):
#     sns.histplot(nhd.radius(r, False).count())


# @pl.api.register_dataframe_namespace("nhd")
# class NeighborhoodAccessor:
#     def __init__(
#         self,
#         df: pl.DataFrame,
#         centroid_column: str | Sequence[str] = "Centroid",
#         index_columns: str | Sequence[str] | None = "label",
#     ):
#         print("init NeighborhoodAccessor")
#         self._nhd = Nhd.from_dataframe(
#             df, index_column=index_columns, centroid_column=centroid_column
#         )
#         self._df = df
#         print("done")

#     def radius(self, r: float, include_self: bool):
#         adjacency_matrix = self._nhd.radius(r=r, include_self=include_self)
#         return NeighborhoodAggregator(
#             # name=f"radius(r={r}, include_self={include_self})",
#             name=f"RADIUS{r}{'s' if include_self else ''}",
#             adjacency_matrix=adjacency_matrix,
#             df=self._df,
#         )

#     def knn(self, k: int, include_self: bool):
#         adjacency_matrix = self._nhd.knn(k=k, include_self=include_self)
#         return NeighborhoodAggregator(
#             name=f"knn(k={k}, include_self={include_self})",
#             adjacency_matrix=adjacency_matrix,
#             df=self._df,
#         )

#     def touch(self, steps: int = 1, include_self: bool = False, threshold: float = 0.0):
#         raise NotImplementedError()
