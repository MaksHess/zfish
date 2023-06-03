"""
Functions to compute radius, knn or touch adjacejcy matrices based on KDTrees or label images in case of touch neighborhood.
"""
# %%
from collections import abc
from functools import singledispatchmethod
from typing import Callable, Iterable, Sequence, TypeAlias, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.neighbors import KDTree

from zfish.features.label import get_position_and_orientation_features
from zfish.features.neighborhood import aggregation_functions as agg
from zfish.features.polars_utils import unnest_all_structs
from zfish.features.types import LabelImage


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



AggFn: TypeAlias = Callable[[NDArray], NDArray]

class NeighborhoodAggregator:
    def __init__(
        self,
        name: str | Iterable[str],
        adjacency_matrix: NDArray | Iterable[NDArray],
        features: pl.DataFrame | None,
        index: pl.DataFrame | None,
    ):
        if isinstance(name, list) and isinstance(adjacency_matrix, list):
            self.neighborhoods: dict[str, NDArray] = {
                n: adj for n, adj in zip(name, adjacency_matrix)
            }
        else:
            self.neighborhoods = {cast(str, name): cast(NDArray, adjacency_matrix)}
        self._df = features
        self._index = index

    def agg(
        self,
        func: AggFn | Sequence[AggFn],
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
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
            for name, nhd in self.neighborhoods.items():
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
        return pl.concat(results, how='horizontal')
    
    def quantile(
            self,
            feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
            *qs: float,
    ) -> pl.DataFrame:
        return self.agg(list(agg.quantile_factory(*qs)), feature_selection=feature_selection)
    
    def mode(
            self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg.Mode, feature_selection=feature_selection)

    def mean(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg.Mean, feature_selection=feature_selection)

    def median(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg.Median, feature_selection=feature_selection)

    def sum(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg.Sum, feature_selection=feature_selection)

    def max(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg.Max, feature_selection=feature_selection)

    def min(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg.Min, feature_selection=feature_selection)

    def var(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg.Var, feature_selection=feature_selection)

    def std(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg.Std, feature_selection=feature_selection)

    def circmean(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg.CircMean, feature_selection=feature_selection)

    def circvar(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg.CircVar, feature_selection=feature_selection)

    def circr(
        self,
        feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
    ) -> pl.DataFrame:
        return self.agg(agg.CircR, feature_selection=feature_selection)

    def count(self) -> pl.DataFrame:
        return pl.concat(
            [
                pl.DataFrame(
                    np.nansum(nhd, axis=1, keepdims=True),
                    schema=[f"{name}_count"],
                )
                for name, nhd in self.neighborhoods.items()
            ],
            how='horizontal',
        )




class Nhd:
    def __init__(
        self,
        df: pl.DataFrame,
        kd_tree: KDTree | None = None,
        weighed_touch_matrix: NDArray | None = None,
        #TODO: Should this be none?
        meta_columns: tuple[str, ...] = ('label', 'Centroid'),
        feature_columns: str | Iterable[str] | pl.Expr | None = None
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
        centroid_column: str | Sequence[str] = "Centroid",
    ) -> "Nhd":
        if "roi" in df:
            assert len(df.select("roi").unique()) == 1, "`df` contains multiple 'roi'."
        if "object" in df:
            assert (
                len(df.select("object").unique()) == 1
            ), "`df` contains multiple 'object'."

        if isinstance(centroid_column, str):
            df_centroid = df.select(centroid_column).unnest(centroid_column)
        else:
            df_centroid = df.select(pl.col(e) for e in centroid_column)
        kd_tree = KDTree(df_centroid.select(sorted(df_centroid.columns, reverse=True)))
        return Nhd(df, kd_tree=kd_tree)

    @singledispatchmethod
    def radius(self, r, include_self: bool) -> "NeighborhoodAggregator":
        raise NotImplementedError()

    @radius.register(float)
    @radius.register(int)
    def radius_single(self, r: float | int, include_self: bool) -> "NeighborhoodAggregator":
        return self.radius_list([r], include_self=include_self)

    @radius.register(abc.Sequence)
    def radius_list(self, r: Sequence[float | int], include_self: bool) -> "NeighborhoodAggregator":
        if self.kd_tree is None:
            raise ValueError("No KDTree for radius query found.")
        
        names = [f"RADIUS-{_r}{'s' if include_self else ''}" for _r in r]
        adjacency_matrices = [
            radius_adjacency_matrix(kdtree=self.kd_tree, r=_r, include_self=include_self)
            for _r in r
        ]
        #TODO: Use consistent naming
        return NeighborhoodAggregator(
            name=names,
            adjacency_matrix=adjacency_matrices,
            features=self._df,
            index=self.df_meta,
        )
    
    @singledispatchmethod
    def knn(self, k, include_self: bool) -> "NeighborhoodAggregator":
        raise NotImplementedError()
    
    @knn.register(int)
    def knn_single(self, k: int, include_self: bool) -> "NeighborhoodAggregator":
        return self.knn_list([k], include_self=include_self)
    
    @knn.register(abc.Sequence)
    def knn_list(self, k: Sequence[int], include_self: bool) -> "NeighborhoodAggregator":
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
            index=self.df_meta,
        )

    #TODO: Implement touch neighborhood
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

