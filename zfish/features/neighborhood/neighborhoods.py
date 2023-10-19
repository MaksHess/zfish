"""
Functions to compute radius, knn or touch adjacejcy matrices based on KDTrees or label 
images in case of touch neighborhood.
"""
# %%
import functools
import inspect
import logging
import warnings
from collections.abc import Callable, Iterable, Sequence
from itertools import accumulate, product
from typing import Literal, TypeAlias, cast

import networkx as nx
import numpy as np
import polars as pl
from attrs import asdict, define, frozen
from numpy.typing import ArrayLike, NDArray
from scipy import spatial
from scipy.ndimage import map_coordinates
from scipy.sparse import csr_array
from sklearn.neighbors import NearestNeighbors

from zfish.features.label import get_position_and_orientation_features
from zfish.features.neighborhood import aggregation_functions
from zfish.features.neighborhood.neighborhood_aggregation import (
    aggregate_rows_csr,
    aggregate_table_csr,
    aggregate_weighted_table_csr,
)
from zfish.features.neighborhood.neighborhood_matrix_parallel import (
    weighted_anisotropic_touch_matrix,
)
from zfish.features.polars_utils import unnest_all_structs
from zfish.features.types import LabelImage, SpatialImage

logger = logging.getLogger(__name__)

CSRArray: TypeAlias = csr_array
AggFn: TypeAlias = Callable[[NDArray], NDArray]


@frozen
class TouchAdjacency:
    csr: CSRArray
    is_weighted: bool = True


@frozen
class DelaunayAdjacency:
    csr: CSRArray
    is_weighted: bool = False


@frozen
class NeighborhoodQuery:
    pass


@frozen
class AdjacencyQuery(NeighborhoodQuery):
    pass


@frozen
class DistanceQuery(NeighborhoodQuery):
    pass


@frozen
class KernelQuery(NeighborhoodQuery):
    pass


@frozen
class RadiusAdjacencyQuery(AdjacencyQuery):
    r: int
    self_loops: bool = False

    def __str__(self):
        return f"RAD:{self.r}{'s' if self.self_loops else ''}"


@frozen
class KnnAdjacencyQuery(AdjacencyQuery):
    k: int
    self_loops: bool = False

    def __str__(self):
        return f"KNN:{self.k}{'s' if self.self_loops else ''}"


@frozen
class RadiusDistanceQuery(DistanceQuery):
    r: float | int
    self_loops: bool = False

    def __str__(self):
        return f"RADd:{self.r}{'s' if self.self_loops else ''}"


@frozen
class KnnDistanceQuery(DistanceQuery):
    k: int
    self_loops: bool = False

    def __str__(self):
        return f"KNNd:{self.k}{'s' if self.self_loops else ''}"


@frozen
class DelaunayQuery(AdjacencyQuery):
    n_steps: int = 1
    self_loops: bool = False
    threshold: float | None = None

    def __str__(self):
        return f"DELAUNAY{f':th={self.threshold}' if self.threshold is not None else ''}:{self.n_steps}{'s' if self.self_loops else ''}"


@frozen
class TouchQuery(AdjacencyQuery):
    n_steps: int = 1
    self_loops: bool = False
    threshold: float | None = None

    def __str__(self):
        return f"TOUCH{f':th={self.threshold}' if self.threshold is not None else ''}:{self.n_steps}{'s' if self.self_loops else ''}"


@frozen
class RadiusKernelQuery(KernelQuery):
    r: float | int
    self_loops: bool = True
    function: str = "triangular"
    row_normalized: bool = True


@frozen
class KnnKernelQuery(KernelQuery):
    k: int
    self_loops: bool = True
    function: str = "triangular"
    row_normalized: bool = True


class tuple_product:  # type: ignore
    def __init__(self, *decorator_args):
        self.decorator_args = decorator_args

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            function_signature = inspect.signature(func).parameters
            adjusted_kwargs = {
                **{
                    k: v.default
                    for k, v in function_signature.items()
                    if v.default is not inspect._empty
                },  # default arguments from signature
                **dict(
                    zip(function_signature, args)
                ),  # warp positional arguments as kwargs
                **kwargs,
            }

            # package arguments in iterable if they're not already.
            for decorator_arg in self.decorator_args:
                if isinstance(adjusted_kwargs[decorator_arg], str) or not isinstance(
                    adjusted_kwargs[decorator_arg], Iterable
                ):
                    adjusted_kwargs[decorator_arg] = [adjusted_kwargs[decorator_arg]]

            # separate out arguments to iterate over, iterate, return tuple of results.
            out = []
            iter_args = {e: adjusted_kwargs.pop(e) for e in self.decorator_args}
            for e in product(*iter_args.values()):
                params = {**dict(zip(iter_args.keys(), e)), **adjusted_kwargs}
                out.append(func(**params))
            return tuple(out)

        return wrapped_func


@tuple_product("r", "self_loops")
def radius(r, self_loops=False, distance=False) -> tuple[RadiusAdjacencyQuery, ...]:
    if distance:
        return RadiusDistanceQuery(r=r, self_loops=self_loops)
    return RadiusAdjacencyQuery(r=r, self_loops=self_loops)


@tuple_product("k", "self_loops")
def knn(k, self_loops=False, distance=False) -> tuple[KnnAdjacencyQuery, ...]:
    if distance:
        return KnnDistanceQuery(k=k, self_loops=self_loops)
    return KnnAdjacencyQuery(k=k, self_loops=self_loops)


@tuple_product("r", "self_loops", "function")
def radius_kernel(
    r, self_loops=True, function="triangular", row_normalized=True
) -> tuple[RadiusKernelQuery, ...]:
    return RadiusKernelQuery(
        r=r, self_loops=self_loops, function=function, row_normalized=row_normalized
    )


@tuple_product("k", "self_loops", "function")
def knn_kernel(
    k, self_loops=True, function="triangular", row_normalized=True
) -> tuple[RadiusKernelQuery, ...]:
    return KnnKernelQuery(
        k=k, self_loops=self_loops, function=function, row_normalized=row_normalized
    )


@tuple_product("n_steps", "self_loops", "threshold")
def delaunay(
    n_steps,
    self_loops=False,
    threshold=None,
) -> tuple[DelaunayQuery, ...]:
    return DelaunayQuery(n_steps=n_steps, self_loops=self_loops, threshold=threshold)


@tuple_product("n_steps", "self_loops", "threshold")
def touch(n_steps, self_loops=False, threshold=None) -> tuple[DelaunayQuery, ...]:
    return TouchQuery(n_steps=n_steps, self_loops=self_loops, threshold=threshold)


def _concatenate_csr_along_corner(arrs: Iterable[CSRArray]) -> CSRArray:
    coo_arrays = [arr.tocoo() for arr in arrs]
    shapes = [arr.shape for arr in arrs]
    offsets = list(
        accumulate(
            shapes, lambda s0, s1: (s0[0] + s1[0], s0[1] + s1[1]), initial=(0, 0)
        )
    )
    out_shape = offsets[-1]

    row = np.concatenate(
        [arr.row + offset[0] for arr, offset in zip(coo_arrays, offsets)]
    )
    col = np.concatenate(
        [arr.col + offset[1] for arr, offset in zip(coo_arrays, offsets)]
    )
    data = np.concatenate([arr.data for arr in arrs])

    # return data, indices, indptrs
    return csr_array((data, (row, col)), shape=out_shape)


def _create_ranges(start, stop, N, endpoint=True):
    if endpoint == 1:
        divisor = N - 1
    else:
        divisor = N
    steps = (1.0 / divisor) * (stop - start)
    return steps[:, None] * np.arange(N) + start[:, None]


def multilineprofile(
    img: SpatialImage | NDArray,
    starts: NDArray,
    ends: NDArray,
    order: int = 3,
    n_samples: int = 30,
    scale: Sequence[int] | np.ndarray | None = None,
):
    """Interpolate an image between start- and end-points, i. e. compute lineprofiles.
    Points are assumed to be in physical coordinates! The image scale is taken from
    `SpatialImage.meta.scale` or can be provided manually using the scale argument.
    Be aware that samples on long lines are spaced further apart.


    Parameters
    ----------
    img : SpatialImage | NDArray
        Image from which to interpolate. If it's a `SpatialImage` the image scale is
        used to transform the points into pixel coordinates for interpolation.
    starts : NDArray (n_lines, n_dim)
        Start points of the lineprofiles, physical coordinates (SpatialImage),
        px coordinates (NDArray).
    ends : NDArray (n_lines, n_dim)
        End poitns of the lineprofile, physical coordinates (SpatialImage),
        px coordinates (NDArray).
    order : int, optional
        Order of interpolation (0-5, see `scipy.ndimage.map_coordinates`), by default 3
    n_samples : int, optional
        Number of samples to draw per lineprofile, by default 30
    scale: Sequence[int], np.ndarray, optional, (n_dim, )
        ([z], y, x) scale of the image. Passing scale overrides the scale read from a
        `SpatialImage`.

    Returns
    -------
    NDArray (n_lines, n_samples)
        Each row corresponds to a lineprofile.
    """
    n_profiles = starts.shape[0]
    ndims = starts.shape[1]
    lines = np.array(
        [_create_ranges(starts[:, i], ends[:, i], n_samples) for i in range(ndims)]
    )
    lines = lines.reshape(ndims, -1)
    # TODO: Do this check with a validator (i. e. is_spatialimage)?
    if isinstance(img, SpatialImage) and scale is None:
        scale = np.asarray(img.meta.scale).reshape(1, -1).T
    elif scale is None:
        scale = np.array([1.0] * img.ndim).reshape(1, -1).T
    else:
        scale = np.array(scale).reshape(1, -1).T
    lines = lines / scale
    lineprofiles = map_coordinates(img, lines, order=order).reshape(
        n_profiles, n_samples
    )
    return lineprofiles


class make_iterable:  # type: ignore
    def __init__(self, *decorator_args):
        self.decorator_args = decorator_args

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            adjusted_kwargs = {
                **dict(
                    zip(inspect.signature(func).parameters, args)
                ),  # warp positional arguments in a dict and pass as kwarg
                **kwargs,
            }
            for decorator_arg in self.decorator_args:
                if not isinstance(adjusted_kwargs[decorator_arg], Iterable):
                    adjusted_kwargs[decorator_arg] = [adjusted_kwargs[decorator_arg]]
            result = func(**adjusted_kwargs)
            return result

        return wrapped_func


class generator_over:  # type: ignore
    def __init__(self, *decorator_args):
        self.decorator_args = decorator_args

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            adjusted_kwargs = {
                **dict(
                    zip(inspect.signature(func).parameters, args)
                ),  # warp positional arguments in a dict and pass as kwarg
                **kwargs,
            }

            # package arguments in iterable if they're not already.
            for decorator_arg in self.decorator_args:
                if not isinstance(adjusted_kwargs[decorator_arg], Iterable):
                    adjusted_kwargs[decorator_arg] = [adjusted_kwargs[decorator_arg]]

            # separate out arguments to iterate over, iterate, yield results.
            iter_args = {e: adjusted_kwargs.pop(e) for e in self.decorator_args}
            for e in product(*iter_args.values()):
                params = {**dict(zip(iter_args.keys(), e)), **adjusted_kwargs}
                yield func(**params)

        return wrapped_func


def _csr_matrix_power(csr: CSRArray, n: int) -> CSRArray:
    out = csr.copy()
    for _ in range(n - 1):
        out = out @ csr
    return out


def _csr_threshold_to_adjacency(csr: CSRArray, threshold: float) -> CSRArray:
    out_csr = csr.copy()
    out_csr.data = np.where(csr.data >= threshold, 1, 0)
    out_csr.eliminate_zeros()
    return out_csr


def _csr_to_edge_indices(adj: CSRArray) -> NDArray:
    adj_coo = adj.tocoo()
    return np.vstack((adj_coo.row, adj_coo.col)).T


def _csr_concatenate_along_corner(arrs: CSRArray | Iterable[CSRArray]) -> CSRArray:
    if isinstance(arrs, CSRArray):
        arrs = [arrs]
    arrs = list(arrs)
    if len(arrs) == 1:
        return arrs[0]

    coo_arrays = [arr.tocoo() for arr in arrs]
    shapes = [arr.shape for arr in arrs]
    offsets = list(
        accumulate(
            shapes, lambda s0, s1: (s0[0] + s1[0], s0[1] + s1[1]), initial=(0, 0)
        )
    )
    out_shape = offsets[-1]

    row = np.concatenate(
        [arr.row + offset[0] for arr, offset in zip(coo_arrays, offsets)]
    )
    col = np.concatenate(
        [arr.col + offset[1] for arr, offset in zip(coo_arrays, offsets)]
    )
    data = np.concatenate([arr.data for arr in coo_arrays])

    # return data, indices, indptrs
    return csr_array((data, (row, col)), shape=out_shape)


def _delaunay_adjacency(
    points: NDArray,
) -> CSRArray:
    delaunay = spatial.Delaunay(points)
    G = nx.Graph()
    for path in delaunay.simplices:
        nx.add_path(G, path)
    return csr_array(nx.adjacency_matrix(G, nodelist=np.arange(points.shape[0])))


# TODO: Weighted by "shared surface"?
def get_delaunay_adjacency(
    points: NDArray,
    mask: SpatialImage | None = None,
    n_samples=50,
):
    adj = _delaunay_adjacency(points)
    if mask is None:
        return DelaunayAdjacency(csr=adj, is_weighted=False)

    adj_coo = adj.tocoo()
    start_idxs, end_idxs = adj_coo.row, adj_coo.col
    start, end = points[start_idxs], points[end_idxs]
    percentage_in_mask = multilineprofile(
        mask, start, end, order=0, n_samples=n_samples
    ).mean(axis=1)
    adj.data = percentage_in_mask
    return DelaunayAdjacency(csr=adj, is_weighted=True)


# TODO: implement
def get_touch_adjacency(label_image: SpatialImage, absolute_surface=True):
    touch_matrix = weighted_anisotropic_touch_matrix(
        label_image.to_numpy().astype('int32'), *label_image.meta.scale
    )
    np.fill_diagonal(touch_matrix, 0)
    labels = np.unique(label_image)[1:]
    if not absolute_surface:
        touch_matrix = touch_matrix / touch_matrix.sum(axis=1, keepdims=True)
    adj = csr_array(touch_matrix[labels][:, labels])
    return TouchAdjacency(csr=adj, is_weighted=True)


@generator_over("neighbors")
def query_radius_adjacency(
    neighbors: NearestNeighbors,
    r: float,
    self_loops: bool = False,
) -> CSRArray:
    X = neighbors._fit_X if self_loops else None

    query = neighbors.radius_neighbors_graph(
        X=X,
        radius=r,
        mode="connectivity",
    )
    return csr_array(query)


@generator_over("neighbors")
def query_radius_distance(
    neighbors: NearestNeighbors,
    r: float,
    self_loops: bool = False,
) -> CSRArray:
    X = neighbors._fit_X if self_loops else None

    query = neighbors.radius_neighbors_graph(
        X=X,
        radius=r,
        mode="distance",
    )
    return csr_array(query)


@generator_over("neighbors")
def query_knn_adjacency(
    neighbors: NearestNeighbors,
    k: int | Iterable[int],
    self_loops: bool = False,
) -> CSRArray:
    n_objects = neighbors._fit_X.shape[0]
    if k > (n_objects-1):
        logger.warn(
            f"k={k} > (n_objects-1)={(n_objects-1)}; setting k to {n_objects-1}"
        )
        k = n_objects-1
    X = neighbors._fit_X if self_loops else None
    query = neighbors.kneighbors_graph(
        X=X,
        n_neighbors=k + int(self_loops),  # Query k+1 neighbors in case of self-loops
        mode="connectivity",
    )
    return csr_array(query)


@generator_over("neighbors")
def query_knn_distance(
    neighbors: NearestNeighbors,
    k: int | Iterable[int],
    self_loops: bool = False,
) -> CSRArray:
    n_objects = neighbors._fit_X.shape[0]
    if k > (n_objects-1):
        logger.warn(
            f"k={k} > (n_objects-1)={(n_objects-1)}; setting k to {n_objects-1}"
        )
        k = n_objects-1
    X = neighbors._fit_X if self_loops else None
    query = neighbors.kneighbors_graph(
        X=X,
        n_neighbors=k + int(self_loops),
        mode="distance",
    )
    return csr_array(query)


def _kernel_triangular(z):
    return 1 - np.abs(z)


def _kernel_uniform(z):
    return np.full_like(z, 0.5)


def _kernel_gaussian(z):
    return (2 * np.pi) ** (-0.5) * np.exp(-(z**2) / 2)


KERNEL_FUNCTIONS = {
    "uniform": _kernel_uniform,
    "triangular": _kernel_triangular,
    "gaussian": _kernel_gaussian,
}


def _normalize_rows_csr(arr: CSRArray, inplace=False) -> CSRArray:
    row_weights = aggregate_rows_csr(arr, aggregation_functions.Sum, keepdims=False)
    distributed_row_weights = np.repeat(row_weights, np.diff(arr.indptr))
    if inplace:
        arr.data = arr.data / distributed_row_weights
        return arr
    else:
        return csr_array(
            (arr.data / distributed_row_weights, arr.indices, arr.indptr), arr.shape
        )


@generator_over("neighbors")
def query_radius_kernel(
    neighbors: NearestNeighbors,
    r: float,
    self_loops: bool = True,
    function: str = "triangular",
    row_normalized: bool = True,
) -> CSRArray:
    """i. e. fixed bandwidth"""
    distance_matrix = next(
        query_radius_distance(neighbors=neighbors, r=r, self_loops=self_loops)
    )
    z = distance_matrix.data / r
    # Adjust the weights according to the kernel function in-place
    distance_matrix.data = KERNEL_FUNCTIONS[function](z)
    # Apply row normalization
    if row_normalized:
        distance_matrix = _normalize_rows_csr(distance_matrix)
    return distance_matrix


@generator_over("neighbors")
def query_knn_kernel(
    neighbors: NearestNeighbors,
    k: int,
    self_loops: bool = True,
    function: str = "triangular",
    row_normalized: bool = True,
) -> CSRArray:
    """i. e. adaptive bandwidth"""
    n_dist, n_idx = neighbors.kneighbors(n_neighbors=k)
    bandwidth = n_dist[:, -1]
    distance_matrix = next(
        query_knn_distance(neighbors=neighbors, k=k, self_loops=self_loops)
    )
    z = distance_matrix.data / np.repeat(bandwidth, k + int(self_loops))
    distance_matrix.data = KERNEL_FUNCTIONS[function](z)
    if row_normalized:
        distance_matrix = _normalize_rows_csr(distance_matrix)
    return distance_matrix


# TODO: Delaunay distance queries?
def query_delaunay_adjacency(
    adj: CSRArray,
    n_steps: int,
    self_loops: bool = False,
    threshold: float | None = None,
) -> CSRArray:
    adj_out = adj.copy()

    if threshold is not None:
        adj_out.data = np.where(adj_out.data >= threshold, 1, 0)
        adj_out.eliminate_zeros()

    adj_out = _csr_matrix_power(adj_out, n_steps)
    adj_out.data = np.ones_like(adj_out.data)
    if self_loops:
        adj_out.setdiag(1)
    else:
        adj_out.setdiag(0)
        adj_out.eliminate_zeros()

    return adj_out


# @generator_over("n_steps", "self_loops")
# def query_neighborhood_from_adjacency(
#     adj: CSRArray,
#     n_steps: int | Iterable[int],
#     self_loops: bool = False,
# ) -> CSRArray:
#     adj_out = _csr_matrix_power(adj, n_steps)
#     adj_out.data = np.ones_like(adj_out.data)
#     if self_loops:
#         adj_out.setdiag(1)
#     else:
#         adj_out.setdiag(0)
#         adj_out.eliminate_zeros()

#     return adj_out


class NeighborhoodQueryObject:
    def __init__(
        self,
        neighbors: dict[str, NearestNeighbors],
        touch_adjacency: TouchAdjacency | None = None,
        delaunay_adjacency: DelaunayAdjacency | None = None,
        label: pl.DataFrame | None = None,
        queries: "tuple[NeighborhoodQuery, ...]" = tuple(),
    ):
        self.neighbors = neighbors
        self.touch_adjacency = touch_adjacency
        self.delaunay_adjacency = delaunay_adjacency
        self.label = label
        self.queries = queries

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame,
        label_columns: Sequence[str] | str | None = "label",
        centroid_column: Sequence[str] | str = "^Centroid(.[xyz])?$",
        region_id_column: str | None = None,
        touch_adjacency: TouchAdjacency | None = None,
        delaunay_adjacency: DelaunayAdjacency | None = None,
        neighbors_algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
    ) -> "NeighborhoodQueryObject":
        if label_columns is None:
            label = pl.Series("label", np.arange(len(df))).to_frame()
        else:
            if isinstance(label_columns, str):
                label_columns = [label_columns]
            for label_column in label_columns:
                assert label_column in df, f"'{label_column}' not found."
            label = df.select(label_columns)
            assert (
                label.unique().height == df.height
            ), f"None unique labels when using label_columns: `{tuple(label_columns)}`"

        if region_id_column is not None:
            assert (
                region_id_column in df
            ), f"`region_id_column` '{region_id_column}' not found."
            assert df[
                region_id_column
            ].is_sorted(), "`df` must be sorted by `region_id_column`."
            dfs = df.groupby(region_id_column, maintain_order=True)
        else:
            dfs = [("site_0", df)]

        neighbors = dict()
        delaunays = []
        for site_name, df in dfs:
            if isinstance(centroid_column, str):
                df_centroid = df.select(centroid_column).pipe(unnest_all_structs)
            else:
                df_centroid = df.select(pl.col(e) for e in centroid_column)

            points = df_centroid.select(
                sorted(df_centroid.columns, reverse=True)
            ).to_numpy()

            neighbors[site_name] = NearestNeighbors(
                n_neighbors=5, radius=40, n_jobs=-1, algorithm=neighbors_algorithm
            ).fit(points)

            if delaunay_adjacency is None:
                delaunays.append(get_delaunay_adjacency(points))

        if delaunay_adjacency is None:
            delaunay_adjacency = DelaunayAdjacency(
                csr=_csr_concatenate_along_corner(e.csr for e in delaunays),
                is_weighted=False,
            )

        return NeighborhoodQueryObject(
            neighbors=neighbors,
            label=label,
            touch_adjacency=touch_adjacency,
            delaunay_adjacency=delaunay_adjacency,
        )

    @classmethod
    def from_labelimage(
        cls,
        lbl: LabelImage | dict[str, LabelImage],
        mask: LabelImage | dict[str, LabelImage] | None = None,
        mask_n_samples: int = 100,
    ) -> "NeighborhoodQueryObject":
        if isinstance(lbl, dict):
            if mask is not None:
                assert isinstance(
                    mask, dict
                ), "You need to pass a dict of mask's if you pass a dict of lbl's"
                assert set(lbl.keys()) == set(
                    mask.keys()
                ), "Dictionaries have inconsistent keys."

            dfs = []
            delaunays = []
            for roi_id, lb in lbl.items():
                df = get_position_and_orientation_features(lb).with_columns(
                    roi=pl.lit(roi_id)
                )
                dfs.append(df)
                if mask is not None:
                    points = (
                        df.pipe(unnest_all_structs, sep=".")
                        .select(["Centroid.z", "Centroid.y", "Centroid.x"])
                        .to_numpy()
                    )
                    delaunays.append(
                        get_delaunay_adjacency(
                            points, mask[roi_id], n_samples=mask_n_samples
                        )
                    )
            df = pl.concat(dfs)
            delaunay_adjacency = DelaunayAdjacency(
                csr=_csr_concatenate_along_corner(e.csr for e in delaunays),
                is_weighted=True,
            )

        else:
            df = get_position_and_orientation_features(lbl)
            if mask is not None:
                points = (
                    df.pipe(unnest_all_structs, sep=".")
                    .select(["Centroid.z", "Centroid.y", "Centroid.x"])
                    .to_numpy()
                )
                delaunay_adjacency = get_delaunay_adjacency(
                    points, mask, n_samples=mask_n_samples
                )
            else:
                delaunay_adjacency = None
        touch_adjacency = get_touch_adjacency(lbl, absolute_surface=False)

        return cls.from_dataframe(
            df, delaunay_adjacency=delaunay_adjacency, touch_adjacency=touch_adjacency
        )

    @classmethod
    def from_numpy(
        cls,
        points: NDArray,
        schema: tuple[str, ...] = ("Centroid.z", "Centroid.y", "Centroid.x"),
        mask: LabelImage | None = None,
        mask_n_samples: int = 100,
    ):
        df = pl.DataFrame(data=points, schema=schema).with_row_count(name="label")
        if mask is not None:
            delaunay_adjacency = get_delaunay_adjacency(
                points, mask, n_samples=mask_n_samples
            )
        return NeighborhoodQueryObject.from_dataframe(
            df,
            centroid_column=schema,
            label_columns="label",
            delaunay_adjacency=delaunay_adjacency,
        )

    # TODO: functools.cached_property?
    @property
    def points_per_site(self):
        return {
            site_id: neighbors._fit_X for site_id, neighbors in self.neighbors.items()
        }

    @property
    def points(self):
        return np.concatenate([e._fit_X for e in self.neighbors.values()], axis=0)

    @property
    def index(self):
        return self.label.with_row_count(name="index").select(pl.col("index"))

    def add_queries(self, queries: "tuple[NeighborhoodQuery, ...]"):
        return NeighborhoodQueryObject(
            neighbors=self.neighbors,
            touch_adjacency=self.touch_adjacency,
            delaunay_adjacency=self.delaunay_adjacency,
            label=self.label,
            queries=self.queries + queries,
        )

    def radius(
        self,
        r: float | Iterable[float],
        self_loops: bool = False,
        distance: bool = False,
    ) -> "NeighborhoodQueryObject":
        return self.add_queries(radius(r=r, self_loops=self_loops, distance=distance))

    def knn(
        self, k: int | Iterable[int], self_loops: bool = False, distance: bool = False
    ) -> "NeighborhoodQueryObject":
        return self.add_queries(knn(k=k, self_loops=self_loops, distance=distance))

    def delaunay(
        self,
        n_steps: int | Iterable[int] = 1,
        self_loops: bool = False,
        threshold: float | None = None,
    ) -> "NeighborhoodQueryObject":
        return self.add_queries(
            delaunay(n_steps=n_steps, self_loops=self_loops, threshold=threshold)
        )

    def touch(
        self,
        n_steps: int | Iterable[int] = 1,
        self_loops: bool = False,
        threshold: float | None = None,
    ) -> "NeighborhoodQueryObject":
        return self.add_queries(
            touch(n_steps=n_steps, self_loops=self_loops, threshold=threshold)
        )

    def generate_neighborhood(self, query: NeighborhoodQuery) -> CSRArray:
        if isinstance(query, RadiusAdjacencyQuery):
            arrays = query_radius_adjacency(self.neighbors.values(), **asdict(query))
        elif isinstance(query, KnnAdjacencyQuery):
            arrays = query_knn_adjacency(self.neighbors.values(), **asdict(query))
        elif isinstance(query, RadiusDistanceQuery):
            arrays = query_radius_distance(self.neighbors.values(), **asdict(query))
        elif isinstance(query, KnnDistanceQuery):
            arrays = query_knn_distance(self.neighbors.values(), **asdict(query))
        elif isinstance(query, DelaunayQuery):
            if not self.delaunay_adjacency:
                raise RuntimeError(
                    "`NeighborhoodQueryObject.delauny_adjacency` not found!"
                )
            arrays = query_delaunay_adjacency(
                self.delaunay_adjacency.csr, **asdict(query)
            )
        elif isinstance(query, TouchQuery):
            if not self.touch_adjacency:
                raise RuntimeError(
                    "`NeighborhoodQueryObject.touch_adjacency` not found!"
                )
            arrays = query_delaunay_adjacency(self.touch_adjacency.csr, **asdict(query))
        else:
            raise ValueError(f"Unknown `NeighborhoodQuery` {query}")
        return _csr_concatenate_along_corner(arrays)

    @property
    def neighborhoods(self) -> Iterable[CSRArray]:
        for query in self.queries:
            yield self.generate_neighborhood(query)

    @property
    def query_neighborhoods(
        self,
    ) -> Iterable[tuple[NeighborhoodQuery, CSRArray]]:
        for query in self.queries:
            yield query, self.generate_neighborhood(query)

    def aggregate(
        self,
        func: AggFn | Sequence[AggFn],
        df: pl.DataFrame | pl.Series | None = None,
        drop_label_columns_from_df: bool = True,
        return_label: bool = False,
        edge_weights: bool = False,
    ) -> pl.DataFrame:
        if not isinstance(func, Sequence):
            funcs = [cast(AggFn, func)]
        else:
            funcs = func

        if isinstance(df, pl.DataFrame):
            df = unnest_all_structs(df)
        elif isinstance(df, pl.Series):
            df = unnest_all_structs(df.to_frame())
        elif df is None:
            for func in funcs:
                assert (
                    func.__name__ in aggregation_functions.NEIGHBORS
                ), f"Cannot to '{func.__name__}' aggregation without `df`."
        else:
            raise ValueError(
                f"`df` must be polars.Series or polars.DataFrame, not `{type(df)}`"
            )

        if df is not None and all(c in df for c in self.label.columns):
            rows_before = df.height
            df = self.label.join(df, how="left", on=self.label.columns)
            if df.height != rows_before:
                warnings.warn(
                    "Joining `NeighborhoodQueryObject.label` with the provided `df` "
                    f"lead to dropping of {rows_before-df.height}/{rows_before} rows."
                )
            if drop_label_columns_from_df:
                for c in self.label.columns:
                    df.drop_in_place(c)

        results = []
        for query, adj in self.query_neighborhoods:
            for func in funcs:
                name = str(query)
                base_name = f"{name}_{func.__name__}"

                if func.__name__ == "Count":
                    aggregated_table = aggregate_table_csr(
                        adj, np.ones((adj.shape[0], 1)), func
                    )

                    results.append(
                        pl.DataFrame(
                            data=aggregated_table,
                            schema=[base_name],
                        ).select(pl.all().cast(pl.UInt32))
                    )

                elif func.__name__ == "Neighbors" or func.__name__ == "NeighborIndices":
                    if self.label is None:
                        # FIXME: is this necessary, or should label be enfoced
                        # upon creation?
                        raise ValueError(
                            "Provide a `meta` dataframe with `label` column."
                        )
                    results.append(
                        self.index.join(
                            pl.DataFrame(
                                _csr_to_edge_indices(adj),
                                schema={
                                    "index": pl.UInt32,
                                    "neighbor": pl.UInt32,
                                },
                            )
                            .groupby("index", maintain_order=True)
                            .agg(
                                (
                                    pl.col("neighbor")
                                    .map_dict(
                                        dict(
                                            zip(
                                                *self.label.select(
                                                    pl.struct(pl.all())
                                                ).with_row_count()
                                            )
                                        )
                                    )
                                    .alias(base_name)
                                    if func.__name__ == "Neighbors"
                                    else pl.col("neighbor").alias(base_name)
                                )
                            ),
                            on="index",
                            how="left",
                        )
                        .drop("index")
                        .select(pl.all().fill_null([]))
                    )

                else:
                    prefix_str = f"{base_name}__"

                    if edge_weights:
                        aggregated_table = aggregate_weighted_table_csr(
                            adj, np.asarray(df), func
                        )
                    else:
                        aggregated_table = aggregate_table_csr(
                            adj, np.asarray(df), func
                        )

                    results.append(
                        pl.DataFrame(
                            data=aggregated_table,
                            schema=df.columns,
                        ).select(pl.all().prefix(prefix_str))
                    )

        if return_label:
            return self.label.with_columns(
                pl.concat(results, how="horizontal") if results else pl.DataFrame()
            )
        else:
            if results:
                return pl.concat(results, how="horizontal")
            else:
                return pl.DataFrame()

    def aggregate_weights(
        self,
        func: AggFn | Sequence[AggFn] = aggregation_functions.Sum,
        return_label: bool = False,
    ) -> pl.DataFrame:
        if not isinstance(func, Sequence):
            funcs = [cast(AggFn, func)]
        else:
            funcs = func

        results = []
        for query, adj in self.query_neighborhoods:
            for func in funcs:
                column_name = f"{str(query)}_{func.__name__}"

                aggregated_weights = aggregate_rows_csr(adj, func)

                results.append(
                    pl.DataFrame(
                        data=aggregated_weights,
                        schema=[column_name],
                    )
                )

        if return_label:
            return self.label.with_columns(
                pl.concat(results, how="horizontal") if results else pl.DataFrame()
            )
        else:
            if results:
                return pl.concat(results, how="horizontal")
            else:
                return pl.DataFrame()


# %%
if __name__ == "__main__":
    from pathlib import Path

    import polars as pl

    model = "Linear(loss='huber', features=['MediumPath', 'EmbryoPath'])"
    obj = "nucleiRaw3"

    df_raw_lazy = pl.scan_parquet(
        Path(
            r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\intensity_normalization\nuc_features_all_models.parquet"
        )
    )
    df_raw = (
        df_raw_lazy.filter(pl.col("model") == model)
        .filter(pl.col("object") == obj)
        .collect()
    )
    from zfish.features.neighborhood import aggregation_functions as agg_funcs
    from zfish.features.polars_selector import sel

    n = 1
    rois = df_raw["roi"].unique().to_list()[:n]

    df = (
        df_raw.filter(pl.col("roi").is_in(rois))
        .select("roi", "label", "Centroid", "Elongation", "PhysicalSize", "Roundness")
        .pipe(unnest_all_structs, sep=".")
    )
    df
    # %%

    nq = NeighborhoodQueryObject.from_dataframe(
        df, label_columns=["roi", "label"], region_id_column="roi"
    )

    # res = nq.radius(np.arange(10, 110, 10)).aggregate([agg_funcs.Count])
    # res
    # %%
    nq.generate_neighborhood(DelaunayQuery(n_steps=2))
    # %%
    nq.radius([10, 50, 100], distance=True).aggregate_weights(
        [agg_funcs.Sum, agg_funcs.Count], return_label=True
    )
    # %%
    nq.radius([10, 40, 100], distance=False).aggregate([agg_funcs.Neighbors])

    # %%
    nq.delaunay([1, 2]).aggregate(agg_funcs.Count)
    # %%
    nq.radius([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200]).aggregate(
        agg_funcs.Count
    )

    # %%
    nq.knn([2, 5, 10, 20, 30, 50], distance=True).aggregate_weights(agg_funcs.Max)
    nq.knn([2, 5, 10, 20, 30, 50], distance=True).aggregate_weights(agg_funcs.Median)
    # %%
    np.array(np.arange(9).reshape((3, 3))) @ np.array(np.arange(9).reshape((3, 3)))
# def length_clipped_delaunay_adjacency_matrix(
#     kdtree: KDTree,
#     k_neighbors: int,
#     include_self: bool = False,
#     not_adjacent_value: float = np.nan,
#     distance_cutoff_median_multiple: float = 2.5,
# ):
#     delaunay = spatial.Delaunay(kdtree.data)
#     G = nx.Graph()
#     for path in delaunay.simplices:
#         nx.add_path(G, path)
#     touching = np.asarray(
#         nx.adjacency_matrix(G, nodelist=np.arange(len(kdtree.data))).todense()
#     )
#     np.fill_diagonal(touching, 1)
#     k_touching = np.linalg.matrix_power(touching, k_neighbors)
#     adjacency_matrix = np.where(k_touching, 1.0, not_adjacent_value)
#     distance_matrix = np.asarray(
#         kdtree.sparse_distance_matrix(kdtree, np.inf).todense()
#     )
#     distances = distance_matrix[np.where(adjacency_matrix)]
#     distance_cutoff = np.median(distances) * distance_cutoff_median_multiple
#     adjacency_matrix = np.where(
#         distance_matrix < distance_cutoff, adjacency_matrix, not_adjacent_value
#     )
#     if not include_self:
#         np.fill_diagonal(adjacency_matrix, not_adjacent_value)
#     return adjacency_matrix


# def delaunay_neighborhood(points):
#     delaunay = spatial.Delaunay(points, furthest_site=False)
#     G = nx.Graph()
#     for path in delaunay.simplices:
#         nx.add_path(G, path)
#     adjacency_matrix = nx.adjacency_matrix(G).todense()
#     order_points = points[np.array(G.nodes())]
#     distance_matrix = spatial.distance_matrix(order_points, order_points)
#     return order_points, adjacency_matrix, distance_matrix


# # TODO: Handle large k values
# def touch_adjacency_matrix(
#     weighted_touch_matrix: NDArray,
#     k_neighbors: int,
#     include_self: bool = False,
#     threshold_area: float = 0.0,
#     not_adjacent_value: float = np.nan,
# ):
#     touching = weighted_touch_matrix > threshold_area
#     k_touching = np.linalg.matrix_power(touching, k_neighbors)
#     adjacency_matrix = np.where(k_touching, 1.0, not_adjacent_value)
#     if not include_self:
#         np.fill_diagonal(adjacency_matrix, not_adjacent_value)
#     return adjacency_matrix


# # TODO: Make stuff more memory efficient (i. e. generator for adjacency_matrix?)
# class NeighborhoodAggregator:
#     def __init__(
#         self,
#         name: str | Iterable[str],
#         adjacency_matrix: NDArray | Iterable[NDArray],
#         features: pl.DataFrame | None,
#         meta: pl.DataFrame | None,
#     ):
#         if isinstance(name, list) and isinstance(adjacency_matrix, list):
#             self.adjacency_matrices: dict[str, NDArray] = {
#                 n: adj for n, adj in zip(name, adjacency_matrix)
#             }
#         else:
#             self.adjacency_matrices = {cast(str, name): cast(NDArray, adjacency_matrix)}
#         self._df = features
#         self._meta = meta

#     @property
#     def index(self) -> pl.DataFrame | None:
#         if self._meta is not None and "label" in self._meta:
#             return self._meta.select("label")

#     def agg(
#         self,
#         func: AggFn | Sequence[AggFn],
#         feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
#         out_dtype: DTypeLike | None = None,
#     ) -> pl.DataFrame:
#         if isinstance(feature_selection, pl.DataFrame):
#             features = unnest_all_structs(feature_selection)
#         else:
#             assert self._df is not None, "No dataframe to aggregate."
#             features = unnest_all_structs(self._df.select(feature_selection))

#         if not isinstance(func, Sequence):
#             funcs = [cast(AggFn, func)]
#         else:
#             funcs = func

#         results = []
#         for func in funcs:
#             for name, nhd in self.adjacency_matrices.items():
#                 prefix_str = f"{name}_{func.__name__}__"
#                 suffix_str = ""

#                 results.append(
#                     pl.DataFrame(
#                         data=aggregate_table_dense_parallel(
#                             nhd, np.asarray(features), func
#                         ),
#                         schema=features.columns,
#                     )
#                     .select(pl.all().prefix(prefix_str))
#                     .select(pl.all().suffix(suffix_str))
#                 )
#         return pl.concat(results, how="horizontal")

#     def quantile(
#         self,
#         feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
#         *qs: float,
#     ) -> pl.DataFrame:
#         return self.agg(
#             list(agg_funcs.quantile_factory(*qs)), feature_selection=feature_selection
#         )

#     def mode(
#         self,
#         feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
#     ) -> pl.DataFrame:
#         return self.agg(agg_funcs.Mode, feature_selection=feature_selection)

#     def mean(
#         self,
#         feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
#     ) -> pl.DataFrame:
#         return self.agg(agg_funcs.Mean, feature_selection=feature_selection)

#     def median(
#         self,
#         feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
#     ) -> pl.DataFrame:
#         return self.agg(agg_funcs.Median, feature_selection=feature_selection)

#     def sum(
#         self,
#         feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
#     ) -> pl.DataFrame:
#         return self.agg(agg_funcs.Sum, feature_selection=feature_selection)

#     def max(
#         self,
#         feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
#     ) -> pl.DataFrame:
#         return self.agg(agg_funcs.Max, feature_selection=feature_selection)

#     def min(
#         self,
#         feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
#     ) -> pl.DataFrame:
#         return self.agg(agg_funcs.Min, feature_selection=feature_selection)

#     def var(
#         self,
#         feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
#     ) -> pl.DataFrame:
#         return self.agg(agg_funcs.Var, feature_selection=feature_selection)

#     def std(
#         self,
#         feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
#     ) -> pl.DataFrame:
#         return self.agg(agg_funcs.Std, feature_selection=feature_selection)

#     def circmean(
#         self,
#         feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
#     ) -> pl.DataFrame:
#         return self.agg(agg_funcs.CircMean, feature_selection=feature_selection)

#     def circvar(
#         self,
#         feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
#     ) -> pl.DataFrame:
#         return self.agg(agg_funcs.CircVar, feature_selection=feature_selection)

#     def circr(
#         self,
#         feature_selection: str | Sequence[str] | pl.Expr | pl.DataFrame,
#     ) -> pl.DataFrame:
#         return self.agg(agg_funcs.CircR, feature_selection=feature_selection)

#     # def neighbors(
#     #     self,
#     # ) -> pl.DataFrame:
#     #     if self.index is None:
#     #         raise ValueError("Provide a `meta` dataframe with `label` column.")
#     #     res = []
#     #     for name, nhd in self.adjacency_matrices.items():
#     #         res.append(
#     #             pl.Series(
#     #                 f"{name}_Neighbors",
#     #                 [
#     #                     self.index.filter(np.nan_to_num(nhd_idxs).astype(bool))
#     #                     .to_series()
#     #                     .to_numpy()
#     #                     for nhd_idxs in nhd
#     #                 ],
#     #                 dtype=pl.List(self.index.dtypes[0]),
#     #             )
#     #         )

#     #     return pl.concat(res)

#     def neighbors(
#         self,
#     ) -> pl.DataFrame:
#         # print('reloaded')
#         if self.index is None:
#             raise ValueError("Provide a `meta` dataframe with `label` column.")
#         res = []
#         for name, nhd in self.adjacency_matrices.items():
#             res.append(
#                 self.index.join(
#                     pl.DataFrame(
#                         np.vstack(np.where(np.nan_to_num(nhd))).T,
#                         schema={
#                             "label": self.index.dtypes[0],
#                             "neighbors": self.index.dtypes[0],
#                         },
#                     )
#                     .select(pl.all().map_dict(dict(zip(*self.index.with_row_count()))))
#                     .groupby("label", maintain_order=True)
#                     .agg(pl.col("neighbors").alias(f"{name}_Neighbors")),
#                     on="label",
#                     how="outer",
#                 ).drop("label")
#             )
#         return pl.concat(res, how="horizontal")

#     def count(self) -> pl.DataFrame:
#         return pl.concat(
#             [
#                 pl.DataFrame(
#                     np.nansum(nhd, axis=1, keepdims=True),
#                     schema=[f"{name}_count"],
#                 )
#                 for name, nhd in self.adjacency_matrices.items()
#             ],
#             how="horizontal",
#         )


# class Nhd:
#     def __init__(
#         self,
#         df: pl.DataFrame,
#         kd_tree: KDTree | None = None,
#         weighed_touch_matrix: NDArray | None = None,
#         # TODO: Should this be none?
#         meta_columns: tuple[str, ...] = ("label", "^Centroid(-[xyz])?$"),
#         feature_columns: str | Iterable[str] | pl.Expr | None = None,
#     ):
#         self._df = df
#         self.kd_tree = kd_tree
#         self.weighed_touch_matrix = weighed_touch_matrix
#         self.df_meta = df.select(pl.col(e) for e in meta_columns)

#     @classmethod
#     def from_labelimage(cls, lbl: LabelImage) -> "Nhd":
#         df = get_position_and_orientation_features(lbl)
#         return cls.from_dataframe(df)

#     @classmethod
#     def from_dataframe(
#         cls,
#         df: pl.DataFrame,
#         index_column: str = "label",
#         centroid_column: str | Sequence[str] = "^Centroid(-[xyz])?$",
#     ) -> "Nhd":
#         if "roi" in df:
#             assert len(df.select("roi").unique()) == 1, "`df` contains multiple 'roi'."
#         if "object" in df:
#             assert (
#                 len(df.select("object").unique()) == 1
#             ), "`df` contains multiple 'object'."

#         if isinstance(centroid_column, str):
#             df_centroid = df.select(centroid_column).pipe(unnest_all_structs)
#         else:
#             df_centroid = df.select(pl.col(e) for e in centroid_column)
#         kd_tree = KDTree(df_centroid.select(sorted(df_centroid.columns, reverse=True)))
#         return Nhd(df, kd_tree=kd_tree)

#     @singledispatchmethod
#     def radius(self, r, include_self: bool) -> "NeighborhoodAggregator":
#         raise NotImplementedError()

#     @radius.register(float)
#     @radius.register(int)
#     def radius_single(
#         self, r: float | int, include_self: bool
#     ) -> "NeighborhoodAggregator":
#         return self.radius_list([r], include_self=include_self)

#     @radius.register(abc.Sequence)
#     def radius_list(
#         self, r: Sequence[float | int], include_self: bool
#     ) -> "NeighborhoodAggregator":
#         if self.kd_tree is None:
#             raise ValueError("No KDTree for radius query found.")

#         names = [f"RADIUS-{_r}{'s' if include_self else ''}" for _r in r]
#         adjacency_matrices = [
#             radius_adjacency_matrix(
#                 kdtree=self.kd_tree, r=_r, include_self=include_self
#             )
#             for _r in r
#         ]
#         # TODO: Use consistent naming
#         return NeighborhoodAggregator(
#             name=names,
#             adjacency_matrix=adjacency_matrices,
#             features=self._df,
#             meta=self.df_meta,
#         )

#     @singledispatchmethod
#     def knn(self, k, include_self: bool) -> "NeighborhoodAggregator":
#         raise NotImplementedError()

#     @knn.register(int)
#     def knn_single(self, k: int, include_self: bool) -> "NeighborhoodAggregator":
#         return self.knn_list([k], include_self=include_self)

#     @knn.register(abc.Sequence)
#     def knn_list(
#         self, k: Sequence[int], include_self: bool
#     ) -> "NeighborhoodAggregator":
#         if self.kd_tree is None:
#             raise ValueError("No KDTree for knn query found.")

#         names = [f"KNN-{_k}{'s' if include_self else ''}" for _k in k]
#         adjacency_matrices = [
#             knn_adjacency_matrix(kdtree=self.kd_tree, k=_k, include_self=include_self)
#             for _k in k
#         ]
#         # return list(zip(names, adjacency_matrices))
#         return NeighborhoodAggregator(
#             name=names,
#             adjacency_matrix=adjacency_matrices,
#             features=self._df,
#             meta=self.df_meta,
#         )

#     # TODO: Implement delaunay neighborhood
#     # TODO: Implement touch neighborhood
#     def touch(self, s: int, include_self: bool, threshold: float = 0.0):
#         if self.weighed_touch_matrix is None:
#             raise ValueError("No weighted touch matrix for touch query found.")
#         return touch_adjacency_matrix(
#             weighted_touch_matrix=self.weighed_touch_matrix,
#             k_neighbors=s,
#             include_self=include_self,
#             threshold_area=threshold,
#         )


# df = pl.read_csv(r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\features\ccp\Ccp_cycle10.csv")
# df_one = df.filter(pl.col('roi')=='B07_px+2788_py+0295')
# points = df_one.select(sorted(df_one.select(pl.col('^Centroid-[xyz]$')).columns, reverse=True))
# tree = KDTree(points)


# def radius_adjacency_matrix(
#     kdtree: KDTree,
#     r: float,
#     include_self: bool = False,
#     not_adjacent_value: float = np.nan,
# ):
#     query = kdtree.query_radius(kdtree.data, r=r, return_distance=False)

#     adjacency_matrix = np.empty((kdtree.data.shape[0], kdtree.data.shape[0]))
#     adjacency_matrix.fill(not_adjacent_value)
#     for i, j in enumerate(query):
#         adjacency_matrix[i, j] = 1.0
#     if not include_self:
#         np.fill_diagonal(adjacency_matrix, not_adjacent_value)
#     return adjacency_matrix


# def knn_adjacency_matrix(
#     kdtree: KDTree,
#     k: int,
#     include_self: bool = False,
#     not_adjacent_value: float = np.nan,
# ):
#     if k > (kdtree.data.shape[0] - 1):
#         print(
#             f"k = {k} is larger than number of objects - 1 ({kdtree.data.shape[0]-1}), setting k = {kdtree.data.shape[0]-1}"
#         )
#         k = kdtree.data.shape[0] - 1

#     query = kdtree.query(kdtree.data, k=k + 1, return_distance=False)

#     adjacency_matrix = np.empty((kdtree.data.shape[0], kdtree.data.shape[0]))
#     adjacency_matrix.fill(not_adjacent_value)
#     for i, j in enumerate(query):
#         adjacency_matrix[i, j] = 1.0
#     if not include_self:
#         np.fill_diagonal(adjacency_matrix, not_adjacent_value)
#     return adjacency_matrix


# def delaunay_adjacency_matrix(
#     kdtree: KDTree,
#     k_neighbors: int,
#     include_self: bool = False,
#     not_adjacent_value: float = np.nan,
# ):
#     delaunay = spatial.Delaunay(kdtree.data)
#     G = nx.Graph()
#     for path in delaunay.simplices:
#         nx.add_path(G, path)
#     touching = np.asarray(
#         nx.adjacency_matrix(G, nodelist=np.arange(len(kdtree.data))).todense()
#     )
#     np.fill_diagonal(touching, 1)
#     k_touching = np.linalg.matrix_power(touching, k_neighbors)
#     adjacency_matrix = np.where(k_touching, 1.0, not_adjacent_value)
#     return adjacency_matrix

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
