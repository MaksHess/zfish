# %%
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd
import polars as pl

from zfish.features.polars_utils import unnest_all_structs, unnest_structs

if TYPE_CHECKING:
    import napari
    from numpy.typing import NDArray

    from zfish.features.neighborhood.neighborhoods import NeighborhoodAggregator, Nhd

def _napari_centroids(df: pl.DataFrame, centroid_column: str = 'Centroid') -> np.array:
    if centroid_column in df.columns:
        df = df.pipe(unnest_structs, [centroid_column])
    centroid_cols = sorted(df.select(pl.col(f"^{centroid_column}.*$")).columns, reverse=True)
    return df.select(centroid_cols).to_numpy()


def napari_centroids(
    df: pl.DataFrame | pd.DataFrame,
    centroid_column: str = 'Centroid',
    features=["PhysicalSize", "Roundness", "Elongation"],
    **kwargs,
) -> dict[str, Any]:
    if isinstance(df, pd.DataFrame):
        df = pl.DataFrame(df)
    return {
        **{
            "data": _napari_centroids(df, centroid_column=centroid_column),
            "size": 5,
            "face_color": features[0],
            "edge_color": "white",
            # "shading": "spherical",
            "features": df.select(features).to_pandas().copy(),
        },
        **kwargs,
    }


def napari_points_to_vectors(
    df: pl.DataFrame, columns: str | Sequence[str] | None = None, scale_factors_zyx=None
) -> dict[str, Any]:
    if columns is None:
        columns = ["*"]
    elif isinstance(columns, str):
        columns = [columns]
    df_vec = df.select([pl.col(e) for e in columns]).pipe(unnest_all_structs)
    start = df_vec.select(
        sorted(df_vec.columns[: df_vec.shape[1] // 2], reverse=True)
    ).to_numpy()
    end = df_vec.select(
        sorted(df_vec.columns[df_vec.shape[1] // 2 :], reverse=True)
    ).to_numpy()
    if scale_factors_zyx is None:
        scale_factors_zyx = np.ones((1, start.shape[1]))
    else:
        scale_factors_zyx = np.array(scale_factors_zyx).reshape((1, -1))
    start = start * scale_factors_zyx
    end = end * scale_factors_zyx
    direction = end - start
    return {"data": np.stack([start, direction], axis=1)}


def napari_principal_axes(
    df: pl.DataFrame,
    # scaler=lambda x: 2 * np.sqrt(x),
    scaler=lambda x: x / 2,
    axes_column: str = "EquivalentEllipsoidDiameter",
) -> tuple[dict[str, Any], ...]:
    """
    Convert a DataFrame with 'PricipalAxes' into a tuple of dictionaries that can be individually passed to napari vectors layer.
    """
    ndims = len(df[axes_column].struct.fields)
    if ndims == 2:
        colors = ["blue", "red"]
        pa_colss = (("a-y", "a-x"), ("b-y", "b-x"))
        pm_cols = ("a", "b")
    else:
        colors = ["blue", "green", "red"]
        pa_colss = (("a-z", "a-y", "a-x"), ("b-z", "b-y", "b-x"), ("c-z", "c-y", "c-x"))
        pm_cols = ("a", "b", "c")
    axes = []
    for color, pa_cols, pm_col in zip(colors, pa_colss, pm_cols):
        direction = (
            df.select("PrincipalAxes")
            .unnest("PrincipalAxes")
            .select([pl.col(e) for e in pa_cols])
            .to_numpy()
        )
        centroids = _napari_centroids(df)
        length_scaler = scaler(
            df.select(axes_column).unnest(axes_column).select(pm_col).to_numpy()
        )
        print(centroids.shape, length_scaler.shape)
        data = np.stack(
            np.broadcast_arrays(centroids, direction * length_scaler), axis=1
        )
        axes.append({"data": data, "edge_color": color})
    return tuple(axes)


def napari_adjacency(points: "NDArray[Any]", adjacency_matrix: "NDArray[Any]", highlight_idx: int | None = None, highlight_alpha=1.0, alpha=0.1):
    start_idxs, end_idxs = np.where(np.nan_to_num(adjacency_matrix))
    highlight_idxs = np.sort(
        np.concatenate(np.where(start_idxs==highlight_idx) + np.where(end_idxs==highlight_idx))
    )
    start = points[start_idxs]
    end = points[end_idxs]
    direction = end - start
    lines = np.stack([start, direction], axis=1)

    reds = np.ones(len(lines))
    greens = np.zeros(len(lines))
    blues = np.zeros(len(lines))
    alphas = np.ones(len(lines)) * alpha

    reds[highlight_idxs] = 0.0
    blues[highlight_idxs] = 1.0
    alphas[highlight_idxs] = highlight_alpha
    
    colors = np.vstack([reds, greens, blues, alphas]).T

    return {'data': lines, 'edge_color': colors}


def napari_weighted_adjacency(points: "NDArray[Any]", adjacency_matrix: "NDArray[Any]",  weight_matrix: "NDArray[float]"):
    start_idxs, end_idxs = np.where(adjacency_matrix)

    start = points[start_idxs]
    end = points[end_idxs]
    direction = end - start
    lines = np.stack([start, direction], axis=1)
    weights = weight_matrix[start_idxs, end_idxs]

    return {'data': lines, 'features': {'distances': weights}, 'edge_colormap': 'viridis'}

def napari_gradients(
        df, 
        centroid_column: str | Sequence[str] = '^Centroid-.*$', 
        gradient_column: str | Sequence[str] = '^.*Gradient.*$',
        normalize: bool = True,
        color_norm: bool = True,
        translate_column: str | None = None,
        translate_yx: tuple[float, float] = (700.0, 700.0),
        **kwargs,
        ) -> dict[str, Any]:
    more_args = {}
    if normalize or color_norm:
        df = df.with_columns(pl.concat_list(gradient_column).arr.eval(pl.all().pow(2).sum().sqrt()).explode().alias('Norm'))
    if normalize:
        df = df.with_columns(pl.col(gradient_column) / pl.col('Norm'))
    if color_norm:
        more_args['features'] = df.select(pl.col('Norm')).to_pandas()
        more_args['edge_color'] = 'Norm'

    centroid_columns = sorted(df.select(pl.col(centroid_column)).columns, reverse=True)
    gradient_columns = sorted(df.select(pl.col(gradient_column)).columns, reverse=True)
    start = df.select(centroid_columns)
    direction = df.select(gradient_columns)
    assert start.shape[1] in [2, 3]
    assert direction.shape == start.shape
    vecs = np.stack([start, direction], axis=1)
    return {
    **{
        'data': vecs,
        'length': 10
    },
    **more_args,
    **kwargs,
    }

def translate_on_grid(df, groupby='roi', dx=700, dy=700, n_rows=None):
    rois = df[groupby].unique(maintain_order=True)
    n_embryos = len(rois)
    if n_rows is None:
        n_rows = int(np.ceil(np.sqrt(n_embryos)))
    n_cols = int(np.ceil(n_embryos) / n_rows)

    trans_x_idx = pl.Series('trans_x', [i for i in range(n_rows) for _ in range(n_cols)][:n_embryos])
    trans_y_idx = pl.Series('trans_y', [i for _ in range(n_rows) for i in range(n_cols)][:n_embryos])

    trans_x = trans_x_idx * dx
    trans_y = trans_y_idx * dy
    trans = pl.concat([rois.to_frame(), trans_x.to_frame(), trans_y.to_frame()], how='horizontal')
    df = (
    df.join(trans, on='roi')
    .with_columns([
        (pl.col('Centroid-x') + pl.col('trans_x')).alias('CentroidTrans-x'),
        (pl.col('Centroid-y') + pl.col('trans_y')).alias('CentroidTrans-y'),
        pl.col('Centroid-z').alias('CentroidTrans-z'),
    ]).drop(['trans_x', 'trans_y'])
)
    return df



def napari_neighborhood(nhd: "Nhd", nhd_agg: "NeighborhoodAggregator", label: int):
    assert nhd.df_meta is not None, "No `df_meta` dataframe found."
    assert "Centroid" in nhd.df_meta, "No 'Centroid's found in `df_meta`."
    assert "label" in nhd.df_meta, "No 'label's found in `df_meta`."
    assert nhd_agg.adjacency_matrices is not None, "No `neighborhoods` found."
    assert label in nhd.df_meta["label"], f"The requested label: {label} was not found."

    nhood_lines = []
    for i, (name, nhood) in enumerate(nhd_agg.adjacency_matrices.items()):
        df = (
            pl.DataFrame(
                data=nhood, schema=nhd.df_meta["label"].cast(pl.Utf8).to_list()
            )
            .select(pl.col(str(label)).alias("neighbor"))
            .with_columns(nhd.df_meta.select(pl.col("label").alias("neighbor_label")))
            .filter(pl.col("neighbor").is_not_nan())
            .with_columns(pl.lit(label).cast(pl.Int64).alias("label"))
            .drop("neighbor")
            .select(["label", "neighbor_label"])
            # .with_columns(pl.struct('label', 'neighbor_label').alias('edges'))
            .join(nhd.df_meta, on="label")
            .join(
                nhd.df_meta,
                left_on="neighbor_label",
                right_on="label",
                suffix="_neighbor",
            )
            .pipe(unnest_all_structs)
        )

        centroid_columns = sorted(
            df.select("^Centroid-[xyz]$").columns, reverse=True
        )  # ([z], y, x)
        centroid_neighbor_columns = sorted(
            df.select("^Centroid_neighbor-[xyz]$").columns, reverse=True
        )  # ([z], y, x)

        start = df.select(centroid_columns).to_numpy()
        end = df.select(centroid_neighbor_columns).to_numpy()
        direction = end - start
        lines = np.stack([start, direction], axis=1)
        nhood_lines.append(lines)
    return nhood_lines


def add_time_dim(lines, t=0):
    return np.concatenate([np.ones((len(lines), 2, 1)) * t, lines], axis=-1)


# %%
def visualize_nhds_of_varying_size(df: pl.DataFrame, nhd: "Nhd", nhd_agg: "NeighborhoodAggregator", label: int, viewer: "napari.Viewer | None" = None, name: str = None, **kwargs):
    import napari
    if viewer is None:
        viewer = napari.Viewer()
    all_neighborhoods = np.concatenate(
        [
            add_time_dim(nhood, i)
            for i, nhood in enumerate(napari_neighborhood(nhd, nhd_agg, label))
        ]
    )
    viewer.add_vectors(all_neighborhoods, name=name, **kwargs)
    return viewer



def example_visualize_nhds_of_varying_size():
    import napari

    from zfish.features.neighborhood.neighborhoods import Nhd
    from zfish.roi.spatial_roi import Roi
    from zfish.roi.visualize import imshow_roi

    CENTER_NUCLEUS = 958
    BORDER_NUCLEUS = 1462

    fn = r"M:\marvwy\20220721_ZE4i2_aligned\imgs\B05_px+0262_py+0108.h5"
    roi = Roi.from_file(fn, level=1, features_root=r"M:\marvwy\20220721_ZE4i2_aligned\features_old\B05_px+0262_py+0108")
    df = roi.tables["nucleiRaw3"]
    nhd = Nhd.from_dataframe(df)

    agg = nhd.knn(list(range(0, 500, 20)), True)

    all_neighborhoods_center = np.concatenate(
        [
            add_time_dim(nhood, i)
            for i, nhood in enumerate(napari_neighborhood(nhd, agg, CENTER_NUCLEUS))
        ]
    )
    all_neighborhoods_border = np.concatenate(
        [
            add_time_dim(nhood, i)
            for i, nhood in enumerate(napari_neighborhood(nhd, agg, BORDER_NUCLEUS))
        ]
    )

    viewer = napari.Viewer()
    viewer.add_points(**napari_centroids(df))
    viewer.add_vectors(all_neighborhoods_center)
    viewer.add_vectors(all_neighborhoods_border)


# %%
def example_visualize_all_nhds_using_time_axis():
    # not finished
    all_neighborhoods = [
        napari_neighborhood(nhd, nhd.radius(40, True), i)[0]
        for i in nhd.df_meta["label"]
    ]
    timed_neighborhoods = [
        add_time_dim(nhood, i) for i, nhood in enumerate(all_neighborhoods)
    ]
    # %%
    viewer = napari.Viewer()
    viewer = imshow_roi(nucs, viewer=viewer)
    viewer.add_points(**napari_centroids(df))
    viewer.add_vectors(np.concatenate(all_neighborhoods_center))
    viewer.add_vectors(np.concatenate(all_neighborhoods_border))
