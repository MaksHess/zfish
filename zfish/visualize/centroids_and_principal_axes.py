# %%
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from zfish.features.polars_utils import unnest_all_structs

if TYPE_CHECKING:
    from zfish.features.neighborhood.neighborhoods import NeighborhoodAggregator, Nhd


def _napari_centroids(df: pl.DataFrame, centroid: str = "Centroid") -> np.array:
    DIMS = ("z", "y", "x")
    ndim = len(df["Centroid"].struct.fields)
    sel = [pl.col(d) for d in DIMS[-ndim:]]
    return df.select(centroid).unnest(centroid).select(sel).to_numpy()


def napari_centroids(
    df: pl.DataFrame,
    features=["PhysicalSize", "Roundness", "Elongation"],
) -> dict[str, Any]:
    return {
        "data": _napari_centroids(df),
        "size": 5,
        "face_color": features[0],
        "edge_color": "white",
        # "shading": "spherical",
        "features": df.select(features).to_pandas().copy(),
    }


def napari_points_to_vectors(
        df: pl.DataFrame,
        columns: str | Sequence[str] | None = None,
        scale_factors_zyx = None
) -> dict[str, Any]:
    if columns is None:
        columns = ['*']
    elif isinstance(columns, str):
        columns = [columns]
    df_vec = (
        df.select([
            pl.col(e) for e in columns
            ])
            .pipe(unnest_all_structs)
    )
    start = df_vec.select(sorted(df_vec.columns[:df_vec.shape[1]//2], reverse=True)).to_numpy()
    end = df_vec.select(sorted(df_vec.columns[df_vec.shape[1]//2:], reverse=True)).to_numpy()
    if scale_factors_zyx is None:
        scale_factors_zyx = np.ones((1, start.shape[1]))
    else:
        scale_factors_zyx = np.array(scale_factors_zyx).reshape((1, -1))
    start = start * scale_factors_zyx
    end = end * scale_factors_zyx
    direction = end - start
    return {'data': np.stack([start, direction], axis=1)}
            
    

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


def napari_neighborhood(nhd: "Nhd", nhd_agg: "NeighborhoodAggregator", label: int):
    assert nhd.df_meta is not None, "No `df_meta` dataframe found."
    assert "Centroid" in nhd.df_meta, "No 'Centroid's found in `df_meta`."
    assert "label" in nhd.df_meta, "No 'label's found in `df_meta`."
    assert nhd_agg.neighborhoods is not None, "No `neighborhoods` found."
    assert label in nhd.df_meta["label"], f"The requested label: {label} was not found."

    nhood_lines = []
    for i, (name, nhood) in enumerate(nhd_agg.neighborhoods.items()):
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
        start = df.select(["Centroid-z", "Centroid-y", "Centroid-x"]).to_numpy()
        end = df.select(
            ["Centroid_neighbor-z", "Centroid_neighbor-y", "Centroid_neighbor-x"]
        ).to_numpy()
        direction = end - start
        lines = np.stack([start, direction], axis=1)
        nhood_lines.append(lines)
    return nhood_lines


def add_time_dim(lines, t=0):
    return np.concatenate([np.ones((len(lines), 2, 1)) * t, lines], axis=-1)


# %%
def example_visualize_nhds_of_varying_size():
    import napari

    from zfish.features.neighborhood.neighborhoods import Nhd
    from zfish.roi.spatial_roi import Roi
    from zfish.roi.visualize import imshow_roi

    CENTER_NUCLEUS = 958
    BORDER_NUCLEUS = 1462

    fn = r"M:\marvwy\20220721_ZE4i2_aligned\imgs\B05_px+0262_py+0108.h5"
    roi = Roi.from_file(fn, level=1)
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
        napari_neighborhood(nhd, nhd.radius(40, True), i)[0] for i in nhd.df_meta["label"]
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
