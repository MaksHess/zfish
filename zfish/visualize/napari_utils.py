# %%
import warnings
from typing import TYPE_CHECKING, Any, Sequence

import cmap
import napari
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from polars.type_aliases import SelectorType
from scipy.sparse import coo_array, csr_array

from zfish.features.polars_utils import nest_structs, unnest_all_structs, unnest_structs

COLORMAPS = {"circular": "cet_CET_C9s", "diverging": "cet_bkr"}

if TYPE_CHECKING:
    from typing import Generator, TypeAlias

    from numpy.typing import NDArray
    from scipy.sparse import csr_array

    from zfish.features.neighborhood.neighborhoods import NeighborhoodQueryObject

    SparseArray: TypeAlias = csr_array | coo_array


def _napari_centroids_data_and_features(
    df: pl.DataFrame,
    features: SelectorType | Sequence[str] | None = cs.numeric(),
    centroid_column_base: str = "[Cc]entroid",
    translate_group: str | None = None,
    translate_sort: str | None = None,
    translate_n_rows: int | None = None,
    translate_zyx: tuple[float, float, float] = (0.0, 700.0, 700.0),
) -> tuple[np.ndarray, pd.DataFrame | None]:
    if centroid_column_base in df.columns:
        df = df.pipe(unnest_structs, [centroid_column_base], sep=".")
    centroid_columns = sorted(
        cs.expand_selector(df, cs.matches(f"^{centroid_column_base}\W[xyz]$")),
        reverse=True,
        # df.select(pl.col(f"^{centroid_column_base}\\W[xyz]$")).columns, reverse=True
    )
    if features is None:
        feature_columns = []
    elif isinstance(features, str):
        feature_columns = [features]
    else:
        feature_columns = cs.expand_selector(df, features)

    if translate_group is not None:
        if translate_group not in df.columns:
            raise ValueError(
                f"translate_group: {translate_group!r} not found in columns: {df.columns}"
            )
        if translate_sort is not None:
            if translate_sort not in df.columns:
                raise ValueError(
                    f"translate_sort: {translate_sort!r} not found in columns: {df.columns}"
                )
            df = df.sort(translate_sort)

        df_trans = _translate_on_grid(
            df.select([translate_group, *centroid_columns, *feature_columns]),
            centroid_column_base=centroid_column_base,
            group=translate_group,
            dx=translate_zyx[2],
            dy=translate_zyx[1],
            n_rows=translate_n_rows,
        )
    else:
        df_trans = df
    return (
        df_trans.select(centroid_columns).to_numpy(),
        df.select(feature_columns).to_pandas(),
    )


def _translate_on_grid(
    df,
    centroid_column_base: str = "Centroid",
    group="roi",
    dx=700,
    dy=700,
    n_rows=None,
    passthrough: "SelectorType | None" = None,
):
    rois = df[group].unique(maintain_order=True)
    n_embryos = len(rois)
    if n_rows is None:
        n_rows = int(np.ceil(np.sqrt(n_embryos)))
    n_cols = int(np.ceil(n_embryos / n_rows))

    trans_x_idx = pl.Series(
        "trans_x", [i for i in range(n_rows) for _ in range(n_cols)][:n_embryos]
    )
    trans_y_idx = pl.Series(
        "trans_y", [i for _ in range(n_rows) for i in range(n_cols)][:n_embryos]
    )

    trans_x = trans_x_idx * dx
    trans_y = trans_y_idx * dy
    df_trans = pl.concat(
        [rois.to_frame(), trans_x.to_frame(), trans_y.to_frame()], how="horizontal"
    )


    if passthrough is None:
        return (
            df.join(df_trans, on=group).with_columns(
                pl.col(f"^{centroid_column_base}\Wz$"),
                pl.col(f"^{centroid_column_base}\Wy$") + pl.col("trans_y"),
                pl.col(f"^{centroid_column_base}\Wx$") + pl.col("trans_x"),
            )
        ).select(pl.col(f"^{centroid_column_base}\W[xyz]$"))
    else:
        return (
            df.join(df_trans, on=group).with_columns(
                pl.col(f"^{centroid_column_base}\Wz$"),
                pl.col(f"^{centroid_column_base}\Wy$") + pl.col("trans_y"),
                pl.col(f"^{centroid_column_base}\Wx$") + pl.col("trans_x"),
            )
        ).select(pl.col(f"^{centroid_column_base}\W[xyz]$"), passthrough)

    # centroid_columns = sorted(df.select(pl.col(f"^{centroid_column_base}\W[xyz]$")).columns, reverse=True)

    # df.select(centroid_columns).with_columns(pl.col(f'^{centroid_column}'))

    # df = (
    #     df.join(trans, on="roi")
    #     .with_columns(
    #         [
    #             (pl.col("Centroid-x") + pl.col("trans_x")).alias("CentroidTrans-x"),
    #             (pl.col("Centroid-y") + pl.col("trans_y")).alias("CentroidTrans-y"),
    #             pl.col("Centroid-z").alias("CentroidTrans-z"),
    #         ]
    #     )
    #     .drop(["trans_x", "trans_y"])
    # )
    # return df


def napari_centroids(
    df: pl.DataFrame | pd.DataFrame,
    centroid_column: str = "[Cc]entroid",
    features: SelectorType | Sequence[str] | None = cs.numeric()
    - cs.by_dtype(pl.UInt8, pl.UInt16),
    face_colormap="turbo",
    face_color_cycle="set1",
    translate_group: str | None = None,
    translate_sort: str | None = None,
    translate_n_rows: int | None = None,
    translate_zyx: tuple[float, float, float] = (0.0, 700.0, 700.0),
    **kwargs,
) -> dict[str, Any]:
    if isinstance(df, pd.DataFrame):
        df = pl.DataFrame(df)
    data, df_features = _napari_centroids_data_and_features(
        df,
        features=features,
        centroid_column_base=centroid_column,
        translate_group=translate_group,
        translate_sort=translate_sort,
        translate_n_rows=translate_n_rows,
        translate_zyx=translate_zyx,
    )
    return {
        **{
            "data": data,
            "features": df_features,
            "size": 10,
            "face_color": df_features.columns[0] if df_features is not None else None,
            "face_colormap": cmap.Colormap(face_colormap).to_vispy(),
            "face_color_cycle": cmap.Colormap(face_color_cycle).to_mpl().colors,
            # "face_contrast_limits": None,
            "edge_color": "white",
            "edge_width": 0.01,
            # "shading": "spherical",
        },
        **kwargs,
    }


def napari_gradients(
    df,
    # centroid_column: str | Sequence[str] = "^[Cc]entroid.[zyx]$",
    centroid_column_base: str = "centroid",
    gradient_column_base: str | Sequence[str] = "grad",
    normalize: bool = True,
    color_norm: bool = True,
    translate_group: str | None = None,
    translate_sort: str | None = None,
    translate_n_rows: int | None = None,
    translate_zyx: tuple[float, float, float] = (0.0, 700.0, 700.0),
    **kwargs,
) -> dict[str, Any]:
    centroid_column = f"^{centroid_column_base}.[zyx]$"
    gradient_column = f"^{gradient_column_base}.[zyx]$"

    more_args = {}
    if normalize or color_norm:
        df = df.with_columns(
            pl.concat_list(gradient_column)
            .list.eval(pl.all().pow(2).sum().sqrt())
            .explode()
            .alias("Norm")
        )
    if normalize:
        df = df.with_columns(pl.col(gradient_column) / pl.col("Norm"))
    if color_norm:
        more_args["features"] = df.select(pl.col("Norm")).to_pandas()
        more_args["edge_color"] = "Norm"

    if translate_group is not None:
        if translate_group not in df.columns:
            raise ValueError(
                f"translate_group: {translate_group!r} not found in columns: {df.columns}"
            )
        if translate_sort is not None:
            if translate_sort not in df.columns:
                raise ValueError(
                    f"translate_sort: {translate_sort!r} not found in columns: {df.columns}"
                )
            df = df.sort(translate_sort)
        df = _translate_on_grid(
            df,
            centroid_column_base=centroid_column_base,
            group=translate_group,
            dx=translate_zyx[-1],
            dy=translate_zyx[-2],
            n_rows=translate_n_rows,
            passthrough="^grad.[zyx]$",
        )
    centroid_columns = sorted(df.select(pl.col(centroid_column)).columns, reverse=True)
    gradient_columns = sorted(df.select(pl.col(gradient_column)).columns, reverse=True)
    # print(centroid_columns)
    # print(gradient_columns)
    start = df.select(centroid_columns)
    direction = df.select(gradient_columns)
    print(start, direction)

    assert start.shape[1] in [2, 3]
    assert direction.shape == start.shape
    vecs = np.stack([start, direction], axis=1)
    # print(vecs)

    return {
        **{"data": vecs, "length": 10},
        **more_args,
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
        centroids, _ = _napari_centroids_data_and_features(df)
        length_scaler = scaler(
            df.select(axes_column).unnest(axes_column).select(pm_col).to_numpy()
        )
        print(centroids.shape, length_scaler.shape)
        data = np.stack(
            np.broadcast_arrays(centroids, direction * length_scaler), axis=1
        )
        axes.append({"data": data, "edge_color": color})
    return tuple(axes)


from itertools import cycle


def napari_adjacency(
    points: "NDArray[Any]",
    adjacency_matrix: "SparseArray",
    highlight_idxs: Sequence[int] | int | None = None,
    highlight_cmap: cmap.Colormap = cmap.Colormap("Dark2"),
    highlight_alpha=1.0,
    background_color: cmap.Color = cmap.Color((0.5, 0.5, 0.5, 0.5)),
    **napari_vector_kwargs,
):
    coo = adjacency_matrix.tocoo()
    start_idxs, end_idxs = coo.row, coo.col
    start = points[start_idxs]
    end = points[end_idxs]
    direction = end - start
    lines = np.stack([start, direction], axis=1)

    colors = np.repeat(np.asarray(background_color).reshape(1, -1), len(lines), axis=0)

    if highlight_idxs is not None:
        if isinstance(highlight_idxs, int):
            highlight_idxs = (highlight_idxs,)

        for idx, color in zip(highlight_idxs, cycle(highlight_cmap.lut())):
            color[-1] = highlight_alpha
            colors[start_idxs == idx, :] = color

    return {
        "data": lines,
        "edge_color": colors,
        "vector_style": "line",
        **napari_vector_kwargs,
    }


def napari_weighted_adjacency(
    points: "NDArray[Any]",
    adjacency_matrix: "SparseArray",
    **napari_vector_kwargs,
):
    coo = adjacency_matrix.tocoo()
    start_idxs, end_idxs = coo.row, coo.col

    start = points[start_idxs]
    end = points[end_idxs]
    direction = end - start
    lines = np.stack([start, direction], axis=1)
    weights = coo.data

    return {
        "data": lines,
        "features": {"distances": weights},
        "edge_colormap": "viridis",
        "edge_color": "distances",
        **napari_vector_kwargs,
    }


def napari_neighborhoods_with_bg(
    nqo: "NeighborhoodQueryObject",
    highlight_idxs=None,
    highlight_cmap: cmap.Colormap = cmap.Colormap("Dark2"),
    highlight_alpha=1.0,
    background_color: cmap.Color = cmap.Color((0.5, 0.5, 0.5, 0.1)),
    **napari_vector_kwargs,
) -> "Generator[dict[str, Any], None, None]":
    for query, csr in nqo.query_neighborhoods:
        yield napari_adjacency(
            nqo.points,
            csr,
            name=str(query),
            highlight_idxs=highlight_idxs,
            highlight_cmap=highlight_cmap,
            highlight_alpha=highlight_alpha,
            background_color=background_color,
            **napari_vector_kwargs,
        )


def napari_neighborhoods(
    nqo: "NeighborhoodQueryObject",
    highlight_idxs=None,
    highlight_perc: float = 0.05,
    highlight_cmap: cmap.Colormap = cmap.Colormap("Dark2"),
    highlight_alpha=1.0,
    background_color: cmap.Color | None = cmap.Color((0.5, 0.5, 0.5, 0.1)),
    points: "NDArray[Any] | None" = None,
    **napari_vector_kwargs,
) -> "Generator[tuple[dict[str, Any], dict[str, Any] | None], None, None]":
    for query, csr in nqo.query_neighborhoods:
        if highlight_idxs is None:
            n_samples = max(round(highlight_perc * nqo.index.height), 1)
            highlight_idxs = nqo.index.sample(n_samples)["index"].to_list()
        if points is None:
            points = nqo.points

        coo = csr.tocoo()
        mask = np.isin(coo.row, highlight_idxs)
        coo_highlight = coo_array(
            (coo.data[mask], (coo.row[mask], coo.col[mask])), shape=coo.shape
        )
        # return coo, coo_highlight, highlight_idxs
        vecs_highlight = napari_adjacency(
            points,
            coo_highlight,
            name=str(query),
            highlight_idxs=highlight_idxs,
            highlight_cmap=highlight_cmap,
            highlight_alpha=highlight_alpha,
            **napari_vector_kwargs,
        )
        if background_color is None:
            vecs_bg = None
        else:
            vecs_bg = napari_adjacency(
                points,
                coo,
                name=f"{str(query)}_bg",
                highlight_idxs=None,
                background_color=background_color,
                **napari_vector_kwargs,
            )

        yield (vecs_highlight, vecs_bg)


def show_neighborhoods(
    nqo: "NeighborhoodQueryObject",
    viewer: "napari.Viewer | None" = None,
    highlight_labels=None,
    highlight_idxs=None,
    highlight_perc: float = 0.05,
    highlight_cmap: cmap.Colormap = cmap.Colormap("Dark2"),
    highlight_alpha=1.0,
    background_color: cmap.Color = cmap.Color((0.5, 0.5, 0.5, 0.1)),
    background_show: bool = False,
    points_show: bool = True,
    points_size=5,
    vecs_edge_width=1.0,
    points: "NDArray[Any] | None" = None,
    **napari_vector_kwargs,
) -> "napari.Viewer | None":
    if len(nqo.queries) == 0:
        warnings.warn("No queries found!")
        return
    if viewer is None:
        viewer = napari.Viewer()

    if highlight_labels is not None:
        highlight_idxs = (
            pl.concat([nqo.label, nqo.index], how="horizontal")
            .filter(pl.col("label").is_in(highlight_labels))["index"]
            .to_list()
        )
    if highlight_idxs is None:
        n_samples = max(round(highlight_perc * nqo.index.height), 1)
        highlight_idxs = nqo.index.sample(n_samples)["index"].to_list()

    if not background_show:
        background_color = None

    if points is None:
        points = nqo.points
    else:
        assert (
            points.shape == nqo.points.shape
        ), f"Wrong shape in points provided: {points.shape} vs {nqo.points.shape}"

    for vecs_fg, vecs_bg in napari_neighborhoods(
        nqo,
        highlight_idxs=highlight_idxs,
        highlight_cmap=highlight_cmap,
        highlight_alpha=highlight_alpha,
        background_color=background_color,
        points=points,
    ):
        if vecs_bg is not None:
            viewer.add_vectors(
                **{
                    **vecs_bg,
                    **napari_vector_kwargs,
                    "opacity": 0.4,
                    "blending": "additive",
                    "edge_width": vecs_edge_width * 0.1,
                }
            )

        viewer.add_vectors(
            **{**vecs_fg, **napari_vector_kwargs, "edge_width": vecs_edge_width}
        )

    if points_show:
        viewer.add_points(points, size=points_size, shading="spherical")

    return viewer


def show_multi_neighborhoods(
    nqo: "NeighborhoodQueryObject",
    viewer: "napari.Viewer | None" = None,
    highlight_labels=None,
    highlight_idxs=None,
    highlight_perc: float = 0.05,
    highlight_cmap: cmap.Colormap = cmap.Colormap("Dark2"),
    highlight_alpha=1.0,
    background_color: cmap.Color = cmap.Color((0.5, 0.5, 0.5, 0.1)),
    background_show: bool = False,
    points_show: bool = True,
    points_size=5,
    vecs_edge_width=1.0,
    translate_zyx: tuple[float, float, float] = (0, 120, 120),
    translate_n_rows: int | None = None,
    translate_group: str | None = "roi",
    **napari_vector_kwargs,
):
    points = nqo.points.copy()
    df_regions = pl.DataFrame(
        points, schema=["Centroid.z", "Centroid.y", "Centroid.x"]
    ).with_columns(nqo.region_ids)
    points_trans, _ = _napari_centroids_data_and_features(
        df_regions,
        features=None,
        translate_group=translate_group,
        translate_n_rows=translate_n_rows,
        translate_zyx=translate_zyx,
    )
    return show_neighborhoods(
        nqo,
        viewer=viewer,
        highlight_labels=highlight_labels,
        highlight_idxs=highlight_idxs,
        highlight_perc=highlight_perc,
        highlight_cmap=highlight_cmap,
        highlight_alpha=highlight_alpha,
        background_color=background_color,
        background_show=background_show,
        points_show=points_show,
        points_size=points_size,
        vecs_edge_width=vecs_edge_width,
        points=points_trans,
    )


# def show_neighborhoods(
#     nqo: NeighborhoodQueryObject,
#     df: pl.DataFrame,
#     highlight_labels=None,


# )


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
def visualize_nhds_of_varying_size(
    df: pl.DataFrame,
    nhd: "Nhd",
    nhd_agg: "NeighborhoodAggregator",
    label: int,
    viewer: "napari.Viewer | None" = None,
    name: str = None,
    **kwargs,
):
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
    roi = Roi.from_file(
        fn,
        level=1,
        features_root=r"M:\marvwy\20220721_ZE4i2_aligned\features_old\B05_px+0262_py+0108",
    )
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
