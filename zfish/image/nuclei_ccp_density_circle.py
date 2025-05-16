# %%
from typing import NewType, Sequence

import numpy as np
import polars as pl
from pingouin import circ_mean

from zfish.features.neighborhood import aggregation_functions as agg_funcs
from zfish.features.neighborhood.neighborhoods import NeighborhoodQueryObject
from zfish.image.coordinates import (
    Image,
    _map_feature_to_spaces,
    get_racecar_lane,
    lane_number_generator,
)
from zfish.multi_table.raster_io_v2 import (
    _aggregate_label_object_paths,
    _images_to_queries,
    _label_images_to_queries,
    _label_objects_to_queries,
    get_quantile_extent,
)
from zfish.multi_table.tables_io import scan_resources

# CCP_QUICKSAVE = r"C:\Users\hessm\Documents\Programming\Python\zfish\paper\nuclei_features_with_ccp_and_pseudotime.parquet"
# CCP_QUICKSAVE = r"C:\Users\hessm\Documents\Programming\Python\zfish\paper\linear_models_in_r_features.parquet"
CCP_QUICKSAVE = r"C:\Users\hessm\Documents\zfish_local\ccp_models\latest_models\ccp_separate_per_cycle_res.parquet"


object_type = "nucleiRaw3"
# channels = ["PCNA.0", "bCatenin.1", "DAPI.1"]
channels = ["PCNA.0", "DAPI.1", "Pol-II-S2P.0"]
# channel_masks = ["nucleiRaw3", "cells", "nucleiRaw3"]
# channel_masks = ["nucleiRaw3", "nucleiRaw3", "nucleiRaw3"]
channel_masks = None
img_id_columns = ["roi", "m", "o", "label"]
m = 1

z_model = "ExpPos-2D"
# z_model = None
# t_model = "Exp"
t_model = "Exp"

feature = "NormalizedCCP"

roi_names = [  # cy10
    "F06_px+2620_py-0693",
    "F07_px+0166_py+2613",
    "G03_px+2426_py-0008",
    "E04_px+0133_py+2206",
    "D03_px+1967_py-0299",
    "G07_px+1761_py+0108",
    "E05_px-0106_py+1980",
    "F04_px+2084_py-0112",
    "F07_px+2568_py+0269",
    "G05_px+2051_py-0486",
    "F05_px+1412_py-0002",
    "D02_px+2103_py+0282",
]

roi_names = [  # cy9
    "F03_px-0248_py-1668",
    "D07_px+1980_py+0295",
    "D05_px-0073_py-2017",
    "B06_px-0261_py-1410",
    "E07_px-0280_py-1306",
    "C06_px-1120_py+1896",
    "G03_px+0101_py-2178",
    "D02_px-0015_py-1539",
    "E05_px-0538_py-2178",
]

roi_names = [  # cy10 early
    "F07_px+2568_py+0269",
    "F04_px+2084_py-0112",
    "F04_px-0067_py-0738",
    "C04_px+1341_py-1074",
    "E06_px+2071_py-0054",
    "F03_px+2058_py+0095",
    "B06_px-0118_py+2019",
    "F07_px+0166_py+2613",
    "G05_px+2051_py-0486",
    "B06_px+1929_py-0209",
    "F05_px+1412_py-0002",
    "D05_px-0112_py+2000",
    "G02_px+2432_py+0702",
    "F06_px+2620_py-0693",
    "D02_px+2103_py+0282",
    "E04_px+2742_py-0060",
]

roi_names = [  # cy11 full range
    "G02_px+1825_py-1662",
    "D03_px-1933_py+0424",
    "G05_px-2108_py+0398",
    "E06_px-2120_py+0631",
    "E05_px-0106_py+1980",
]
# roi_names = [  # c10 early subset
#     "F04_px+2084_py-0112",
#     "G05_px+2051_py-0486",
#     "D02_px+2103_py+0282",
#     "D05_px-0112_py+2000",
# ]

df = pl.read_parquet(CCP_QUICKSAVE)
df_rois = df.filter(pl.col("roi").is_in(roi_names))


# r = rl.collect().pipe_tables(
#     lambda x: x.with_columns(
#         pl.col("path").str.replace(
#             r"Z:\\hmax\\MARVWY_RESTORED\\20220721_ZE4i2_aligned1",
#             r"C:\Users\hessm\Documents\zfish_local",
#         )
#     )
# )
# r = rl.collect()
rl = scan_resources()
r = rl.collect()
r_roi = r.filter(pl.col("roi").is_in(roi_names))

# %%
df_imgs, df_label_objects = r_roi.pipe(
    _aggregate_label_object_paths,
    object_type,
    channels=channels,
    channel_masks=channel_masks,
    multiscale_level=m,
    z_model=z_model,
    t_model=t_model,
)
df_label_objects = df_label_objects.with_columns(
    pl.concat_str(*img_id_columns, separator="__").alias("id")
)

df_label_objects

# %%
df_imgs

# %%
lo_queries = _label_objects_to_queries(
    df_label_objects,
    df_imgs,
    strategy="numpy",
    fuzzy_mask_sigma=2,
    dask_chunk_size=(50, 1000, 1000),
)

lo_imgs = {e.object_id: e.compute() for e in lo_queries}


imgs = {
    name: Image.from_array_meta((array.max(axis=1, keepdims=True), meta)).center()
    for name, (array, meta) in lo_imgs.items()
}


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# df_rois = df.filter(pl.col("roi").is_in(roi_names)).with_columns(
#     pl.col("roi").cast(pl.String)
# )

sns.scatterplot(
    df_rois.to_pandas(),  # .filter(pl.col("roi").str.starts_with("F05")).to_pandas(),
    x="NormalizedCCP",
    y="nuc_PCNA.0_Mean",
    hue="roi",
    alpha=0.5,
    s=8,
    legend=False,
)
# sns.scatterplot(df_rois.to_pandas(), x="NormalizedCCP", y="nuc_PhysicalSize")

plt.gca().set_yscale("log")
# %%
from scipy.stats import circmean

from zfish.multi_table.schemas_v2 import sel

df_all = (
    (
        df_label_objects.cast({"roi": pl.String, "o": pl.String})
        .with_row_index()
        .join(
            df.select(
                # pl.col("roi").cast(pl.Categorical),
                # pl.col("object").cast(pl.Categorical).alias("o"),
                # pl.col('o').cast(pl.Categorical),
                pl.col("roi").cast(pl.String),
                # pl.col("o").cast(pl.String),
                pl.col("label"),
                pl.col("NormalizedCCP"),
            ),
            on=["roi", "label"],
            how="left",
        )
        .join(
            df.group_by("roi")
            .agg(
                pl.col.NormalizedCCP.filter(pl.col.NormalizedCCP.is_not_null())
                .map_elements(circmean, return_dtype=pl.Float64)
                .alias("NormalizedCCP_circmean")
            )
            .with_columns(pl.col("roi").cast(pl.String)),
            on=["roi"],
            how="left",
        )
        .join(
            r.label_objects.cast({"roi": pl.String, "o": pl.String}).select(
                "roi", "o", "label", "centroid.z"
            ),
            on=["roi", "o", "label"],
            how="left",
        )
        .filter(pl.col("NormalizedCCP").is_not_null())
        # .sample(300)
        # .sort("NormalizedCCP")
        .sort(feature)
        # .filter(pl.col("roi").is_in(["G02_px+2432_py+0702"]))
    )
    .join(
        df_rois.select(sel.idx, "nuc__Count").cast({"roi": pl.String}),
        on=["roi", "label"],
    )
    .sort(pl.col.nuc__Count)
)
# all_index = df_all["index"]
extents = get_quantile_extent(
    df_all,
    r.multiscale_levels,
    match_length_of_last_two_axes=True,
    quantile=0.95,
)
extent = max(extents[1:])


import numpy as np

from zfish.image.coordinates import (
    ChannelCoord,
    arrange_in_circle_by_feature,
    canvas_from_images,
    imshow,
    translate_images,
)


def individual_canvases_per_group(
    df_all: pl.DataFrame,
    imgs: dict[str, Image],
    group_column="roi",
    feature_range=(0, 2 * np.pi),
    feature="NormalizedCCP",
    id_column: str = "id",
    extent=12.35,
):
    canvases = []
    grids = []

    for name, df_site in df_all.group_by(group_column, maintain_order=True):
        df_sort = df_site.sort(feature)
        # feature_range = df_sort[feature].min(), df_sort[feature].max()

        # print(feature_range)
        arr = arrange_in_circle_by_feature(
            df_sort.select("id", feature),
            feature_range=feature_range,
            tolerance=0.1,
            space_multiplier=1,
        )
        cell_arr = arr.filter(pl.col("index_samples").is_not_null()).with_columns(
            pl.col("y", "x") * extent
        )
        grid = arr.select(pl.col("y", "x") * extent)
        imgs_sort = [imgs[id_] for id_ in df_site["id"]]
        imgs_trans = translate_images(
            imgs_sort, cell_arr.select("y", "x").rows(named=True)
        )
        canvas = canvas_from_images(imgs_trans)

        canvases.append(
            canvas._from_template(
                c=ChannelCoord([f"{'__'.join(name)}__{c}" for c in canvas.c.coord])
            )
        )
        grids.append(grid)
    return canvases, grids, cell_arr


def one_canvas_for_all(
    df_all: pl.DataFrame,
    imgs: dict[str, Image],
    feature_range=(0, 2 * np.pi),
    feature="NormalizedCCP",
    id_column: str = "id",
    extent=12.35,
    tolerance=0.1,
):
    df_sort = df_all.sort(feature)
    arr = arrange_in_circle_by_feature(
        df_sort.select(id_column, feature),
        feature_range=feature_range,
        tolerance=tolerance,
        space_multiplier=1,
        max_iterations=1000,
    )
    cell_arr = arr.filter(pl.col("index_samples").is_not_null()).with_columns(
        pl.col("y", "x") * extent
    )

    grid = arr.select(pl.col("y", "x") * extent, "id")
    imgs_sort = [imgs[id_] for id_ in df_sort["id"]]
    imgs_trans = translate_images(imgs_sort, cell_arr.select("y", "x").rows(named=True))
    canvas_huge = canvas_from_images(imgs_trans)
    return canvas_huge, grid


def imshow_canvases(canvases, n_cols=4, dx=700.0, dy=700.0, viewer=None):
    import napari

    if viewer is None:
        viewer = napari.Viewer()

    n_cols = 4
    dx = 700.0
    dy = 700.0
    for i, canvas in enumerate(canvases):
        if i == 0:
            visible = True
        else:
            visible = True

        imshow(
            canvas.translate(x=dx * (i % n_cols), y=dy * (i // n_cols)),
            viewer,
            contrast_limits=[(0, 2800), (0, 1300)],
            colormap=["red", "blue"],
            visible=visible,
        )
    return viewer


def simple_zoom_animation(
    canvas, initial_zoom=0.3, final_zoom=3.5, out_file="test.mov"
):
    import napari
    from napari_animation import Animation
    from napari_animation.easing import Easing

    INITIAL_ZOOM = initial_zoom
    FINAL_ZOOM = final_zoom
    viewer = napari.Viewer()
    animation = Animation(viewer)

    imshow(canvas, viewer, colormap=["red", "blue"])
    viewer.camera.zoom = INITIAL_ZOOM
    animation.capture_keyframe()
    animation.capture_keyframe(steps=30)
    viewer.camera.zoom = FINAL_ZOOM
    animation.capture_keyframe(steps=60, ease=Easing.QUADRATIC)
    animation.capture_keyframe(steps=60)
    viewer.camera.zoom = INITIAL_ZOOM
    animation.capture_keyframe(steps=60, ease=Easing.QUADRATIC)
    animation.capture_keyframe(steps=30)
    animation.animate(
        out_file,
    )


def groupwise_concentric_circles(
    n_spaces=200, n_groups=4, n_init=12, lane_length=0, return_meta=True
):
    groups = []

    for i in range(n_groups):
        lanes = []
        if i == 0:
            total_spaces = 0
            for n_lane, length_lane, r in lane_number_generator(
                n_samples=n_init, r=1.0, lane_length=lane_length
            ):
                total_spaces += n_lane
                # print(n_lane, length_lane, r)
                lanes.append((n_lane, length_lane, r))
                dt = length_lane / n_lane
                if total_spaces > n_spaces:
                    break
        else:
            total_spaces = 0
            last_lane_n, _, last_lane_r = groups[-1][-1]
            for i, (n_lane, length_lane, r) in enumerate(
                lane_number_generator(
                    n_samples=last_lane_n, r=last_lane_r, lane_length=lane_length
                )
            ):
                if i == 0:
                    continue
                total_spaces += n_lane
                lanes.append((n_lane, length_lane, r))
                if total_spaces > n_spaces:
                    break
        groups.append(lanes)

    arrs = []
    for i, group in enumerate(groups):
        arrs.append(
            pl.concat(
                [
                    get_racecar_lane(n, r, lane_length=lane_length).with_columns(
                        pl.lit(i).alias("lane")
                    )
                    for i, (n, _, r) in enumerate(group)
                ],
                how="vertical",
            ).with_columns(pl.lit(i).alias("group"), pl.col("y", "x") / dt)
        )
    return arrs, pl.concat(
        [
            pl.DataFrame(
                e,
                schema={
                    "n_samples": pl.UInt32,
                    "lane_lenth": pl.Float64,
                    "r": pl.Float64,
                },
                orient="row",
            )
            .with_columns(((pl.col("r") + dt / 2) / dt).alias("r_outer"))
            .with_columns(pl.col("r") / dt, pl.lit(i).alias("group"))
            for i, e in enumerate(groups)
        ]
    )

    # df_sort[feature],
    # feature_range=feature_range,
    # tolerance=0.1,
    # space_multiplier=1,


def arrange_in_groupwise_concentric_circles(
    df_all: pl.DataFrame,  # feature, group
    feature_range: tuple[float, float] = (0.0, 2 * np.pi),
    feature: str = "NormalizedCCP",
    group_column: str = "roi",
    index_column: str = "index_global",  # autogenerated if not provided
    n_spaces=200,
    n_init=12,
    lane_length=0,
    max_iterations: int = 1000,
    tolerance: float = 0.2,
):
    groups = df_all[group_column].unique(maintain_order=True)
    spaces, space_metas = groupwise_concentric_circles(
        n_spaces=n_spaces, n_groups=len(groups), n_init=n_init, lane_length=lane_length
    )

    if index_column not in df_all:
        df_all = df_all.with_row_index(index_column)
    # remappings = []
    df_groups = []
    for space, (name_group, df_group) in zip(
        spaces,
        df_all.select(index_column, feature, group_column)
        # .with_row_index(index_column)
        .group_by(group_column, maintain_order=True),
    ):
        df_group_sorted = df_group.sort(feature).with_row_index("index_samples")
        remapping = _map_feature_to_spaces(
            df_group_sorted[feature],
            space,
            feature_range=feature_range,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        df_groups.append(
            df_group_sorted.join(remapping, on="index_samples").join(
                space.with_row_index("index_spaces"), on="index_spaces"
            )
        )

    return (
        pl.concat(df_groups, how="vertical").select(
            index_column, "group", "x", "y", "t2"
        ),
        pl.concat(spaces, how="vertical"),
        space_metas,
    )


# ID = NewType("ID", str)
# # init_radius = 2.570796326794896
# init_radius = 1.0
# lane_length = 0
# # init_n = 31
# init_n = 12
# n_spaces = 200
# n_groups = 6
# # %%

# # %%
# canvas, grid = one_canvas_for_all(df_all, imgs)
# %%
import polars.selectors as cs

from zfish.analysis.plots import plot_correlation
from zfish.features.polars_utils import drop_null_columns, drop_null_rows
from zfish.multi_table.tables_io import scan_features, to_wide

fl = scan_features()

df_density = (
    fl.density_count.filter(pl.col("o") == "nucleiRaw3").collect().pipe(to_wide)
)
df_plot = (
    df_density.join(df.select(sel.idx, "log2_nuc__Count"), on=["roi", "label"])
    .with_columns(cs.starts_with("RAD").log(2))
    .drop("label")
    .group_by("roi")
    .agg(
        cs.numeric().median(),
        cs.numeric().quantile(0.25).name.suffix("_q25"),
        cs.numeric().quantile(0.75).name.suffix("_q75"),
    )
    # .select(cs.numeric().fill_nan(0))
)  # .corr().pipe(drop_null_columns).pipe(drop_null_rows).pipe(plot_correlation)


y_err = [
    df_plot.select(pl.col("RAD:250.0_Count_q25") * -1 + pl.col("RAD:250.0_Count"))[
        "RAD:250.0_Count_q25"
    ],
    df_plot.select(pl.col("RAD:250.0_Count_q75") - pl.col("RAD:250.0_Count"))[
        "RAD:250.0_Count_q75"
    ],
]
sns.scatterplot(df_plot.to_pandas(), x="log2_nuc__Count", y="RAD:250.0_Count")
plt.gca().set_aspect("equal")
# plt.errorbar(
#     x=df_plot["log2_nuc__Count"],
#     y=df_plot["RAD:250.0_Count"],
#     yerr=y_err,
#     fmt="none",
#     c="black",
#     capsize=2,
# )
# sns.scatterplot(df_plot, x='log2_nuc__Count', y='RAD:250.0_Count_q25', size=2, color='black')
# sns.scatterplot(df_plot, x='log2_nuc__Count', y='RAD:250.0_Count_q75', size=2, color='black')
# sns.scatterplot(df_plot, x='log2_nuc__Count', y='RAD:250.0_Count')
# %%
df_plot.plot.scatter(x="log2_nuc__Count", y="RAD\:200\.0_Count", tooltip="roi")
# %%

groups, space, space_metas = arrange_in_groupwise_concentric_circles(
    df_all.sort("NormalizedCCP_circmean", descending=True).with_columns(
        pl.concat_str(img_id_columns, separator="__").alias("id")
    )[5:],
    n_init=30,
    index_column="id",
    n_spaces=500,
    tolerance=0.4,
)
group_points = (
    space_metas.group_by("group", maintain_order=True)
    .agg(pl.col.r_outer.last())
    .with_columns(pl.lit(0.0).alias("x"), pl.lit(0.0).alias("y"))
).join(
    groups.group_by("group", maintain_order=True)
    .agg(pl.col.id.first())
    .with_columns(pl.col.id.str.split("__").list.get(0)),
    on="group",
)

group_edge_shape = (
    np.asarray(
        group_points.select(
            pl.concat_list(
                pl.concat_list("y", "x").list.to_array(2).alias("pos"),
                pl.concat_list(pl.col("r_outer"), pl.col("r_outer"))
                .list.to_array(2)
                .alias("radius"),
            ).list.to_array(2)
        )["pos"]
    )
    * extent
)

density_outliers = [
    "F07_px+0166_py+2613",
    "F05_px+1412_py-0002",
    "D05_px-0112_py+2000",
]

color_cycle = ["red", "green", "blue"]

shape_layer_groups = {
    "data": group_edge_shape,
    # "text": text,
    "features": group_points.with_columns(
        pl.col("id").is_in(density_outliers)
    ).to_pandas(),
    "edge_width": 1,
    "edge_color": "id",
    "edge_color_cycle": ["white", "red"],
    "face_color": "#FFFFFF00",
    "opacity": 0.08,
    "shape_type": "ellipse",
}

text = {
    "string": "{id}",
    "size": 6,
    "color": "white",
    # "anchor": "center",
    "translation": [10, 0],
    # "size": {"feature": "size"},
    # "color": "green",
    # "color": {"feature": "good_point", "colormap": color_cycle},
}
# %%
imgs_sorted = [imgs[id_] for id_ in groups["id"]]
imgs_trans = translate_images(
    imgs_sorted, groups.select(pl.col("y", "x") * extent).rows(named=True)
)
canvas = canvas_from_images(imgs_trans)
# %%
canvas_all, space = one_canvas_for_all(df_all, imgs=imgs)
# %%
import napari

viewer = napari.Viewer()
viewer.add_shapes(
    **shape_layer_groups,
)
viewer.add_points(
    **{
        "data": group_points.select(
            pl.col("r_outer") * -1 * extent, pl.col("x")
        ).to_numpy(),
        "features": group_points.select("id").to_pandas(),
        "text": {**text, **{"size": 2}},
        "face_color": "#FFFFFF00",
        "size": 2,
        # "edge_color": "white",
        # "opacity": 0.0,
    }
)
imshow(canvas, viewer, colormap=["red", "blue", "green"])
# %%
import napari

cell_space = space.filter(pl.col("id").is_not_null(), pl.col("y", "x") * extent)

viewer = napari.Viewer()
viewer.add_points(
    **{
        "data": cell_space["y", "x"].to_numpy(),
        "features": cell_space.select("id").to_pandas(),
        "text": text,
        "face_color": "#FFFFFF00",
        "edge_color": "white",
        "opacity": 0.0,
    }
)
imshow(canvas_all, viewer, colormap=["red", "blue", "green"])

# %%
canvas_all

# sns.scatterplot(space, x="x", y="y", size=4, color="0.7")
# sns.scatterplot(groups.with_columns(pl.col('x', 'y')*extent), x="x", y="y", hue="group", palette="Set1")
# plt.gca().set_aspect("equal")
# # %%
# arrs = groupwise_concentric_circles()
# arr = pl.concat(arrs, how="vertical")


# sns.scatterplot(arr, x="x", y="y", hue="group")
# plt.gca().set_aspect("equal")
