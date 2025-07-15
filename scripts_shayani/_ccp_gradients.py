# %%
from itertools import chain, islice, product
from pathlib import Path

import napari
import polars as pl
import polars.selectors as cs
import tqdm

from zfish.features.neighborhood.gradient_meyer_et_al_3d import (
    circular_gradients,
    gradients,
)
from zfish.features.neighborhood.neighborhoods import NeighborhoodQueryObject
from zfish.image.coordinates import (
    Image,
    canvas_from_images,
    get_column_grid,
    imshow,
    translate_images,
)
from zfish.multi_table.raster_io_v2 import (
    _aggregate_image_paths,
    _aggregate_label_object_paths,
    _images_to_queries,
    _label_objects_to_queries,
)
from zfish.multi_table.tables_io import scan_features, scan_resources
from zfish.plot.plot_commons import BASE, UMAP, cmaps
from zfish.visualize.napari_utils import napari_centroids, napari_gradients

# CCP_QUICKSAVE = r"C:\Users\hessm\Documents\Programming\Python\zfish\paper\linear_models_in_r_features.parquet"
CCP_QUICKSAVE = (
    r"C:\Users\hessm\Documents\Programming\Python\thesis\Data\ccp1_and_ccp2.parquet"
)

feature = "NormalizedCCP"

lr = scan_resources()

df = pl.read_parquet(CCP_QUICKSAVE).join(
    lr.label_objects.filter(pl.col("o") == "nucleiRaw3")
    .select("roi", "o", "label", cs.starts_with("centroid"))
    .collect(),
    on=["roi", "label"],
)

df_all_raw = (
    df
    # df.filter(pl.col("cycle") > 9).filter(pl.col("cycle") < 11).sort("nuc__Count")
    # df.filter(pl.col("cycle") == 8)
    # df.filter(
    #     pl.col.roi.is_in(
    #         [
    #             "F07_px-0080_py-0060",
    #             "C04_px+0314_py-2230",
    #             # "E05_px-0538_py-2178",
    #             # "B05_px-0493_py-1526",
    #         ]
    #     )
    # )
)


df_all_out = df_all_raw.filter(pl.col("NormalizedCCP").is_null())
df_all = df_all_raw.filter(pl.col("NormalizedCCP").is_not_null())

r_all = (
    lr.collect()
    .filter(pl.col.roi.is_in(df_all_raw["roi"].unique().to_list()))
    .collect()
)

# %%
viewer = napari.Viewer()

viewer.add_points(
    **napari_centroids(
        df_all,
        "centroid",
        features=cs.by_name("NormalizedCCP"),
        translate_group="roi",
        translate_sort="log2_nuc__Count",
        translate_n_rows=20,
        size=15,
        face_colormap=cmaps.ccp.to_mpl(),
    )
)

viewer.add_points(
    **napari_centroids(
        df_all_out.with_columns(pl.col("NormalizedCCP").is_null().alias("is_outlier")),
        "centroid",
        features=cs.matches("is_outlier"),
        translate_group="roi",
        translate_sort="log2_nuc__Count",
        translate_n_rows=20,
        size=10,
        face_color="grey",
    )
)


# %%
import numpy as np
from scipy.stats import circmean, circstd, circvar

n_bins = 14
start = 6.3
ccp_col = "NormalizedCCP"
sort_by = ["cycle", "log2_nuc__Count_corr"]

df_emb = df_all.group_by("roi", maintain_order=True).agg(
    [
        pl.col("log2_nuc__Count_corr").first(),
        pl.col("log2_nuc__Count").first(),
        pl.col("cycle").first(),
        pl.col("_cycle").first(),
    ]
)


df_order = df_emb.sort(sort_by).with_row_index("order")
df_with_meta = df_all.join(df_order, on="roi")


df_outlier = (
    df_with_meta.sort("order")
    .group_by(["roi", "order"], maintain_order=True)
    .agg(pl.len().log(2))
    .with_columns(pl.median("len").rolling("order", period="5i").alias("len_window"))
    .with_columns((pl.col("len") < pl.col("len_window") * 0.9).alias("outlier"))
)

df_with_meta = df_with_meta.join(df_outlier.select("roi", "outlier"), on="roi").filter(
    pl.col("outlier").not_()
)

df_select = (
    df_order.with_columns(
        ((pl.col("log2_nuc__Count_corr") - (pl.col("cycle") - 1)) * 2 * np.pi)
        .cut(
            np.linspace(0, 2 * np.pi, n_bins, endpoint=False),
            labels=[f"{i}" for i in range(n_bins + 1)],
        )
        .cast(pl.String)
        .cast(pl.Int64)
        .alias("CcpGroup")
    )
    .group_by("cycle", "CcpGroup", maintain_order=True)
    .agg(pl.all().first())
)

df_compute = df_with_meta.filter(pl.col("roi").is_in(df_select["roi"]))
df_visualize = df_compute.join(
    df_select.select("roi", "cycle", "CcpGroup"), on="roi"
).with_columns(
    pl.col("centroid.y") + ((pl.col("cycle") - pl.col("cycle").min()) * 700.0),
    pl.col("centroid.x") + ((pl.col("CcpGroup") - pl.col("CcpGroup").min()) * 700.0),
)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(
    df_order.with_columns(
        pl.col("cycle") + np.random.randn(df_order.height) * 0.1,
        pl.col("_cycle") + np.random.randn(df_order.height) * 0.1,
    ),
    x="cycle",
    y="_cycle",
)
# %%
fig, ax = plt.subplots()
sns.scatterplot(df_outlier, x="order", y="len", hue="outlier")
sns.scatterplot(df_outlier, x="order", y="len_window", color="0.5")

fig, ax = plt.subplots()
sns.scatterplot(
    df_order.with_columns(
        pl.col("log2_nuc__Count_corr")
        .cut(np.linspace(7, 14, 64))
        .alias("devtime_class")
    ).to_pandas(),
    x="log2_nuc__Count_corr",
    y="log2_nuc__Count",
    hue="devtime_class",
    # palette=cmaps.ccp.to_mpl(),
    palette="Set1",
    legend=False,
)
# fig, ax = plt.subplots()
# sns.scatterplot(
#     df_order.with_columns(
#         pl.col("log2_nuc__Count_corr")
#         .cut(np.linspace(7, 14, 64))
#         .alias("devtime_class")
#     )
#     .with_columns(cs.starts_with("nuc_RAD").cbrt())
#     .to_pandas(),
#     x="log2_nuc__Count_corr",
#     y="nuc_RAD:250.0_Count",
#     hue="devtime_class",
#     # palette=cmaps.ccp.to_mpl(),
#     palette="Set1",
#     legend=False,
# )
# fig, ax = plt.subplots()
# sns.scatterplot(
#     df_with_meta.with_columns(
#         pl.col("log2_nuc__Count_corr")
#         .cut(np.linspace(7, 14, 64))
#         .alias("devtime_class")
#     )
#     .with_columns(cs.starts_with("nuc_RAD").cbrt())
#     .group_by("roi")
#     .agg(
#         pl.col("log2_nuc__Count_corr").first(),
#         cs.starts_with("nuc_RAD").median(),
#         pl.col("devtime_class").first(),
#     )
#     .to_pandas(),
#     x="log2_nuc__Count_corr",
#     y="nuc_RAD:50.0_Count",
#     hue="devtime_class",
#     # palette=cmaps.ccp.to_mpl(),
#     palette="Set1",
#     legend=False,
# )
fig, ax = plt.subplots()
sns.histplot(
    df_with_meta.with_columns(
        pl.col("log2_nuc__Count_corr")
        .cut(np.linspace(7, 14, 64))
        .alias("devtime_class")
    )
    .with_columns(cs.starts_with("nuc_RAD").cbrt())
    .group_by("roi")
    .agg(
        pl.col("log2_nuc__Count_corr").first(),
        cs.starts_with("nuc_RAD").median(),
        pl.col("devtime_class").first(),
    )
    .to_pandas(),
    x="log2_nuc__Count_corr",
    # y="nuc_RAD:100.0_Count",
    # hue="devtime_class",
    # palette=cmaps.ccp.to_mpl(),
    palette="Set1",
    legend=False,
    bins=np.linspace(7, 14, 64),
)

# %%
n_cols = 12

pre_feature = feature
# sort_by = ["cycle", "Ccp__Mean"]
sort_by = "order"
viewer = napari.Viewer()
viewer.add_points(
    **napari_centroids(
        df_visualize.sort(sort_by),
        face_colormap=cmaps.ccp.to_mpl(),
        centroid_column="centroid",
        features=cs.by_name(pre_feature),
        size=(1 / df_visualize.sort(sort_by)["log2_nuc__Count_corr"])
        / (1 / df_visualize.sort(sort_by)["log2_nuc__Count_corr"].max())
        * 15,
        # translate_group="roi",
        translate_group=None,
        translate_sort=None,
        translate_n_rows=n_cols,
    ),
    face_contrast_limits=(0, 2 * np.pi),
)
# viewer.add_points(
#     **napari_centroids(
#         df_all_out.join(df_emb, on="roi")
#         .sort(sort_by)
#         .with_columns(pl.lit(True).alias("is_outlier")),
#         centroid_column="centroid",
#         features=cs.by_name(pre_feature),
#         size=5,
#         translate_group="roi",
#         # translate_sort=sort_by,
#         translate_n_rows=n_cols,
#         face_color="grey",
#         # colormap='grey'
#     ),
#     # face_contrast_limits=(0, 2 * np.pi),
# )
# for grad in grads:
#     viewer.add_vectors(
#         **napari_gradients(
#             df_with_meta.sort(sort_by).with_columns(
#                 pl.DataFrame(grad, orient="row", schema=["grad.z", "grad.y", "grad.x"])
#             ),
#             length=30,
#             edge_width=4,
#             translate_group="roi",
#             # translate_sort=sort_by,
#             translate_n_rows=n_cols,
#         )
#     )
# imshow(canvas, viewer, colormap=["red", "blue", "green"])
# %%
from zfish.multi_table.raster_io_v2 import (
    _aggregate_image_paths,
    _aggregate_label_image_paths,
    _images_to_queries,
    _label_images_to_queries,
)

df_some = df_visualize.filter(pl.col("cycle") < 13)
df_embryo_masks = _aggregate_label_image_paths(
    r_all.filter(pl.col("roi").is_in(df_some["roi"].unique(maintain_order=True))),
    object_types=["embryoRaw"],
    multiscale_level=1,
    # channel_masks=["embryoRaw"] * 3,
    # z_model="ExpPos-1D",
    # t_model="Exp",
)


# %%
from zfish.multi_table.tables_io import scan_resources

r = scan_resources().collect()
df_imgs = _aggregate_image_paths(
    r.filter(pl.col("roi").is_in(df_some["roi"].unique(maintain_order=True))),
    channels=["DAPI.1", "PCNA."],
    multiscale_level=1,
)

# %%

# img_queries = _images_to_queries(df_imgs, dask_chunk_size=(100, 1000, 1000))

# imgs = {name: Image.from_array_meta(q.compute()) for name, q in img_queries.items()}
# %%
lbl_queries = _label_images_to_queries(
    df_embryo_masks, dask_chunk_size=(100, 1000, 1000)
)
lbls = {}
for k, v in tqdm.tqdm(lbl_queries.items()):
    lbls[k] = Image.from_array_meta(v.compute())
# %%
from zfish.features.neighborhood.neighborhoods import (
    DelaunayAdjacency,
    _csr_concatenate_along_corner,
    _delaunay_adjacency,
    get_delaunay_adjacency,
    multilineprofile,
)
from zfish.multi_table.feature_query_schemas import _to_si

# %%

adjs = []
for name, df_roi in df_compute.filter(pl.col("cycle") < 13).group_by(
    "roi", maintain_order=True
):
    print(name)
    lbl_img = lbls["__".join(name + ("embryoRaw",))]
    lbl = _to_si(
        (
            lbl_img.data,
            lbl_img.meta.from_template(
                type_="label", dims=("z", "y", "x"), name="embryoRaw"
            ),
        )
    )
    adjs.append(
        get_delaunay_adjacency(
            df_roi["centroid.z", "centroid.y", "centroid.x"].to_numpy(),
            lbl,
            n_samples=100,
        )
    )
adj_delaunay = DelaunayAdjacency(
    csr=_csr_concatenate_along_corner([e.csr for e in adjs]), is_weighted=True
)
# %%
from zfish.features.neighborhood import aggregation_functions as agg_funcs

nqo = NeighborhoodQueryObject.from_dataframe(
    df_some,
    label_columns=("roi", "label"),
    centroid_column="^centroid.*$",
    delaunay_adjacency=adj_delaunay,
)

df_smooth = (
    nqo.radius([0, 80], self_loops=True)
    # .knn([10, 25, 45], self_loops=True)
    .knn([10, 25], self_loops=True)
    .aggregate(agg_funcs.CircMean, df_some["NormalizedCCP"])
)

adj = next(nqo.delaunay(1).neighborhoods)


# adj = next(nqo.delaunay(1, threshold=0.6).neighborhoods)
# knn_adj = next(nqo.knn(3).neighborhoods)
# adj[np.where(adj.todense().sum(axis=1) < 3)] = knn_adj[
#     np.where(adj.todense().sum(axis=1) < 3)
# ]


pre_features = df_smooth.columns[-1:]

grads = []
divs = []
curls = []
for pre_feature in pre_features:
    print(pre_feature)
    grad = circular_gradients(
        nqo.points, adj.todense(), df_smooth[pre_feature].to_numpy()
    )
    grad_dxdxyz = gradients(nqo.points, adj.todense(), grad[:, 0])
    grad_dydxyz = gradients(nqo.points, adj.todense(), grad[:, 1])
    grad_dzdxyz = gradients(nqo.points, adj.todense(), grad[:, 2])

    grad_dx = grad_dxdxyz[:, 0]
    grad_dy = grad_dydxyz[:, 1]
    grad_dz = grad_dzdxyz[:, 2]

    curl_x = grad_dzdxyz[:, 1] - grad_dydxyz[:, 2]
    curl_y = grad_dxdxyz[:, 2] - grad_dzdxyz[:, 0]
    curl_z = grad_dydxyz[:, 0] - grad_dxdxyz[:, 1]

    # div = gradients(nqo.points, adj.todense(), grad)
    grads.append(grad)
    divs.append(grad_dx + grad_dy + grad_dz)
    curls.append(np.stack([curl_x, curl_y, curl_z], axis=1))


# div = (
#     nqo.delaunay(1, threshold=0.6)
#     .aggregate(agg_funcs.Sum, pl.DataFrame(grad, schema=["grad.z", "grad.y", "grad.x"]))
#     .select(pl.sum_horizontal(pl.all()).alias("div"))
# )

# %%
df_grad = (
    df_some.with_columns(
        pl.DataFrame(grads[0], schema=["grad.z", "grad.y", "grad.x"]),
        # pl.Series("div", divs[0]),
    )
    .with_columns(
        pl.DataFrame(curls[0], schema=["curl.z", "curl.y", "curl.x"]),
    )
    .with_columns(
        pl.Series("div", divs[0]),
    )
    # .with_columns(
    #     (pl.col("cycle_corr") + pl.col("Ccp__Mean") / (2 * np.pi)).alias(
    #         "log2_nuc__Count_corr"
    #     )
    # )
)

CCP_GRAD_QUICKSAVE = (
    r"C:\Users\hessm\Documents\Programming\Python\thesis\Data\ccp_with_grad.parquet"
)
df_grad.select(
    [
        "o",
        "roi",
        "label",
        "celltype_pred",
        "CCP",
        "NormalizedCCP",
        "DistanceToCircleCCP",
        "log2_nuc__Count",
        "log2_nuc__Count_corr",
        "_CCP",
        "_NormalizedCCP",
        "_cycle",
        "cycle",
        "centroid.z",
        "centroid.y",
        "centroid.x",
        "grad.z",
        "grad.y",
        "grad.x",
        "curl.z",
        "curl.y",
        "curl.x",
        "div",
    ]
).write_parquet(CCP_GRAD_QUICKSAVE)

# %%
import polars as pl
import polars.selectors as cs

CCP_GRAD_QUICKSAVE = (
    r"C:\Users\hessm\Documents\Programming\Python\thesis\Data\ccp_with_grad.parquet"
)
df_grad_ = pl.read_parquet(CCP_GRAD_QUICKSAVE)  # .rename({"grad.y.": "grad.y"})
# %%
import seaborn as sns

sns.scatterplot(
    df_grad_.to_pandas(),
    x="log2_nuc__Count_corr",
    y="log2_nuc__Count",
    # hue="Ccp__Mean",
)

# %%
n_cols = 6
import napari
import numpy as np
from scipy import linalg

from zfish.plot.plot_commons import cmaps
from zfish.visualize.napari_utils import napari_centroids, napari_gradients

pre_feature = "NormalizedCCP"
# sort_by = ["cycle", "Ccp__Mean"]
# sort_by = "order"
length_scaler = 0.5
size_scaler = 0.5

size = (
    (1 / df_grad_["log2_nuc__Count_corr"])
    / (1 / df_grad_["log2_nuc__Count_corr"].max())
    * 15
    * size_scaler
)


viewer = napari.Viewer()
viewer.add_points(
    **napari_centroids(
        df_grad_,
        # .with_columns(
        #     pl.Series(
        #         "div", grad_dx.sum(axis=1) + grad_dy.sum(axis=1) + grad_dz.sum(axis=1)
        #     )
        # ),
        # face_colormap=cmaps.ccp.to_mpl(),
        centroid_column="centroid",
        features=cs.by_name("NormalizedCCP"),
        size=size,
        # translate_group="roi",
        translate_group=None,
        translate_sort=None,
        translate_n_rows=n_cols,
        # face_contrast_limits=(-0.005, 0.005),
        face_contrast_limits=(0, 2 * np.pi),
        face_colormap=cmaps.ccp.to_mpl(),
        # face_colormap="PiYG",
    ),
    # face_contrast_limits=(0, 2 * np.pi),
)
viewer.add_points(
    **napari_centroids(
        df_grad_,
        # face_colormap=cmaps.ccp.to_mpl(),
        centroid_column="centroid",
        features=cs.by_name("div"),
        size=size,
        # translate_group="roi",
        translate_group=None,
        translate_sort=None,
        translate_n_rows=n_cols,
        face_contrast_limits=(-0.0001, 0.0001),
        # face_contrast_limits=(0, 2 * np.pi),
        # face_colormap=cmaps.ccp.to_mpl(),
        face_colormap="PiYG",
    ),
    # face_contrast_limits=(0, 2 * np.pi),
)
# viewer.add_points(
#     **napari_centroids(
#         df_grad,
#         # face_colormap=cmaps.ccp.to_mpl(),
#         centroid_column="centroid",
#         features=cs.by_name("curl_mag"),
#         size=14,
#         # translate_group="roi",
#         translate_group=None,
#         translate_sort=None,
#         translate_n_rows=n_cols,
#         face_contrast_limits=(-0.0001, 0.0001),
#         # face_contrast_limits=(0, 2 * np.pi),
#         # face_colormap=cmaps.ccp.to_mpl(),
#         face_colormap="PiYG",
#     ),
#     # face_contrast_limits=(0, 2 * np.pi),
# )
# viewer.add_points(
#     **napari_centroids(
#         df_some.with_columns(
#             # pl.Series(linalg.norm(curls[0], axis=1))
#             pl.Series(curls[0][:, 0])
#             # .is_between(-0.00015, 0.00015)
#             .alias("curl_mag")
#         ),
#         # face_colormap=cmaps.ccp.to_mpl(),
#         centroid_column="centroid",
#         features=cs.by_name("celltype_pred"),
#         size=14,
#         # translate_group="roi",
#         translate_group=None,
#         translate_sort=None,
#         translate_n_rows=n_cols,
#         face_contrast_limits=(-0.0001, 0.0001),
#         # face_contrast_limits=(0, 2 * np.pi),
#         # face_colormap=cmaps.ccp.to_mpl(),
#         face_colormap="PiYG",
#     ),
#     # face_contrast_limits=(0, 2 * np.pi),
# )
# viewer.add_points(
#     **napari_centroids(
#         df_all_out.join(df_emb, on="roi")
#         .sort(sort_by)
#         .with_columns(pl.lit(True).alias("is_outlier")),
#         centroid_column="centroid",
#         features=cs.by_name(pre_feature),
#         size=5,
#         translate_group="roi",
#         # translate_sort=sort_by,
#         translate_n_rows=n_cols,
#         face_color="grey",
#         # colormap='grey'
#     ),
#     # face_contrast_limits=(0, 2 * np.pi),
# )
viewer.add_vectors(
    **{
        **napari_gradients(
            df_grad_.select(cs.starts_with("centroid"), cs.starts_with("grad")),
            name="grad",
            length=40,
            # length=3*size,
            # edge_width=(size/3).to_frame().to_numpy(),
            edge_width=4,
            translate_group=None,
            # translate_group="roi",
            # translate_sort=sort_by,
            # translate_n_rows=n_cols,
        ),
        **{"edge_color": "#DDDDDDFF", "features": None},
    }
)

# %%
import pyvista as pv

from zfish.visualize.pyvista_utils import (
    _get_streamplot_components,
    compute_norm,
    stream_plot,
)

# %%

rois = [
    "B02_px+0198_py-2152",
    "F05_px-0015_py-1358",
    "E04_px+2742_py-0060",
    "F03_px+0159_py+2187",
]

subplots = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
]

size_scale_stat = 2.5
size_scale = 1.5

n_lines_stat = 2000

sizes = [
    12,
    10,
    8,
    6,
]
default_cpos = [
    (-1302.5324833545992, 482.93383621972845, 50.71494986009872),
    (114.42073220492563, 342.67707889727745, 345.55821403064544),
    (-0.12186830802510701, -0.985650006753266, 0.11679974180786815),
]
pv.global_theme.font.family = "arial"
po_stat = pv.Plotter(
    line_smoothing=True,
    point_smoothing=True,
    polygon_smoothing=True,
    shape=(2, 2),
    window_size=(800, 800),
    image_scale=4,
    off_screen=True,
)
po_dyn = pv.Plotter(
    line_smoothing=True,
    point_smoothing=True,
    polygon_smoothing=True,
    shape=(2, 2),
    window_size=(800, 800),
    image_scale=4,
    off_screen=False,
)

po_stat.parallel_projection = True
po_dyn.parallel_projection = True


for i, roi in enumerate(rois):
    df_one = (
        df_grad_.with_columns(
            (compute_norm(("grad.z", "grad.y", "grad.x"))).alias("norm")
        )
        .with_columns((pl.col("norm").log()).alias("log_norm"))
        .filter(pl.col("roi") == roi)
    )

    po_stat.subplot(*subplots[i])
    po_dyn.subplot(*subplots[i])

    # po.reset_camera(clipping_range=True)  # Optional, to fix clipping
    comps, p_stat = stream_plot(
        df_one,
        points=True,
        # stream_source_n_points=500,
        stream_source_n_points=n_lines_stat,
        point_size=sizes[i] * size_scale_stat,
        convex_hull=False,
        vector_scale="log_norm",
        centroid_columns=("nuc_Centroid-z", "nuc_Centroid-y", "nuc_Centroid-x"),
        vector_components=("grad.z", "grad.y", "grad.x"),
        point_hue="NormalizedCCP",
        # point_hue_norm=(0, 2*np.pi),
        # stream_hue="IntegrationTime",
        stream_hue=None,
        stream_opacity=0.3,
        # stream_hue_norm=(-400, 400),
        stream_render_lines_as_tubes=False,
        stream_line_width=3,
        return_components=True,
        stream=True,
        quiver_points=False,
        po=po_stat,
    )
    comps, p_dyn = stream_plot(
        df_one,
        points=True,
        # stream_source_n_points=500,
        stream_source_n_points=1000,
        point_size=sizes[i] * size_scale,
        convex_hull=False,
        vector_scale="log_norm",
        centroid_columns=("nuc_Centroid-z", "nuc_Centroid-y", "nuc_Centroid-x"),
        vector_components=("grad.z", "grad.y", "grad.x"),
        point_hue="NormalizedCCP",
        # point_hue_norm=(0, 2*np.pi),
        # stream_hue="IntegrationTime",
        stream_hue=None,
        stream_opacity=0.3,
        # stream_hue_norm=(-400, 400),
        stream_render_lines_as_tubes=False,
        stream_line_width=3,
        return_components=True,
        stream=True,
        quiver_points=False,
        po=po_dyn,
    )
    for po in [po_stat, po_dyn]:
        po.camera_position = [
            (-1188.8848612793934, 468.43133859485954, 54.90920175263625),
            (109.05490893125534, 339.9550971984863, 324.98784255981445),
            (-0.12186830802510701, -0.985650006753266, 0.11679974180786815),
        ]
        po.camera_set = True  # prevent auto-reset of camera

for po in [po_stat, po_dyn]:
    po.link_views()

zoom_factor = 1.3
po_stat.reset_camera()
po_stat.camera.zoom(zoom_factor)
# for i in range(len(rois)):
#     po.subplot(*subplots[i])
#     po.camera.zoom(zoom_factor)


out_cpos = po_stat.show(
    return_cpos=True,
)
po_stat.screenshot(
    r"C:\Users\hessm\Documents\Programming\Python\thesis\Figures\Plots\ccp_gradient_stream_8-11_static.png"
)


po_dyn.export_html(
    r"C:\Users\hessm\Documents\Programming\Python\thesis\Figures\Plots\ccp_gradient_stream_8-11.html"
)
# %%

# %%
po = pv.Plotter(
    line_smoothing=True,
    point_smoothing=True,
    polygon_smoothing=True,
    # shape=(2, 2),
    window_size=(800, 800),
    image_scale=4,
)
po.parallel_projection = True
# po.subplot(0, 1)
comps, p = stream_plot(
    df_one,
    points=True,
    stream_source_n_points=1000,
    point_size=12,
    convex_hull=False,
    vector_scale="log_norm",
    centroid_columns=("nuc_Centroid-z", "nuc_Centroid-y", "nuc_Centroid-x"),
    vector_components=("grad.z", "grad.y", "grad.x"),
    point_hue="NormalizedCCP",
    # point_hue_norm=(0, 2*np.pi),
    # stream_hue="IntegrationTime",
    stream_hue=None,
    # stream_hue_norm=(-400, 400),
    stream_render_lines_as_tubes=True,
    stream_line_width=4,
    return_components=True,
    stream=True,
    quiver_points=False,
    po=po,
)

po.link_views()

# p.export_html(
#     r"C:\Users\hessm\Documents\Programming\Python\thesis\Figures\Plots\ccp_gradient_stream.html"
# )
# p.parallel_projection = True
# p.show()
po.show(cpos=default_cpos)

# %%

comps = []
for roi in [
    "G06_px-0577_py-1410",
    "F07_px-0080_py-0060",
    "C04_px+0314_py-2230",
    "B06_px-0261_py-1410",
    "C06_px-0351_py-0906",
    "F05_px-0015_py-1358",
    "E06_px-0118_py-2462",
    "F04_px+0178_py-2140",
    "E05_px-0538_py-2178",
]:
    df_one = (
        df_grad_.filter(pl.col("roi") == roi)
        .with_columns((compute_norm(("grad.z", "grad.y", "grad.x"))).alias("norm"))
        .with_columns((pl.col("norm").log()).alias("log_norm"))
    )

    comp = _get_streamplot_components(
        df_one,
        vector_scale="norm",
        centroid_columns=("nuc_Centroid-z", "nuc_Centroid-y", "nuc_Centroid-x"),
        vector_components=("grad.z", "grad.y", "grad.x"),
        # vector_scale_factor=1e5,
        stream_terminal_speed=0.001,
        stream_source_n_points=3000,
        # stream_hue=None,
        # point_size=30,
    )
    comps.append(comp)

# %%


comp = comps[4]
po = pv.Plotter()

# po.add_volume(
#     comps["interp"], opacity=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.8), scalars="norm"
# )
po.add_mesh(comp["stream"], scalars="IntegrationTime")
po.add_mesh(comp["points"], scalars="norm")
# po.add_mesh(comps["quiver_points"], scalars="norm")

po.show()


# %%
comps = _get_streamplot_components(
    df_one,
    vector_scale="norm",
    centroid_columns=("nuc_Centroid-z", "nuc_Centroid-y", "nuc_Centroid-x"),
    vector_components=("grad.z", "grad.y", "grad.x"),
    # vector_scale_factor=1e5,
    stream_terminal_speed=0.001,
    stream_source_n_points=1000,
)

po = pv.Plotter()

# po.add_volume(
#     comps["interp"], opacity=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.8), scalars="norm"
# )
po.add_mesh(comps["stream"], scalars="IntegrationTime")
po.add_mesh(comps["points"], scalars="norm")
# po.add_mesh(comps["quiver_points"], scalars="norm")

po.show()


# %%

# %%

z_min, z_max, y_min, y_max, x_min, x_max = comps["points"].bounds
max_extent = max([z_max - z_min, y_max - y_min, x_max - x_min])
radius = max_extent / 2
default_kwargs = {
    "source_radius": radius,
    "terminal_speed": 0.1,
    "n_points": 3000,
}
streamline_kwargs = {}
streamline_kwargs["return_source"] = True
# return interp, interp_raw, hull, grid, point_cloud
stream, src = comps["interp"].streamlines(
    "vectors",
    **{
        **default_kwargs,
        **streamline_kwargs,
    },
)

pv.PolyData(stream.points).plot()

# %%
from pyvista import examples

mesh = examples.download_carotid()

streamlines, src = mesh.streamlines(
    return_source=True,
    max_time=100.0,
    initial_step_length=2.0,
    terminal_speed=0.1,
    n_points=25,
    source_radius=2.0,
    source_center=(133.1, 116.3, 5.0),
)
# %%
for grad, name in zip(grads[-1:], pre_features[-1:]):
    viewer.add_vectors(
        **napari_gradients(
            df_some.with_columns(
                pl.DataFrame(grad, orient="row", schema=["grad.z", "grad.y", "grad.x"])
            ),
            name=name,
            length=30,
            edge_width=4,
            translate_group=None,
            edge_color="#030303",
            # translate_group="roi",
            # translate_sort=sort_by,
            # translate_n_rows=n_cols,
        )
    )

for curl, name in zip(curls[-1:], pre_features[-1:]):
    viewer.add_vectors(
        **napari_gradients(
            df_some.with_columns(
                pl.DataFrame(curl, orient="row", schema=["curl.z", "curl.y", "curl.x"])
            ),
            gradient_column_base="curl",
            name=name,
            length=30,
            edge_width=4,
            translate_group=None,
            edge_color="#030303",
            # translate_group="roi",
            # translate_sort=sort_by,
            # translate_n_rows=n_cols,
        )
    )
# imshow(canvas, viewer, colormap=["red", "blue", "green"])

# gradss = []
# for name, df_one in df_visualize.filter(pl.col("cycle_corr") < 11).group_by("roi"):
#     print(name)
#     nqo = NeighborhoodQueryObject.from_dataframe(
#         df_one, label_columns=("roi", "label"), centroid_column="^centroid.*$"
#     )

#     df_smooth = (
#         nqo.radius([0, 80, 120], self_loops=True)
#         .knn([10, 25, 45], self_loops=True)
#         .aggregate(agg_funcs.CircMean, df_all["NormalizedCCP"])
#     )

#     # adj = next(nqo.knn(18).neighborhoods)
#     adj = next(nqo.delaunay(1).neighborhoods)
#     # adj = next(nqo.knn(12).neighborhoods)

#     pre_features = df_smooth.columns

#     grads = []
#     for pre_feature in pre_features:
#         print(pre_feature)
#         grad = circular_gradients(
#             nqo.points, adj.todense(), df_smooth[pre_feature].to_numpy()
#         )
#         grads.append(grad)
#     gradss.append(grads)
# %%
import matplotlib.pyplot as plt
import napari
import numpy as np
import seaborn as sns

xx, yy = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))


f = lambda x, y: np.cos(y * 6) + np.sin(x * 7)

zz = f(xx, yy)

np.stack([zz.flatten(), yy.flatten(), xx.flatten()]).T

# fdx = lambda x, y: 2 * x * y
# fdy = lambda x, y: x**2 + 3

# divdx = lambda x, y: 2 * y
# divdy = lambda x, y: 0

fdx = lambda x, y: 7 * np.cos(7 * x)
fdy = lambda x, y: -6 * np.sin(6 * y)

divdx = lambda x, y: -49 * np.sin(7 * x)
divdy = lambda x, y: -36 * np.cos(6 * y)


dxx = fdx(xx, yy)
dyy = fdy(xx, yy)


fig, ax = plt.subplots(2, 1, dpi=300)
plt.sca(ax[0])
plt.imshow(zz, extent=(-1, 1, -1, 1), origin="lower")
plt.quiver(xx.flatten(), yy.flatten(), dxx.flatten(), dyy.flatten())
plt.sca(ax[1])
plt.imshow(divdx(xx, yy) + divdy(xx, yy), extent=(-1, 1, -1, 1), origin="lower")
# plt.quiver(yy.flatten(), xx.flatten(), dyy.flatten(), dxx.flatten())
# %%
viewer = napari.Viewer()
viewer.add_points(
    np.stack([zz.flatten(), yy.flatten(), xx.flatten()]).T,
    size=0.01,
    axis_labels=["z", "y", "x"],
)
# imshow_si(dapi, viewer=viewer)

# %%

df_all["roi"].unique().to_list()

df_imgs = _aggregate_image_paths(
    r_all.filter(pl.col("roi").is_in(df_some["roi"].unique(maintain_order=True))),
    channels=["PCNA.0", "DAPI.1", "Pol-II-S2P.0"],
    multiscale_level=3,
    channel_masks=["embryoRaw"] * 3,
    z_model="ExpPos-1D",
    t_model="Exp",
)

img_queries = _images_to_queries(df_imgs, dask_chunk_size=(200, 1000, 1000))

imgs = {}
for k, v in tqdm.tqdm(img_queries.items()):
    imgs[k] = Image.from_array_meta(v.compute())

# %%


extent = 700.0

grid_ = get_column_grid(len(imgs), 8) * extent
grid = (
    df_select.with_columns(
        ((pl.col("cycle_corr") - pl.col("cycle_corr").min()) * extent).alias("y"),
        ((pl.col("CcpGroup") - pl.col("CcpGroup").min()) * extent).alias("x"),
    ).sort(["cycle_corr", "CcpGroup"])
    # .group_by(["cycle_corr", "CcpGroup"], maintain_order=True)
    # .agg(pl.col("centroid.z", "centroid.y", "centroid.x"))
).filter(pl.col("roi").is_in(list(imgs)), pl.col("cycle_corr") < 11)
grid
# %%
# order = df_with_meta.sort(sort_by).unique("roi", maintain_order=True)["roi"]

order = grid["roi"]
imgs_sorted = [imgs[o] for o in order]


imgs_trans = translate_images(
    imgs_sorted, grid[: len(order)].select("x", "y").rows(named=True)
)

# %%
canvas, origins = canvas_from_images(imgs_trans, return_origins=True)

# %%
viewer = napari.Viewer()
viewer.add_points(
    **napari_centroids(
        df_visualize.filter(pl.col("cycle_corr") < 11),
        face_colormap=cmaps.ccp.to_mpl(),
        centroid_column="centroid",
        features=cs.by_name(pre_feature),
        size=14,
        # translate_group="roi",
        translate_group=None,
        translate_sort=None,
        translate_n_rows=n_cols,
    ),
    face_contrast_limits=(0, 2 * np.pi),
)
for grad, name in zip(grads, pre_features):
    viewer.add_vectors(
        **napari_gradients(
            df_visualize.filter(pl.col("cycle_corr") < 11).with_columns(
                pl.DataFrame(grad, orient="row", schema=["grad.z", "grad.y", "grad.x"])
            ),
            name=name,
            length=30,
            edge_width=4,
            translate_group=None,
            # translate_group="roi",
            # translate_sort=sort_by,
            # translate_n_rows=n_cols,
        )
    )

imshow(canvas, viewer=viewer, colormap=["red", "blue", "green"])

# %%
sort_by = ["cycle", "Ccp__Mean"]
sort_by = "order"
n_cols = 8

viewer = napari.Viewer()
viewer.add_points(
    **napari_centroids(
        df_with_meta.sort(sort_by)
        .with_columns(
            pl.DataFrame(grad, orient="row", schema=["grad.z", "grad.y", "grad.x"])
        )
        .with_columns(
            pl.sum_horizontal("grad.z", "grad.y", "grad.x").alias("Ccp__Div")
        ),
        centroid_column="centroid",
        features=cs.by_name("NormalizedCCP"),
        size=15,
        translate_group="roi",
        # translate_sort=sort_by,
        translate_n_rows=n_cols,
        # face_contrast_limits=(-0.015, 0.015),
        face_contrast_limits=(0, 2 * np.pi),
        edge_width=0.01,
        face_colormap=cmaps.ccp.to_mpl(),
    ),
)

viewer.add_vectors(
    **napari_gradients(
        df_with_meta.sort(sort_by).with_columns(
            pl.DataFrame(grad, orient="row", schema=["grad.z", "grad.y", "grad.x"])
        ),
        length=30,
        edge_width=4,
        translate_group="roi",
        # translate_sort=sort_by,
        translate_n_rows=n_cols,
    )
)
imshow(canvas, viewer)
