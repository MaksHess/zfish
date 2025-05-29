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
