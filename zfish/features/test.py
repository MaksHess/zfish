# %%
from zfish.features._base import unnest_all_structs
from zfish.features.correlation import get_colocalization_features
from zfish.features.distance import get_distance_features
from zfish.features.intensity import get_intensity_features
from zfish.features.label import get_label_features
from zfish.features.polars_utils import set_index_dtypes
from zfish.image.image import load_roi
from zfish.io.datasource import hierarchical_labels, image_si
from zfish.visualize.imshow import imshow

# %%
fn = r"M:\marvwy\20220721_ZE4i2_aligned\imgs\B02_px+1825_py+1851.h5"
imgs_raw = load_roi(fn, {"img_type": "intensity", "level": 1})
lbls_raw = load_roi(fn, {"img_type": "label", "level": 1})

# %%
# shape = (288, 288)
# scale = (1.2, 0.8)
shape = (144, 288, 288)
scale = (0.6, 0.8, 0.7)

# imgs = image_si((5, 40, 400, 400), dims=("c", "z", "y", "x"))
# imgs = image_si((3, 400, 400), scale=(1.2, 1.2), dims=("c", "y", "x"))
imgs_raw = image_si((3, *shape), scale=scale, dims=("c", "z", "y", "x"))
# lbls = label_si((3, 40, 400, 400), dims=("c", "z", "y", "x"))
# lbls = hierarchical_labels((40, 400, 400))
# lbls = hierarchical_labels((400, 400), scale=(1.2, 1.2))
lbls_raw = hierarchical_labels(shape, scale=scale, objects=("emb", "cell"))

# imgs = imgs.isel(x=slice(0, 200), y=slice(0, 200)).copy()
# lbls = lbls.isel(x=slice(0, 200), y=slice(0, 200)).copy()

# imgs = imgs.isel(x=slice(0, 200), y=slice(0, 200), z=slice(0, 50)).copy()
# lbls = lbls.isel(x=slice(0, 200), y=slice(0, 200), z=slice(0, 50)).copy()


# %%
from itertools import chain, combinations

objects = ("embryoRaw", "nucleiRaw3", "cells")
channels = ("DAPI-1", "Pol-II-S2P-0", "PCNA-0", "Pol-II-S5P-2")
corr_channels = (("DAPI-0", "DAPI-1"), ("DAPI-1", "DAPI-2"))
distance_objects = (("embryoRaw", 1),)


# objects = tuple(lbls_raw.c.to_numpy())
# channels = tuple(imgs_raw.c.to_numpy())
# corr_channels = tuple(pair for pair in combinations(channels, 2) if "ch1" in pair)
# distance_objects = (("emb", 1),)

all_objects = objects + tuple(e[0] for e in distance_objects if e[0] not in objects)
all_channels = channels + tuple(
    e for e in chain.from_iterable(corr_channels) if e not in channels
)
lbls = lbls_raw.sel(c=list(all_objects)).compute()
imgs = imgs_raw.sel(c=list(all_channels)).compute()
# %%

from zfish.features.correlation import COLOC_FEATURES

coloc_metrics = {k: COLOC_FEATURES[k] for k in ["PearsonR", "SpearmanR", "KendallTau"]}
lbl = lbls.sel(c="cell")
channel1 = imgs.sel(c="ch0")
channel2 = imgs.sel(c="ch1")
get_colocalization_features(lbl, channel1, channel2, coloc_metrics)
# %%


from zfish.features.correlation import COLOC_FEATURES

coloc_metrics = {k: COLOC_FEATURES[k] for k in ["PearsonR", "SpearmanR", "KendallTau"]}

features = {}
for obj in objects:
    features[obj] = {"label": [], "intensity": [], "correlation": [], "distance": []}

label_features = []
intensity_features = []
coloc_features = []
distance_features = []
for obj in objects:
    print(f"Feature extraction {obj}")
    print("-" * 50)
    print(f"Extracting {obj}", end=" ")
    lbl = lbls.sel(c=obj)
    print("label features...")
    features[obj]["label"].append(get_label_features(lbl))
    # label_features.append(get_label_features(lbl))
    for ch in channels:
        print(f"Extracting {obj}/{ch}", end=" ")
        channel = imgs.sel(c=ch)
        print("intensity features...")
        # intensity_features.append(get_intensity_features(lbl, channel))
        features[obj]["intensity"].append(get_intensity_features(lbl, channel))
    for ch1, ch2 in corr_channels:
        print(f"Extracting {obj}/{ch1}-{ch2}", end=" ")
        channel1 = imgs.sel(c=ch1)
        channel2 = imgs.sel(c=ch2)
        print("correlation features...")
        # coloc_features.append(
        #     get_colocalization_features(lbl, channel1, channel2, coloc_metrics)
        # )
        features[obj]["correlation"].append(
            get_colocalization_features(lbl, channel1, channel2, coloc_metrics)
        )

    for obj_to, lbl_id in distance_objects:
        print(f"Extracting {obj}/{obj_to}-{lbl_id}", end=" ")
        lbl_to = lbls.sel(c=obj_to)
        print("distance features...")
        # distance_features.append(get_distance_features(lbl, lbl_to, lbl_id))
        features[obj]["distance"].append(get_distance_features(lbl, lbl_to, lbl_id))
    print()

# %%
import polars as pl


def concatenate_features(
    dfs: Sequence[pl.DataFrame], index_col="index", how="horizontal"
):
    index = dfs[0].select(pl.col(index_col))
    return pl.concat([index, *[e.drop(index_col) for e in dfs]], how=how)


feature_types = ["label", "intensity", "correlation", "distance"]
features_concat = {}
for obj in objects:
    features_concat[obj] = {}

for obj in objects:
    for feature_type in feature_types:
        features_concat[obj][feature_type] = concatenate_features(
            features[obj][feature_type]
        )


# %%
from itertools import product

import numpy as np
import pandas as pd

from zfish.features.correlation import COLOC_FEATURES
from zfish.features.distance import DISTANCE_ITK_FEATURES, DISTANCE_TRANSFORMS
from zfish.features.intensity import INTENSITY_FEATURES
from zfish.features.label import LABEL_FEATURES

COLOC_FEATURES = set(COLOC_FEATURES.keys())
DISTANCE_FEATURES = {
    "".join(e) for e in product(DISTANCE_ITK_FEATURES, DISTANCE_TRANSFORMS)
}


def get_labelfeatures(n_objects=100, object_type="nuc", seed=42):
    rng = np.random.default_rng(seed=seed)
    data = rng.random(size=(n_objects, len(LABEL_FEATURES)))
    df = pd.DataFrame(data=data, columns=list(LABEL_FEATURES))
    df["feature_type"] = "label"
    df["label"] = range(1, n_objects + 1)
    df["object"] = object_type
    df["channel"] = [[]] * n_objects
    df["object_to"] = [[]] * n_objects
    return df


def get_intensityfeatures(n_objects=100, object_type="nuc", channel="ch0", seed=42):
    rng = np.random.default_rng(seed=seed)
    data = rng.random(size=(n_objects, len(INTENSITY_FEATURES)))
    df = pd.DataFrame(data=data, columns=list(INTENSITY_FEATURES))
    df["feature_type"] = "intensity"
    df["label"] = range(1, n_objects + 1)
    df["object"] = object_type
    df["channel"] = [[channel]] * n_objects
    df["object_to"] = [[]] * n_objects
    return df


def get_correlationfeatures(
    n_objects=100, object_type="nuc", channel1="ch0", channel2="ch1", seed=42
):
    rng = np.random.default_rng(seed=seed)
    data = rng.random(size=(n_objects, len(COLOC_FEATURES)))
    df = pd.DataFrame(data=data, columns=list(COLOC_FEATURES))
    df["feature_type"] = "correlation"
    df["label"] = range(1, n_objects + 1)
    df["object"] = object_type
    df["channel"] = [list(sorted([channel1, channel2]))] * n_objects
    df["object_to"] = [[]] * n_objects
    return df


def get_distancefeatures(
    n_objects=100, object_type="nuc", object_type_to="emb", object_lbl_to=1, seed=42
):
    rng = np.random.default_rng(seed=seed)
    data = rng.random(size=(n_objects, len(COLOC_FEATURES)))
    df = pd.DataFrame(data=data, columns=list(COLOC_FEATURES))
    df["feature_type"] = "distance"
    df["label"] = range(1, n_objects + 1)
    df["object"] = object_type
    df["channel"] = [[]] * n_objects
    df["object_to"] = [[object_type_to, str(object_lbl_to)]] * n_objects
    return df


fl = get_labelfeatures()
fi = get_intensityfeatures()
fd = get_distancefeatures()
fc = get_correlationfeatures()
# %%
from zfish.features.object_hierarchy import get_full_object_hierarchy

HIERARCHY = {
    "embryoRaw": (),
    "cells": ("embryoRaw",),
    "nucleiRaw3": (
        "embryoRaw",
        "cells",
    ),
}
hierarchy = get_full_object_hierarchy(lbls, raw_hierarchy=HIERARCHY)
# %%
get_label_features(lbl)
# %%

import numpy as np
import polars as pl

obbx_origin = (
    label_features[1]
    .select(pl.col("^.*BoundingBox.*$"))[0]["OrientedBoundingBoxOrigin"]
    .item()
)
obbx_dir = (
    label_features[1]
    .select(pl.col("^.*BoundingBox.*$"))[0]["OrientedBoundingBoxDirection"]
    .item()
)
obbx_size = (
    label_features[1]
    .select(pl.col("^.*BoundingBox.*$"))[0]["OrientedBoundingBoxSize"]
    .item()
)
obbx_vertices = (
    label_features[1]
    .select(pl.col("^.*BoundingBox.*$"))[0]["OrientedBoundingBoxVertices"]
    .item()
)
# obbx_vertices = np.array(
#     [
#         tuple(obbx_vertices_raw.GetElement(i))[::-1]
#         for i in range(obbx_vertices_raw.Size())
#     ]
# )
bbx = label_features[1].select(pl.col("^.*BoundingBox.*$"))[0]["BoundingBox"].item()
# n1, n2 = (np.array(tuple(obbx_size.values())[::-1])).round().astype(int)

lbl = lbls.sel(c="cell")
# scale = min(lbl.meta.scale)
scale = 2.4

ALL_SPATIAL_DIMS = ("z", "y", "x")
ALL_LOCAL_DIMS = ("c", "b", "a")

origin = np.array(tuple(obbx_origin[e] for e in ALL_SPATIAL_DIMS if e in obbx_origin))
ndim = len(origin)
spatial_dims = ALL_SPATIAL_DIMS[-ndim:]
local_dims = ALL_LOCAL_DIMS[-ndim:]


# %%
from itertools import product

size = np.array(tuple(obbx_size[e] for e in local_dims if e in obbx_size))
ns = np.ceil(size / scale).astype(int)
axes = {}
diffs = {}
coords = {}

for local_dim, n in zip(local_dims, ns):
    direction = np.array(
        [[obbx_dir[f"{local_dim}-{spatial_dim}"] for spatial_dim in spatial_dims]]
    )
    _range = np.arange(n).reshape(-1, 1)
    axes[local_dim] = _range * direction * scale + origin

for k, ax in axes.items():
    diff = np.cumsum(np.diff(ax, axis=0), axis=0)
    diff = np.append([[0] * ndim], diff, axis=0)
    coord = np.linalg.norm(diff, axis=1)
    diffs[k] = diff
    coords[k] = coord
# %%
viewer = imshow(lbl)
for c, (k, ax) in zip(["red", "green", "blue"], axes.items()):
    viewer.add_points(ax, face_color=c, size=2)
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
ax.set_aspect("equal")
lbl.plot()
for k, ax in axes.items():
    plt.scatter(ax[:, 1], ax[:, 0], s=0.05)
# %%


for direction_name in spatial_dims:
    for component in ("z", "y", "x"):
        k = f"{direction_name}-{component}"
        if k in obbx_dir:
            direction.append(obbx_dir[k])
    if not direction:
        continue
    print(np.array(direction))

# ax1 = np.linspace(
#     *np.array(
#         [
#             [obbx_vertices["a-y"], obbx_vertices["a-x"]],
#             [obbx_vertices["b-y"], obbx_vertices["b-x"]],
#         ]
#     ),
#     n1,
# )

# ax2 = np.linspace(
#     *np.array(
#         [
#             [obbx_vertices["a-y"], obbx_vertices["a-x"]],
#             [obbx_vertices["c-y"], obbx_vertices["c-x"]],
#         ]
#     ),
#     n2,
# )

import xarray as xr

# ax1 = np.linspace(*obbx_vertices[:2], n1)
ax1_diffs = np.cumsum(np.diff(ax1, axis=0), axis=0)
ax1_diffs = np.append([[0, 0]], ax1_diffs, axis=0)
ax1_coords = np.linalg.norm(ax1_diffs, axis=1)

# ax2 = np.linspace(*obbx_vertices[[0, 2]], n2)
ax2_diffs = np.cumsum(np.diff(ax2, axis=0), axis=0)
ax2_diffs = np.append([[0, 0]], ax2_diffs, axis=0)
ax2_coords = np.linalg.norm(ax2_diffs, axis=1)

# grid = np.expand_dims(ax1, axis=-1) + np.expand_dims(ax2_diffs.T, axis=0)
# points = np.swapaxes(grid, 1, 2).reshape(n1 * n2, 2)
# y = grid[:, 0, :].T
# x = grid[:, 1, :].T
# # res = [ax1 + e for e in ax2_diffs]
# y = np.stack(res)[..., 0]
# x = np.stack(res)[..., 1]
y = xr.DataArray(y, dims=("u", "v"), coords={"u": ax2_coords, "v": ax1_coords})
x = xr.DataArray(x, dims=("u", "v"), coords={"u": ax2_coords, "v": ax1_coords})
# points2 = np.concatenate(res, axis=0)

# dir1 = np.diff(ax1[:2], axis=0)
# dir1 = np.diff(ax1[:2], axis=0)
# mg = np.meshgrid(ax1, ax2)
# p1 = np.stack([mg[0].flatten(), mg[1].flatten()], axis=1)

lbli = lbl.interp(x=x, y=y, method="nearest", kwargs=dict(fill_value=0)).astype(
    lbl.dtype
)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1)
plt.sca(ax[0])
ax[0].set_aspect("equal")
lbl.plot()
plt.sca(ax[1])
ax[1].set_aspect("equal")
lbli.plot()
# out3 = lbl.interp({'y': ax1, 'x': ax2})

# %%

# %%
fig, ax = plt.subplots()
lbl.plot(yincrease=False)
plt.scatter(ax1[:, 1], ax1[:, 0])
plt.scatter(ax2[:, 1], ax2[:, 0])
# obbx_vertices = label_features[1].select(pl.col("^Oriented.*Vertices$"))[0].item()
# %%
x = xr.DataArray(ax1[:, 1], dims=("z",))
y = xr.DataArray(ax1[:, 0], dims=("z",))
# %%
y
# %%
viewer = imshow(lbl)
# viewer.add_points(obbx_vertices[:2])
# viewer.add_points(obbx_vertices[[0, 2]])
# viewer.add_points(ax1)
# viewer.add_points(ax2, face_color="green")
viewer.add_points(points, face_color="red", size=0.2)
viewer = imshow(lbli, viewer)
# viewer.add_points(points2, face_color="green", size=3)
# viewer = imshow(out, viewer)
# viewer = imshow(out2, viewer)
# viewer = imshow(out3, viewer)
# viewer = imshow(out4, viewer)


# viewer = imshow(out3, viewer)
# %%
u = np.expand_dims(np.arange(len(ax1)), axis=-1) * np.expand_dims(
    np.ones(len(ax2)), axis=0
)
v = np.expand_dims(np.ones(len(ax1)), axis=-1) * np.expand_dims(
    np.arange(len(ax2)), axis=0
)

u = xr.DataArray(u, dims=["y", "x"], coords={"y": ax1[:, 0], "x": ax2[:, 0]})
v = xr.DataArray(v, dims=["y", "x"], coords={"y": ax1[:, 1], "x": ax2[:, 1]})
import matplotlib.pyplot as plt

# %%
import numpy as np
import pandas as pd
import xarray as xr

# %%
ds = xr.tutorial.open_dataset("air_temperature").isel(time=0)

fig, axes = plt.subplots(ncols=2, figsize=(10, 4))

ds.air.plot(ax=axes[0])

axes[0].set_title("Raw data")

new_lon = np.linspace(ds.lon[0], ds.lon[-1], ds.dims["lon"] * 4)

new_lat = np.linspace(ds.lat[0], ds.lat[-1], ds.dims["lat"] * 4)

dsi = ds.interp(lat=new_lat, lon=new_lon)

dsi.air.plot(ax=axes[1])

axes[1].set_title("Interpolated data")


x = np.linspace(240, 300, 100)

z = np.linspace(20, 70, 100)

lat = xr.DataArray(z, dims=["z"], coords={"z": z})

lon = xr.DataArray(
    (x[:, np.newaxis] - 270) / np.cos(z * np.pi / 180) + 270,
    dims=["x", "z"],
    coords={"x": x, "z": z},
)

fig, axes = plt.subplots(ncols=2, figsize=(10, 4))

ds.air.plot(ax=axes[0])

for idx in [0, 33, 66, 99]:
    axes[0].plot(lon.isel(x=idx), lat, "--k")

for idx in [0, 33, 66, 99]:
    axes[0].plot(*xr.broadcast(lon.isel(z=idx), lat.isel(z=idx)), "--k")

axes[0].set_title("Raw data")

# dsi = ds.interp(lon=lon, lat=lat.expand_dims('x'))
dsi = ds.interp(lon=lon).interp(lat=lat)

dsi.air.plot(ax=axes[1])

axes[1].set_title("Remapped data")

# %%

# %%
import polars as pl

from zfish.features.object_hierarchy import get_full_object_hierarchy
from zfish.features.typing import LabelImage, MultichannelLabelImage

HIERARCHY = {
    "emb": (),
    "cell": ("emb",),
    "nuc": ("cell", "emb"),
    "mem": ("cell", "emb"),
    "cyto": ("cell", "emb"),
    "loc": ("nuc", "cyto", "cell", "emb"),
}


res = get_full_object_hierarchy(lbls)
res

# lbls.sel(c=())

# def overlapping_labels(lbl, lbl2):
#     label, count = np.unique(lbl2[np.where(lbl)], return_counts=True)
#     order = np.array(list(reversed(np.argsort(count))))
#     ordered_label = label[order]
#     percentage = count[order] / np.sum(lbl)
#     return {k: v for k, v in zip(ordered_label, percentage)}


# def parent_label(lbl: np.array, lbl2: np.array):
#     labels, counts = np.unique(lbl2[np.where(lbl)], return_counts=True)
#     parent_label = labels[np.argsort(counts)[-1]]
#     return int(parent_label)


# def get_parent(lbl: LabelImage, lbl2: LabelImage) -> pl.DataFrame:
#     obj = lbl.c.item()
#     other_obj = lbl2.c.item()
#     props = regionprops(lbl, lbl2, extra_properties=(parent_label,))
#     return {
#         obj: [prop.label for prop in props],
#         other_obj: [prop.parent_label for prop in props],
#     }


# pl.DataFrame(
#     regionprops_table(
#         lbl3, lbl2, properties=("label",), extra_properties=(parent_label, overlapping_labels,)
#     )
# )

# %%
import polars as pl

c_features = []
for features in all_features:
    c_features.append(
        pl.concat(
            [features[0].select("index")] + [e.drop("index") for e in features],
            how="horizontal",
        )
    )
# %%
out = pl.concat(c_features)
# %%
unnest_all_structs(out).select(
    [
        pl.col(("index-object", "index-Label")),
        pl.exclude(("index-object", "index-Label")),
    ]
)

# %%
lbl = lbls.isel(c=0)
img1 = imgs.isel(c=0)
img2 = imgs.isel(c=2)

lbl_2d = lbl.isel(z=100)
img1_2d = img1.isel(z=100)
img2_2d = img2.isel(z=100)
# %%
df = get_label_features(lbl)
df
# %%
lbls.z
# %%
lbl.compute()
# %%
from zfish.visualize.imshow import imshow, lbshow

# %%
lbshow(lbls.sel(c=["/lbl_cells", "/lbl_nuc_raw3"]).compute())
# %%
from itertools import combinations

from zfish.features.correlation import ColocMetrics

all_features = []
coloc_metrics = {k: ColocMetrics[k] for k in ["PearsonR", "SpearmanR", "KendallTau"]}

for lbl in lbls:
    features = []
    features.append(get_label_features(lbl))
    for img in imgs:
        features.append(get_intensity_features(lbl, img))
    for img1, img2 in combinations(imgs, 2):
        features.append(
            get_colocalization_features(
                lbl, img1, img2, correlation_metrics=coloc_metrics
            )
        )
    all_features.append(features)
# %%
from typing import Sequence

import polars as pl


def join(
    dfs: Sequence[pl.DataFrame],
    left_on=None,
    right_on=None,
    on=None,
    how="inner",
    suffix="_right",
) -> pl.DataFrame:
    df = dfs[0]
    for df_other in dfs[1:]:
        df = df.join(
            df_other, left_on=left_on, right_on=right_on, on=on, how=how, suffix=suffix
        )
    return df


# %%
from base import unnest_all_structs

# %%
d1 = all_features[0][0].select(
    [
        pl.col("index").apply(lambda x: f"{x['object']}-{x['Label']}"),
        pl.exclude("index"),
    ]
)
d2 = all_features[0][1].select(
    [
        pl.col("index").apply(lambda x: f"{x['object']}-{x['Label']}"),
        pl.exclude("index"),
    ]
)
# %%
unnest_all_structs(
    all_features[0][1]
)  # .to_pandas().set_index(['index-object', 'index-Label']).to_parquet('test.parquet')
# %%
import pandas as pd

pd.read_parquet("test.parquet")
# %%
d1.join(d2, on="index")
# %%
import matplotlib.pyplot as plt

plt.imshow(lbl)
plt.scatter(df["Centroid"].struct.field("x"), df["Centroid"].struct.field("y"))

# %%
img.name
