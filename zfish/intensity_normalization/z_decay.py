# %%
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from zfish.features.polars_utils import (
    get_metadata,
    set_index_dtypes,
    stack_correlation_metric_by_acquisition,
)
from zfish.intensity_normalization.models import (
    ExponentialModel,
    ExponentialModelFitLinear,
    ExponentialModelNoOffset,
    LinearModel,
    Model,
)

pl.toggle_string_cache(True)

# %%
df_intensity = set_index_dtypes(
    pl.read_parquet(
        r"C:\Users\hessm\Documents\Programming\Python\zfish\data\features\intensity.parquet"
    )
)

df_label = set_index_dtypes(
    pl.read_parquet(
        r"C:\Users\hessm\Documents\Programming\Python\zfish\data\features\label.parquet"
    )
)
df_corr = set_index_dtypes(
    pl.read_parquet(
        r"C:\Users\hessm\Documents\Programming\Python\zfish\data\features\coloc.parquet"
    )
)

df_distance = set_index_dtypes(
    pl.read_parquet(
        r"C:\Users\hessm\Documents\Programming\Python\zfish\data\features\distance.parquet"
    )
)

df_alignment = stack_correlation_metric_by_acquisition(df_corr)


# %%
SPHERICAL_RADIUS_CUTOFF = (4, 8)
ax = sns.kdeplot(df_label.to_pandas(), x="EquivalentSphericalRadius")
y_lims = ax.get_ylim()
plt.vlines(
    x=SPHERICAL_RADIUS_CUTOFF,
    ymin=y_lims[0],
    ymax=y_lims[1],
    colors=["k"],
    linestyles=["dotted", "dashed"],
)

df_label_clean1 = df_label.filter(
    pl.col("EquivalentSphericalRadius").is_between(*SPHERICAL_RADIUS_CUTOFF)
)

# %% Remove non-interphase cells by roundness
ROUNDNESS_CUTOFF = 0.992
ax = sns.kdeplot(df_label.to_pandas(), x="Roundness")
y_lims = ax.get_ylim()
plt.vlines(
    x=ROUNDNESS_CUTOFF,
    ymin=y_lims[0],
    ymax=y_lims[1],
    colors=["k"],
    linestyles=["dotted", "dashed"],
)

df_label_clean = df_label_clean1.filter(pl.col("Roundness") > ROUNDNESS_CUTOFF)


# %% Remove early stage embryos (< cycle 10)
control_wells = ["B07", "C07", "D07", "E07"]
df_meta = get_metadata(df_label_clean, control_wells=control_wells)

df_meta_clean = df_meta

# %% Merge relevant columns of dataframes
index = ["roi", "structure", "label", "channel", "stain", "acquisition"]
obj_index = ["roi", "structure", "label"]


df = (
    df_intensity.with_columns(
        [
            pl.col("channel")
            .cast(pl.Utf8)
            .str.split(".")
            .arr.get(0)
            .cast(pl.Categorical)
            .alias("stain"),
            pl.col("channel")
            .cast(pl.Utf8)
            .str.split(".")
            .arr.get(1)
            .cast(pl.UInt16)
            .alias("acquisition"),
        ]
    )
    .select([pl.col(index), pl.col("Mean"), pl.col("Mean").log(10).alias("log10_Mean")])
    .join(
        df_label_clean.select(
            obj_index
            + ["Centroid-z", "PhysicalSize", "EquivalentSphericalRadius", "Roundness"]
        ),
        on=obj_index,
    )
    .join(df_alignment, on=["roi", "structure", "label", "acquisition"])
    .join(
        df_distance.select(
            obj_index
            + ["embryoRaw.1_CentroidDistToBorder", "embryoRaw.1_CentroidDistAlongZ"]
        ),
        on=obj_index,
    )
    .join(df_meta_clean, on=["roi"])
    .with_columns(
        [
            (pl.col("Centroid-z") - pl.col("embryoRaw.1_CentroidDistAlongZ")).alias(
                "mediumPath"
            ),
            pl.col("embryoRaw.1_CentroidDistAlongZ").alias("embryoPath"),
        ]
    )
)


# %% Remove individual misaligned cells
ALIGNMENT_SCORE_CUTOFF = 0.85
ax = sns.kdeplot(
    df_alignment.filter(pl.col("acquisition") != 1)
    .select(
        [
            pl.col("acquisition").cast(pl.Utf8).cast(pl.Categorical),
            pl.col("alignmentScore"),
        ]
    )
    .to_pandas(),
    hue="acquisition",
    x="alignmentScore",
)
y_lims = ax.get_ylim()
plt.vlines(
    x=ALIGNMENT_SCORE_CUTOFF,
    ymin=y_lims[0],
    ymax=y_lims[1],
    colors=["k"],
    linestyles=["dotted"],
)

df_clean = df.filter(pl.col("alignmentScore") > ALIGNMENT_SCORE_CUTOFF).filter(
    pl.col("is_control_well") != True
)

# %%
import pandas as pd


def stratified_sample(df, attr, n="max", s=5):
    strat = pd.cut(
        df[attr],
        bins=np.linspace(df[attr].min() - 1, df[attr].max() + 1, s + 1),
        labels=np.arange(s),
    )
    if n == "max":
        n = strat.value_counts().min() * s
    sample = df.groupby(strat, group_keys=False).apply(lambda x: x.sample(n // s))
    return sample


# %%
import numpy as np

to_plot = df_clean.to_pandas()

mdl_colors = ["#dd966d", "#6ddd96", "#966ddd"]
scatter_colors = ["#8e8e8e"]

loss = "huber"
mdl_types = [
    # LinearModel,
    ExponentialModelFitLinear,
    # ExponentialModelNoOffset,
    # ExponentialModel,
]

channels = [
    "DAPI.0",
    "FLAG.0",
    "PCNA.0",
    "Pol-II-S2P.0",
    "DAPI.1",
    "H3K27Ac.1",
    "bCatenin.1",
    "pH3.1",
    "DAPI.2",
    "ALYREF.2",
    "H2B.2",
    "Pol-II-S5P.2",
    "DAPI.3",
    "Nanog.3",
    "XRN2.3",
    "YAP.3",
]

n_cols = 4
n_rows = len(channels) // n_cols

fig, axs = plt.subplots(
    n_rows, n_cols, sharex=True, sharey=False, squeeze=True, figsize=(20, 20), dpi=200
)

models = dict()

for i, (channel, ax) in enumerate(zip(channels, axs.flatten())):
    print(channel)
    models[channel] = dict()
    df_channel = to_plot[to_plot.channel == channel]
    df_sample = df_channel.sample(10000)  # stratified_sample(df_channel, "Centroid-z")
    X = df_sample[["Centroid-z"]]
    y = df_sample["Mean"]
    X_fit = np.linspace(
        df_channel["Centroid-z"].min(), df_channel["Centroid-z"].max(), 200
    ).reshape(-1, 1)

    plt.sca(ax)
    if i % n_cols == 0:
        plt.ylabel("Mean Intensity [AU]")
    if i // n_cols == (n_rows - 1):
        plt.xlabel("Centroid-z [um]")
    plt.title(f"{channel}")
    # plt.scatter(X, y, color=scatter_colors[0], s=1, alpha=0.3, label=f"{channel} Mean")
    sns.scatterplot(
        x=X.values.flatten(),
        y=y,
        color=scatter_colors[0],
        # hue=df_sample["Roundness"],
        s=10,
        alpha=0.3,
        # label=f"{channel} Mean",
    )
    y_top = np.quantile(y, 0.999) * 1.5

    # plt.ylim(bottom=0, top=y_top)
    plt.yscale("log")
    plt.ylim(bottom=1, top=10**3.5)

    for mdl_type, mdl_color in zip(mdl_types, mdl_colors):
        mdl = mdl_type(loss=loss)
        name = mdl_type.__name__
        try:
            mdl.fit(X, y)
        except RuntimeError as e:
            print(e)
            print(f"no convergance for `{name}` in channel `{channel}`")
            models[channel][name] = None
            continue
        models[channel][name] = mdl

        y_mdl = mdl.predict(X_fit)
        plt.plot(X_fit, y_mdl, color=mdl_color, label=name, linewidth=2)

    # plt.legend()


# %%
import copy
from pathlib import Path
from typing import Sequence

import h5py

from zfish.image.image import SpatialImage, to_si
from zfish.io import h5
from zfish.visualize.imshow import imshow

to_plot = df_clean.to_pandas()


def load_channel(
    fn: Path,
    channel: str,
    level: int = 1,
) -> tuple[SpatialImage, SpatialImage]:
    with h5py.File(fn) as f:
        stain, acquisition = channel.split(".")
        channel_si = to_si(
            h5.select(f, {"stain": stain, "cycle": int(acquisition), "level": level})[0]
        ).compute()
        embryo_si = to_si(h5.select(f, {"stain": "embryoRaw"})[0]).compute()
    return channel_si, embryo_si


def fit_model_to_channel(
    df: pd.DataFrame,
    channel: str,
    model: Model,
    X_columns: Sequence[str] = ("mediumPath", "embryoPath"),
    y_column: str = "Mean",
    n_samples: int | None = None,
    in_place: bool = False,
) -> Model:
    for column in list(X_columns) + [y_column]:
        assert column in df, f"column `{column}` not in `df`"

    if in_place:
        model_copy = model
    else:
        model_copy = copy.deepcopy(model)

    df_channel = df[df.channel == channel]
    if n_samples:
        df_sample = df_channel.sample(n_samples)
    else:
        df_sample = df_channel
    X = df_sample[list(X_columns)]
    y = df_sample[y_column]
    model_copy.fit(X, y)
    return model_copy


def apply_model_to_channel(
    model: Model,
    channel_si: SpatialImage,
    label_si: SpatialImage | None = None,
) -> SpatialImage:
    if model._feature_names == ["Centroid-z"]:
        return (
            channel_si
            * model._correction_factor(channel_si.coords["z"].expand_dims(["_"], -1))
        ).rename("image")

    if model._feature_names == ["mediumPath", "embryoPath"]:
        assert (
            label_si is not None
        ), "Provide embryo segmentation for 2-step correction."
        embryo_path_si = (label_si * label_si.meta.scale[0]).cumsum(dim="z")
        medium_path_si = (~label_si * label_si.meta.scale[0]).cumsum(dim="z")
        correction_factors = model._correction_factor(
            np.concatenate([medium_path_si, embryo_path_si]).reshape(2, -1).T
        ).T.reshape(channel_si.shape)
        return channel_si * correction_factors


channel = "DAPI.1"
X_columns_2step = ["mediumPath", "embryoPath"]
# X_columns = ["Centroid-z"]
model = ExponentialModelFitLinear(loss="huber")
y_column = "Mean"
n_samples = None

mdl_2step = fit_model_to_channel(
    to_plot,
    channel,
    model,
    X_columns=X_columns_2step,
    y_column=y_column,
    n_samples=n_samples,
    in_place=False,
)

X_columns_1step = ["Centroid-z"]

mdl_1step = fit_model_to_channel(
    to_plot,
    channel,
    model,
    X_columns=X_columns_1step,
    y_column=y_column,
    n_samples=n_samples,
    in_place=False,
)
# sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y_corr, s=1)
# plt.axis("equal")
# plt.show()

# plt.scatter(X, y, color=scatter_colors[0], s=1, alpha=0.1)
# y_999 = np.quantile(y, 0.999)
# plt.ylim(bottom=0, top=2 * y_999)

# %%
fn = r"M:\marvwy\20220721_ZE4i2_aligned\imgs\B02_px-2133_py+0159.h5"
img, lbl = load_channel(fn, channel)
img_corr_1step = apply_model_to_channel(mdl_1step, img)
img_corr_2step = apply_model_to_channel(mdl_2step, img, lbl)
# %%
import napari
viewer = napari.Viewer()
imshow(img, viewer)
imshow(img_corr_1step, viewer)
imshow(img_corr_2step, viewer)
# %%
y_corr = y * mdl._correction_factor(X)


sns.histplot(y.apply(np.log2))
sns.histplot(y_corr.apply(np.log2))
print(y.std() / y.mean())
print(y_corr.std() / y_corr.mean())
print(y.apply(np.log2).std() / y.apply(np.log2).mean())
print(y_corr.apply(np.log2).std() / y_corr.apply(np.log2).mean())
# %%
sns.scatterplot(x=df_sample["Centroid-z"], y=y.apply(np.log2), s=1, alpha=0.1)
sns.scatterplot(x=df_sample["Centroid-z"], y=y_corr.apply(np.log2), s=1, alpha=0.1)
# %%
n_grid = 10
X_grid = (
    np.stack(np.meshgrid(*np.linspace(X.min(), X.max(), n_grid).T)).reshape(2, -1).T
)
y_grid = mdl.predict(X_grid)
y_grid2 = mdl2.predict(X_grid)
y_grid3 = mdl3.predict(X_grid)
# mdl.predict(np.stack(np.meshgrid(*np.linspace(X.min(), X.max(), 10).T)).reshape(2, -1).T).reshape((10, 10))
# %%
sample_idxs = np.random.randint(len(y), size=50000)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))

ax.plot_surface(
    X_grid[:, 0].reshape(n_grid, n_grid),
    X_grid[:, 1].reshape(n_grid, n_grid),
    y_grid.reshape(n_grid, n_grid),
    alpha=0.2,
)
ax.plot_surface(
    X_grid[:, 0].reshape(n_grid, n_grid),
    X_grid[:, 1].reshape(n_grid, n_grid),
    y_grid2.reshape(n_grid, n_grid),
    alpha=0.2,
)
ax.plot_surface(
    X_grid[:, 0].reshape(n_grid, n_grid),
    X_grid[:, 1].reshape(n_grid, n_grid),
    y_grid3.reshape(n_grid, n_grid),
    alpha=0.2,
)
ax.scatter(
    X.iloc[sample_idxs, 0], X.iloc[sample_idxs, 1], y.iloc[sample_idxs], s=1, alpha=0.2
)

# %%
