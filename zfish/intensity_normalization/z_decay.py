# %% Imports and definitions
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
import seaborn as sns

from zfish.features.polars_preprocessing import get_metadata
from zfish.features.polars_selector import sel
from zfish.features.polars_utils import (
    drop_null_columns,
    pipe_df_nulls,
    read_table,
    split_channel_column,
    split_channel_pair_column,
    stack_column_name_to_column,
    unnest_all_structs,
    unnest_structs,
    unstack_column_to_column_name,
)
from zfish.intensity_normalization.models import (
    Exp,
    ExpNoOffset,
    Linear,
    LogLinear,
)
from zfish.preprocessing.outlier_ranges import (
    RngOutlier,
    remove_outliers,
)
from zfish.roi.spatial_roi import Roi, RoiMap, load_roi_tables

# polars < 0.19
# pl.enable_string_cache(True)
pl.enable_string_cache()

DEBUG = True
# INDEX = ("roi", "object", "label")
# COL_INDEX = [pl.col(e) for e in INDEX]
CONTROL_WELLS = ["B07", "C07", "D07", "E07"]

# %% Load data of nuclei
fld_features_uncorr = Path(
    r"C:\Users\hessm\Documents\zfish_local\features_v2\NoCorrection"
)
df_nuc = read_table(fld_features_uncorr, _object="nucleiRaw3")


# %% Discard outliers

outlier_ranges = [
    {
        "feature": "EquivalentSphericalRadius",  # um
        "lower": 4,
        "upper": 8,
    },
    {
        "feature": "Roundness",
        "lower": 0.95,
    },
]

outliers = [RngOutlier(**e) for e in outlier_ranges]

# df_nuc_clean = remove_outliers(df_nuc, outliers=outliers)

df_nuc_clean = df_nuc.pipe(remove_outliers, outliers=outliers, plot_results=True)

# %%

df_meta = get_metadata(
    df_nuc_clean, control_wells=CONTROL_WELLS, drop_objects_with_no_parents=True
)
df_meta = df_meta.with_columns(
    [
        pl.col("site")
        .cast(pl.Utf8)
        .str.extract("px([+-]\d+)")
        .cast(float)
        .alias("site_x"),
        pl.col("site")
        .cast(pl.Utf8)
        .str.extract("py([+-]\d+)")
        .cast(float)
        .alias("site_y"),
    ]
).drop(cs.starts_with("parent."))


# %% Select relevant features and compute medium & embryo path length.

features = [
    "Centroid",
    "PhysicalSize",
    "EquivalentSphericalRadius",
    "Roundness",
    "embryoRaw-1_CentroidDistanceToBorder",
    "embryoRaw-1_CentroidDistanceAlongZ",
]
intensity_features = "^.*_Mean$"
corr_features = "^.*_PearsonR$"

embryo_path_name = "EmbryoPath"
medium_path_name = "MediumPath"

path_selector = cs.matches("EmbryoPath|MediumPath")

df = (
    df_nuc_clean.select(
        sel.index,
        *[pl.col(f) for f in features],
        cs.matches(intensity_features),
        cs.matches(corr_features),
    )
    .pipe(unnest_structs, cols=["Centroid"])
    .with_columns(
        [
            (pl.col("Centroid-z") - pl.col("embryoRaw-1_CentroidDistanceAlongZ")).alias(
                medium_path_name
            ),
            pl.col("embryoRaw-1_CentroidDistanceAlongZ").alias(embryo_path_name),
        ]
    )
    # .pipe(pipe_df_nulls)
    # .pipe(stack_column_name_to_column, index=cs.expand_selector(df_nuc_clean, sel.index))
    # .pipe(pipe_df_nulls)
)
df
# %% Convert tables into tall (tidy) format.

df_label = df.select(sel.index, sel.label, path_selector)

df_intensity = (
    df.select(sel.index - sel.hierarchy, sel.intensity)
    .pipe(stack_column_name_to_column, index=sel.index, column_name="channel")
    .pipe(split_channel_column)
)

df_corr = (
    df.select(sel.index - sel.hierarchy, sel.correlation)
    .pipe(stack_column_name_to_column, index=sel.index, column_name="channel_pair")
    .pipe(split_channel_pair_column)
)

df_distance = df.select(sel.index - sel.hierarchy, sel.distance).pipe(
    stack_column_name_to_column, index=sel.index, column_name="object_to"
)
# %% Join all tables
df_merge = (
    df_label.join(df_intensity, on=["roi", "object", "label"], how="left")
    .join(df_corr, on=["roi", "object", "label", "acquisition"], how="left")
    .join(df_distance, on=["roi", "object", "label"], how="left")
    .with_columns(cs.by_name("PearsonR").fill_null(1.0))
    .select(sel.index, path_selector, sel.features)
)

# %% Remove misaligned cells
registration_outlier_ranges = [
    {
        "feature": "PearsonR",
        "lower": 0.9,
    },
]

registration_outliers = [RngOutlier(**e) for e in registration_outlier_ranges]

df_merge_clean = df_merge.pipe(remove_outliers, registration_outliers)

# %%
import re
from collections import defaultdict

import numpy as np
from scipy import stats

from zfish.intensity_normalization.models import apply_model_to_channel, fit_model_to_df


def get_X_grid(X, n_samples: int = 200):
    return np.meshgrid(*np.linspace(X.min(axis=0), X.max(axis=0), n_samples).T)


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


models = [
    Linear(loss="huber"),
    LogLinear(loss="huber"),
    Exp(loss="huber", pos_offset=True),
    Exp(loss="huber", pos_offset=False),
]
model_colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
scatter_color = "#8e8e8e"
scatter_color_kde = False
n_samples_kde = 10000

X_columnss = [["Centroid-z"], ["MediumPath", "EmbryoPath"]]
n_featuress = [len(X_columns) for X_columns in X_columnss]
n_samples_model = None
n_samples_plot = 20000

# %%
if n_samples_kde > n_samples_plot:
    print("`n_samples_kde` must be <= `n_samples_plot`.")
    n_samples_kde = n_samples_plot

n_cols = 4
n_rows = len(channels) // n_cols

figs = [
    plt.figure(
        layout="constrained",
        figsize=(3.75 * n_cols, 2.25 * n_rows * n_features),
        dpi=200,
    )
    for n_features in n_featuress
]
# subfigss = [fig.subfigures(n_rows, n_cols, wspace=0.07, squeeze=False).flatten() for fig in figs]
axss = [
    fig.subplots(
        n_rows,
        n_cols,
        subplot_kw={"projection": "3d"} if n_features == 2 else {},
        squeeze=False,
    ).flatten()
    for fig, n_features in zip(figs, n_featuress)
]

all_models = defaultdict(dict)
for j, (model, model_color) in enumerate(zip(models, model_colors)):
    print(model)
    channel_models = {}
    for i, channel in enumerate(channels):
        print(channel)

        df_channel = (
            df_merge_clean.filter(pl.col("channel") == channel)
            .with_columns(pl.col("Mean").name.prefix(f"{channel}_"))
            .to_pandas()
        )

        y_column = f"{channel}_Mean"

        if n_samples_plot is None:
            df_plot = df_channel
        else:
            df_plot = df_channel.sample(
                n_samples_plot, random_state=42
            )  # stratified_sample(df_channel, "Centroid-z")

        for X_columns, axs in zip(X_columnss, axss):
            model_fit = fit_model_to_df(
                df=df_channel,
                channel=channel,
                model=model,
                X_columns=X_columns,
                y_column=y_column,
                n_samples=n_samples_model,
                in_place=False,
            )
            name = str(model_fit)
            all_models[name][channel] = model_fit
            print(name)

            X = df_plot[X_columns]
            y = df_plot[y_column]
            if scatter_color_kde and j == 0:
                values = np.concatenate(
                    [X.values, np.expand_dims(y.values, axis=1)], axis=1
                )
                sample_idxs = np.random.choice(
                    np.arange(len(values)), n_samples_kde, replace=False
                )
                values_sample = values[sample_idxs, :]
                kernel = stats.gaussian_kde(values_sample.T)
                scatter_color = kernel(values.T)

            if model_fit is not None:
                if len(X_columns) == 2:
                    X_grid_fit = get_X_grid(X)
                    y_grid_pred = model_fit.grid_predict(X_grid_fit)
                    y_top = np.quantile(y_grid_pred, 0.999) * 1.5
                else:
                    X_fit = np.linspace(X.min(), X.max(), 200)
                    y_pred = model_fit.predict(X_fit)
                    y_top = np.quantile(y_pred, 0.999) * 1.5

            if len(X_columns) == 2:
                ax = axs[i]
                plt.sca(ax)
                if j == 0:
                    ax.set_title(f"{channel}")
                    ax.scatter(
                        X.values[:, 0],
                        X.values[:, 1],
                        y,
                        c=scatter_color,
                        # hue=df_sample["Roundness"],
                        s=1,
                        alpha=0.3,
                        # label=f"{channel} Mean",
                    )
                label = "\n  ".join(
                    [e for e in re.split("[\(, \)]", str(model_fit)) if e != ""]
                )
                ax.plot_surface(
                    X_grid_fit[0],
                    X_grid_fit[1],
                    y_grid_pred,
                    color=model_color,
                    alpha=0.5,
                    label=label,
                )
                ax.set_zlim(0, y_top)
                # if model_fit is not None:
                #     ax.

            else:
                ax = axs[i]
                plt.sca(ax)
                if j == 0:
                    ax.set_title(f"{channel}")
                    sns.scatterplot(
                        x=X.values.flatten(),
                        y=y,
                        c=scatter_color,
                        # hue=df_sample["Roundness"],
                        s=1,
                        alpha=0.3,
                        # label=f"{channel} Mean",
                    )
                if model_fit is not None:
                    label = "\n  ".join(
                        [e for e in re.split("[\(, \)]", str(model_fit)) if e != ""]
                    )
                    plt.plot(
                        X_fit.flatten(),
                        y_pred,
                        color=model_color,
                        linewidth=1.5,
                        label=label,
                    )
                plt.ylim(0, y_top)

        print()
    print()
plt.sca(axss[0][-1])
plt.legend()
plt.sca(axss[1][-1])
plt.legend()

more_models = {}
more_models["z_decay"] = all_models
# %%
from zfish.roi.spatial_roi import write_models

fld_models = fld_features_uncorr.parent.parent / "models_v3"
fld_plots = fld_models / "__plots/z_decay"
fld_plots.mkdir(exist_ok=True, parents=True)

write_models(more_models, root=fld_models)

figs[0].savefig(fld_plots / "overview__one_step.svg")
figs[1].savefig(fld_plots / "overview__two_step.svg")
# %%

sns.scatterplot(
    r.t_models.filter(pl.col("c") == "DAPI.1").join(
        f.intensity.filter(pl.col("c") == "DAPI.1")
        .group_by("roi")
        .agg(pl.col("Mean").mean(), pl.col("o", "label", "c").first())
        .select(pl.col(["roi", "o", "label", "c", "Mean"]))
        .with_columns(pl.col("Mean")),
        on=["c", "roi"],
        how="left",
    ).with_columns(pl.col('Mean')/pl.col('correction_factor')),
    x="delta_t_min",
    y="Mean",
)
# %%
import holoviews as hv

from zfish.intensity_normalization.plots import data_range, plot_model_, plot_models
from zfish.roi.spatial_roi import read_models

hv.extension("plotly")

# fld_models = Path('C:/Users/hessm/Documents/zfish_local/models_v3')

# more_models = read_models(fld_models)


NEW_NAMES = {
    "Linear(features=Centroid-z, loss=huber)": "one_step__Linear",
    "Linear(features=MediumPath;EmbryoPath, loss=huber)": "two_step__Linear",
    "LogLinear(features=Centroid-z, loss=huber)": "one_step__LogLinear",
    "LogLinear(features=MediumPath;EmbryoPath, loss=huber)": "two_step__LogLinear",
    "Exp(features=Centroid-z, loss=huber, pos_offset=True)": "one_step__ExpPos",
    "Exp(features=MediumPath;EmbryoPath, loss=huber, pos_offset=True)": "two_step__ExpPos",
    "Exp(features=Centroid-z, loss=huber, pos_offset=False)": "one_step__Exp",
    "Exp(features=MediumPath;EmbryoPath, loss=huber, pos_offset=False)": "two_step__Exp",
}

for channel in channels:
    print(channel)
    df_channel = df_merge_clean.filter(pl.col("channel") == channel)
    models = {}
    for model_name in NEW_NAMES:
        dd = more_models["z_decay"][model_name]
        if channel in dd:
            models[NEW_NAMES[model_name]] = dd[channel]
    figure = (
        plot_models(models, df_channel, target_column="Mean", n_samples=10_000)
        # .opts(
        #     opts.Scatter3D(height=300),
        #     opts.Scatter(height=300),
        #     opts.Distribution(height=300),
        # )
        .opts(title=f"channel = {channel}")
    )
    out_fn = fld_plots / f"individual_{channel}.html"
    hv.save(figure, out_fn)

# %%
import plotly.graph_objects as go
import plotly.io as pio

df_plot


fig = go.Figure(
    data=[
        go.Scatter3d(
            x=df_plot["MediumPath"],
            y=df_plot["EmbryoPath"],
            z=df_plot["Mean"],
            mode="markers",
            marker=dict(size=2, opacity=0.5, color="rgb(150, 150, 150)"),
        )
    ]
)
fig = go.Figure()
fig.update_layout(template="simple_white+gridon")
# fig.update_xaxes(minor=dict(showgrid=True), overwrite=True)
# fig.update_yaxes(minor=dict(showgrid=True), overwrite=True)
# fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(zaxis=dict(range=[0, 200])))
fig.update_layout(xaxis=dict(ticks="inside"), yaxis=dict(ticks="inside"))
fig.update_layout(
    scene=dict(
        xaxis=dict(ticks="inside"),
        yaxis=dict(ticks="inside"),
        zaxis=dict(ticks="inside"),
    )
)
fig.show()


# %%
# def stratified_sample(df, attr, n="max", s=5):
#     strat = pd.cut(
#         df[attr],
#         bins=np.linspace(df[attr].min() - 1, df[attr].max() + 1, s + 1),

#     )
#     if n == "max":
#         n = strat.value_counts().min() * s
#     sample = df.groupby(strat, group_keys=False).apply(lambda x: x.sample(n // s))
#     return sample


# # %%
# import numpy as np

# to_plot = df_merge_clean.to_pandas()

# mdl_colors = ["#dd966d", "#6ddd96", "#966ddd"]
# scatter_colors = ["#8e8e8e"]

# loss = "linear"
# mdl_types = [
#     Linear,
#     # ExpFitLinear,
#     ExpNoOffset,
#     Exp,
# ]
# X_columns = ['MediumPath', 'EmbryoPath']
# # X_columns = ["Centroid-z"]

# channels = [
#     "DAPI.0",
#     "FLAG.0",
#     "PCNA.0",
#     "Pol-II-S2P.0",
#     "DAPI.1",
#     "H3K27Ac.1",
#     "bCatenin.1",
#     "pH3.1",
#     "DAPI.2",
#     "ALYREF.2",
#     "H2B.2",
#     "Pol-II-S5P.2",
#     "DAPI.3",
#     "Nanog.3",
#     "XRN2.3",
#     "YAP.3",
# ]

# n_cols = 4
# n_rows = len(channels) // n_cols
# n_features = len(X_columns)

# fig = plt.figure(layout="constrained", figsize=(15, 9 * n_features), dpi=200)
# subfigs = fig.subfigures(n_rows, n_cols, wspace=0.07, squeeze=False).flatten()

# models = dict()

# for i, (channel, subfig) in enumerate(zip(channels, subfigs)):
#     print(channel)
#     models[channel] = dict()
#     df_channel = to_plot[to_plot.channel == channel]
#     df_sample = df_channel.sample(10000)  # stratified_sample(df_channel, "Centroid-z")
#     # df_sample = df_channel
#     X = df_sample[X_columns]
#     y = df_sample["Mean"]
#     X_fit = np.linspace(X.min(), X.max(), 200)

#     axs = subfig.subplots(n_features, 1, sharex=True, squeeze=False).flatten()
#     for j, ax in enumerate(axs):
#         plt.sca(ax)
#         if j % n_features == 0:
#             plt.ylabel("Mean Intensity [AU]")
#         if j // n_features == 0:
#             plt.xlabel(f"{X_columns[j]} [um]")
#         if j == 0:
#             plt.title(f"{channel}")
#         # plt.scatter(X, y, color=scatter_colors[0], s=1, alpha=0.3, label=f"{channel} Mean")
#         sns.scatterplot(
#             x=X.values[:, j],
#             y=y,
#             color=scatter_colors[0],
#             # hue=df_sample["Roundness"],
#             s=3,
#             alpha=0.3,
#             # label=f"{channel} Mean",
#         )
#         y_top = np.quantile(y, 0.999) * 1.5

#         plt.ylim(bottom=0, top=y_top)
#         # plt.yscale("log")
#         # plt.ylim(bottom=1, top=10**3.5)

#     for mdl_type, mdl_color in zip(mdl_types, mdl_colors):
#         mdl = mdl_type(loss=loss)
#         try:
#             mdl.fit(X, y)
#         except RuntimeError as e:
#             print(e)
#             print(f"no convergance for `{name}` in channel `{channel}`")
#             models[channel][name] = None
#             continue
#         name = str(mdl)
#         models[channel][name] = mdl
#         y_mdl = mdl.predict(X_fit)

#         for j, ax in enumerate(axs):
#             plt.sca(ax)
#             plt.plot(X_fit[:, j], y_mdl, color=mdl_color, label=name, linewidth=1.5)

#     # plt.legend()
# # %%
# df_models = (
#     pl.DataFrame(models)
#     .pipe(unnest_all_structs)
#     .pipe(stack_column_name_to_column, sep="-", column_name="channel", index=tuple())
#     .pipe(unnest_all_structs)
#     .pipe(
#         stack_column_name_to_column,
#         sep="-",
#         column_name="model_type",
#         index=("channel",),
#         return_list=False,
#     )
# )
# # %%
# df_models.select("_params").apply(list)
# pl.concat(df_models[1:], how='diagonal')
# pl.concat(df_models, how="vertical")
# df_models
# pl.concat(
# a = (
#     df_models
#     .select('^LinearModel.*$')
#     .pipe(stack_column_name_to_column, sep='-', index=tuple(), column_name='model_type')
# )
# b = (
#     df_models
#     .select('^ExponentialModel-.*$')
#     .pipe(stack_column_name_to_column, sep='-', index=tuple(), column_name='model_type')
# )

# (
#     df_models.select("channel", "^LinearModel.*$").pipe(
#     stack_column_name_to_column,
#     sep="-",
#     column_name="model_type",
#     index=tuple(),
#     return_list=True,
# ),
# # how='horizontal'
# )[0]
# # %%
# import copy
# from pathlib import Path
# from typing import Sequence

# import dask.array as da
# import h5py
# import numpy as np
# import pandas as pd

# from zfish.image.image import SpatialImage, to_si
# from zfish.io import h5
# from zfish.visualize.imshow import imshow


# def load_channel(
#     fn: Path | str,
#     channel: str,
#     level: int = 1,
# ) -> tuple[SpatialImage, SpatialImage]:
#     with h5py.File(fn) as f:
#         stain, acquisition = channel.split(".")
#         channel_si = to_si(
#             h5.select(f, {"stain": stain, "cycle": int(acquisition), "level": level})[0]
#         ).compute()
#         embryo_si = to_si(h5.select(f, {"stain": "embryoRaw"})[0]).compute()
#     return channel_si.squeeze(), embryo_si.squeeze()


# def load_nuclei(
#     fn: Path | str, level: int = 1, nuc_name: str = "nucleiRaw3"
# ) -> SpatialImage:
#     with h5py.File(fn) as f:
#         nuclei_si = to_si(
#             h5.select(f, {"stain": nuc_name, "level": level})[0]
#         ).compute()
#     return nuclei_si.squeeze()


# # %%
# df_channel = df_merge_clean.with_columns(pl.col("Mean")).to_pandas()

# TODO: Check code below, might still be usefull for model testing.
# # %%
# from dataclasses import dataclass
# from typing import Callable, Sequence

# import numpy as np
# from models import exponential_model_glm, exponential_model_with_offset, linear_model


# @dataclass
# class RegressionData:
#     data: pd.DataFrame = None
#     X_cols: list[str] = None
#     y_col: str = None

#     def __repr__(self):
#         return (
#             f"{self.__class__.__name__}\nX = {self.X_cols} y = {self.y_col!r}\n\n"
#             + repr(self.data.head(2))
#             + "\n..."
#         )

#     @property
#     def X(self) -> pd.DataFrame:
#         return self.data[self.X_cols]

#     @property
#     def y(self) -> pd.Series:
#         return self.data[self.y_col]

#     def X_linspace(self, idx: int = 0, n_samples: int = 200):
#         """Return X with all columns except .iloc[i] set to zero."""
#         X_out = np.zeros((n_samples, self.X.shape[1]), dtype=self.X.dtype)
#         x = self.X[:, idx]
#         X_out[:, idx] = np.linspace(x.min(), x.max(), n_samples)
#         return X_out

#     def X_gridspace(self, n_samples: int = 200):
#         if self.X.shape[1] != 2:
#             raise ValueError(
#                 f"Gridspace can only be computed for 2d X. Found {self.X.shape[1]} dimensions instead."
#             )
#         return np.meshgrid(
#             *np.linspace(self.X.min(axis=0), self.X.max(axis=0), n_samples).T
#         )


# @dataclass
# class SyntheticRegressionData(RegressionData):
#     model: Callable = None
#     noise_model: Callable = None
#     params: np.ndarray = None

#     def __repr__(self):
#         return (
#             f"{self.__class__.__name__}\nModel: {self.model}, Noise: {self.noise_model}, Params: {self.params}\nX = {self.X_cols}, y = {self.y_col!r}\n\n"
#             + repr(self.data.head(2))
#             + "\n..."
#         )

#     @property
#     def y_true(self) -> pd.Series:
#         return self.data[self.y_col_true].values


# def make_linear_regression(
#     params: np.ndarray = np.array([10, -1, -3]),
#     n_obs: int = 200,
#     noise_model: Callable = np.random.randn,
# ) -> SyntheticRegressionData:
#     X = np.random.rand(n_obs, len(params[1:]))
#     y_true = linear_model(X, *params)
#     y = y_true + noise_model(*y_true.shape)

#     X_cols = [f"x_{i}" for i in range(X.shape[1])]
#     y_col = "y"
#     cols = X_cols + [y_col]
#     return SyntheticRegressionData(
#         data=pd.DataFrame(
#             data=np.concatenate([X, np.expand_dims(y, 1)], axis=1), columns=cols
#         ),
#         X_cols=X_cols,
#         y_col=y_col,
#         model=linear_model,
#         noise_model=noise_model,
#         params=params,
#     )


# def make_exponential_regression(
#     params: np.ndarray = np.array([10, -1, -3]),
#     n_obs: int = 200,
#     noise_model: Callable = np.random.randn,
# ) -> SyntheticRegressionData:
#     X = np.random.rand(n_obs, len(params[1:]))
#     y_true = exponential_model_glm(X, *params)
#     y = y_true + noise_model(*y_true.shape)

#     X_cols = [f"x_{i}" for i in range(X.shape[1])]
#     y_col = "y"
#     cols = X_cols + [y_col]
#     return SyntheticRegressionData(
#         data=pd.DataFrame(
#             data=np.concatenate([X, np.expand_dims(y, 1)], axis=1), columns=cols
#         ),
#         X_cols=X_cols,
#         y_col=y_col,
#         model=exponential_model_glm,
#         noise_model=noise_model,
#         params=params,
#     )


# def make_exponential_regression_with_offset(
#     params: np.ndarray = np.array([10, 1, -1, -3]),
#     n_obs: int = 200,
#     noise_model: Callable = np.random.randn,
# ) -> SyntheticRegressionData:
#     X = np.random.rand(n_obs, len(params[2:])) * 200
#     y_true = exponential_model_with_offset(X, *params)
#     y = y_true + noise_model(*y_true.shape)

#     X_cols = [f"x_{i}" for i in range(X.shape[1])]
#     y_col = "y"
#     cols = X_cols + [y_col]
#     return SyntheticRegressionData(
#         data=pd.DataFrame(
#             data=np.concatenate([X, np.expand_dims(y, 1)], axis=1), columns=cols
#         ),
#         X_cols=X_cols,
#         y_col=y_col,
#         model=exponential_model_with_offset,
#         noise_model=noise_model,
#         params=params,
#     )


# # %%
# from copy import deepcopy

# from zfish.intensity_normalization.models import apply_model_to_channel, fit_model_to_df

# # data = make_exponential_regression(params=[10, -1, -4])
# data = make_exponential_regression_with_offset(params=[10, 15, -0.1, -0.4])
# # model = Exp(loss='huber', enforce_positive_offset=True)
# model = ExpNoOffset(loss="huber")
# model_fit = deepcopy(model).fit(data.X, data.y)

# if data.X.shape[1] == 1:
#     ax = scatter_plot(data.X.values, data.y.values)
#     ax = model_plot(data.X.values, model_fit, ax=ax)
# elif data.X.shape[1] == 2:
#     ax = scatter_plot_3d(data.X.values, data.y.values)
#     ax = model_plot_3d(data.X.values, model_fit, ax=ax)
# # %%
# from zfish.intensity_normalization.models import apply_model_to_channel, fit_model_to_df

# channel = "pH3.1"

# model = Exp(loss="huber", enforce_positive_offset=False)
# # model = ExpFitLinear(loss='huber')
# # model = Linear(loss='huber')
# # model = ExpNoOffset(loss='huber')

# X_columns = ["MediumPath", "EmbryoPath"]
# # X_columns = ['Centroid-z']
# y_column = f"{channel}_Mean"
# n_samples = None

# df_channel = (
#     df_merge_clean.filter(pl.col("channel") == channel)
#     .with_columns((pl.col("Mean") / pl.col("Mean").mean()).prefix(f"{channel}_"))
#     .to_pandas()
# )
# if n_samples is not None:
#     df_channel = df_channel.sample(n_samples)

# X = df_channel[X_columns].to_numpy()
# y = df_channel[y_column].to_numpy()

# model_fit = fit_model_to_df(
#     df_channel, channel=channel, model=model, X_columns=X_columns, y_column=y_column
# )

# if X.shape[1] == 1:
#     ax = scatter_plot(X, y)
#     ax = model_plot(X, model_fit, ax=ax)

#     ax = scatter_plot_corr(X, y, model_fit)
#     ax = model_plot_corr(X, model_fit, ax=ax)

# elif X.shape[1] == 2:
#     ax = scatter_plot_3d(X, y)
#     ax = model_plot_3d(X, model_fit, ax=ax)

#     ax = scatter_plot_3d_corr(X, y, model_fit)
#     ax = model_plot_3d_corr(X, model_fit, ax=ax)
# # %%

# # %%
# a = 3
# # %%
# from mayavi import mlab


# def scatter_plot_3d(
#     X, y, ax=None, subsample: int | None = 10000, c="#8e8e8e", **kwargs
# ):
#     if ax is None:
#         fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#     y_top = np.quantile(y, 0.999) * 1.5
#     ax.set_zlim(0, y_top)
#     if subsample:
#         np.random.seed(42)
#         idxs = np.random.choice(np.arange(len(X)), subsample, replace=False)
#         X = X[idxs, :]
#         y = y[idxs]
#         if not isinstance(c, str):
#             c = c[idxs]
#     ax.scatter(X[:, 0], X[:, 1], y, s=3, alpha=0.5, c=c, **kwargs)
#     return ax


# def mscatter_plot_3d(X, y):
#     mlab.points3d(X[:, 0], X[:, 1], y, scale_factor=0.1)
#     # y_top = np.quantile(y, 0.999) * 1.5
#     # ax.set_zlim(0, y_top)
#     # return ax


# def model_plot_3d(X, model, ax=None, **kwargs):
#     if ax is None:
#         fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, alpha=0.5)
#     X_grid = get_X_grid(X)
#     y_pred = model.grid_predict(X_grid)
#     ax.plot_surface(X_grid[0], X_grid[1], y_pred, alpha=0.5, **kwargs)
#     return ax


# def scatter_plot_3d_corr(X, y, model, ax=None):
#     if ax is None:
#         fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#     y_corr = model.correct(X, y)
#     scatter_plot_3d(X, y_corr, ax=ax)
#     return ax


# def model_plot_3d_corr(X, model, ax=None):
#     if ax is None:
#         fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, alpha=0.5)
#     X_grid = get_X_grid(X)
#     y_pred = model.grid_predict(X_grid)
#     y_pred_corr = model.grid_correct(X_grid, y_pred)
#     ax.plot_surface(X_grid[0], X_grid[1], y_pred_corr, alpha=0.5)
#     return ax


# def scatter_plot(X, y, ax=None):
#     if ax is None:
#         fig, ax = plt.subplots()
#     ax.scatter(X[:, 0], y, s=3, alpha=0.5, color="#8e8e8e")
#     return ax


# def model_plot(X, model, ax=None):
#     if ax is None:
#         fig, ax = plt.subplots()
#     X_fit = np.linspace(X.min(axis=0), X.max(axis=0), 200)
#     y_pred = model.predict(X_fit)
#     ax.plot(X_fit, y_pred)
#     return ax


# def scatter_plot_corr(X, y, model, ax=None):
#     if ax is None:
#         fig, ax = plt.subplots()
#     y_corr = model.correct(X, y)
#     scatter_plot(X, y_corr, ax=ax)
#     return ax


# def model_plot_corr(X, model, ax=None):
#     if ax is None:
#         fig, ax = plt.subplots()
#     X_fit = np.linspace(X.min(axis=0), X.max(axis=0), 200)
#     y_pred = model.predict(X_fit)
#     y_pred_corr = model.correct(X_fit, y_pred)
#     ax.plot(X_fit, y_pred_corr)
#     return ax


# # %%
# from typing import Mapping


# def load_channel_corr_roi(
#     roi: Roi, channel: str, models: str | Sequence[str]
# ) -> list[SpatialImage]:
#     if isinstance(models, str):
#         models = [models]


# def load_channel_corr(
#     roi: Roi,
#     channel: str,
#     models: Model | Mapping[str, Model],
#     label_image: str | None = "embryoRaw",
#     return_raw: bool = False,
# ) -> list[SpatialImage]:
#     print(f"loading channel {channel}")
#     if isinstance(models, Model):
#         models = {models.__class__.__name__: models}

#     channel_raw = roi.sel(c=channel).images
#     old_name = channel_raw.c.item()

#     channels = []
#     if return_raw:
#         channels.append(channel_raw)

#     for name, model in models.items():
#         print(name)
#         if model._feature_names is None:
#             raise ValueError(
#                 "missing `model._feature_names`. Either model is not fit or fit on a numpy array."
#             )
#         new_name = f"{name}_{old_name}"
#         if len(model._feature_names) == 1:
#             channel_corr = apply_model_to_channel(model, channel_raw)
#         elif len(model._feature_names) == 2:
#             label_embryo = roi.sel(l=label_image).labels
#             channel_corr = apply_model_to_channel(model, channel_raw, label_embryo)
#         else:
#             raise ValueError("max two features allowed.")
#         channel_corr["c"] = new_name
#         channels.append(channel_corr)
#     return channels


# # %%
# from pathlib import Path

# from zfish.intensity_normalization.models import apply_model_to_channel
# from zfish.roi.spatial_roi import Roi, RoiMap

# fld = r"M:\marvwy\20220721_ZE4i2_aligned\imgs"
# dater = RoiMap.from_files(list(Path(fld).glob("*.h5"))[3::80], level=1)
# # %%
# roi = dater.first()
# # %%
# # TODO: ensure consistent behaviour with degenerate dims.
# r = roi.sel(c=["DAPI.0", "DAPI.1"], l=["embryoRaw"])
# r2 = roi.sel(c=["DAPI.0", "DAPI.1"], l="embryoRaw")
# r3 = roi.sel(c="DAPI.0", l=["embryoRaw"])
# r4 = roi.sel(c="DAPI.0", l="embryoRaw")
# r5 = roi.sel(c=["DAPI.0"], l=["embryoRaw"])
# r6 = roi.sel(c=["DAPI.0"], l="embryoRaw")
# # %%
# r3.labels.sel(z=[10]).squeeze("l")
# # %%
# # r4.compute("Exp(loss='huber', features=['MediumPath', 'EmbryoPath'], p_offset=True)")
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots()
# r2.compute("Exp(loss='huber', features=['Centroid-z'], p_offset=True)").drop_dim(
#     "l"
# ).sel(y=300, c="DAPI.1", method="nearest").images.plot()
# fig, ax = plt.subplots()
# r2.compute().drop_dim("l").sel(y=300, c="DAPI.1", method="nearest").images.plot()
# # %%
# r = roi.sel(l=["embryoRaw"]).sel(c=["DAPI.0", "DAPI.1"]).compute()
# r2 = (
#     roi.sel(l=["embryoRaw"])
#     .sel(c=["DAPI.0", "DAPI.1"])
#     .compute("Exp(loss='huber', features=['MediumPath', 'EmbryoPath'], p_offset=True)")
# )
# # %%
# r2.labels.expand_dims()
# # %%
# import napari

# from zfish.roi.visualize import imshow_roi

# viewer = napari.Viewer()
# imshow_roi(r, viewer)
# imshow_roi(r2, viewer)
# # %%
# channel = "PCNA.0"
# lbl = roi.sel(l="embryoRaw").labels
# img = roi.sel(c=channel).images
# # mdl = models_fit['ExponentialModelFalse_huber_Centroid-z']
# # mdl = models_fit['ExponentialModelFalse_huber_MediumPath|EmbryoPath']
# mdl = roi.models["z_decay"][channel][
#     "Exp(loss='huber', features=['MediumPath', 'EmbryoPath'], p_offset=True)"
# ]
# mdl2 = roi.models["z_decay"][channel][
#     "Exp(loss='huber', features=['Centroid-z'], p_offset=True)"
# ]
# # %%
# res = apply_model_to_channel(mdl, img, lbl)
# res2 = apply_model_to_channel(mdl2, img, lbl)
# # %%
# img_ = img.compute()
# # %%
# res_ = res.compute()
# # %%
# res2_ = res2.compute()
# # %%
# import napari

# from zfish.visualize.imshow import imshow_spatial_image

# viewer = napari.Viewer()
# imshow_spatial_image(img_, viewer, **{"contrast_limits": (0, 2800)})
# imshow_spatial_image(res_, viewer, **{"contrast_limits": (0, 2800)})
# imshow_spatial_image(res2_, viewer, **{"contrast_limits": (0, 2800)})
# # %%
# corrs = load_channel_corr(roi, channel=channel, models=models_fit, return_raw=True)
# corrs_c = [e.compute() for e in corrs]

# # %%
# import napari

# from zfish.features.intensity import get_distribution_features
# from zfish.features.label import get_position_and_orientation_features
# from zfish.visualize.imshow import imshow_spatial_image

# print("label features")
# df = get_position_and_orientation_features(lbl_nuc)
# for corr in corrs:
#     print(f"intensity features {corr.c.item()}")
#     df = df.idx.join(get_distribution_features(lbl_nuc, corr))
# # %%
# df_tall = (
#     df.select("Centroid", "^.*_Mean$")
#     .pipe(stack_column_name_to_column, column_name="model_type", index=("Centroid",))
#     .select(
#         [
#             pl.col("Centroid"),
#         ]
#     )
#     .with_columns([pl.col("model").str.replace_all("^(.*Model)(True|False)", "$1-$2")])
#     # .with_columns(df.select(pl.col('Centroid')))
#     .pipe(unnest_all_structs)
#     .fill_null("")
# ).sort(by=["features", "model", "loss"])


# # %%
# df_tall
# # %%
# from scipy import stats

# n = len(df_tall.groupby(["features", "model", "loss", "channel"]).count())
# ncols = 5
# nrows = int(np.ceil(n / ncols))


# fig, axs = plt.subplots(
#     nrows=nrows,
#     ncols=ncols,
#     figsize=(ncols * 3, nrows * 3),
#     sharex=True,
#     sharey=True,
#     dpi=200,
#     layout="constrained",
# )
# plt.suptitle(channel, fontsize=20, ha="left", x=0)
# for (group_columns, df_model), ax in zip(
#     df_tall.groupby(
#         ["features", "model", "loss", "channel"], maintain_order=True
#     ).__iter__(),
#     axs.flatten(),
# ):
#     name = "\n".join(group_columns[:3])

#     df_plot = df_model.with_columns(
#         df_model["Centroid-z"].cut(bins=[50, 100, 150, 200])["category"]
#     ).to_pandas()
#     x = df_plot["Centroid-z"]
#     y = df_plot["Mean"] / df_plot["Mean"].mean()
#     values = np.vstack([x, y])
#     kernel = stats.gaussian_kde(values)
#     density = kernel(values)

#     plt.sca(ax)
#     plt.title(name)
#     sns.kdeplot(x=y, hue=df_plot["category"], common_norm=True, palette="viridis")
#     # sns.scatterplot(
#     #     df_model.to_pandas(),
#     #     x=x,
#     #     y=y,
#     #     s=4,
#     #     alpha=1.0,
#     #     # label=name,
#     #     # legend="full",
#     #     c=density,
#     #     cmap="inferno",
#     # )
#     # plt.ylim(bottom=0)
#     # plt.ylabel(f"{group_columns[-1]}_{y.name}")
#     # plt.text(0, 1500, , fontdict={})
# # print("intensity raw")
# # df_img = (
# #     get_distribution_features(lbl_nuc, img).idx.join(df_label).pipe(unnest_all_structs)
# # )
# # print("intensity one step")
# # df_img_corr_1step = (
# #     get_distribution_features(lbl_nuc, img_corr_1step)
# #     .idx.join(df_label)
# #     .pipe(unnest_all_structs)
# # )
# # print("intensity two step")
# # df_img_corr_2step = (
# #     get_distribution_features(lbl_nuc, img_corr_2step)
# #     .idx.join(df_label)
# #     .pipe(unnest_all_structs)
# # )
# # %%
# from unidip import UniDip

# # create bi-modal distribution
# dat = np.concatenate([np.random.randn(200) - 3, np.random.randn(200) + 3])
# dat = np.random.randn(200)

# # sort data so returned indices are meaningful
# dat = np.msort(dat)

# # get start and stop indices of peaks
# intervals = UniDip(dat).run()
# # %%
# fig, axs = plt.subplots(3, 1, figsize=(5, 10), squeeze=False, sharex=True)

# for df_plot, label, ax in zip(
#     [df_img.to_pandas(), df_img_corr_1step.to_pandas(), df_img_corr_2step.to_pandas()],
#     ["raw", "one step", "two step"],
#     axs.flatten(),
# ):
#     plt.sca(ax)
#     plt.title(label)
#     sns.scatterplot(df_plot, x="Centroid-z", y="DAPI.1_Mean", s=4, alpha=0.3)
# # %%
# df_plot = df_merge_clean.filter(pl.col("channel") == "DAPI.1").to_pandas()
# x = df_plot[["Centroid-z"]]
# x2 = df_plot[["MediumPath", "EmbryoPath"]]
# y = df_plot[f"Mean"]
# y_corr_1step = mdl_1step.correct(x, y)
# y_corr_2step = mdl_2step.correct(x2, y)

# fig, axs = plt.subplots(3, 1, figsize=(5, 10), squeeze=False, sharex=True)
# axs = axs.flatten()

# plt.sca(axs[0])
# sns.scatterplot(x=x.squeeze(), y=y, s=4, alpha=0.3)

# plt.sca(axs[1])
# df_plot = df_img_corr_1step.to_pandas()
# x = df_plot[["Centroid-z"]]
# sns.scatterplot(df_plot, x="Centroid-z", y="Mean", s=4, alpha=0.3, label="image corr")
# sns.scatterplot(x=x.squeeze(), y=y_corr_1step, s=4, alpha=0.3, label="feature corr")

# df_plot = df_img_corr_2step.to_pandas()
# x = df_plot[["Centroid-z"]]
# sns.scatterplot(df_plot, x="Centroid-z", y="Mean", s=4, alpha=0.3, label="image corr")
# sns.scatterplot(x=x.squeeze(), y=y_corr_2step, s=4, alpha=0.3, label="feature corr")
# # %%
# y_corr = y * mdl._correction_factor(X)


# sns.histplot(y.apply(np.log2))
# sns.histplot(y_corr.apply(np.log2))
# print(y.std() / y.mean())
# print(y_corr.std() / y_corr.mean())
# print(y.apply(np.log2).std() / y.apply(np.log2).mean())
# print(y_corr.apply(np.log2).std() / y_corr.apply(np.log2).mean())
# # %%
# sns.scatterplot(x=df_sample["Centroid-z"], y=y.apply(np.log2), s=1, alpha=0.1)
# sns.scatterplot(x=df_sample["Centroid-z"], y=y_corr.apply(np.log2), s=1, alpha=0.1)
# # %%
# n_grid = 10
# X_grid = (
#     np.stack(np.meshgrid(*np.linspace(X.min(), X.max(), n_grid).T)).reshape(2, -1).T
# )
# y_grid = mdl.predict(X_grid)
# y_grid2 = mdl2.predict(X_grid)
# y_grid3 = mdl3.predict(X_grid)
# # mdl.predict(np.stack(np.meshgrid(*np.linspace(X.min(), X.max(), 10).T)).reshape(2, -1).T).reshape((10, 10))
# # %%
# sample_idxs = np.random.randint(len(y), size=50000)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))

# ax.plot_surface(
#     X_grid[:, 0].reshape(n_grid, n_grid),
#     X_grid[:, 1].reshape(n_grid, n_grid),
#     y_grid.reshape(n_grid, n_grid),
#     alpha=0.2,
# )
# ax.plot_surface(
#     X_grid[:, 0].reshape(n_grid, n_grid),
#     X_grid[:, 1].reshape(n_grid, n_grid),
#     y_grid2.reshape(n_grid, n_grid),
#     alpha=0.2,
# )
# ax.plot_surface(
#     X_grid[:, 0].reshape(n_grid, n_grid),
#     X_grid[:, 1].reshape(n_grid, n_grid),
#     y_grid3.reshape(n_grid, n_grid),
#     alpha=0.2,
# )
# ax.scatter(
#     X.iloc[sample_idxs, 0], X.iloc[sample_idxs, 1], y.iloc[sample_idxs], s=1, alpha=0.2
# )

# # %%

# %%
