# %% Imports and definitions
from os import PathLike
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import tqdm

from zfish.features.polars_utils import (
    INTENSITY_FEATURE_COLUMNS,
    get_metadata,
    set_index_dtypes,
    stack_channels,
    stack_correlation_metric_by_acquisition,
    unnest_all_structs,
    unnest_structs,
    unstack_channels,
)
from zfish.intensity_normalization.models import (
    ExponentialModel,
    ExponentialModelFitLinear,
    ExponentialModelNoOffset,
    LinearModel,
    Model,
)
from zfish.roi.spatial_roi import Roi, RoiMap, load_roi_tables

pl.enable_string_cache(True)

DEBUG = True
INDEX = ('roi', 'object', 'label')
COL_INDEX = [pl.col(e) for e in INDEX]
CONTROL_WELLS = ["B07", "C07", "D07", "E07"]

from typing import TypeVar

T_AnyFrame = TypeVar('T_AnyFrame', pl.DataFrame, pl.LazyFrame)

def index_first(df: T_AnyFrame) -> T_AnyFrame:
    return df.select([pl.col(INDEX), pl.exclude(INDEX)])

def read_table(root: PathLike[str], _object: str | None = "", use_pyarrow=True) -> pl.DataFrame:
    fns = list(Path(root).rglob(f'{_object}*.parquet'))
    tables = []
    for fn in tqdm.tqdm(fns):
        table = pl.read_parquet(fn, use_pyarrow=use_pyarrow).with_columns(pl.lit(fn.parent.name).alias('roi'), pl.lit(fn.stem).alias('object'))
        tables.append(table)
    return pl.concat(tables, how='diagonal')

def scan_table(root: PathLike[str], _object: str | None = "") -> pl.LazyFrame:
    fns = list(Path(root).rglob(f'{_object}*.parquet'))
    tables = [
        pl.scan_parquet(fn)
        .with_columns(
            pl.lit(fn.parent.name).alias('roi'), 
            pl.lit(fn.stem).alias('object')
            ) for fn in tqdm.tqdm(fns)]

    return pl.concat(tables, how='diagonal')


# %% Helper function for range rejection with plot
def discard_bounds(df: pl.LazyFrame, feature: str, lower: float | None = None, upper: float | None = None, debug=DEBUG, ax=None) -> pl.LazyFrame:
    if isinstance(df, pl.DataFrame):
        df = df.lazy()

    if lower is None and upper is None:
        out = df
    elif lower is None:
        x = [upper]
        linestyles=['dashed']
        out = df.filter(pl.col(feature).lt(upper))
    elif upper is None:
        x = [lower]
        linestyles=['dotted'] 
        out = df.filter(pl.col(feature).gt(lower))
    else:
        x = [lower, upper]
        linestyles=['dotted', 'dashed']
        out = df.filter(pl.col(feature).is_between(lower, upper))
    
    if debug:
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        ax = sns.kdeplot(
            df.select([
                pl.col(INDEX), 
                pl.col(feature),
                ]).collect().to_pandas(), 
            x=feature,
            ax=ax)

        y_lims = ax.get_ylim()
        plt.vlines(
            x=x,
            ymin=y_lims[0],
            ymax=y_lims[1],
            colors=["k"],
            linestyles=linestyles,
        )
    return out

def remove_outliers(df: pl.LazyFrame, outliers: list[dict[str, Any]]) -> pl.LazyFrame:
    for outlier in outliers:
        df = partial(discard_bounds, **outlier)(df)
    return df

# %% Load data
fld_features = Path(r"M:\marvwy\20220721_ZE4i2_aligned\features")
fld_features_block = Path(r"M:\marvwy\20220721_ZE4i2_aligned\features_block")


# df_files = read_table(fld_features, _object="", use_pyarrow=True)
df_block = pl.read_parquet(fld_features_block, use_pyarrow=True).pipe(index_first)
# df_lazy_files = scan_table(fld_features)
df_lazy_block = pl.scan_parquet(fld_features_block / 'full.parquet').pipe(index_first)

# %% Select nuclei measurements
df_full = df_block.filter(pl.col('object')=='nucleiRaw3')
df_full_lazy = df_lazy_block.filter(pl.col('object')=='nucleiRaw3')

# %% Discard outliers
from functools import partial, reduce
from typing import Any, Callable, TypeVar

outliers = [
    {
    'feature': "EquivalentSphericalRadius", # um
    'lower': 4,
    'upper': 8,
    },
    {
    'feature': "Roundness",
    'upper': 0.922,
    },
]

df_clean = remove_outliers(df_full_lazy, outliers=outliers)
# %%

df_meta = get_metadata(df_clean.select([pl.col('roi', 'object')]).collect(), control_wells=CONTROL_WELLS)

# %% Select and join relevant columns
features = ["Centroid", "PhysicalSize", "EquivalentSphericalRadius", "Roundness", "embryoRaw-1_CentroidDistToBorder", "embryoRaw-1_CentroidDistAlongZ"]


df = (
df_clean
.select([
    pl.col(INDEX),
    pl.col(features),
    pl.col('^.*_Mean$'),
    pl.col('^.*PearsonR$'),
])
.join(df_meta.lazy(), on=["roi"])
.collect()
.pipe(unnest_structs, cols=['Centroid'])
.with_columns(
        [
            (pl.col("Centroid-z") - pl.col("embryoRaw-1_CentroidDistAlongZ")).alias(
                "mediumPath"
            ),
            pl.col("embryoRaw-1_CentroidDistAlongZ").alias("embryoPath"),
        ]
    )
)
df
# %%
from zfish.features.polars_utils import (
    _split_channel_column,
    _split_feature_name,
    _split_single_channel,
)

_split_feature_name(df_clean.columns)

# %%
_split_feature_name(df.select(pl.col("^.*Pearson.*$")).columns)
# %%
channel_patterns = ['^DAPI-0_.*$', '^DAPI-1_.*$', '^Pol-II-S5P-2_.*$', '^bCatenin-3_.*$']
CORRELATION_FEATURE_PATTERNS = ['^.*PearsonR$']
df_test = df_files.select(list(INDEX) + ['BoundingBox', 'Centroid'] + channel_patterns + list(CORRELATION_FEATURE_PATTERNS)).filter(pl.col('roi').is_in(['G03_px+2426_py-0008', 'E07_px+2226_py+1238', 'E05_px-1798_py-0764']))
# %%
CORRELATION_FEATURE_PATTERNS = ['^.*PearsonR$']
ref_channel = 'DAPI-1'
df_corr = df

from zfish.features.polars_utils import (
    INTENSITY_FEATURE_PATTERNS,
    _split_feature_name,
)

# %%
stack_channels(df, patterns=INTENSITY_FEATURE_PATTERNS)
# %%
# stack_channels(df.select([pl.col(INDEX), *[pl.col(pattern) for pattern in CORRELATION_FEATURE_PATTERNS]]))
# _split_single_channel(df.select(pl.col(INDEX), pl.col('^DAPI-0-DAPI-1.*$')))
df.select([pl.col(e) for e in list(INTENSITY_FEATURE_PATTERNS) + CORRELATION_FEATURE_PATTERNS])
# %%
df.select()
# %% Stack channels
# df.select(pl.exclude('^.*[0, 2, 3]_Mean$'))
df_tall = stack_channels(df)
# %%
df.select(pl.col('^.*PearsonR$'))
# %%
REF_CHANNEL = 'DAPI-1'
(
    pl.Series(
    name='features',
    values=df.select(pl.col('^.*PearsonR$')).columns,
    )
    .to_frame()
    .select(
        [
            pl.all().str.split('-').arr.slice(0, 2).arr.join('-').alias('p1'),
            pl.all().str.split('-').arr.slice(2, None).arr.join('-').alias('p2'),
        ]
    )
    .select(
        [
            pl.concat_list(pl.all()).arr.join('*')
        ]
    )
    .select([
        pl.all().str.split('_').arr.eval(pl.element().str.split('*').arr.explode())
    ])
    .select([
        pl.all().arr.get(0).alias('channel0'),
        pl.all().arr.get(1).alias('channel1'),
        pl.all().arr.get(2).alias('feature'),
    ])
    .select([
        pl.when(pl.col('channel0')=='DAPI-1').then(pl.col('channel1')).otherwise(pl.col('channel0')).alias('channel'),
        pl.lit(REF_CHANNEL).alias('ref_channel'),
        pl.col('feature'),
    ])
    # .select([
    #     pl.all().arr.eval(pl.element())
    # ])
#     .select([
#         pl.all().arr.slice(0, 2).alias('channels').arr.contains(REF_CHANNEL),
#         pl.all().arr.last().alias('feature'),
# # 
#     ])
        
)
# df.select(pl.col('^.*PearsonR$')).columns
    
# %% Remove individual misaligned cells
ALIGNMENT_SCORE_CUTOFF = 0.85
ax = sns.kdeplot(
    df_tall.filter(pl.col("acquisition") != 1)
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
