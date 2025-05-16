# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns

from zfish.features.polars_preprocessing import get_metadata
from zfish.features.polars_selector import sel
from zfish.features.polars_utils import (
    drop_null_columns,
    read_table,
    set_index_dtypes,
    split_channel_column,
    split_channel_pair_column,
    stack_column_name_to_column,
    stack_correlation_metric_by_acquisition,
    unnest_all_structs,
)
from zfish.intensity_normalization.models import (
    Exp,
    ExpFitLinear,
    ExpNoOffset,
    Linear,
)
from zfish.preprocessing.outlier_ranges import (
    RngOutlier,
    remove_outliers,
)
from zfish.preprocessing.sliding_window_samples import stratified_sample

pl.enable_string_cache()

# %% Outliers ranges for nuclei
SEGMENTATION_OUTLIER_RANGES = [
    {"feature": "EquivalentSphericalRadius", "lower": 4, "upper": 8},
    {"feature": "parent.embryoRaw", "lower": 1},  # Remove 'orphan nuclei'
]
segmentation_outliers = [RngOutlier(**e) for e in SEGMENTATION_OUTLIER_RANGES]

# %% Load from server
feature_fld = Path(
    r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\features_v2\NoCorrection"
)
df_wide = read_table(feature_fld, _object="nucleiRaw3")
# %% Load local save file
df_wide = pl.read_parquet(
    r"C:\Users\hessm\Documents\zfish_local\features\nuclei_all.parquet"
)


# %% Remove segmentation outliers and generate metadata tables
control_wells = ["B07", "C07", "D07", "E07"]
df_clean_seg = remove_outliers(df_wide.pipe(unnest_all_structs), segmentation_outliers)
df_meta = get_metadata(df_clean_seg, control_wells=control_wells).drop(
    "parent.embryoRaw"
)
df_alignment = stack_correlation_metric_by_acquisition(
    df_wide.with_columns(pl.lit(1.0).alias("DAPI.1|DAPI.1_PearsonR"))
)
# %% Load acquisition timepoints
df_timepoints = set_index_dtypes(
    pl.DataFrame(
        pd.read_parquet(
            r"C:\Users\hessm\Documents\Programming\Python\zfish\data\acquisition_times.parquet"
        ).drop("acquisitionTime", axis=1)
    )
)

# %%
df_label = df_clean_seg.select(sel.index - sel.hierarchy, sel.label)

df_intensity = (
    df_clean_seg.select(sel.index, sel.intensity)
    .pipe(stack_column_name_to_column, index=sel.index, column_name="channel")
    .pipe(split_channel_column)
)

# %% Merge relevant columns of dataframes
df = (
    df_intensity.select([sel.index - sel.hierarchy, pl.col("Mean")])
    .with_columns(pl.col("acquisition").cast(pl.UInt16))
    .join(
        df_label.pipe(unnest_all_structs).select([sel.index, pl.col("Centroid.z")]),
        on=["roi", "object", "label"],
    )
    .join(df_alignment, on=["roi", "object", "label", "acquisition"])
    .join(
        df_timepoints.select(["roi", "acquisition", "timeDeltaMinutes"]).with_columns(
            pl.col("roi").cast(pl.Utf8)
        ),
        on=["roi", "acquisition"],
    )
    .join(df_meta, on=["roi"])
)
# %% Remove outliers
QC_OUTLIER_RANGES = [
    {"feature": "Centroid.z", "upper": 150},  # Only consider cells -150um deep
    {"feature": "alignmentScore", "lower": 0.9},
    {"feature": "cycle", "lower": 6},
]
qc_outliers = [RngOutlier(**e) for e in QC_OUTLIER_RANGES]

df_clean = (
    remove_outliers(df, qc_outliers)
    # remove control wells per acquisition
    .filter(pl.col("acquisition") < pl.col("control_well_for_acquisition"))
)


# %% Aggregate intensity measurements per site
df_mean_per_embryo = (
    df_clean.group_by(["roi", "channel", "timeDeltaMinutes", "cycle"])
    .agg(
        pl.col("Mean").mean(),
        pl.col("Mean").std().alias("Std"),
        pl.col("Mean").count().alias("nuc_count"),
    )
    .with_columns(
        [
            ((pl.col("Mean") - pl.col("Mean").mean()) / pl.col("Mean").std())
            .over("channel")
            .alias("Mean_ZScore"),
            ((pl.col("Mean") - pl.col("Mean").mean()) / pl.col("Mean").std())
            .over(["channel", "cycle"])
            .alias("Mean_CycleZScore"),
            ((pl.col("Mean") / pl.col("Mean").mean()))
            .over(["channel", "cycle"])
            .alias("Mean_CycleMeanNorm"),
        ]
    )
)

# %%
df_mean_per_embryo = (
    df_clean.with_columns(
        (pl.col("Mean") / (pl.col("Mean").mean().over(["channel", "cycle"]))).alias(
            "MeanCycleNorm"
        )
    )
    .group_by(["roi", "channel", "timeDeltaMinutes", "cycle"])
    .agg(
        pl.col("Mean").median().name.suffix("_median"),
        pl.col("MeanCycleNorm").median().name.suffix("_median"),
        pl.col("Mean").quantile(0.1).name.suffix("_q10"),
        pl.col("MeanCycleNorm").quantile(0.1).name.suffix("_q10"),
        pl.col("Mean").quantile(0.9).name.suffix("_q90"),
        pl.col("MeanCycleNorm").quantile(0.9).name.suffix("_q90"),
    )
    # .agg(
    #     pl.col("Mean").mean().name.suffix("_mean"),
    #     pl.col("Mean").std().name.suffix("_std"),
    #     pl.col("MeanCycleNorm").mean().name.suffix("_mean"),
    #     pl.col("MeanCycleNorm").std().name.suffix("_std"),
    #     pl.col("Mean").count().alias("nuc_count"),
    # )
)


# %%
# to_plot = df_mean_per_embryo.sample(100_000, seed=42).to_pandas()

to_plot = df_mean_per_embryo.filter(pl.col("cycle") > 10).to_pandas()
fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
agg_feature = "Mean"

df_channel = to_plot[to_plot.channel == "Pol-II-S5P.2"]
sns.scatterplot(
    data=df_channel,
    x="timeDeltaMinutes",
    y=f"{agg_feature}_q10",
    # s=10,
    # hue="cycle",
    palette="Spectral",
)
sns.scatterplot(
    data=df_channel,
    x="timeDeltaMinutes",
    y=f"{agg_feature}_median",
    # s=10,
    # hue="cycle",
    palette="Spectral",
)
sns.scatterplot(
    data=df_channel,
    x="timeDeltaMinutes",
    y=f"{agg_feature}_q90",
    # s=10,
    # hue="cycle",
    palette="Spectral",
)

# ax.set_yscale('log')
# %%
image_root = r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\imgs"

df_multiscale_meta = (
    pl.read_parquet(
        r"C:\Users\hessm\Documents\Programming\Python\zfish\data\image_meta.parquet"
    )
    .with_columns(
        pl.lit(image_root).alias("image_root"),
        pl.col("root_path").str.split("\\").list.get(-1).alias("h5_file"),
    )
    .with_columns(
        pl.concat_str(pl.col("image_root"), pl.col("h5_file"), separator="\\").alias(
            "root_path"
        )
    )
    # .select(pl.exclude("image_root", "root_path"))
    .drop("image_root")
    .select(
        [
            pl.col(["root_path", "h5_file", "h5_path"]),
            pl.exclude(["root_path", "h5_file", "h5_path"]),
        ]
    )
    .with_columns(
        [
            pl.concat_str(
                [pl.col("stain"), pl.col("cycle").cast(pl.Utf8)], separator="."
            ).alias("channel")
        ]
    )
    .rename({"filename_prefix": "roi", "cycle": "acquisition"})
    .join(
        df_timepoints.select(["roi", "acquisition", "timeDeltaMinutes"]).with_columns(
            pl.col("roi").cast(pl.Utf8), pl.col("acquisition").cast(pl.Int64)
        ),
        on=["roi", "acquisition"],
    )
)

df_channel_meta = (
    df_multiscale_meta.filter(pl.col("level") == 0)
    .with_columns(
        pl.col("h5_path")
        .str.split("/")
        .list.slice(0, 2)
        .list.join("/")
        .alias("h5_group")
    )
    .select(
        [
            "roi",
            "h5_group",
            "channel",
            "stain",
            "acquisition",
            "wavelength",
            "timeDeltaMinutes",
            "root_path",
        ]
    )
)
# %% Fit models to data. After trying all three decicded on Exponential model on cycles 10, 11 and 12, mean-normalization per cycle.
mdl_colors = ["#dd966d", "#6ddd96", "#966ddd"]
scatter_colors = ["#8e8e8e"]

import numpy as np

# mld_types = [LinearModel, ExponentialModelNoOffset, ExponentialModel]
mdl_types = [Exp, Linear]
loss = "huber"
# y_name = "MeanCycleNorm_median"
y_name = "MeanCycleNorm"

to_plot = (
    df_clean.with_columns(
        (pl.col("Mean") / (pl.col("Mean").mean().over(["channel", "cycle"]))).alias(
            "MeanCycleNorm"
        )
    )
    .filter(pl.col("cycle") > 10)
    .to_pandas()
)  # .sample(100_000).to_pandas()

# channels = sorted(
#     list(to_plot["channel"].unique()),
#     key=lambda x: (int(x.split(".")[-1]), x.split(".")[0]),
# )

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
    models[channel] = dict()
    df_channel = to_plot[to_plot.channel == channel]
    X = df_channel[["timeDeltaMinutes"]]
    y = df_channel[y_name]
    X_fit = np.linspace(
        df_channel["timeDeltaMinutes"].min(), df_channel["timeDeltaMinutes"].max(), 200
    ).reshape(-1, 1)

    plt.sca(ax)
    if i % n_cols == 0:
        plt.ylabel("Mean Intensity [AU]")
    if i // n_cols == (n_rows - 1):
        plt.xlabel("Acquisition Time [min]")
    plt.title(f"{channel}")
    sns.scatterplot(
        x=X.values.flatten(),
        y=y.values,
        color=scatter_colors[0],
        s=10,
        label=f"{channel} Mean",
    )
    # sns.scatterplot(
    #     x=X.values.flatten(),
    #     y=y,
    #     hue=df_channel["cycle"],
    #     s=10,
    #     # label=f"{channel} Mean",
    #     palette="cividis_r",
    # )
    plt.ylim(bottom=0)

    for mdl_type, mdl_color in zip(mdl_types, mdl_colors):
        try:
            mdl = mdl_type(loss=loss).fit(X, y)
            y_mdl = mdl.predict(X_fit)
            plt.plot(
                X_fit, y_mdl, color=mdl_color, label=mdl_type.__name__, linewidth=2
            )
            models[channel][mdl_type.__name__] = mdl
        except RuntimeError:
            print(f"{mdl_type.__name__} not converged on {channel}")
            models[channel][mdl_type.__name__] = None
    plt.legend()

# %%
results = []
for ch in models.keys():
    res = df_channel_meta.filter(pl.col("channel") == ch)
    print(ch)
    for mdl_name in models[ch]:
        print(mdl_name)
        mdl = models[ch][mdl_name]
        new_values = mdl._correction_factor(res[["timeDeltaMinutes"]].to_numpy())
        new_column = pl.Series(name=f"correctionFactor-{mdl_name}", values=new_values)
        res = res.with_columns([new_column])
    results.append(res)
    print()
df_t_correction = pl.concat(results).with_columns(
    pl.col("correctionFactor-Exp").alias("correctionFactor")
)

# %%
df_t_correction_old = pl.read_parquet(
    r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\models\t_decay\correction_factors.parquet"
)
# %%
df_t_correction.sort(["channel", "timeDeltaMinutes"])
# %%
n_cols = 4
n_rows = len(channels) // n_cols

fig, axs = plt.subplots(
    n_rows, n_cols, sharex=False, sharey=False, squeeze=True, figsize=(20, 20), dpi=200
)
to_plot = df_t_correction.join(
    df_t_correction_old.select(
        ["roi", "channel", pl.col("correctionFactor-Exp").name.suffix("_old")]
    ),
    on=["roi", "channel"],
).to_pandas()

for i, (channel, ax) in enumerate(zip(channels, axs.flatten())):
    plt.sca(ax)

    df_channel = to_plot[to_plot.channel == channel]
    sns.scatterplot(df_channel, x="correctionFactor-Exp", y="correctionFactor-Exp_old")
    ax.set_aspect("equal")
    ax.set(title=channel)
# %%
channel = 'Pol-II-S2P.0'
feature = f"{channel}_Mean"

df_plot = (
    df.select(["roi", "object", "label", feature])
    .join(
        df_t_correction.filter(pl.col("channel") == channel).select(
            ["roi", "correctionFactor-Exp"]
        ),
        on="roi",
    )
    # .with_columns(pl.col(feature) * pl.col("correctionFactor-Exp"))
    .with_columns(pl.col("roi").str.split("_").list.get(0).alias("well"))
    .with_columns(pl.col("well").str.slice(0, 1).alias("row"))
)


selector = pl.col(feature).log()

ax = sns.kdeplot(
    df_plot.with_columns(selector),
    x=feature,
    hue="row",
    common_norm=False,
    palette="Spectral",
)

ax.set_xlim(*iqr_range(df.select(selector).to_series(), q_lower=0.1, q_upper=0.9, r=3))
# %%
import polars as pl
import tqdm

from zfish.roi.spatial_roi import Roi

df_t_correction = pl.read_parquet(r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\models\t_decay\correction_factors.parquet")

level = 3

selection = [
    "B02_px+0385_py-0060",
    "B02_px+1825_py+1851",
    "G03_px+1322_py-1474",
    "G03_px-1410_py+1399",
]

selection_paths = df_t_correction.filter(pl.col("roi").is_in(selection))[
    "root_path"
].unique()

selection_paths = df_t_correction[
    "root_path"
].unique()

lazy_rois = [Roi.from_file(p, level=level) for p in tqdm.tqdm(selection_paths)]
# %%
from zfish.roi.spatial_roi import apply_t_decay_factors

corr_rois = [apply_t_decay_factors(r, df_t_correction) for r in lazy_rois]

# %%

channels = [
    "DAPI.1",
    "H2B.2",
    "Pol-II-S5P.2",
]

raw_rois = [r.sel(c=channels).compute() for r in lazy_rois]
sel_rois = [r.sel(c=channels).compute() for r in corr_rois]

# %%
from zfish.visualize.grids import arrange_on_grid, rect_grid

images_raw = {r.name: r.images for r in raw_rois}
images = {r.name: r.images for r in sel_rois}

order = (
    df_meta.sort("log2_nucleiRaw3_Count")
    .select("roi")
    .with_row_count()
    .to_pandas()
    .set_index("roi")["row_nr"]
)
canvas = arrange_on_grid(images, order)
canvas_raw = arrange_on_grid(images_raw, order)

# %%
import napari

upper_limit = 2000

viewer = napari.Viewer()
viewer.add_image(canvas, colormap="inferno", contrast_limits=(0, upper_limit))
# viewer.add_image(canvas_linear, colormap="inferno", contrast_limits=(0, upper_limit))
viewer.add_image(canvas_raw, colormap="inferno", contrast_limits=(0, upper_limit))
# %%
# %%
from typing import Any

import xarray as xr
from numpy.typing import NDArray
from tqdm import tqdm

from zfish.features.types import SpatialImage
from zfish.image.h5_io import load_channel


def load_channels(
    df: pl.DataFrame, correction_factor: str | None = None
) -> dict[str, SpatialImage]:
    if correction_factor is None:
        df_iter = df.select(
            [pl.col("root_path"), pl.col("h5_path"), pl.col("roi")]
        ).with_columns([pl.lit(1.0).alias("correction_factor")])
    else:
        df_iter = df.select(
            [
                pl.col("root_path"),
                pl.col("h5_path"),
                pl.col("roi"),
                pl.col(correction_factor).alias("correction_factor"),
            ]
        )
    channels: dict[str, SpatialImage] = dict()
    for root_path, h5_path, roi, cf in tqdm(df_iter.iter_rows()):
        channels[roi] = (
            (load_channel(root_path, h5_path) * cf).expand_dims("t").to_numpy()
        )
    return channels


# %%
# %%

# channel = "Pol-II-S5P.2"
# channel = "DAPI.1"
channel = 'H2B.2'
level = 3
acquisition = int(channel.split(".")[1])

channels_exp = load_channels(
    results.filter(pl.col("level") == level).filter(pl.col("channel") == channel),
    "correctionFactor-Exp",
)
# channels_linear = load_channels(
#     results.filter(pl.col("level") == level).filter(pl.col("channel") == channel),
#     "correctionFactor-LinearModel",
# )
channels_raw = load_channels(
    results.filter(pl.col("level") == level).filter(pl.col("channel") == channel), None
)


# %%
order = (
    df_timepoints.filter(pl.col("acquisition") == acquisition)
    .to_pandas()
    .set_index("roi")["timeDeltaMinutes"]
)

canvas_raw = arrange_on_grid(channels_raw, order=order)
# canvas_linear = arrange_on_grid(
#     channels_linear,
#     order=order,
# )
canvas_exp = arrange_on_grid(channels_exp, order=order)



# %%
channels_raw[list(channels_raw.keys())[0]].shape
df_timepoints.to_pandas().set_index("roi")["timeDeltaMinutes"]

# %%
