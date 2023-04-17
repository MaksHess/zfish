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
)

pl.toggle_string_cache(True)

# %%
df_intensity = set_index_dtypes(
    pl.read_parquet(
        r"C:\Users\hessm\Documents\Programming\Python\zfish\data\features\intensity.parquet"
    )
)
df_timepoints = set_index_dtypes(
    pl.read_parquet(
        r"C:\Users\hessm\Documents\Programming\Python\zfish\data\acquisition_times.parquet"
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

df_alignment = stack_correlation_metric_by_acquisition(df_corr)


# %% Remove objects outsize of size range
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

df_label_clean = df_label.filter(
    pl.col("EquivalentSphericalRadius").is_between(*SPHERICAL_RADIUS_CUTOFF)
)

# %% Remove early stage embryos (< cycle 10)
control_wells = ["B07", "C07", "D07", "E07"]
df_meta = get_metadata(df_label_clean, control_wells=control_wells)

df_meta_clean = df_meta.filter(pl.col("cycle") > 9)
# df_meta_clean = df_meta

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
    .select(
        [
            pl.col(index),
            pl.col("Mean"),
        ]
    )
    .join(
        df_label_clean.select(
            obj_index + ["Centroid-z", "PhysicalSize", "EquivalentSphericalRadius"]
        ),
        on=obj_index,
    )
    .join(df_alignment, on=["roi", "structure", "label", "acquisition"])
    .join(
        df_timepoints.select(["roi", "acquisition", "timeDeltaMinutes"]),
        on=["roi", "acquisition"],
    )
    .join(df_meta_clean, on=["roi"])
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


# %% Aggregate intensity measurements per site
df_mean_per_embryo = (
    df_clean.groupby(["roi", "channel", "timeDeltaMinutes", "cycle"])
    .agg(pl.col("Mean").mean())
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

to_plot = df_mean_per_embryo.to_pandas()
# %%
df_channel = to_plot[to_plot.channel == "PCNA.0"]
sns.scatterplot(
    data=df_channel, x="timeDeltaMinutes", y="Mean", hue="cycle", palette="Set1"
)
# %%
df_channel_meta = pl.read_parquet(
    r"C:\Users\hessm\Documents\Programming\Python\zfish\data\image_meta.parquet"
)
df_channel_meta = set_index_dtypes(
    df_channel_meta.with_columns(
        [
            pl.concat_str(
                [pl.col("stain"), pl.col("cycle").cast(pl.Utf8)], separator="."
            ).alias("channel")
        ]
    ).rename({"filename_prefix": "roi", "cycle": "acquisition"})
)
df_channel_meta = df_channel_meta.join(
    df_timepoints.select(["roi", "acquisition", "timeDeltaMinutes"]),
    on=["roi", "acquisition"],
)

# %% Fit models to data. After trying all three decicded on Exponential model on cycles 10, 11 and 12, mean-normalization per cycle.
mdl_colors = ["#dd966d", "#6ddd96", "#966ddd"]
scatter_colors = ["#8e8e8e"]

import numpy as np

# mld_types = [LinearModel, ExponentialModelNoOffset, ExponentialModel]
mdl_types = [ExponentialModel]
loss = "linear"
y_name = "Mean_CycleMeanNorm"

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
results = pl.concat(results)

# %%
from typing import Any

from numpy.typing import NDArray
from tqdm import tqdm

from zfish.features.types import SpatialImage
from zfish.image.image import load_channel


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
from zfish.visualize.grids import arrange_on_grid

channel = "Pol-II-S5P.2"
level = 4
acquisition = int(channel.split(".")[1])

channels_exp = load_channels(
    results.filter(pl.col("level") == level).filter(pl.col("channel") == channel),
    "correctionFactor-ExponentialModel",
)
channels_linear = load_channels(
    results.filter(pl.col("level") == level).filter(pl.col("channel") == channel),
    "correctionFactor-LinearModel",
)
channels_raw = load_channels(
    results.filter(pl.col("level") == level).filter(pl.col("channel") == channel), None
)

order = (
    df_timepoints.filter(pl.col("acquisition") == acquisition)
    .to_pandas()
    .set_index("roi")["timeDeltaMinutes"]
)

canvas_raw = arrange_on_grid(channels_raw, order=order)
canvas_linear = arrange_on_grid(
    channels_linear,
    order=order,
)
canvas_exp = arrange_on_grid(channels_exp, order=order)

# %%
import napari

upper_limit = 500

viewer = napari.Viewer()
viewer.add_image(canvas_raw, colormap="inferno", contrast_limits=(0, upper_limit))
viewer.add_image(canvas_linear, colormap="inferno", contrast_limits=(0, upper_limit))
viewer.add_image(canvas_exp, colormap="inferno", contrast_limits=(0, upper_limit))

# %%
channels_raw[list(channels_raw.keys())[0]].shape
df_timepoints.to_pandas().set_index("roi")["timeDeltaMinutes"]

# %%
