# %%

import polars as pl
import polars.selectors as cs
from sklccp.ccp_pipeline_gridsearch_cluster import SCALERS, fit_pipeline, get_pipeline
from sklccp.ccp_transformers import PolarsSelector

from zfish.features.ccp.ccp_normalize_align import (
    cut_normalized_ccp,
    find_normalized_ccp_cuts,
)
from zfish.preprocessing.sliding_window_samples import stratified_sample

ANN_ORDER = ["M ana", "M telo", "S early", "S", "M pro", "M meta"]

# %% Load a pipeline
pip = get_pipeline()
pip

# %% All available parameters:
pip.get_params()

# %%
# Feature set for embedding
features = [
    "nucleiRaw3_PhysicalSize",
    "nucleiRaw3_Elongation",
    "nucleiRaw3_Flatness",
    # "nucleiRaw3_Roundness",
    "nucleiRaw3_FeretDiameter",
    "nucleiRaw3_Perimeter",
    "nucleiRaw3_EquivalentSphericalPerimeter",
    "nucleiRaw3_EquivalentSphericalRadius",
    "nucleiRaw3_EquivalentEllipsoidDiameter-a",
    "nucleiRaw3_EquivalentEllipsoidDiameter-b",
    "nucleiRaw3_EquivalentEllipsoidDiameter-c",
    "nucleiRaw3_PCNA.0_Mean",
    "nucleiRaw3_PCNA.0_Variance",
    "nucleiRaw3_PCNA.0_StandardDeviation",
    "nucleiRaw3_PCNA.0_Skewness",
    "nucleiRaw3_PCNA.0_Kurtosis",
    "nucleiRaw3_PCNA.0_Sum",
    "nucleiRaw3_DAPI.1_Mean",
    "nucleiRaw3_DAPI.1_Variance",
    "nucleiRaw3_DAPI.1_StandardDeviation",
    "nucleiRaw3_DAPI.1_Skewness",
    "nucleiRaw3_DAPI.1_Kurtosis",
    "nucleiRaw3_DAPI.1_Sum",
    # "nucleiRaw3_pH3.1_Mean", # excluded because of background
    # "nucleiRaw3_pH3.1_Variance",
    # "nucleiRaw3_pH3.1_StandardDeviation",
    # "nucleiRaw3_pH3.1_Skewness",
    # "nucleiRaw3_pH3.1_Kurtosis",
    # "nucleiRaw3_pH3.1_Sum",
]

# Preprocessing/Sampling Params (passed as kw args to fit_pipeline)
strata_params = {
    "exclude_celltypes": ["YSL"],  # exclude celltype from model
    "sample": "15k",  # "name" of a dataframe (sampled outside pipeline)
    "strata_ccp": "cycle_pooled",  # stratification of the ccp-embedding (division cycle 7-9 pooled)
    "strata_emb": None,  # stratification of the pre-embedding
}

# Pipeline params (set on the pipeline via pipeline.set_params(**params))
pipeline_params = {
    # ccp transformer params
    "ccpt__transformer__ccp_inference": "nearest",
    "ccpt__transformer__circle_lambda": 1.0,
    "ccpt__transformer__circle_mu": 1.0,
    "ccpt__transformer__n_spline_samples": 30_000,
    "ccpt__transformer__return_distance": True,
    # pre-embedding (UMAP) params
    "embedder__transformer__min_dist": 0.1,
    "embedder__transformer__spread": 1.0,
    "embedder__transformer__n_components": 3,
    "embedder__transformer__n_neighbors": 15,
    # feature scaler (needs to by sklearn compatible component i.e. `StandardScaler`)
    "scaler": SCALERS["power"],
    # feature selector
    "selector": PolarsSelector(selector=cs.by_name(features)),
}

# Set pipeline params
pip.set_params(**pipeline_params)
# %%
df_nuc = pl.read_parquet("features/aggregated/nucleiRaw3_CCP.parquet")
df_emb = pl.read_parquet("features/aggregated/embryoRaw_CCP.parquet")
df_ann = pl.read_parquet("features/aggregated/ann_cellcycle_clean.parquet")
df_full = df_nuc.join(df_emb, on="roi", how="left").join(
    df_ann, on=["roi", "o", "label"], how="left"
)

# %% Sampled Tables
dfs = {
    "15k": df_full.filter(stratified_sample(by="cycle", n=15_000, seed=42)),
    "45k": df_full.filter(stratified_sample(by="cycle", n=45_000, seed=42)),
    "full": df_full,
    "labeled": df_full.filter(pl.col("ann0").is_not_null()),
}

# %%
# df_model
res, pop, scores = fit_pipeline(
    pipeline=pip.set_params(**pipeline_params),
    dfs=dfs,
    n_scores=1,
    **strata_params,
)


# %%
from sklccp.ccp_align_and_shift import (
    align_ccps,
    align_ccps_better,
    normalize_ccp,
    plot_extrema,
    stitch_normalized_ccps,
)
from sklccp.ccp_plot import (
    plot_ccp_transformers,
    plot_umap_3d_projections,
)

from zfish.features.ccp.ccp_normalize_align import (
    _align,
    _find_all_turning_points,
    align,
)
from zfish.multi_table.schemas_v2 import sel
from zfish.plot.plot_commons import BASE, UMAP

# %%
df_res = (
    pl.concat(
        [
            df_full.select(sel.idx),
            res,
            df_full.select(
                "log2_nucleiRaw3__Count",
                # "log2_nuc__Count_corr",
                # "cycle_corr",
                # "cycle_sc",
                "ann0",
                # "dev_time",
                # "cycle_corr_pooled",
                # "cycle_sc_pooled",
                "cycle_pooled",
                "nucleiRaw3_PCNA.0_Mean",
                "nucleiRaw3_PhysicalSize",
                "nucleiRaw3_Roundness",
                "nucleiRaw3_DAPI.1_Mean",
                "nucleiRaw3_DAPI.1_Sum",
                "nucleiRaw3_PCNA.0_Sum",
                "nucleiRaw3_pH3.1_Mean",
                "nucleiRaw3_pH3.1_Sum",
                "nucleiRaw3_Pol-II-S2P.0_Mean",
                "nucleiRaw3_Pol-II-S5P.2_Mean",
                "nucleiRaw3_Pol-II-S2P.0_Sum",
                "nucleiRaw3_Pol-II-S5P.2_Sum",
                "nucleiRaw3_Pol-II-S2P.0_Skewness",
                "nucleiRaw3_Pol-II-S5P.2_Skewness",
            ),
        ],
        how="horizontal",
    ).with_row_index("index")
    # .join(
    #     pl.concat([e.with_row_index() for e in res_ccp]).rename(
    #         {
    #             "cycle_ccp": "cycle_sc_pooled",
    #             "CCP": "CCP_sc",
    #             "DistanceToCircleCCP": "DistanceToCircleCCP_sc",
    #         }
    #     ),
    #     on=("index", "cycle_sc_pooled"),
    # )
    # .join(
    #     df_all_train.select(
    #         pl.col("roi").cast(pl.String),
    #         pl.col("label").cast(pl.UInt64),
    #         pl.col("CCP").alias("_CCP"),
    #         pl.col("NormalizedCCP").alias("_NormalizedCCP"),
    #     ),
    #     on=["roi", "label"],
    # )
)
import seaborn as sns

ax = sns.kdeplot(
    df_res.to_pandas(),
    x="DistanceToCircleCCP",
    # hue="cycle_sc_pooled",
    hue="cycle_pooled",
    common_norm=False,
)
distance_cutoff = 4.0
distance_cutoff_12 = 0.8
ax.axvline(distance_cutoff, color="k", ls="--")
# ax.axvline(distance_cutoff_12, color="k", ls=":")

# %%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2)
features = "nucleiRaw3_PCNA.0_Mean"
cycles = range(9, 13)

with plt.style.context(BASE, UMAP):
    for i, cycle in enumerate(cycles):
        df_one = df_res.filter(pl.col("cycle_pooled") == cycle).sample(10_000)
        ax = axs.flatten()[i]
        plt.sca(ax)
        sns.scatterplot(
            df_one.to_pandas(),
            x="umap_0",
            y="umap_1",
            hue=features,
            s=1,
            alpha=0.3,
            legend=False,
            palette="turbo",
        )
    # plt.gca().set_aspect("equal")
    plt.show()


# %%
df_ccp, extrema = align(df_res, group_column="cycle_pooled")

df_res_clean = df_res.with_columns(df_ccp)
# %%
fig, axs = plt.subplots(2, 3, figsize=(6, 4), sharey=True, sharex=True)
f = "nucleiRaw3_Pol-II-S2P.0_Mean"
f = "nucleiRaw3_DAPI.1_Sum"
# f = "nucleiRaw3_PhysicalSize"
f = "nucleiRaw3_PCNA.0_Mean"
for cyc, ax in zip(range(9, 13), axs.flatten()):
    plt.sca(ax)
    ax = sns.scatterplot(
        # df_res.with_columns(df_res.select("NormalizedCCP", "DistanceToCircleCCP"))
        # .with_columns(df_ccp)
        df_res_clean.filter(pl.col("cycle_pooled") == cyc)
        .filter(stratified_sample("cycle_pooled", 10000))
        .to_pandas(),
        x="NormalizedCCP",
        y=f,
        hue="DistanceToCircleCCP",
        s=0.5,
        alpha=0.3,
        legend=False,
        palette="turbo",
    )
    for k, v in extrema[(cyc,)].items():
        if k != "division":
            # continue
            pass
        ax.axvline(v, color="k", ls="--", lw=0.4)
# ax.set_ylim(0, 3e6)
ax.set_yscale("log", base=10)
# %%

df_one


# %%
def dist_kdeplot(df, ax=None, subsample=10_000, sample_strata="cycle_corr"):
    if ax is None:
        fig, ax = plt.subplots()
    if subsample is not None:
        df_samp = df.filter(stratified_sample(sample_strata, subsample))
    else:
        df_samp = df
    ax = sns.kdeplot(
        df_samp.to_pandas(),
        x="DistanceToCircleCCP",
        cumulative=True,
    )
    ax.set_xlim((0, 1))
    return df


def quick_umap(
    df, hue="DistanceToCircleCCP", hue_norm=(0, 1), ax=None, palette="turbo"
):
    if ax is None:
        fig, ax = plt.subplots()
    ax = sns.scatterplot(
        df.to_pandas(),
        x="umap_0",
        y="umap_1",
        s=1,
        legend=False,
        alpha=0.5,
        palette=palette,
        hue=hue,
        hue_norm=hue_norm,
    )
    return df


# %%
# ax = sns.scatterplot(
#     df_plot.with_columns(pl.col("NormalizedCCP")),
#     x="umap_0",
#     y="umap_2",
#     s=1,
#     alpha=0.5,
#     hue="NormalizedCCP",
#     legend=False,
#     palette=cmaps.ccp.to_mpl(),
# )
# t_cycles = [e for e in [cycle + 1, cycle - 1] if e >= 9 and e <= 13]

# with plt.style.context(UMAP):
#     fig = plot_umap_3d_projections(
#         # df_res.join(df_model, on=["roi", "o", "label"], how="anti"),
#         df_res.join(df_model, on=["roi", "o", "label"], how="semi"),
#         pop,
#         cycle=13,
#         transformer_plot_cycles=tuple(range(9, 14)),
#         # transformer_plot_cycles=(),
#         # hue_norm=(0, 1),
#         hue_norm=hue_norm,
#         n_arrow_to_node=0.3,
#         fig_width=3,
#         s=0.8,
#         alpha=0.3,
#     )

#     ax.set_yscale("log", base=2)

#     ax = fig.get_axes()


# with plt.style.context(UMAP):
#     fig = plt.figure()
#     axs = fig.subplots(2, 2, width_ratios=(1.0, 0.5), height_ratios=(1.0, 0.5))
#     plt.sca(axs[0, 0])
def compute_range(df):
    extent = df.select(
        pl.max_horizontal(
            df.select(cs.starts_with("umap")).max()
            - df.select(cs.starts_with("umap")).min()
        ).ceil()
    )

    rng = (
        df.select(cs.starts_with("umap")).max()
        - df.select(cs.starts_with("umap")).min()
    )

    minimum = df.select(cs.starts_with("umap").min())
    maximum = df.select(cs.starts_with("umap").max())

    target = extent.with_columns(
        pl.col("umap_0").alias("umap_1"), pl.col("umap_0").alias("umap_2")
    )
    padding = (target - rng) / 2

    minima = (minimum - padding).transpose()
    maxima = (maximum + padding).transpose()

    return list(zip(minima["column_0"], maxima["column_0"]))


# %%

import plotly.express as px
import plotly.graph_objects as go
import polars.selectors as cs

from zfish.plot.plot_commons import BASE, cmaps

SAFE_FIG = False


def plot_3d(
    df,
    x="umap_0",
    y="umap_1",
    z="umap_2",
    name=None,
    sel=cs.all(),
    filt=pl.lit(True),
    hue="DistanceToCircleCCP",
):
    fig = px.scatter_3d(
        data_frame=df,
        x=x,
        y=y,
        z=z,
        color=hue,
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig.update_traces(marker=dict(size=1, opacity=0.3))
    return fig


df_plot_3d = (
    df_res.join(dfs["15k"], on=["roi", "o", "label"], how="semi")
    .filter(stratified_sample("cycle_pooled", 6854))
    .with_columns(pl.col("cycle_pooled").cast(pl.String).alias("division cycle"))
    .rename({"umap_0": "umap 0", "umap_1": "umap 1", "umap_2": "umap 2"})
)
fig = px.scatter_3d(
    df_plot_3d,
    x="umap 0",
    y="umap 1",
    z="umap 2",
    # color="DistanceToCircleCCP",
    color="division cycle",
    color_discrete_sequence=cmaps.division_cycle_hex_5,
    category_orders={"division cycle": ["9", "10", "11", "12", "13"]},
    # s=1,
)

fig.update_traces(marker=dict(size=1.5, opacity=0.3))
# fig.update_traces(legend_group=e)

for c, (n, ccpt) in zip(cmaps.division_cycle_hex_5, pop["ccpt"].transformers_.items()):
    fig.add_trace(
        go.Scatter3d(
            {
                **dict(zip(["x", "y", "z"], ccpt.spline_samples_[::100].T)),
            },
            name=f"ccpt {n[0]}",
            legendgroup=f"{n[0]}",
            marker=go.scatter3d.Marker(size=2, color=c),
            line=None,
        )
    )


rng_0, rng_1, rng_2 = compute_range(df_res)
fig.update_layout(
    scene=dict(
        camera=dict(
            eye=dict(x=-1.5, y=1.5, z=1.5),
            # zoom=0.5,
            # eye=dict(x=1.15, y=1.15, z=0.8)
        ),  # the default values are 1.25, 1.25, 1.25
        xaxis=dict(
            range=rng_0,
            ticks="",
            backgroundcolor="white",
            showbackground=True,
            showticklabels=False,
            gridcolor="rgb(200, 200, 200)",
        ),
        yaxis=dict(
            range=rng_1,
            backgroundcolor="white",
            showbackground=True,
            showticklabels=False,
            gridcolor="rgb(200, 200, 200)",
        ),
        zaxis=dict(
            range=rng_2,
            backgroundcolor="white",
            showbackground=True,
            showticklabels=False,
            gridcolor="rgb(200, 200, 200)",
        ),
        aspectmode="cube",  # this string can be 'data', 'cube', 'auto', 'manual'
        # a custom aspectratio is defined as follows:
        aspectratio=dict(x=1, y=1, z=1),
    ),
    font=dict(family="Bahnschrift", color="rgb(10, 10, 10)"),
    legend=dict(
        # x=0,
        y=0.5,
        # traceorder="reversed",
        # font_family="Bahnschrift",
        font=dict(family="Bahnschrift", size=12, color="black"),
        # bgcolor="LightSteelBlue",
        # bordercolor="Black",
        # borderwidth=2,
    ),
    width=700,
    height=450,
    margin=dict(r=0, l=0, b=0, t=0),
)

if SAFE_FIG:
    fig.write_html(tpaths.ccp / "ccp_umap_3d_with_ccpt.html")
fig.show(renderer="browser")

# %%

df_ann_here = df_res.filter(pl.col("ann0").is_not_null())
df_ann_color = df_ann_here.with_columns(
    pl.col("ann0")
    .replace_strict(dict(zip(ANN_ORDER, range(len(ANN_ORDER)))), return_dtype=pl.Int32)
    .alias("ann0int")
).sort("ann0int")

fig_ = px.scatter_3d(
    # df_res.join(df_model, on=["roi", "o", "label"], how="semi")
    # .filter(stratified_sample("cycle_corr_pooled", 6854, seed=42))
    # .with_columns(pl.col("cycle_corr_pooled").cast(pl.String).alias("division cycle"))
    df_ann_color.rename(
        {
            "umap_0": "umap 0",
            "umap_1": "umap 1",
            "umap_2": "umap 2",
            "ann0": "manual annotation",
        }
    ),
    x="umap 0",
    y="umap 1",
    z="umap 2",
    color="manual annotation",
    # color="division cycle",
    color_discrete_sequence=cmaps.ccp_hex,
    category_orders={"manual annotation": ANN_ORDER},
    # alpha=0.8,
    # s=1,
)

fig_.update_traces(marker=dict(size=4, opacity=1.0))
fig_.add_trace(
    go.Scatter3d(
        {
            ax: df_plot_3d[k].to_numpy()
            for ax, k in zip(["z", "y", "x"], ["umap 2", "umap 1", "umap 0"])
        },
        name="sampled cells",
        marker=go.scatter3d.Marker(size=1.0, color="gray", opacity=0.3),
        legendgroup="embedded points",
        mode="markers",
    )
)

fig_.update_layout(
    scene=dict(
        camera=dict(
            eye=dict(x=-1.5, y=1.5, z=1.5),
            # zoom=0.5,
            # eye=dict(x=1.15, y=1.15, z=0.8)
        ),  # the default values are 1.25, 1.25, 1.25
        xaxis=dict(
            range=rng_0,
            ticks="",
            backgroundcolor="white",
            showbackground=True,
            showticklabels=False,
            gridcolor="rgb(200, 200, 200)",
        ),
        yaxis=dict(
            range=rng_1,
            backgroundcolor="white",
            showbackground=True,
            showticklabels=False,
            gridcolor="rgb(200, 200, 200)",
        ),
        zaxis=dict(
            range=rng_2,
            backgroundcolor="white",
            showbackground=True,
            showticklabels=False,
            gridcolor="rgb(200, 200, 200)",
        ),
        aspectmode="cube",  # this string can be 'data', 'cube', 'auto', 'manual'
        # a custom aspectratio is defined as follows:
        aspectratio=dict(x=1, y=1, z=1),
    ),
    font=dict(family="Bahnschrift", color="rgb(10, 10, 10)"),
    legend=dict(
        # x=0,
        y=0.5,
        # traceorder="reversed",
        # font_family="Bahnschrift",
        font=dict(family="Bahnschrift", size=12, color="black"),
        # bgcolor="LightSteelBlue",
        # bordercolor="Black",
        # borderwidth=2,
    ),
    width=700,
    height=450,
    margin=dict(r=0, l=0, b=0, t=0),
)


fig_.update_layout()
# fig_.update_xaxes(font_family="Bahnschrift")
if SAFE_FIG:
    fig_.write_html(tpaths.ccp / "ccp_umap_3d_with_annotations.html")

fig_.show(renderer="browser")


# %%
cycle = 9
ax = sns.scatterplot(
    df_res_clean.filter(pl.col("cycle_pooled") == cycle).filter(
        stratified_sample("cycle_pooled", 50_000)
    ),
    x="NormalizedCCP",
    y="nucleiRaw3_PCNA.0_Mean",
    # y="nucleiRaw3_DAPI.1_Sum",
    s=2,
    alpha=0.3,
    c="0.7",
)
ax = sns.scatterplot(
    df_res_clean.filter(pl.col("cycle_pooled") == cycle),
    x="NormalizedCCP",
    y="nucleiRaw3_PCNA.0_Mean",
    # y="nucleiRaw3_DAPI.1_Sum",
    s=10,
    alpha=0.8,
    hue="ann0",
    hue_order=["M ana", "M telo", "S early", "S", "M pro", "M meta"],
    # palette="turbo",
    palette=cmaps.ccp_hex,
    legend=False,
)
ax.set_yscale("log", base=10)
ax.set_ylim(10**1, 10**3.5)
# plot_extrema(shifts[(10, )])
