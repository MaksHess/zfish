# %%
from functools import reduce
from operator import or_
from pathlib import Path

import cmap

# import arviz as az
# import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import seaborn as sns
from zfish.plot.plot_commons import BASE, UMAP, cmaps, get_circular_palette
from sklccp.analyse_gridsearch_results.utils import feature_columns, plot, print_frame
from sklccp.ccp_feature_normalization import add_pooled_cycle
from sklccp.ccp_plot import plot_ccp_pipeline, plot_ccp_transformer

from zfish.features.polars_preprocessing import get_metadata
from zfish.features.polars_utils import read_table
from zfish.plot.datashader import dsscatter
from zfish.plot.umap import plot_umap_ds

# %%
gridsearch_best_results_fld = Path(
    r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\ccp_gridsearch\run3_best2"
)

df_gridsearch_scores = pl.read_parquet(
    [f for f in sorted(gridsearch_best_results_fld.glob("*score.parquet"))], use_pyarrow=True
)

single_score_selector = reduce(or_, [cs.matches(f"^score_{i}$") for i in range(7, 14)])
bg_columns = cs.expand_selector(df_gridsearch_scores, single_score_selector)


# %%
# fld_features = r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\features_tcorr_v3\Exp(features=MediumPath;EmbryoPath, loss=huber, pos_offset=False)"

# df_nuc_raw = read_table(fld_features, _object='nucleiRaw3')
# %%
# df_nuc.write_parquet('quicksave.parquet')
df_nuc_raw = pl.read_parquet(
    r"C:\Users\hessm\Documents\Programming\Python\scikit-ccp\sklccp\quicksave.parquet"
)
# %%
df_debris = pl.read_parquet(
    r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\classifiers\predictions\debris_pred.parquet"
)
df_celltype = pl.read_parquet(
    r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\classifiers\predictions\celltype_pred.parquet"
)
df_nuc_no_debris = df_nuc_raw.join(df_debris, on=["roi", "object", "label"]).filter(
    pl.col("debris_proba") < 0.3
)
df_meta = get_metadata(
    df_nuc_no_debris, control_wells=["B07", "C07", "D07", "E07"]
).pipe(add_pooled_cycle)

df_nuc = df_nuc_no_debris.join(df_meta.drop("parent.embryoRaw"), on=["roi"]).join(
    df_celltype, on=["roi", "object", "label"]
)
df_ann = (
    pl.read_parquet(
        r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\classifier\ann_cellcycle_clean.parquet"
    )
    .filter(pl.col("ann0int") < 6)
    .with_columns(pl.col("label").cast(pl.Int64()))
    .join(
        df_nuc.select(["roi", "object", "label", "cycle", "cycle_pooled"]),
        on=["roi", "object", "label"],
        how="left",
    )
).drop_nulls("cycle_pooled")


# %% Load embedding data for one of the best models
job_idx = 461  # 2D embedding
# job_idx = 1280  # 3D embedding

df_score = pl.read_parquet(gridsearch_best_results_fld / f"{job_idx:04}_score.parquet")
df_index = pl.read_parquet(gridsearch_best_results_fld / "index.parquet")
df_res = pl.read_parquet(gridsearch_best_results_fld / f"{job_idx:04}_res.parquet")

# %%
df_ccp = (
    pl.concat([df_index, df_res], how="horizontal")
    .join(df_nuc, on=df_index.columns[1:], how="left")
    .filter(pl.col("cycle") < 13)
    .filter(pl.col("cycle").is_not_null())
)
print(df_ccp["cycle"].value_counts())
print(df_ccp["celltype_pred"].value_counts())


# %% 2D Plot setup

from zfish.preprocessing.sliding_window_samples import stratified_sample

feature_selector = (
    cs.matches("umap")
    | cs.matches("CCP")
    | cs.matches("cycle")
    | cs.matches("debris")
    | cs.matches("celltype")
    | cs.matches("nucleiRaw3")
)

df_plot = df_ccp.filter(stratified_sample(by="cycle", n=10_000)).select(
    ["roi", "object", "label", feature_selector]
)

df_plot_celltype = df_ccp.filter(
    stratified_sample(by="celltype_pred", n=15_000)
).select(["roi", "object", "label", feature_selector])

# %% UMAP division cycle (time) datashader full

# %matplotlib inline
plt.style.use([BASE, UMAP])
# plt.style.use('default')

fig, ax = plt.subplots(dpi=300)

artist = dsscatter(
    df_ccp.with_columns(pl.col("cycle_pooled_name")),
    x="umap_0",
    y="umap_1",
    # hue="cycle_pooled",
    hue="cycle_pooled",
    spread_via="agg_hook_min",
    # is_categorical=True,
    is_categorical=False,
    ax=ax,
    palette=cmaps.division_cycle.to_mpl(),
    # palette=cmaps.division_cycle_hex_4,
    color_key=cmaps.division_cycle_hex_4,
    spread_px=0,
)

ax.set_aspect("equal")
# ax.legend(artist.get_legend_elements())
# ['#9E0142', '#FDBF6F', '#BFE5A0', '#5E4FA2']
# %% UMAP division cycle (time) subsampled
# %matplotlib inline
plt.style.use([BASE, UMAP])
# plt.style.use('default')

fig, ax = plt.subplots(dpi=300)

sns.scatterplot(
    df_plot.filter(pl.col("cycle") < 20).to_pandas(),
    x="umap_0",
    y="umap_1",
    hue="cycle_pooled",
    # hue='log2_nucleiRaw3_Count',
    # hue_order=["12", "11", "10", "7-8-9"],
    ax=ax,
    # palette=list(cmaps.division_cycle.lut(6)),
    palette=cmaps.division_cycle.to_mpl(),
    legend=False,
    s=0.5,
    alpha=0.4,
)
# sns.move_legend(ax, loc='bottom_right', marker_scale=10)
ax.set_aspect("equal")

# LEGEND!!!
# leg = plt.legend()
# for lh in leg.legendHandles:
#     lh.set_alpha(0.7)

# sns.move_legend(
#     ax,
#     "upper left",
#     bbox_to_anchor=(1, 1),
#     markerscale=6,
#     title=None,
#     labels=["7-9", "10", "11", "12"],
#     frameon=False,
# )

# %% UMAP vs Celltype

fig, ax = plt.subplots(dpi=300)

sns.scatterplot(
    df_plot_celltype.to_pandas(),
    x="umap_0",
    y="umap_1",
    hue="celltype_pred",
    # hue='log2_nucleiRaw3_Count',
    # hue_order=["12", "11", "10", "7-8-9"],
    ax=ax,
    # palette=list(cmaps.division_cycle.lut(6)),
    palette=cmaps.celltype_hex,
    legend=False,
    s=0.3,
    alpha=0.4,
)
# sns.move_legend(ax, loc='bottom_right', marker_scale=10)
ax.set_aspect("equal")

# # LEGEND!!!
# leg = plt.legend()
# for lh in leg.legendHandles:
#     lh.set_alpha(0.7)

# sns.move_legend(
#     ax,
#     "upper left",
#     bbox_to_anchor=(1, 1),
#     markerscale=6,
#     title=None,
#     # labels=["7-9", "10", "11", "12"],
#     frameon=False,
# )


# %% Annotations all in one
fig, ax = plt.subplots()
plot_umap_ds(
    x=df_ccp["umap_0"],
    y=df_ccp["umap_1"],
    hue=df_ccp.join(
        df_ann.select(["roi", "object", "label", "ann0", "ann0int"]),
        on=["roi", "object", "label"],
        how="left",
    )["ann0int"],
    ax=ax,
    s=5,
    alpha=0.8,
    palette=get_circular_palette(base_palette="cet_CET_C9s", n_out=0),
    legend=False,
    dsscatter_kwargs=dict(norm="eq_hist", palette=["0.8", "0.3"]),
)
ax.set_aspect("equal")
# %% Annotations all in one with legend
fig, ax = plt.subplots()
plot_umap_ds(
    x=df_ccp["umap_0"],
    y=df_ccp["umap_1"],
    hue=df_ccp.join(
        df_ann.select(["roi", "object", "label", "ann0", "ann0int"]),
        on=["roi", "object", "label"],
        how="left",
    )["ann0"],
    hue_order=["M pro", "M meta", "M ana", "M telo", "S early", "S"],
    ax=ax,
    s=6,
    palette=get_circular_palette(base_palette="cet_CET_C9s", n_out=0),
    legend=True,
    dsscatter_kwargs=dict(norm="eq_hist", palette=["0.8", "0.3"]),
)
ax.set_aspect("equal")

# %% Annotations per group
df_this_plot = df_ccp.join(
    df_ann.select(["roi", "object", "label", "ann0", "ann0int"]),
    on=["roi", "object", "label"],
    how="left",
)
with plt.style.context([BASE, UMAP]):
    # fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(3.6, 2.8))
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(3.6, 2.8))
    for group, ax in zip(["7-8-9", "10", "11", "12"], axs.flatten()):
        df_group = df_this_plot.filter(pl.col("cycle_pooled_name") == group)
        plot_umap_ds(
            x=df_group["umap_0"],
            y=df_group["umap_1"],
            hue=df_group.join(
                df_ann.select(["roi", "object", "label", "ann0", "ann0int"]),
                on=["roi", "object", "label"],
                how="left",
            )["ann0"],
            hue_order=["M pro", "M meta", "M ana", "M telo", "S early", "S"],
            ax=ax,
            s=5,
            alpha=0.9,
            palette=get_circular_palette(base_palette="cet_CET_C9s", n_out=0),
            legend=False,
            title=group,
            dsscatter_kwargs={'norm': 'eq_hist', 'palette': ['0.8', '0.3']}
        )
        ax.set_aspect("equal")

