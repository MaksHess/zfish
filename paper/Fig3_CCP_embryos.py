# %%
from functools import reduce
from operator import or_
from pathlib import Path

import cmap
import holoviews as hv
import hvplot.pandas  # noqa
import hvplot.polars  # noqa

# import arviz as az
# import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import seaborn as sns
from holoviews import opts
from sklccp.analyse_gridsearch_results.utils import feature_columns, plot, print_frame
from sklccp.ccp_align_and_shift import aligned_and_normalized_ccp
from sklccp.ccp_feature_normalization import (
    load_features_with_preprocessing,
)
from sklccp.ccp_plot import plot_ccp_pipeline, plot_ccp_transformer

from zfish.features.io import scan_tables
from zfish.features.polars_preprocessing import get_metadata
from zfish.features.polars_selector import sel
from zfish.features.polars_utils import read_table, unnest_all_structs
from zfish.plot.datashader import dsscatter
from zfish.plot.plot_commons import BASE, UMAP, cmaps, get_circular_palette
from zfish.plot.umap import plot_umap_ds
from zfish.preprocessing.sliding_window_samples import stratified_sample

# %%
gridsearch_best_results_fld = Path(
    r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\ccp_gridsearch\run4_best"
)

df_gridsearch_scores = pl.read_parquet(
    [f for f in sorted(gridsearch_best_results_fld.glob("*score.parquet"))]
)

single_score_selector = reduce(or_, [cs.matches(f"^score_{i}$") for i in range(7, 14)])
bg_columns = cs.expand_selector(df_gridsearch_scores, single_score_selector)

plot(
    df_gridsearch_scores.group_by("job_idx").agg(
        cs.numeric().median(), (~cs.numeric()).first()
    ),
    sort=["score_groups_mean"],
    hover_cols=["job_idx", "exclude_celltypes"],
)
# %%
fld_features = Path(
    r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\features_tcorr_v3_consolidated\z_model=Exp__TwoStep"
)

# df_nuc = load_features_with_preprocessing(fld_features)
dfs = scan_tables(fld_features)
# # fld_features = r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\features_tcorr_v3\Exp(features=MediumPath;EmbryoPath, loss=huber, pos_offset=False)"

df_nuc = (
    dfs["nucleiRaw3"]
    .select(
        (sel.index - sel.hierarchy)
        | sel.label & cs.numeric()
        | cs.by_name("Centroid")
        | sel.intensity & (cs.matches("_Mean") | cs.matches("_Sum"))
        | sel.density
        | cs.by_name(["ann0"])
        | cs.matches("pred")
    )
    .join(
        dfs["meta"]
        .select(
            "roi",
            "cycle",
            "cycle_pooled",
            "control_well_for_acquisition",
            "nucleiRaw3_Count",
        )
        .lazy(),
        on="roi",
    )
    # .collect()
).collect()


# %% Load embedding data for one of the best models
# Run3
# job_idx = 461  # 2D embedding
# job_idx = 1280  # 3D embedding

# Run 4
job_idxs = [
    1280,
    9359,
    9365,
    9368,
    9476,
    10169,
    10172,
    10280,
    10283,
    10286,
    10868,
    11033,
    11144,
    11160,
    11162,
    11189,
    11897,
    11919,
    11973,
    12942,
    15124,
    15232,
    15259,
    15283,
    15286,
    15367,
    15421,
    15424,
    15472,
    15475,
]
job_idx = job_idxs[10]
# 10172, 10283

index_map = {"()": "", "('YSL',)": "YSL", "('YSL', 'EVL')": "YSL, EVL"}

df_score = pl.read_parquet(gridsearch_best_results_fld / f"{job_idx:05}_score.parquet")
df_res = pl.read_parquet(gridsearch_best_results_fld / f"{job_idx:05}_res.parquet")
df_index = pl.read_parquet(
    gridsearch_best_results_fld
    / f"index_{index_map[df_score['exclude_celltypes'][0]]}.parquet"
)

# %%
df_ccp = (
    pl.concat([df_index, df_res], how="horizontal")
    .join(df_nuc, on=["roi", "object", "label"], how="left")
    .filter(pl.col("cycle") < 13)
    .filter(pl.col("cycle").is_not_null())
)
# df_ccp = df_ccp.with_columns(aligned_and_normalized_ccp(df_ccp))

print(df_ccp["cycle"].value_counts())
print(df_ccp["celltype_pred"].value_counts())

df_ccp.filter(stratified_sample("cycle", 10_000, seed=42)).hvplot.scatter(
    "CCP",
    # "NormalizedCCP",
    "PCNA.0_Mean",
    # "PhysicalSize",
    by="cycle",
    subplots=True,
    size=2,
    # logy=False,
    logy=True,
    # color="PCNA.0_Mean",
    color="PhysicalSize",
    hover_cols="roi",
).cols(1)
# %%
SHIFTS_RUN4 = {}


SHIFTS_RUN4[10868] = {
    9: (-4.95, False),
    10: (-3.5, True),
    11: (-1.51, True),
    12: (-3.648, True),
}
# %%
SHIFTS_RUN4[9368] = {  # ??
    9: (-3.155, False),
    10: (-3.27, True),
    11: (-3.268, False),
    12: (-3.392, False),
}

SHIFTS_RUN4[1280] = {
    9: (-4, False),
    10: (-2.12, True),
    11: (-2.3, False),
    12: (-1.87, False),
}

SHIFTS_RUN4[11162] = {
    9: (-5.8, False),
    10: (-0.15, True),
    11: (-0.0, False),
    12: (-2.5, False),
}

SHIFTS_RUN4[12942] = {
    9: (-5.8, False),
    10: (-0.15, True),
    11: (-0.0, False),
    12: (-2.5, False),
}

SHIFTS_RUN4[15259] = {
    9: (-3.45, True),
    10: (-2.9, False),
    11: (-3.55, True),
    12: (-2.51, False),
}

SHIFTS_RUN4[15283] = {
    9: (-3.52, True),
    10: (-3.0, False),
    11: (-1.6, True),
    12: (-3.775, True),
}

SHIFTS_RUN4[15286] = {
    9: (-3.52, True),
    10: (-2.69, False),
    11: (-3.55, True),
    12: (-2.76, False),
}

SHIFTS_RUN4[15367] = {
    9: (-3.025, False),
    10: (-3.16, False),
    11: (-3.4, True),
    12: (-1.64, True),
}

SHIFTS_RUN4[15472] = {
    9: (-3.376, True),
    10: (-3.362, True),
    11: (-3.4, True),
    12: (-3.64, False),
}
# %%
from sklccp.ccp_align_and_shift import SHIFTS_RUN3, apply_shift, stitch_normalized_ccps

feature = "PCNA.0_Mean"
color = "PhysicalSize"
# color = "PCNA.0_Mean"
# color = "RAD:100.0_Count"
feature = "PhysicalSize"
color = "PCNA.0_Mean"
# feature = "Roundness"
logy = False

df_ccp.with_columns(
    apply_shift(
        df_ccp,
        SHIFTS_RUN4[job_idx],
        group="cycle_pooled",
    )
).filter(stratified_sample("cycle", 10_000, seed=42)).with_columns(
    cs.matches("RAD.*_Count").sqrt()
).hvplot.scatter(
    "NormalizedCCP",
    feature,
    by="cycle",
    subplots=True,
    size=2,
    logy=logy,
    color=color,
    # color='celltype_pred',
).cols(
    1
)

# %%
feature = "PhysicalSize"
color = "PCNA.0_Mean"
logy = False

df_ccp.with_columns(
    apply_shift(
        df_ccp,
        SHIFTS_RUN4[job_idx],
        group="cycle_pooled",
    )
).filter(stratified_sample("cycle", 10_000, seed=42)).with_columns(
    cs.matches("RAD.*_Count").sqrt()
).pipe(
    stitch_normalized_ccps
).hvplot.scatter(
    "SingleCellPseudotime",
    feature,
    size=2,
    logy=logy,
    color=color,
    # color='celltype_pred',
    width=1000,
)


# %%
df_plot = (
    df_ccp.with_columns(
        apply_shift(
            df_ccp,
            SHIFTS_RUN4[job_idx],
            group="cycle_pooled",
        )
    )
    .filter(stratified_sample("cycle", 10_000, seed=42))
    .with_columns(cs.matches("RAD.*_Count").sqrt())
    .pipe(stitch_normalized_ccps)
)


df_live = pl.read_parquet(
    r"C:\Users\hessm\Documents\Programming\Python\tracking\tracking\tracks_clean.parquet"
).with_columns(
    (pl.col("delta_min") - pl.col("delta_min").min().over("ID")).alias("track_time"),
    pl.col('PhysicalSize') * (0.65 * 0.65),
    (pl.col('generation') + 6).alias('cycle'),
    
).with_columns(
    ((pl.col('track_time') / pl.col('track_time').max()) * 2 * np.pi).over('ID').alias('NormalizedCCP')
    ).filter(~pl.col('ID').is_in([3533]))
# %%
df_plot.with_columns(pl.col('nucleiRaw3_Count').log(2)).hvplot.scatter(x='SingleCellPseudotime', y='PhysicalSize', color='RAD:100.0_Count', hover_cols=['ID'], s=2)
# %%
# fixed = df_plot.hvplot.scatter(x="SingleCellPseudotime", y="EquivalentSphericalRadius", subplots=True, cmap="Set1", hover_cols=["ID"], s=1)
fixed_size = df_plot.hvplot.scatter(x="SingleCellPseudotime", y="EquivalentSphericalRadius", subplots=True, cmap="Set1", hover_cols=["ID"], s=1, alpha=0.3)
# %%
live_size = df_live.with_columns((pl.col('PhysicalSize') * 3 / np.pi / 4).cbrt().alias('EquivalentSphericalRadius')).filter(pl.col('cycle').is_between(7, 12)).pipe(stitch_normalized_ccps).hvplot.scatter(
    x="SingleCellPseudotime", y="EquivalentSphericalRadius", subplots=True, cmap="Set1", hover_cols=["ID"], color='k', s=4
)
# %%
size = fixed_size * live_size
size
# %%
# fixed = df_plot.hvplot.scatter(x="SingleCellPseudotime", y="EquivalentSphericalRadius", subplots=True, cmap="Set1", hover_cols=["ID"], s=1)
fixed_roundness = df_plot.with_columns(pl.col('Roundness') * 0.99).hvplot.scatter(x="SingleCellPseudotime", y="Roundness", subplots=True, cmap="Set1", hover_cols=["ID"], s=1, alpha=0.3)
# %%
live_roundness = df_live.with_columns((pl.col('PhysicalSize').cbrt().alias('EquivalentSphericalRadius'))).filter(pl.col('cycle').is_between(7, 12)).pipe(stitch_normalized_ccps).hvplot.scatter(
    x="SingleCellPseudotime", y="Roundness", subplots=True, cmap="Set1", hover_cols=["ID"], color='k', s=4
)
# %%
fixed_pcna = df_plot.with_columns((pl.col('PhysicalSize').cbrt().alias('EquivalentSphericalRadius'))).hvplot.scatter(x="SingleCellPseudotime", y="PCNA.0_Mean", logy=True, subplots=True, cmap="Set1", hover_cols=["ID"], s=1, alpha=0.3)
fixed_pol2 = df_plot.with_columns((pl.col('PhysicalSize').cbrt().alias('EquivalentSphericalRadius'))).hvplot.scatter(x="SingleCellPseudotime", y="Pol-II-S2P.0_Mean", logy=True, subplots=True, cmap="Set1", hover_cols=["ID"], s=1, alpha=0.3)

# %%
roundness = fixed_roundness * live_roundness
# %%
hvplot.extension('bokeh')
(size + roundness + fixed_pcna + fixed_pol2).opts(width=1000).cols(1)

# %%
# TODO: density, spatial asynchrony vs. density
# TODO: Supp movies colored by generation / root
# TODO: Global Asynchrony
# TODO: Local Asynchrony
# TODO: Mitotic waves

# sample_embryos = [
#     "B02_px+0385_py-0060",
#     "C04_px-0002_py+2310", 
#     "C04_px-2146_py+0224",
#     "C06_px-2276_py-0867",
#     "E04_px-1823_py-0854",
#     "E04_px-2424_py+0579",
#     "E05_px+2168_py+1386",
#     "E05_px-1798_py-0764",
#     "E06_px-0118_py-2462",
#     "E06_px-1759_py-0725",
#     "E07_px-0280_py-1306",
#     "F05_px+0166_py+1787",
#     "F05_px+1948_py-1778",
#     "F05_px-0015_py-1358",
#     "F05_px-1746_py-0396",
#     "G03_px+2426_py-0008",
#     "G03_px-0093_py-0054",
#     "G05_px-0370_py+1942",
# ]

# df_meta = get_metadata(df_nuc, control_wells=CONTROL_WELLS)
# df_emb = df_meta.join(
#     df_emb_raw, left_on=["roi", "parent.embryoRaw"], right_on=["roi", "label"]
# )
sample_embryos = (
    df_meta.filter(pl.col("control_well_for_acquisition") > 1)
    .filter(stratified_sample(by="cycle", n=3, seed=42))
    .filter(pl.col("cycle") < 13)
)["roi"].to_list()

sample_embryos_sml = (
    df_meta.filter(pl.col("control_well_for_acquisition") > 1)
    .filter(stratified_sample(by="cycle", n=3, seed=42))
    .filter(pl.col("cycle") < 13)
    .filter(stratified_sample(by="cycle", n=2, seed=42))
)["roi"].to_list()

# df_plot = df_nuc.join(df_meta, on=["roi"])
df_plot = df_ccp
df_one = df_plot.filter(pl.col("roi") == df_plot["roi"][0])
df_ten = df_plot.filter(pl.col("roi").is_in(df_meta["roi"][:10]))
df_sample = df_plot.filter(pl.col("roi").is_in(sample_embryos))
df_sample_sml = df_plot.filter(pl.col("roi").is_in(sample_embryos_sml))
# %% Fig 2J: Embryo Overview Celltype 3x6
import cmap
import napari
from napari.components import Camera
from napari.settings import get_settings

from zfish.visualize.napari import napari_centroids

# viewer.add_points(
#     data=df_one[["Centroid.z", "Centroid.y", "Centroid.x"]],
#     features=df_one[["celltype_pred"]].to_pandas(),
#     face_color="celltype_pred",
#     face_color_cycle=cmap.Colormap('set1').to_mpl().colors,
# )

angles = (0.0, 0.0, 135.0)


viewer = napari.Viewer(ndisplay=3, axis_labels=["z", "y", "x"])
# settings = get_settings()
# settings.appearance.theme = 'light'
# settings.application.window_fullscreen = True

viewer.add_points(
    **napari_centroids(
        df_sample,
        # features=["CCP"],
        features=["NormalizedCCP"],
        translate_group="roi",
        translate_n_rows=6,
        translate_sort="nucleiRaw3_Count",
        edge_color="black",
        edge_width=0,
        size=16,
        face_colormap=cmaps.ccp.to_mpl(),
        face_contrast_limits=(0, 2 * np.pi),
    )
)

viewer.camera.angles = angles

# %% Fig 2J: Embryo Overview Celltype 4x3
import cmap
import napari
from napari.components import Camera
from napari.settings import get_settings

# from zfish.visualize.napari import napari_centroids

# viewer.add_points(
#     data=df_one[["Centroid.z", "Centroid.y", "Centroid.x"]],
#     features=df_one[["celltype_pred"]].to_pandas(),
#     face_color="celltype_pred",
#     face_color_cycle=cmap.Colormap('set1').to_mpl().colors,
# )


viewer = napari.Viewer(ndisplay=3, axis_labels=["z", "y", "x"])
# settings = get_settings()
# settings.appearance.theme = 'light'
# settings.application.window_fullscreen = True

viewer.add_points(
    **napari_centroids(
        df_sample_sml,
        features=["NormalizedCCP"],
        translate_group="roi",
        translate_n_rows=3,
        translate_sort="nucleiRaw3_Count",
        edge_color="black",
        edge_width=0,
        size=16,
        face_colormap=cmaps.ccp.to_mpl(),
        face_contrast_limits=(0, 2 * np.pi),
    )
)

viewer.camera.angles = angles
# %% Fig 3: Single Embryo with Image
from zfish.roi.spatial_roi import Roi

fn = r"C:\Users\hessm\Documents\zfish_local\imgs\B02_px+0198_py-2152.h5"
lazy_roi = Roi.from_file(fn, level=1)
roi = lazy_roi.sel(l="nucleiRaw3", c=["DAPI.1", "bCatenin.1", "PCNA.0"]).compute()


def show_roi_and_ccp(roi, df_roi):
    viewer = napari.Viewer()
    imshow_roi(roi, viewer=viewer)
    viewer.add_points(
        **napari_centroids(
            df_roi,
            features=["NormalizedCCP"],
            translate_group="roi",
            translate_n_rows=3,
            translate_sort="nucleiRaw3_Count",
            edge_color="black",
            edge_width=0,
            size=16,
            face_colormap=cmaps.ccp.to_mpl(),
            face_contrast_limits=(0, 2 * np.pi),
            opacity=0.3,
            out_of_slice_display=True,
        )
    )


# %%
scatter_c8 = (
    df_plot.filter(pl.col("cycle") == 8)
    .hvplot.scatter(
        x="NormalizedCCP",
        y="PCNA.0_Mean",
        # y="PhysicalSize",
        # y='Roundness',
        color="NormalizedCCP",
        cmap=cmaps.ccp.to_mpl(),
        size=1,
        logy=True,
    )
    .opts(clim=(0, 2 * np.pi))
)


# %%
import hvplot.polars

from zfish.roi.visualize import imshow_roi

df_roi = df_plot.filter(pl.col("roi") == roi.name)

scatter_roi = df_roi.hvplot.scatter(
    "NormalizedCCP",
    "PCNA.0_Mean",
    # clim=(0, 1.0),
    xlim=(0, 2 * np.pi),
    color="black",
    s=8,
    # color="NormalizedCCP",
    # cmap=cmaps.ccp.to_mpl(),
).opts(clim=(0, 2 * np.pi))

scatter_bg = (
    df_plot.filter(pl.col("cycle") == df_roi["cycle"].unique())
    .hvplot.scatter(
        x="NormalizedCCP",
        y="PCNA.0_Mean",
        # y="PhysicalSize",
        # y='Roundness',
        color="NormalizedCCP",
        cmap=cmaps.ccp.to_mpl(),
        size=1,
        logy=True,
    )
    .opts(clim=(0, 2 * np.pi))
)
scatter_bg * scatter_roi
# %%
viewer = napari.Viewer()
imshow_roi(roi, viewer=viewer)
viewer.add_points(
    **napari_centroids(
        df_roi,
        features=["NormalizedCCP"],
        translate_group="roi",
        translate_n_rows=3,
        translate_sort="nucleiRaw3_Count",
        edge_color="black",
        edge_width=0,
        size=16,
        face_colormap=cmaps.ccp.to_mpl(),
        face_contrast_limits=(0, 2 * np.pi),
    )
)


# %% Another single embryo
img_fld = Path(r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\imgs")
roi_name = r"F05_px+0166_py+1787"

df_roi = df_plot.filter(pl.col("roi") == roi_name)

y_col = "PCNA.0_Mean"
scatter_roi = df_roi.hvplot.scatter(
    "NormalizedCCP",
    y_col,
    # clim=(0, 1.0),
    xlim=(0, 2 * np.pi),
    color="black",
    s=8,
    # color="NormalizedCCP",
    # cmap=cmaps.ccp.to_mpl(),
).opts(clim=(0, 2 * np.pi))
scatter_bg = (
    df_plot.filter(pl.col("cycle") == df_roi["cycle"].unique())
    .hvplot.scatter(
        x="NormalizedCCP",
        y=y_col,
        # y="PhysicalSize",
        # y='Roundness',
        color="NormalizedCCP",
        cmap=cmaps.ccp.to_mpl(),
        size=1,
        logy=True,
    )
    .opts(clim=(0, 2 * np.pi))
)
scatter_bg * scatter_roi
# %%
fn_roi = img_fld / f"{roi_name}.h5"
lazy_roi = Roi.from_file(fn_roi, level=1)
# lazy_roi = Roi.from_file(fn, level=1)
roi = (
    lazy_roi.sel(l="nucleiRaw3", c=["DAPI.1", "bCatenin.1", "PCNA.0"])
    .drop_dim("l")
    .compute()
)

# %%
viewer = napari.Viewer()
imshow_roi(roi, viewer)
viewer.add_points(
    **napari_centroids(
        df_roi,
        features=["NormalizedCCP"],
        translate_group="roi",
        translate_n_rows=3,
        translate_sort="nucleiRaw3_Count",
        edge_color="black",
        edge_width=0,
        size=16,
        face_colormap=cmaps.ccp.to_mpl(),
        face_contrast_limits=(0, 2 * np.pi),
    )
)
# %%

# %% Fig 2G: nuc/cell volume per embryo
with plt.style.context(STYLE_SHEET):
    # fig, ax = plt.subplots(figsize=(2, 1.5), dpi=300)
    fig, ax = plt.subplots()
    # %matplotlib inline
    df_plot = (
        df_emb.filter(pl.col("control_well_for_acquisition") > 1)
        .join(
            df_nuc.group_by(["roi"]).agg(
                pl.col("PhysicalSize").mean().alias("Mean Nuclear Volume")
            ),
            on="roi",
        )
        .join(
            df_cell.group_by(["roi"]).agg(
                pl.col("PhysicalSize").mean().alias("Mean Cell Volume")
            ),
            on="roi",
        )
    )

    sns.scatterplot(
        x=df_plot["nucleiRaw3_Count"],
        y=df_plot["Mean Nuclear Volume"],
        label="nuclei",
        s=5,
        alpha=0.8,
    )
    sns.scatterplot(
        x=df_plot["nucleiRaw3_Count"],
        y=df_plot["Mean Cell Volume"],
        label="cells",
        s=5,
        alpha=0.8,
    )
    ax.set_xlabel("Nuclei Count")
    ax.set_ylabel("Mean Volume $[\mu m^3]$")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_title("Mean volume per embryo")
    ax.set_xticks([2**i for i in range(7, 13)])
    ax.legend()


# top=0.85,
# bottom=0.25,
# left=0.2,
# right=0.95,
# hspace=0.1,
# wspace=0.1
# %% Fig 2H: DNA vs PCNA
df_plot = (
    df_nuc.select(
        [
            "roi",
            "object",
            "label",
            "DAPI.1_Mean",
            "DAPI.1_Sum",
            "PCNA.0_Mean",
            "PCNA.0_Sum",
        ]
    )
    .join(df_emb, on="roi", how="left")
    .filter(pl.col("cycle") == 10)
    # .sample(10_000, seed=42)
    # .join(df_emb, on="roi", how="left")
)

with plt.style.context(STYLE_SHEET):
    # fig, ax = plt.subplots(figsize=(2, 1.5), dpi=300)
    fig, ax = plt.subplots()
    sns.scatterplot(
        df_plot.to_pandas(),
        x="DAPI.1_Sum",
        y="PCNA.0_Mean",
        # hue="cycle",
        s=0.5,
        alpha=0.4,
        palette="viridis",
        legend=False,
    )
    ax.set_title("DNA content vs PCNA concentration")
    # ax.set_xlim(0, 2.8e6)
    ax.set_yscale("log")
    ax.set_ylim(1.2, 9_000)
    ax.set_xscale("log", base=2)
    ax.set_xlim(2**18.5, 2**21.5)
    # ax.set_xlim(2**17, 2**21.2)
    ax.set_xlabel("DAPI Sum Intensity [AU]")
    ax.set_ylabel("PCNA Mean Intensity [AU]")

# %% Fig 2I: Density vs cellcount
with plt.style.context(STYLE_SHEET):
    # fig, ax = plt.subplots(figsize=(2, 1.5), dpi=300)
    fig, ax = plt.subplots()
    # %matplotlib inline
    df_plot = (
        df_emb.filter(pl.col("control_well_for_acquisition") > 1)
        .join(
            df_nuc.group_by(["roi"]).agg(
                pl.col("PhysicalSize").mean().alias("Mean Nuclear Volume")
            ),
            on="roi",
        )
        .join(
            df_cell.group_by(["roi"]).agg(
                pl.col("PhysicalSize").mean().alias("Mean Cell Volume")
            ),
            on="roi",
        )
    )

    sns.scatterplot(
        x=df_plot["nucleiRaw3_Count"],
        y=df_plot["Mean Nuclear Volume"],
        label="nuclei",
        s=5,
        alpha=0.8,
    )
    sns.scatterplot(
        x=df_plot["nucleiRaw3_Count"],
        y=df_plot["Mean Cell Volume"],
        label="cells",
        s=5,
        alpha=0.8,
    )
    ax.set_xlabel("Nuclei Count")
    ax.set_ylabel("Mean Volume $[\mu m^3]$")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_title("Mean volume per embryo")
    ax.set_xticks([2**i for i in range(7, 13)])
    ax.legend()

# %%
from zfish.features.polars_selector import sel

df_nuc.select(sel.density)
# %%
df_nuc.join(df_emb, on="roi").sort("cycle").hvplot.kde("KNNd:50_Mean", by="cycle")
# %%
density = "KNNd:50_Mean"
density = "RAD:250.0_Count"
density = "RAD:200.0_Count"
df_plot = (
    df_nuc.group_by(pl.col("roi"))
    .agg(
        pl.col(density).median(),
        (pl.col(density).median() - pl.col(density).quantile(0.25)).name.suffix(
            "q0.25"
        ),
        (pl.col(density).quantile(0.75) - pl.col(density).median()).name.suffix(
            "q0.75"
        ),
    )
    .join(df_emb, on="roi")
    .sort("cycle")
)
scatter = df_plot.hvplot.scatter("log2_nucleiRaw3_Count", y=density, size=4)
# scatter_low = df_plot.hvplot.scatter('log2_nucleiRaw3_Count', f"{density}q0.9", size=2)
error = df_plot.hvplot.errorbars(
    x="log2_nucleiRaw3_Count",
    y=density,
    yerr1=f"{density}q0.25",
    yerr2=f"{density}q0.75",
    size=2,
)
# %%
(scatter * error).opts(logy=True)  # .opts(opts.ErrorBars(line_alpha=0.5))
# %%
scatter = df_plot.hvplot.scatter(
    "log2_nucleiRaw3_Count",
    y=density,
    transforms={density: np.log(hv.dim(density))},
    size=4,
)
scatter
# %%

scatter = df_plot.hvplot.scatter(
    "log2_nucleiRaw3_Count",
    y=density,
    transforms={density: hv.dim(density) ** (1 / 3)},
    size=4,
)
scatter
# %%

scatter = df_plot.hvplot.scatter(
    "nucleiRaw3_Count",
    y=density,
    transforms={density: hv.dim(density) ** (1 / 1)},
    size=4,
)
scatter
