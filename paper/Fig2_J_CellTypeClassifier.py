# %%
import holoviews as hv
import hvplot.pandas  # noqa
import hvplot.polars  # noqa
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from holoviews import opts

from zfish.features.polars_preprocessing import get_metadata
from zfish.features.polars_selector import sel
from zfish.features.polars_utils import read_table, unnest_all_structs
from zfish.preprocessing.sliding_window_samples import stratified_sample

hv.extension("bokeh", "matplotlib")
# %%
# %matplotlib inline
# %config InlineBackend.print_figure_kwargs = {'bbox_inches':None}
# %%
STYLE_SHEET = (
    r"C:\Users\hessm\Documents\Programming\Python\zfish\paper\mystyle.mplstyle"
)
df_debris = pl.read_parquet(
    r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\classifiers\predictions\debris_pred.parquet"
)
df_celltype = pl.read_parquet(
    r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\classifiers\predictions\celltype_pred.parquet"
)
df_nuc_raw = (
    read_table(
        r"C:\Users\hessm\Documents\zfish_local\features_tcorr\Linear(loss='huber', features=['MediumPath', 'EmbryoPath'])",
        _object="nucleiRaw3",
    )
    .join(df_celltype, on=["roi", "object", "label"], how="left")
    .join(df_debris, on=["roi", "object", "label"], how="left")
)
df_emb_raw = read_table(
    r"C:\Users\hessm\Documents\zfish_local\features_tcorr\Linear(loss='huber', features=['MediumPath', 'EmbryoPath'])",
    _object="embryoRaw",
)
# %%


df_nuc = df_nuc_raw.filter(pl.col("debris_pred") == False)
# %%
CONTROL_WELLS = ["B07", "C07", "D07", "E07"]

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

df_meta = get_metadata(df_nuc, control_wells=CONTROL_WELLS)
df_emb = df_meta.join(
    df_emb_raw, left_on=["roi", "parent.embryoRaw"], right_on=["roi", "label"]
)
sample_embryos = (
    df_meta.filter(pl.col("control_well_for_acquisition") > 1)
    .filter(stratified_sample(by="cycle", n=3, seed=42))
    .filter(pl.col("cycle") < 13)
)["roi"].to_list()

sample_embryos_sml = (
    df_meta.filter(pl.col("control_well_for_acquisition") > 1)
    .filter(stratified_sample(by="cycle", n=3, seed=42))
    .filter(pl.col("cycle") < 13)
    .filter(stratified_sample(by='cycle', n=2, seed=42))
)["roi"].to_list()

df_plot = df_nuc.join(df_meta, on=["roi"])
df_one = df_plot.filter(pl.col("roi") == df_plot["roi"][0])
df_ten = df_plot.filter(pl.col("roi").is_in(df_meta["roi"][:10]))
df_sample = df_plot.filter(pl.col("roi").is_in(sample_embryos))
df_sample_sml = df_plot.filter(pl.col('roi').is_in(sample_embryos_sml))
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

angles=(0.0, 0.0, 135.0)



viewer = napari.Viewer(ndisplay=3, axis_labels=["z", "y", "x"])
# settings = get_settings()
# settings.appearance.theme = 'light'
# settings.application.window_fullscreen = True

viewer.add_points(
    **napari_centroids(
        df_sample,
        features=["celltype_pred"],
        translate_group="roi",
        translate_n_rows=6,
        translate_sort="nucleiRaw3_Count",
        edge_color='black',
        edge_width=0,
        size=16,
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
        features=["celltype_pred"],
        translate_group="roi",
        translate_n_rows=3,
        translate_sort="nucleiRaw3_Count",
        edge_color='black',
        edge_width=0,
        size=16,
    )
)

viewer.camera.angles = angles
# %% Fig 2F: staging by cellcount
# from zfish.plot.features import plot_embryos_by_age

with plt.style.context(STYLE_SHEET):
    # fig, ax = plt.subplots(figsize=(2, 1.5), dpi=300)
    fig, ax = plt.subplots(figsize=(1.8, 1.6))
    # plot_embryos_by_age(df_emb, ax=ax)

    Y_FEATURE = "rank"
    X_FEATURE = "nucleiRaw3_Count"
    to_plot = df_meta.sort(X_FEATURE).with_row_count("rank")

    if ax is None:
        ax = plt.gca()
    ax.plot(to_plot[X_FEATURE], to_plot[Y_FEATURE], color="0.3", lw=0.5)

    scatterplot_kwargs = {}
    default_kwargs = {
        "data": to_plot.with_columns(pl.col("cycle").clip_max(12)).to_pandas(),
        "x": X_FEATURE,
        "y": Y_FEATURE,
        "hue": "cycle",
        "palette": "viridis",
        "zorder": 10,
        "s": 3,
        "alpha": 0.7,
    }
    sns.scatterplot(
        **{
            **default_kwargs,
            **scatterplot_kwargs,
        }
    )
    sns.move_legend(ax, loc="best", markerscale=2.0)
    ax.set_ylabel(f"Nuclei Count Rank (n={len(to_plot)})")
    ax.set_xlabel("Nuclei Count")
    ax.set_yticks([])
    ax.set_title("Staging by Nuclei Count")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2**i for i in range(7, 13)])
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
