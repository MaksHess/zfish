# %%
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
from cmap import Colormap
from scipy.special import logit
from scipy.stats import circmean, circstd, circvar

from zfish.multi_table.features_aggregate import subtract_background
from zfish.multi_table.tables_io import scan_features, scan_resources
from zfish.plot.plot_commons import BASE, UMAP, cmaps, lighten_color, tpaths

plt.style.use(BASE)

GRIDSEARCH_PATH = (
    r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\ccp_gridsearch\run4_best2"
)


CCP_QUICKSAVE = r"C:\Users\hessm\Documents\Programming\Python\zfish\paper\linear_models_in_r_features.parquet"

ANN_ORDER = ["M ana", "M telo", "S early", "S", "M pro", "M meta"]

SAFE_FIG = False

feature = "NormalizedCCP"

lr = scan_resources()
lf = scan_features()

f_nuc_cyto = (
    lf.filter(pl.col("o").is_in(["nucleiRaw3", "cyto"]))
    .pipe_table(
        lambda x: x.filter(
            pl.col("c").is_in(
                ["DAPI.1", "PCNA.0", "pH3.1", "Pol-II-S2P.0", "Pol-II-S5P.2"]
            )
        ),
        "intensity",
    )
    .collect()
)

f_nuc = f_nuc_cyto.filter(pl.col("o") == "nucleiRaw3")
df_bgsub = subtract_background(
    f=f_nuc_cyto.lazy(), r=lr.collect().filter(pl.col("o") == "nucleiRaw3")
)
f_nuc_bgsub = f_nuc.from_template(
    dict(
        intensity=(
            df_bgsub.with_columns(f_nuc.intensity.select(pl.exclude(df_bgsub.columns)))
        )
    )
)

df = (
    pl.read_parquet(CCP_QUICKSAVE)
    .join(
        lr.label_objects.filter(pl.col("o") == "nucleiRaw3")
        .select("roi", "o", "label", cs.starts_with("centroid"))
        .collect(),
        on=["roi", "label"],
    )
    .select(
        "roi",
        "o",
        "label",
        "cycle",
        "nuc__Count",
        "log2_nuc__Count",
        "CCP",
        "NormalizedCCP",
        "SingleCellPseudotime",
        "ClassCCP",
    )
)

df_all_raw = df


df_all_out = df_all_raw.filter(pl.col("NormalizedCCP").is_null())
df_all = df_all_raw.filter(pl.col("NormalizedCCP").is_not_null())

r_all = (
    lr.collect().filter(pl.col.roi.is_in(df_all["roi"].unique().to_list())).collect()
)

# %%
import cmap

from zfish.multi_table.schemas_v2 import sel
from zfish.multi_table.tables_io import to_wide
from zfish.plot.plot_commons import lighten_hex_list


def plot_top_panel(df_all_raw, axs):
    res_emb = (
        df_all_raw.group_by("roi")
        .agg(pl.col("cycle").first())
        .group_by("cycle")
        .agg(pl.len())
        .sort("cycle")
    )
    res_cell = df_all_raw.group_by("cycle").agg(pl.len()).sort("cycle")

    axs[0].set_ylabel("Count")
    xlabels = np.array(list(range(7, 14)))

    cells = res_cell["len"]
    embs = res_emb["len"]
    axs[0].bar(
        xlabels,
        cells,
        label="cell",
        alpha=1.0,
        facecolor="0.8",
        lw=0.5,
        edgecolor="0.2",
    )
    axs[0].bar(
        xlabels,
        embs,
        label="embryo",
        lw=0.5,
        edgecolor="0.2",
        facecolor=lighten_color(cmap.Color("firebrick"), stops=2).hex,
    )

    from zfish.multi_table.table_base_mixin import human_format_safe

    for cell, xlabel in zip(cells, xlabels):
        axs[0].annotate(
            human_format_safe(cell).lower(),
            xycoords="data",
            xy=(xlabel, cell),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            va="bottom",
            size=5,
        )

    for emb, xlabel in zip(embs, xlabels):
        axs[0].annotate(
            human_format_safe(emb).lower(),
            xy=(xlabel, emb),
            xycoords="data",
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            size=5,
        )

    axs[0].legend()
    axs[0].grid(visible=False, axis="y")
    axs[0].set_yticks([10**e for e in range(7)])
    axs[0].set_yscale("log", base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9])
    axs[0].set_ylim(10**0, 10**7)

    total_cells = cells.sum()
    total_embryos = embs.sum()

    cell_label = "$\mathregular{n_{cells}}$"
    emb_label = "$\mathregular{n_{embryos}}$"

    axs[0].annotate(
        f"{cell_label} =\n{emb_label} =",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=5,
    )

    axs[0].annotate(
        f"{total_cells:,}\n{total_embryos:,}",
        xy=(0.4, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=5,
    )


def plot_bottom_panels(df_all_raw, df_all, f_nuc, axs):
    res = (
        df_all_raw.join(
            f_nuc.classifier.select(sel.idx, cs.matches("pred")).pipe(to_wide),
            on=["roi", "label"],
        )
        .group_by("celltype_pred", "cycle", "roi")
        .agg(pl.len())
        .sort("celltype_pred", "cycle")
        .with_columns(pl.col("len") / pl.col("len").sum().over("cycle", "roi"))
    )

    for color, (name, df) in zip(
        lighten_hex_list(cmaps.celltype_hex, stops=2),
        pl.Series("celltype_pred", ["EVL", "Deep", "YSL"], dtype=pl.Categorical)
        .to_frame()
        .with_row_index("order")
        .join(res, on="celltype_pred")
        .sort("order")
        .group_by("celltype_pred", maintain_order=True),
    ):
        axs[1].plot(
            df.join(
                df_all.group_by("roi")
                .agg(pl.col("log2_nuc__Count").first())
                .select(sel.idx, "log2_nuc__Count"),
                how="left",
                on="roi",
            )["log2_nuc__Count"],
            df["len"],
            ".",
            ms=0.8,
            alpha=0.7,
            c=color,
            # label=name[0],
        )
    for color, (name, df_cycle_agg) in zip(
        cmaps.celltype_hex,
        (
            pl.Series("celltype_pred", ["EVL", "Deep", "YSL"], dtype=pl.Categorical)
            .to_frame()
            .with_row_index("order")
            .join(res, on="celltype_pred")
            .sort("celltype_pred")
            .group_by("cycle", "celltype_pred", maintain_order=True)
        )
        .agg(pl.col("len").mean(), pl.col("roi").first())
        .group_by("celltype_pred"),
    ):
        print(name)
        axs[1].plot(
            df_cycle_agg.join(
                df_all.group_by("roi")
                .agg(pl.col("log2_nuc__Count").first())
                .select(sel.idx, "log2_nuc__Count"),
                how="left",
                on="roi",
            )["cycle"],
            df_cycle_agg["len"],
            "-",
            lw=0.7,
            ms=3,
            alpha=1,
            c=color,
            label=name[0].replace("/", "\nM "),
        )
    axs[1].legend()
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel("Ratio")

    res_cellcycle = (
        df_all_raw.join(
            f_nuc.classifier.select(sel.idx, cs.matches("pred")).pipe(to_wide),
            on=["roi", "label"],
        )
        .group_by("cellcycle3c_pred", "cycle", "roi")
        .agg(pl.len())
        .sort("cellcycle3c_pred", "cycle")
        .with_columns(pl.col("len") / pl.col("len").sum().over("cycle", "roi"))
    )
    res_celltype_agg = res_cellcycle.group_by("cycle", "cellcycle3c_pred").agg(
        pl.col("len").mean()
    )

    for color, (name, df) in zip(
        [cmap.Color(e).hex for e in cmaps.ccp([0.1, 0.5, 0.9])],
        pl.Series(
            "cellcycle3c_pred", ["M ana/telo", "S", "M pro/meta"], dtype=pl.Categorical
        )
        .to_frame()
        .with_row_index("order")
        .join(res_cellcycle, on="cellcycle3c_pred")
        .sort("cellcycle3c_pred")
        .group_by("cellcycle3c_pred", maintain_order=True),
    ):
        axs[2].plot(
            df.join(
                df_all.group_by("roi")
                .agg(pl.col("log2_nuc__Count").first())
                .select(sel.idx, "log2_nuc__Count"),
                how="left",
                on="roi",
            )["log2_nuc__Count"],
            df["len"],
            ".",
            ms=0.8,
            alpha=0.7,
            c=color,
            # label=name[0].replace('/', '\nM '),
            # legend=False,
        )
    for color, (name, df_celltype_agg) in zip(
        lighten_hex_list([cmap.Color(e).hex for e in cmaps.ccp([0.1, 0.5, 0.9])], -2),
        (
            pl.Series(
                "cellcycle3c_pred",
                ["M ana/telo", "S", "M pro/meta"],
                dtype=pl.Categorical,
            )
            .to_frame()
            .with_row_index("order")
            .join(res_cellcycle, on="cellcycle3c_pred")
            .sort("cellcycle3c_pred")
            .group_by("cycle", "cellcycle3c_pred", maintain_order=True)
        )
        .agg(pl.col("len").mean(), pl.col("roi").first())
        .group_by("cellcycle3c_pred"),
    ):
        axs[2].plot(
            df_celltype_agg.join(
                df_all.group_by("roi")
                .agg(pl.col("log2_nuc__Count").first())
                .select(sel.idx, "log2_nuc__Count"),
                how="left",
                on="roi",
            )["cycle"],
            df_celltype_agg["len"],
            "-",
            lw=0.7,
            ms=3,
            alpha=1,
            c=color,
            label=name[0].replace("/", "\nM "),
        )
    axs[2].legend()
    axs[2].set_xticks(range(7, 14))
    axs[2].set_ylim(0, 1)
    axs[2].set_xlabel("Division Cycle")
    axs[2].set_ylabel("Ratio")

    for ax in axs:
        ax.grid(visible=False, axis="x")
        # sns.move_legend(axs[1], "center left", title="Cell Type", bbox_to_anchor=(1, 0.5))
        # sns.move_legend(axs[0], "center left", title="Object Type", bbox_to_anchor=(1, 0.5))
        # sns.move_legend(axs[2], "center left", title="Cell Cycle", bbox_to_anchor=(1, 0.5))
        sns.move_legend(
            axs[1],
            "lower center",
            title="Cell Type",
            bbox_to_anchor=(0.5, 1.0),
            ncols=3,
            columnspacing=0.5,
        )
        sns.move_legend(
            axs[0],
            "lower center",
            title="Object Type",
            bbox_to_anchor=(0.5, 1.0),
            ncols=3,
            columnspacing=0.5,
        )
        sns.move_legend(
            axs[2],
            "lower center",
            title="Cell Cycle",
            bbox_to_anchor=(0.5, 1.0),
            ncols=3,
            columnspacing=0.5,
        )

    plt.tight_layout()
    #     plt.subplots_adjust(
    #         top=0.97,
    # bottom=0.07,
    # left=0.135,
    # right=0.76,
    # hspace=0.1,
    # wspace=0.2
    #     )
    plt.subplots_adjust(
        top=0.75, bottom=0.135, left=0.06, right=0.975, hspace=0.1, wspace=0.2
    )
    axs[1].set_xticks(range(7, 14))
    # sns.move_legend(axs[1], "upper left", title="Cell Type", bbox_to_anchor=(1, 1))
    # sns.move_legend(axs[0], "upper left", title="Object Type", bbox_to_anchor=(1, 1))
    # sns.move_legend(axs[2], "upper left", title="Cell Cycle", bbox_to_anchor=(1, 1))


# fig, axs = plt.subplots(3, 1, figsize=(5, 1.6), sharex=True, height_ratios=[5, 4, 4]) # tall layout
fig, axs = plt.subplots(1, 3, figsize=(4, 1.4), sharex=True, width_ratios=[5, 4, 4])


plot_top_panel(df_all_raw, axs)
plot_bottom_panels(df_all_raw, df_all, f_nuc, axs)

if SAFE_FIG:
    plt.savefig(tpaths.classifier / "object_count_and_classifiers.png")
# %%
import numpy as np
import seaborn as sns

ccp_col = "NormalizedCCP"
sort_by = ["cycle_corr", "Ccp__Mean"]
logit_mult = 1 / 10

df_emb = (
    df_all.group_by("roi", maintain_order=True)
    .agg(
        [
            pl.col(ccp_col)
            .map_elements(circmean, return_dtype=pl.Float64)
            .alias("Ccp__CircMean"),
            pl.col(ccp_col)
            .map_elements(circstd, return_dtype=pl.Float64)
            .alias("Ccp__CircStd"),
            pl.col(ccp_col)
            .map_elements(circvar, return_dtype=pl.Float64)
            .alias("Ccp__CircVar"),
            pl.col("nuc__Count").first(),
            # pl.col("^RAD:.*$").mean().name.prefix("nuc_"),
            pl.col("log2_nuc__Count").first(),
            pl.col("cycle").first(),
            # pl.col('cycle').mean()
        ]
    )
    .with_columns(
        (
            pl.col("log2_nuc__Count")
            - logit(pl.col("Ccp__CircMean") / (2 * np.pi))
            * logit_mult  # empirically found
        ).alias("adjusted_nuc_count")
    )
    .with_columns(
        pl.col("adjusted_nuc_count")
        # .cut(np.arange(7.5, 13.6, 1), labels=list(map(str, np.arange(7, 15))))
        .cut(np.arange(7.3, 13.4, 1), labels=list(map(str, np.arange(7, 15))))
        .cast(pl.String)
        .cast(pl.Int64)
        .alias("cycle_corr")
    )
    .with_columns(
        (pl.col("cycle_corr") + pl.col("Ccp__CircMean") / (2 * np.pi) - 1).alias(
            "log2_nuc__Count_corr"
        )
    )
)

fig, ax = plt.subplots(4, 1, sharex=False, figsize=(5, 6))
plt.sca(ax[0])
sns.scatterplot(
    df_emb,
    x="log2_nuc__Count",
    y="Ccp__CircMean",
    hue="cycle",
    palette=cmaps.division_cycle.to_mpl(),
    legend=False,
)
plt.sca(ax[1])
sns.scatterplot(
    df_emb,
    x="log2_nuc__Count",
    y="Ccp__CircMean",
    hue="cycle_corr",
    palette=cmaps.division_cycle.to_mpl(),
    legend=False,
)
plt.sca(ax[2])
sns.scatterplot(
    df_emb.with_columns(
        pl.col("log2_nuc__Count")
        - logit(pl.col("Ccp__CircMean") / (2 * np.pi)) * logit_mult
    ),
    x="log2_nuc__Count",
    y="Ccp__CircMean",
    hue="cycle_corr",
    palette=cmaps.division_cycle.to_mpl(),
    legend=False,
)
plt.sca(ax[3])
sns.scatterplot(
    df_emb.with_columns(
        pl.col("log2_nuc__Count")
        - logit(pl.col("Ccp__CircMean") / (2 * np.pi)) * logit_mult
    ),
    x="log2_nuc__Count_corr",
    y="Ccp__CircMean",
    hue="cycle_corr",
    palette=cmaps.division_cycle.to_mpl(),
    legend=False,
)
# %%
plt.style.use(BASE)
fig, ax = plt.subplots(figsize=(2, 1))


df_plot = df_emb.with_columns(
    logit(pl.col("Ccp__CircMean") / (2 * np.pi)), pl.col("log2_nuc__Count")
)

sns.scatterplot(
    df_plot, x="log2_nuc__Count", y="Ccp__CircMean", hue="Ccp__CircVar", legend=False
)


def line_plot(ax=None, interc=0, m=0, x_orig=None, c="k", **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    if x_orig is None:
        x_orig = x0

    dx = x1 - x_orig
    dx0 = x_orig - x0

    p0 = x_orig, interc
    p1 = x1, m * dx + interc

    ax.plot(*zip(p0, p1), c=c, **kwargs)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


m = 10
interc = -(m / 2)
interc = -10

ax.set_xlim(6.5, 13.5)
# ax.set_ylim(-7, 7)
for e in range(6, 13):
    line_plot(ax, interc=interc, m=m, x_orig=e + 0.5)
    line_plot(ax, interc=interc, m=m, x_orig=e, lw=0.5, ls=":")

    pass
    # line_plot(ax, interc=interc, m=m, x_orig=7.5)

# plt.grid()

# %%
sns.scatterplot(
    df_plot, x="log2_nuc__Count", y="Ccp__CircVar", hue="Ccp__CircMean", legend=False
)


# %%
plt.style.use({"lines.markersize": 2})

fig, axs = plt.subplots(2, 1, figsize=(1.8, 1.6), sharex=True, sharey=True)
plt.sca(axs[0])
sns.scatterplot(
    df_emb,
    x="log2_nuc__Count",
    y="Ccp__CircMean",
    hue="cycle",
    palette=cmaps.division_cycle_hex_7,
    legend=False,
)
plt.sca(axs[1])
sns.scatterplot(
    df_emb,
    x="log2_nuc__Count",
    y="Ccp__CircMean",
    hue="cycle_corr",
    palette=cmaps.division_cycle_hex_7,
    legend=False,
)
axs[1].set_xlabel("$\\mathregular{log_2}$ Nuc Count")
axs[1].set_xticks(range(7, 14))
for ax in axs:
    ax.set_ylabel("Circ Mean Nuc CCP")
    ax.set_yticks([0, np.pi, 2 * np.pi], labels=["0", "π", "2π"])

if SAFE_FIG:
    plt.savefig(tpaths.ccp / "adjusted_staging-method.png")
# %%
# sns.scatterplot(
#     df_emb.filter(pl.col("cycle") == pl.col("cycle_corr")),
#     x="log2_nuc__Count_corr",
#     y="log2_nuc__Count",
#     hue="cycle_corr",
#     palette=cmaps.division_cycle.to_mpl(),
#     legend=False,
# )
hue_name = "Circ Mean CCP Nuc"

plt.style.use(BASE)
plt.style.use({"lines.markersize": 2})

plt.style.use({"axes.grid": True})
fig, ax = plt.subplots()
sns.scatterplot(
    df_emb.with_columns(pl.col("Ccp__CircMean").alias(hue_name)),
    y="log2_nuc__Count",
    x="log2_nuc__Count_corr",
    hue=hue_name,
    palette=cmaps.ccp.to_mpl(),
)
ax.set_aspect("equal")
ax.set_xticks(list(range(7, 14)))
ax.set_yticks(list(range(7, 13)))
ax.set_xlabel("CCP Adjusted Staging")
ax.set_ylabel("Count Based Staging")
# for i in range(7, 13):
#     ax.axvline(i, color="0.7", linestyle=":")
#     ax.axhline(i, color="0.7", linestyle=":")

if SAFE_FIG:
    plt.savefig(tpaths.ccp / "adjusted_staging.png")
# %%

plt.style.use(BASE)
plt.style.use({"lines.markersize": 2})
from zfish.preprocessing.sliding_window_samples import stratified_sample

df_emb_sampled = (
    pl.DataFrame({"corr_nuc_count_samples": np.arange(6, 15, 0.03)})
    .join_asof(
        df_emb.sort("log2_nuc__Count_corr"),
        left_on="corr_nuc_count_samples",
        right_on="log2_nuc__Count_corr",
        strategy="nearest",
    )
    .with_columns(
        (pl.col("corr_nuc_count_samples") - pl.col("log2_nuc__Count_corr"))
        .abs()
        .alias("error")
    )
    .group_by("roi", maintain_order=True)
    .agg(pl.all().sort_by("error").first())
)
# df_emb_sampled = df_emb.filter(stratified_sample(by="cycle_corr", n=6))
fig, ax = plt.subplots()
ax = sns.scatterplot(
    df_emb_sampled.with_columns(pl.col("cycle_corr").alias("adj. pooled cycle")),
    x="log2_nuc__Count_corr",
    y="log2_nuc__Count",
    # hue="Ccp__Mean",
    hue="adj. pooled cycle",
    palette=cmaps.division_cycle.to_mpl(),
)
sns.move_legend(ax, "lower right", ncols=2)
ax.set_xticks(list(range(7, 14)))
ax.set_yticks(list(range(7, 13)))
ax.set_xlabel("CCP Adjusted Staging")
ax.set_ylabel("Count Based Staging")
# for i in range(7, 13):
#     ax.axvline(i, color="0.7", linestyle=":")
#     ax.axhline(i, color="0.7", linestyle=":")
ax.set_aspect("equal")

if SAFE_FIG:
    plt.savefig(
        r"C:\Users\hessm\Documents\Programming\Python\paper_manuscript\figures\Figure 3\adjusted_bins.png"
    )

# %%

plt.style.use(BASE)


def pl_circdiff(x: str, y: str):
    a = (pl.col(x) - pl.col(y)).add(2 * np.pi).mod(2 * np.pi)
    b = (pl.col(y) - pl.col(x)).add(2 * np.pi).mod(2 * np.pi)
    return pl.when(a < b).then(-a).otherwise(b).alias(x)


df_all_train = (
    df_all.with_columns(
        pl.col("NormalizedCCP")
        .map_elements(circmean, return_dtype=pl.Float64)
        .over("roi")
        .alias("CircMean__NormalizedCCP"),
        pl.col("NormalizedCCP")
        .map_elements(circvar, return_dtype=pl.Float64)
        .over("roi")
        .alias("CircVar__NormalizedCCP"),
    )
    .with_columns(
        (pl_circdiff("CircMean__NormalizedCCP", "NormalizedCCP") / (2 * np.pi)).alias(
            "_ccp_shift"
        )
    )
    .join(df_emb.select("roi", "log2_nuc__Count_corr", "cycle_corr"), on="roi")
    .with_columns(
        (pl.col("log2_nuc__Count_corr") + pl.col("_ccp_shift")).alias("dev_time")
    )
    .with_columns(
        pl.when(pl.col("cycle_corr") > 8)
        .then(pl.col("cycle_corr"))
        .otherwise(pl.lit(9))
        .alias("cycle_corr_pooled")
    )
)
sns.kdeplot(
    df_all_train.to_pandas(),
    x="_ccp_shift",
    hue="log2_nuc__Count",
    legend=False,
    cut=True,
    common_norm=False,
    lw=0.2,
    palette="inferno",
)

# %%
import holoviews as hv
import hvplot.polars
from holoviews import opts

x = "CircMean__NormalizedCCP"
y = "log2_nuc__Count_corr"
# y = "log2_nuc__Count"
y_rank_ordering = False
err = "CircVar__NormalizedCCP"
hue = "cycle_corr"


circ_mean = "CircMean__NormalizedCCP"
q_lower = 0.0
q_upper = 1.0
err_q = 0.05
scl = 2 * np.pi
MAX_CYCLE = 13

qs = [1.0, 0.99, 0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.0]
n_qs = len(qs)

quantiles = (qs[qs.index(err_q)], qs[qs.index(round(1 - err_q, 5))])
q_lower_col = f"_ccp_shift_q{min(quantiles)}"
q_upper_col = f"_ccp_shift_q{max(quantiles)}"


df_aggro = (
    df_all_train.group_by("roi")
    .agg(
        pl.col(y).first(),
        pl.col(hue).first(),
        pl.col("CircMean__NormalizedCCP").first(),
        pl.col("CircVar__NormalizedCCP").first(),
        *[
            (pl.col("_ccp_shift") * scl).quantile(e).name.suffix(f"_q{str(e)}")
            for e in qs
        ],
    )
    .sort(y)
    .with_row_index()
    .with_columns(pl.col("index").alias(y if y_rank_ordering else "index"))
    .with_columns(
        pl.when((pl.col("CircMean__NormalizedCCP") - pl.col(q_lower_col).abs()) < 0)
        .then(pl.col("cycle_corr") - 1)
        .when(
            (pl.col("CircMean__NormalizedCCP") + pl.col(q_upper_col).abs()) >= 2 * np.pi
        )
        .then(pl.col("cycle_corr") + 1)
        .otherwise(pl.col("cycle_corr"))
        .clip(0, MAX_CYCLE)
        .alias("cycle_sc")
    )
)

(
    df_aggro.hvplot.scatter(
        x=y,
        y=x,
        color=hue,
        cmap=cmaps.division_cycle_hex_7,
        height=600,
        width=500,
    )
    * df_aggro.with_columns(
        pl.col(f"_ccp_shift_q{q}").abs() for q in qs
    ).hvplot.errorbars(
        x=y,
        y=x,
        # yerr1="CircVar__NormalizedCCP",
        yerr1=q_lower_col,
        yerr2=q_upper_col,
        invert=True,
    )
    * hv.HLines([0.0, 2 * np.pi]).opts(color="black", line_width=1)
)

df_all_train = df_all_train.with_columns(
    pl.when(pl.col(circ_mean) - pl.col(circ_mean).quantile(q_lower) < 0)
    .then(pl.col("cycle_corr") - 1)
    .when(pl.col(circ_mean) - pl.col(circ_mean).quantile(q_upper) > 2 * np.pi)
    .then(pl.col("cycle_corr") + 1)
    .otherwise("cycle_corr")
    .clip(0, MAX_CYCLE)
    .alias("cycle_sc")
).with_columns(
    pl.when(pl.col("cycle_sc") > 8)
    .then(pl.col("cycle_sc"))
    .otherwise(pl.lit(9))
    .alias("cycle_sc_pooled")
)
# %%


df_all_train.filter(stratified_sample("cycle_sc", 10_000)).join(
    f_nuc.intensity.filter(pl.col("c").is_in(["PCNA.0", "DAPI.1"]))
    .select(sel.idx, "Mean", "Sum")
    .pipe(to_wide),
    on=["roi", "label"],
).join(
    f_nuc.label.select(sel.idx, "PhysicalSize", "Roundness").pipe(to_wide),
    on=["roi", "label"],
).sort("cycle_sc").hvplot.violin(by="cycle_sc", y="DAPI.1_Sum")
# %%
from zfish.multi_table.schemas_v2 import sel
from zfish.multi_table.tables_io import to_wide

df_nuc_new = (
    f_nuc.intensity.filter(
        pl.col("c").is_in(["DAPI.1", "PCNA.0", "pH3.1", "Pol-II-S2P.0", "Pol-II-S5P.2"])
    )
    .select(sel.idx, "Mean", "Sum")
    .with_columns(pl.col("roi").cast(pl.String), pl.col("label").cast(pl.UInt64))
    .pipe(to_wide)
)
# %%
fig, ax = plt.subplots(figsize=(5, 2), dpi=300)
sns.scatterplot(
    df_all_train
    # .filter(stratified_sample("cycle_corr", 10_000))
    .join(
        f_nuc.intensity.filter(pl.col("c") == "PCNA.0")
        .select(sel.idx, "Mean")
        .pipe(to_wide),
        on=["roi", "label"],
    )
    # .filter(pl.col('cycle_corr')<12)
    # .with_columns(pl.Series('div', divs[0]).clip(-0.05, 0.05))
    .to_pandas(),
    x="dev_time",
    y="PCNA.0_Mean",
    hue="cycle_sc",
    legend=False,
    s=1,
    alpha=0.3,
    # palette='PiYG',
    # hue_norm=(-0.005, 0.005)
    # cut=True,
)
plt.gca().set_yscale("log", base=2)
# df_one.pipe(lambda x: sns.kdeplot(x.to_pandas(), x="NormalizedCCP"))


# %%

df_cells_sampled = df_all_train.with_columns(
    pl.col("dev_time")
    .cut(
        np.linspace(6.0, 14.0, 80, endpoint=False),
        labels=list(map(str, np.linspace(7.0, 15.0, 80, endpoint=False).round(1)))
        + ["15.0"],
    )
    .alias("dev_time_bins")
).filter(stratified_sample(by="dev_time_bins", n=300))

df_cells_sampled2 = (
    df_all_train.with_columns(
        pl.col("ClassCCP")
        .str.replace("^S[0-2]$", "S0")
        .str.replace("^S[3-5]$", "S1")
        .alias("ccp_bins"),
        pl.col("roi").cast(pl.String),
        pl.col("label").cast(pl.UInt64),
    )
    .filter(stratified_sample(by=("cycle_corr", "ccp_bins"), n=1200))
    .join(df_nuc_new, on=["roi", "label"])
)

df_cells_sampled3 = (
    df_all_train.with_columns(
        pl.col("roi").cast(pl.String),
        pl.col("label").cast(pl.UInt64),
    )
    .filter(stratified_sample(by="cycle_corr_pooled", n=11000))
    .join(df_nuc_new, on=["roi", "label"])
)

df_cells_sampled4 = (
    df_all_train.with_columns(
        pl.col("roi").cast(pl.String),
        pl.col("label").cast(pl.UInt64),
    )
    .filter(stratified_sample(by="cycle_corr_pooled", n=45000))
    .join(df_nuc_new, on=["roi", "label"])
)

df_cells_sampled2b = (
    df_all_train.with_columns(
        pl.col("ClassCCP")
        .str.replace("^S[0-2]$", "S0")
        .str.replace("^S[3-5]$", "S1")
        .alias("ccp_bins"),
        pl.col("roi").cast(pl.String),
        pl.col("label").cast(pl.UInt64),
    )
    .filter(stratified_sample(by=("cycle_sc", "ccp_bins"), n=1200))
    .join(df_nuc_new, on=["roi", "label"])
)

df_cells_sampled3b = (
    df_all_train.with_columns(
        pl.col("roi").cast(pl.String),
        pl.col("label").cast(pl.UInt64),
    )
    .filter(stratified_sample(by="cycle_sc_pooled", n=11000))
    .join(df_nuc_new, on=["roi", "label"])
)

fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
sns.scatterplot(
    df_cells_sampled2.to_pandas(),
    x="dev_time",
    y="PCNA.0_Mean",
    s=5,
    hue="ccp_bins",
    hue_order=[
        "M ana",
        "S early",
        # "S",
        "S0",
        "S1",
        # "S2",
        # "S3",
        # "S4",
        # "S5",
        "S late",
        "M meta",
    ],
    # palette=cmaps.division_cycle.to_mpl(),
    palette=list(cmaps.ccp.lut(11)[:10]),
    legend=False,
    alpha=0.3,
)
plt.gca().set_yscale("log", base=2)

# %%
fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
sns.scatterplot(
    df_cells_sampled3.to_pandas(),
    x="dev_time",
    y="PCNA.0_Mean",
    # palette=cmaps.division_cycle.to_mpl(),
    # palette=list(cmaps.ccp.lut(11)[:10]),
    legend=False,
    alpha=0.3,
    size=0.05,
)
plt.gca().set_yscale("log", base=2)
# %%
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
sns.kdeplot(
    df_cells_sampled.to_pandas(),
    x="dev_time",
    hue="cycle_corr",
    palette=cmaps.division_cycle.to_mpl(),
    cut=0,
)
sns.kdeplot(df_cells_sampled.to_pandas(), x="dev_time", color="0.6", cut=0, linewidth=2)


fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
sns.kdeplot(
    df_cells_sampled2.to_pandas(),
    x="dev_time",
    hue="cycle_corr",
    palette=cmaps.division_cycle.to_mpl(),
    cut=0,
)
sns.kdeplot(
    df_cells_sampled2.to_pandas(), x="dev_time", color="0.6", cut=0, linewidth=2
)


fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
sns.kdeplot(
    df_cells_sampled2b.to_pandas(),
    x="dev_time",
    hue="cycle_sc",
    palette=cmaps.division_cycle.to_mpl(),
    cut=0,
)
sns.kdeplot(
    df_cells_sampled2b.to_pandas(), x="dev_time", color="0.6", cut=0, linewidth=2
)

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
sns.kdeplot(
    df_cells_sampled3.to_pandas(),
    x="dev_time",
    hue="cycle_corr",
    palette=cmaps.division_cycle.to_mpl(),
    cut=0,
)
sns.kdeplot(
    df_cells_sampled3.to_pandas(), x="dev_time", color="0.6", cut=0, linewidth=2
)


fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
sns.kdeplot(
    df_cells_sampled3b.to_pandas(),
    x="dev_time",
    hue="cycle_sc",
    palette=cmaps.division_cycle.to_mpl(),
    cut=0,
)
sns.kdeplot(
    df_cells_sampled3.to_pandas(), x="dev_time", color="0.6", cut=0, linewidth=2
)
# %%
from pathlib import Path

from sklccp.ccp_pipeline_gridsearch_cluster import (
    SCALERS,
    fit_pipeline,
    get_pipeline,
)

from zfish.features.io import CLF_ANNOTATIONS_LOCAL

df_ann = pl.read_parquet(
    Path(CLF_ANNOTATIONS_LOCAL) / "ann_cellcycle_clean.parquet"
).select("roi", pl.col("object").alias("o"), "label", "ann0", "ann0int")

# %%
f_nuc_selected = f_nuc.pipe_table(
    lambda x: x.select(
        sel.idx, "Mean", "Variance", "StandardDeviation", "Skewness", "Kurtosis", "Sum"
    ),
    "intensity",
).pipe_tables(lambda x: x.pipe(to_wide), include_tables=("label", "intensity"))


df_nuc_new = (
    f_nuc_selected.label.join(f_nuc_selected.intensity, on=["roi", "o", "label"])
    .join(
        df_all_train.select(
            "roi",
            "label",
            "log2_nuc__Count",
            "cycle_corr",
            "cycle_sc",
            "dev_time",
            "cycle_corr_pooled",
            "cycle_sc_pooled",
            "log2_nuc__Count_corr",
        ),
        on=["roi", "label"],
    )
    .with_columns(
        pl.col("roi").cast(pl.String),
        pl.col("o").cast(pl.String),
        pl.col("label").cast(pl.UInt64),
    )
    .join(
        df_ann,
        on=["roi", "o", "label"],
        how="left",
    )
)
df_model = df_nuc_new.join(
    # df_cells_sampled3.with_columns(
    df_cells_sampled.with_columns(
        pl.col("roi").cast(pl.String),
        pl.col("o").cast(pl.String),
        pl.col("label").cast(pl.UInt64),
    ),
    how="semi",
    on=["roi", "label"],
)

df_model2 = df_nuc_new.join(
    # df_cells_sampled3.with_columns(
    df_cells_sampled2.with_columns(
        pl.col("roi").cast(pl.String),
        pl.col("o").cast(pl.String),
        pl.col("label").cast(pl.UInt64),
    ),
    how="semi",
    on=["roi", "label"],
)

df_model3 = df_nuc_new.join(
    # df_cells_sampled3.with_columns(
    df_cells_sampled3.with_columns(
        pl.col("roi").cast(pl.String),
        pl.col("o").cast(pl.String),
        pl.col("label").cast(pl.UInt64),
    ),
    how="semi",
    on=["roi", "label"],
)

# %%
# df_train = df_model.select(
#     [
#         "PhysicalSize",
#         "Elongation",
#         "Flatness",
#         "Roundness",
#         "FeretDiameter",
#         "Perimeter",
#         "EquivalentSphericalPerimeter",
#         "EquivalentSphericalRadius",
#         "PCNA.0_Skewness",
#         "PCNA.0_Kurtosis",
#         "PCNA.0_Sum",
#         "DAPI.1_Skewness",
#         "DAPI.1_Kurtosis",
#         "DAPI.1_Sum",
#         "pH3.1_Skewness",
#         "pH3.1_Kurtosis",
#         "pH3.1_Sum",
#         "Centroid-x",
#         "Centroid-y",
#         "Centroid-z",
#         "EquivalentEllipsoidDiameter-a",
#         "EquivalentEllipsoidDiameter-b",
#         "EquivalentEllipsoidDiameter-c",
#         "PCNA.0_Mean",
#         "PCNA.0_Variance",
#         "PCNA.0_StandardDeviation",
#         "DAPI.1_Mean",
#         "DAPI.1_Variance",
#         "DAPI.1_StandardDeviation",
#         "pH3.1_Mean",
#         "pH3.1_Variance",
#         "pH3.1_StandardDeviation",
#         # "log2_nuc__Count_corr",
#     ]
# )


# %%
# umap_pip = make_pipeline(
#     PowerTransformer(),
#     UMAP(n_neighbors=10, n_components=3, densmap=False, min_dist=0.1),
# )
# umap_pip.fit(df_train)
# # %%
# emb = umap_pip.transform(df_train)
# # emb_full = umap_pip.transform(df_nuc.select(df_train.columns))
# from cmap import Colormap
# df_plot = (
#     df_model.with_columns(pl.DataFrame(emb, schema=["umap0", "umap1", "umap2"]))
#     # .filter(stratified_sample("cycle_corr", 10_000))
#     .sort("cycle_corr", descending=True)
# )
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.scatter(
#     df_plot["umap0"],
#     df_plot["umap1"],
#     df_plot["umap2"],
#     s=2,
#     # c=df_plot["DAPI.1_Sum"].log(base=2).clip(19.5, 20.5),
#     # c=df_plot.with_columns(pl.col("celltype_pred")=='EVL')['celltype_pred'],
#     c=df_plot["cycle_corr"].cast(pl.UInt32),
#     cmap=Colormap(cmaps.division_cycle).to_mpl(),
#     # cmap=Colormap(Colormap('set1').lut()[:7, :]).to_mpl(),
# )
# %%
import os
from dataclasses import dataclass

# %%
import hvplot.polars
import polars as pl
from sklccp.ccp_transformers import PolarsSelector
from umap import UMAP

wd = os.getcwd()
zfd = "c:\\Users\\hessm\\Documents\\Programming\\Python\\zfish"
os.chdir(zfd)
from paper.ccp_load_gridsearch_results import (
    load_params,
    plot_scores,
    read_ccp_gridsearch_scores,
    scan_ccp_predictions,
)
from zfish.features.ccp.ccp_normalize_align import (
    cut_normalized_ccp,
    find_normalized_ccp_cuts,
)

os.chdir(wd)


@dataclass
class CcptParams:
    pass


df = read_ccp_gridsearch_scores("run4")
plot_scores(
    df.group_by("job_idx")
    .agg(cs.string().first(), cs.float().mean())
    .filter(pl.col("n_components").cast(pl.Int32) == 3)
    .filter(pl.col("exclude_celltypes") == "('YSL',)")
    .filter(pl.col("strata_emb") == "None")
    .with_columns(
        pl.concat_str(
            pl.col("circle_mu"), pl.col("circle_lambda"), separator=", "
        ).alias("mu, lambda")
    ),
    sort="score_worst",
    hover_cols=("job_idx", *df.select(cs.string() - cs.matches("filename")).columns),
    max_models=1000,
    by="scaler",
    fg_size=12,
    bg_size=1,
)
# %%
scrs = read_ccp_gridsearch_scores("run4_best2")

# plot_scores(
#     scrs.filter(pl.col("n_components").cast(pl.Int32) < 3),
#     sort="score_worst",
#     hover_cols=("job_idx", *df.select(cs.string() - cs.matches("filename")).columns),
# )

plot_scores(
    scrs.filter(pl.col("exclude_celltypes") == "('YSL',)").filter(
        pl.col("n_components").cast(pl.Int32) < 4
    ),
    sort="score_worst",
    hover_cols=("job_idx", *df.select(cs.string() - cs.matches("filename")).columns),
    by="sample",
)
# %%
pip = get_pipeline()

manual_params = True
job_id = 3176  # 3D not in best, manual parameter settings
# job_id = 12949  # 3D
# job_id = 731  # 2D
# job_id = 8405 # 2D
# strata_params, pipeline_params = load_params(
#     rf"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\ccp_gridsearch\run4_best2\{job_id:05d}_params.json"
# )
# %%
if manual_params:
    strata_params = {
        "exclude_celltypes": ["YSL"],
        "sample": "15k",
        "strata_ccp": "cycle_pooled",
        "strata_emb": None,
    }
    pipeline_params = {
        "ccpt__transformer__ccp_inference": "nearest",
        "ccpt__transformer__circle_lambda": 1.0,
        "ccpt__transformer__circle_mu": 1.0,
        "ccpt__transformer__n_spline_samples": 30000,
        "embedder__transformer__min_dist": 0.1,
        # "embedder__transformer__spread": 0.5,
        "embedder__transformer__spread": 1.0,
        "embedder__transformer__n_components": 3,
        "embedder__transformer__n_neighbors": 15,
        "scaler": "power",
    }
else:
    strata_params, pipeline_params = load_params(
        rf"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\ccp_gridsearch\run4\{job_id:05d}_params.json"
    )

    df_res_pip = scan_ccp_predictions(
        run_id="run4_best2",
        job_idx=job_id,
        # rf"{job_id:05d}_res.parquet"
    )

features = [
    "PhysicalSize",
    "Elongation",
    "Flatness",
    # "Roundness",
    "FeretDiameter",
    "Perimeter",
    "EquivalentSphericalPerimeter",
    "EquivalentSphericalRadius",
    "EquivalentEllipsoidDiameter-a",
    "EquivalentEllipsoidDiameter-b",
    "EquivalentEllipsoidDiameter-c",
    "PCNA.0_Mean",
    "PCNA.0_Variance",
    "PCNA.0_StandardDeviation",
    "PCNA.0_Skewness",
    "PCNA.0_Kurtosis",
    "PCNA.0_Sum",
    "DAPI.1_Mean",
    "DAPI.1_Variance",
    "DAPI.1_StandardDeviation",
    "DAPI.1_Skewness",
    "DAPI.1_Kurtosis",
    "DAPI.1_Sum",
    # "pH3.1_Mean",
    # "pH3.1_Variance",
    # "pH3.1_StandardDeviation",
    # "pH3.1_Skewness",
    # "pH3.1_Kurtosis",
    # "pH3.1_Sum",
]
new_params = {
    **strata_params,
    **{
        "strata_ccp": "cycle_corr_pooled",
        # "strata_emb": "cycle_corr_pooled",
        "strata_emb": None,
        "sample": "ss_model2",
    },
}

pipeline_parsed_params = {
    **pipeline_params,
    **{
        "selector": PolarsSelector(selector=cs.by_name(features)),
        "scaler": SCALERS["power"],
        "ccpt__transformer__n_spline_samples": 30_000,
        # "ccpt__transformer__start": (2.0, 6.0, 6.0),
        # "ccpt__transformer__direction": (2.0, 4.0, 6.0),
        "ccpt__transformer__return_distance": True,
        # "ccpt__transformer__circle_lambda": 0.5,
        # "ccpt__transformer__circle_mu": 0.6,
        # "embedder__transformer__min_dist": 0.1,
        # "embedder__transformer__spread": 0.5,
        # "embedder__transformer__n_neighbors": 15,
        # "scaler": "power",
    },
}
if "scaler" in pipeline_params:
    scaler = pipeline_parsed_params["scaler"]
    if isinstance(scaler, str):
        pipeline_parsed_params["scaler"] = SCALERS[pipeline_params["scaler"]]

pip.set_params(**pipeline_parsed_params)
pipeline_parsed_params
# %%
# df_model
res, pop, scores = fit_pipeline(
    pip.set_params(**pipeline_parsed_params),
    {
        "ss_model": df_model,
        "ss_model2": df_model2,
        "ss_model3": df_model3,
        "full": df_nuc_new,
        "labeled": df_nuc_new.filter(pl.col("ann0").is_not_null()),
    },
    n_scores=1,
    **new_params,
)

# %% Scoring variablilty
labelled = df_nuc_new.filter(pl.col("ann0").is_not_null())

embeddings = [pop[:-1].transform(labelled) for _ in range(5)]

ccp_embeddings = [
    pop.transform(labelled, strata_ccp=labelled["cycle_corr_pooled"]) for _ in range(5)
]

from scipy.linalg import norm

df_reembed = (
    pl.DataFrame(
        np.concatenate(
            [
                np.concatenate(
                    [
                        np.arange(len(embeddings[0])).reshape(-1, 1),
                        embeddings[i],
                        np.ones((embeddings[i].shape[0], 1)) * i,
                    ],
                    axis=1,
                )
                for i in range(len(embeddings))
            ]
        ),
        schema=["id", "umap_0", "umap_1", "umap_2", "run"],
    ).with_columns(pl.col("id").cast(pl.Int32), pl.col("run").cast(pl.Int32))
    # .filter(pl.col("run") < 3)
)

df_plot = df_reembed.join(
    pl.concat(
        [
            df_reembed.pivot(index="run", values=[f"umap_{i}"], on="id")
            .select(pl.exclude("run").std())
            .transpose()
            .select(pl.col("column_0").alias(f"err_{i}"))
            for i in range(3)
        ],
        how="horizontal",
    )
    .with_columns(
        pl.sum_horizontal(pl.col(f"err_{i}") for i in range(3)).alias("err_sum")
    )
    .with_row_index("id"),
    on="id",
)

bg = df_plot.filter(pl.col("err_sum") == 0.0).hvplot.scatter(
    "umap_0",
    "umap_1",
    color="gray",
    size=2,
    colorbar=False,
    width=400,
)
outliers = df_plot.filter(pl.col("err_sum") > 0.0).hvplot.scatter(
    "umap_0",
    "umap_1",
    c="id",
    cmap="glasbey",
    hover_cols=["id"],
    colorbar=False,
    alpha=0.7,
    width=400,
)
bg * outliers
# %%
res_ccp = []
for name, ccpt in pop[-1].transformers_.items():
    ccp, dist = ccpt.transform(res.select(cs.starts_with("umap"))).T
    res_ccp.append(
        pl.DataFrame(
            {"cycle_ccp": name[0], "CCP": ccp * 2 * np.pi, "DistanceToCircleCCP": dist}
        )
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
from zfish.plot.plot_commons import BASE, UMAP

# %%
df_res = (
    pl.concat(
        [
            df_nuc_new.select(sel.idx),
            res,
            df_nuc_new.select(
                "log2_nuc__Count",
                "log2_nuc__Count_corr",
                "cycle_corr",
                "cycle_sc",
                "ann0",
                "dev_time",
                "cycle_corr_pooled",
                "cycle_sc_pooled",
                "PCNA.0_Mean",
                "PhysicalSize",
                "Roundness",
                "DAPI.1_Mean",
                "DAPI.1_Sum",
                "PCNA.0_Sum",
                "pH3.1_Mean",
                "pH3.1_Sum",
                "Pol-II-S2P.0_Mean",
                "Pol-II-S5P.2_Mean",
                "Pol-II-S2P.0_Sum",
                "Pol-II-S5P.2_Sum",
                "Pol-II-S2P.0_Skewness",
                "Pol-II-S5P.2_Skewness",
            ),
        ],
        how="horizontal",
    )
    .with_row_index("index")
    .join(
        pl.concat([e.with_row_index() for e in res_ccp]).rename(
            {
                "cycle_ccp": "cycle_sc_pooled",
                "CCP": "CCP_sc",
                "DistanceToCircleCCP": "DistanceToCircleCCP_sc",
            }
        ),
        on=("index", "cycle_sc_pooled"),
    )
    .join(
        df_all_train.select(
            pl.col("roi").cast(pl.String),
            pl.col("label").cast(pl.UInt64),
            pl.col("CCP").alias("_CCP"),
            pl.col("NormalizedCCP").alias("_NormalizedCCP"),
        ),
        on=["roi", "label"],
    )
)
ax = sns.kdeplot(
    df_res.to_pandas(),
    x="DistanceToCircleCCP",
    hue="cycle_sc_pooled",
    common_norm=False,
)
distance_cutoff = 2.2
distance_cutoff_12 = 0.8
ax.axvline(distance_cutoff, color="k", ls="--")
# ax.axvline(distance_cutoff_12, color="k", ls=":")

# %%
fig, axs = plt.subplots(2, 2)
features = "PCNA.0_Mean"
cycles = range(9, 13)

with plt.style.context(BASE, UMAP):
    for i, cycle in enumerate(cycles):
        df_one = df_res.filter(pl.col("cycle_sc_pooled") == cycle).sample(10_000)
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
df_ccp, extrema = align(df_res, group_column="cycle_sc_pooled")


# %%
fig, axs = plt.subplots(2, 3, figsize=(4, 2), sharey=True, sharex=True)
f = "Pol-II-S2P.0_Mean"
f = "DAPI.1_Sum"
# f = "PhysicalSize"
f = "PCNA.0_Mean"
for cyc, ax in zip(range(9, 14), axs.flatten()):
    plt.sca(ax)
    ax = sns.scatterplot(
        df_nuc_new.with_columns(df_res.select("NormalizedCCP", "DistanceToCircleCCP"))
        .with_columns(df_ccp)
        .filter(pl.col("cycle_sc_pooled") == cyc)
        .filter(stratified_sample("cycle_sc_pooled", 10000))
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
    df_res.join(df_model, on=["roi", "o", "label"], how="semi")
    .filter(stratified_sample("cycle_corr_pooled", 6854))
    .with_columns(pl.col("cycle_corr_pooled").cast(pl.String).alias("division cycle"))
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
cycle = 12
ax = sns.scatterplot(
    df_res_clean.filter(pl.col("cycle_corr") == cycle).filter(
        stratified_sample("cycle_corr_pooled", 10_000)
    ),
    x="NormalizedCCP_shift",
    # y="PCNA.0_Mean",
    y="DAPI.1_Sum",
    s=1,
    alpha=0.3,
    c="0.7",
)
ax = sns.scatterplot(
    df_res_clean.filter(pl.col("cycle_corr") == cycle),
    x="NormalizedCCP_shift",
    # y="PCNA.0_Mean",
    y="DAPI.1_Sum",
    s=10,
    alpha=0.8,
    hue="ann0",
    hue_order=["M ana", "M telo", "S early", "S", "M pro", "M meta"],
    # palette="turbo",
    palette=cmaps.ccp_hex,
    legend=False,
)
ax.set_yscale("log", base=2)
# plot_extrema(shifts[(10, )])

# %%

df_devtime = (
    df_res_clean.with_columns(
        pl.col("NormalizedCCP_shift")
        .map_elements(circmean, return_dtype=pl.Float64)
        .over("roi")
        .alias("_ccp_shift")
    )
    .with_columns(
        (pl_circdiff("_ccp_shift", "NormalizedCCP_shift") / (2 * np.pi)),
        pl.col("roi").cast(pl.String),
        pl.col("label").cast(pl.UInt64),
    )
    .join(
        df_emb.with_columns(pl.col("roi").cast(pl.String)).select(
            "roi", "log2_nuc__Count_corr"
        ),
        on="roi",
    )
    .with_columns(
        (pl.col("log2_nuc__Count_corr") + pl.col("_ccp_shift")).alias("dev_time2")
    )
    .join(
        df_res.select(
            sel.idx,
            cs.ends_with("_Mean"),
            cs.ends_with("_Sum"),
            cs.by_name("PhysicalSize", "Roundness"),
        ),
        on=["roi", "label"],
    )
    # .with_columns(
    #     (pl.col("cycle_corr") + pl.col("_ccp_shift")).alias("dev_time2")
    # )
    # .with_columns(
    #     pl.when(pl.col("cycle_corr") > 8)
    #     .then(pl.col("cycle_corr"))
    #     .otherwise(pl.lit(9))
    #     .alias("cycle_corr_pooled")
    # )
    # .filter(pl.col("cycle") == 11)
    .filter(
        pl.col("roi").is_in(
            df_emb.filter(stratified_sample("cycle_corr", 3, seed=2))["roi"]
        )
    )
)
fig, ax = plt.subplots(figsize=(4, 1.5), dpi=300)
ax = sns.scatterplot(
    df_devtime.filter(stratified_sample("roi", 1000)),
    x="dev_time2",
    y="PCNA.0_Mean",
    hue="cycle_corr_pooled",
    hue_order=[9, 10, 11, 12, 13],
    palette="Set1",
    alpha=0.5,
    s=1,
    # legend=False,
)
# ax = sns.scatterplot(
#     df_devtime.filter(stratified_sample("cycle_corr", 10_000)),
#     x="dev_time2",
#     y="PCNA.0_Mean",
#     s=0.5,
#     alpha=0.4,
# )
ax.set_yscale("log")

# %%
from zfish.plot.plot_commons import cmaps

df_scp = stitch_normalized_ccps(
    df_res_clean, group_column="cycle_corr", ccp_column="NormalizedCCP_shift"
)

fig, ax = plt.subplots(figsize=(4, 1.6), dpi=300)
ax = sns.scatterplot(
    df_scp.filter(stratified_sample("cycle_corr", 10_000)).filter(
        pl.col("cycle_corr") < 14
    ),
    x="SingleCellPseudotime",
    # y="PCNA.0_Mean",
    y="DAPI.1_Sum",
    s=0.2,
    alpha=0.3,
    c="0.3",
    # hue="ann0",
    # hue_order=["M ana", "M telo", "S early", "S", "M pro", "M meta"],
    # palette="turbo",
    # palette=cmaps.ccp_hex,
)
ax = sns.scatterplot(
    df_scp,
    x="SingleCellPseudotime",
    # y="PCNA.0_Mean",
    y="DAPI.1_Sum",
    s=5,
    alpha=0.9,
    hue="ann0",
    hue_order=["M ana", "M telo", "S early", "S", "M pro", "M meta"],
    # palette="turbo",
    palette=[
        cmap.Color(e).hex
        for e in cmaps.ccp.reversed()(np.linspace(0, 1, 6, endpoint=False))
    ],
    legend=False,
)
ax.set_yscale("log")

# %%

# %%
ax = sns.scatterplot(
    df_res_clean.filter(pl.col("cycle_corr_pooled") == 13).sample(10_000),
    x="NormalizedCCP_shift",
    # y="PCNA.0_Mean",
    y="DAPI.1_Sum",
    # y="PhysicalSize",
    # y="Roundness",
    s=1,
    alpha=0.2,
    hue="PCNA.0_Mean",
    # hue="DistanceToCircleCCP",
    # hue="cycle_corr_pooled",
    palette="viridis",
    # hue_norm=(0.6, 1.0),
    legend=False,
)
# ax.set_ylim(0, 2000)
ax.set_yscale("log", base=2)
# ax.set_yscale('log')

# %%
(
    pl.concat([e.with_row_index().with_columns(df_nuc_new["roi"]) for e in res_ccp])
    .group_by(["roi", "cycle_corr"])
    .agg(pl.col("dist_to_CCP").mean())
    .sort(["roi", "cycle_corr"])
    .with_columns(pl.col("dist_to_CCP"))
    .with_columns(pl.col("cycle_corr").alias("cycle_estimate"))
    .drop("cycle_corr")
    .with_columns(
        distance_decay(
            "dist_to_CCP", decay_dist=1.0, decay_alpha=4.0, error="mse"
        ).alias("cycle_proba")
    )
    # .group_by("roi", maintain_order=True)
    # .agg(pl.col("dist_to_CCP"))
    # .select(
    #     pl.col("roi"),
    #     (pl.col("dist_to_CCP") / pl.col("dist_to_CCP").list.sum()).alias(
    #         "cycle_probas"
    #     ),
    # )
    .join(
        df_emb.with_columns(pl.col("roi").cast(pl.String)).select(
            "roi", "cycle", "cycle_corr"
        ),
        on="roi",
    )
).group_by("roi").agg(
    pl.col("cycle_estimate").sort_by("cycle_proba", descending=True).first(),
    pl.col("cycle").first(),
    pl.col("cycle_corr").first(),
).sort("cycle_estimate").pipe(
    with_jitter,
    cols=["cycle", "cycle_estimate", "cycle_corr"],
    # ).pipe(
    #     run_plot,
    #     func=lambda x: sns.scatterplot(x.to_pandas(), x="cycle_estimate", y="cycle_corr"),
).pipe(run_plot, lambda x: sns.scatterplot(x, x="cycle", y="cycle_corr"))

# %%

ax = sns.scatterplot(
    stitch_normalized_ccps(
        pl.concat([df_ccp, df_try.drop("CCP", "NormalizedCCP")], how="horizontal"),
        group_column="cycle_corr_pooled",
    ),
    x="SingleCellPseudotime",
    y="PCNA.0_Mean",
    s=1,
    alpha=0.3,
)

ax.set_yscale("log")
# %%


extrema = find_extrema_in_ccp((ccp[:, 0]), offset=0)
ccp_aligned, extrema_aligned = align(ccp[:, 0], extrema, max_value=(2 * np.pi))

# %%
from paper.ccp_normalize_align import cut_normalized_ccp

df_ccp = df_nuc_new.with_columns(
    res.with_columns(
        df_best_cycle.select(
            pl.col("cycle_corr").alias("cycle_corr_ccpt"),
            (pl.col("CCP") * 2 * np.pi).alias("CCP_multi"),
        )
    )
).sort("cycle_corr")


g = sns.FacetGrid(
    data=df_ccp.filter(stratified_sample(by="cycle_corr", n=10_000)).to_pandas(),
    col="cycle_corr",
    col_wrap=3,
    aspect=1,
)
g.map(
    sns.scatterplot,
    "umap_0",
    "umap_1",
    # "PhysicalSize",
    "CCP_multi",
    s=1,
    # palette=cmaps.ccp.to_mpl(),
    palette=Colormap("cmocean:phase").to_mpl(),
    # palette="inferno",
)
# sns.scatterplot(, x='umap_0', y='umap_1')

g.add_legend(markerscale=10.0)
# %%
df_ccp.group_by("roi").agg(pl.col("cycle_corr_ccpt"))["cycle_corr_ccpt"][
    7
].value_counts()

# %%
df_plot = df_ccp.filter(stratified_sample("cycle_corr", 10_000))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

ax.scatter(
    df_plot["umap_0"],
    df_plot["umap_1"],
    df_plot["umap_2"],
    s=1,
    # c=df_plot["DAPI.1_Sum"].log(base=2).clip(19.5, 20.5),
    # c=df_plot.with_columns(pl.col("celltype_pred")=='EVL')['celltype_pred'],
    # c=df_plot["DistanceToCircleCCP"],
    c=df_plot["cycle_corr"].cast(pl.UInt32),
    # cmap=Colormap(cmaps.division_cycle).to_mpl(),
    # cmap=Colormap(Colormap('set1').lut()[:7, :]).to_mpl(),
)
# %%

# %%
df_plot = df_ccp.filter(stratified_sample("cycle_corr", 10_000)).filter(
    pl.col("cycle_corr") == cycle
)
cycle = 13

fig, axs = plt.subplots(2, 2, sharex="col", sharey="row", dpi=300)
axs[0, 0].set_aspect("equal")

axs_ = list(axs.flatten())[:3]
ax_remove = list(axs.flatten())[-1]

umap_cols = [("umap_0", "umap_1"), ("umap_2", "umap_1"), ("umap_0", "umap_2")]
hue = "DistanceToCircleCCP"
palette = "inferno_r"

for ax, umap_c in zip(axs_, umap_cols):
    # hue = [e for e in ['umap_0', 'umap_1', 'umap_2'] if e not in umap_c][0]
    sns.scatterplot(
        df_ccp.filter(stratified_sample("cycle", 10_000))
        .filter(pl.col("cycle_corr") == cycle)
        .to_pandas(),
        x=umap_c[0],
        y=umap_c[1],
        s=2,
        alpha=0.5,
        hue=hue,
        legend=False,
        # palette=Colormap("cmocean:phase").to_mpl(),
        palette=palette,
        # palette='grey',
        ax=ax,
    )

plot_ccp_transformers(pop, ax=axs[0, 0], project_axis=2, emphasize=(cycle,))
plot_ccp_transformers(
    pop, ax=axs[1, 0], project_axis=1, emphasize=(cycle,), transpose=False
)
plot_ccp_transformers(
    pop, ax=axs[0, 1], project_axis=0, emphasize=(cycle,), transpose=True
)
# axs[1, 1].remove()
for ax in axs_:
    ax.set_aspect("equal")
ax_remove.set_aspect("equal")

plt.tight_layout()


# %%
plt.style.use([BASE])
fig, ax = plt.subplots()
sns.kdeplot(
    df_ccp.with_columns((pl.col("CCP")) % (2 * np.pi))
    # .filter(pl.col("CCP").is_between(1.45, 1.55))
    .to_pandas(),
    x="CCP",
    hue="cycle_corr",
    cut=0,
    common_norm=False,
    palette="Set1",
)


# %%


fig, ax = plt.subplots()
sns.scatterplot(
    pl.concat([df_ccp, df_ccp_avg], how="horizontal"),
    x="CCP_mean",
    y="PCNA.0_Mean",
    s=0.4,
    hue="dist_to_CCP_mean",
    legend=False,
)

# %%
pl.concat([df_ccp, df_ccp_avg], how="horizontal").filter(pl.col("CCP_var") > 0.003)


# %%
df_plot = (
    pl.concat(
        [
            df_ccp,
            df_ccp_avg,
            df_best_cycle.select(pl.col("CCP").name.suffix("_closest")),
        ],
        how="horizontal",
    ).with_columns((pl.col("CCP_closest") * 2 * np.pi))
    # .with_columns(pl_circdiff("CCP_closest", "CCP"))
    # .filter(stratified_sample(by="cycle_corr", n=10_000))
)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

ax.scatter(
    df_plot["umap_0"],
    df_plot["umap_1"],
    df_plot["umap_2"],
    s=1,
    # c=df_plot["DAPI.1_Sum"].log(base=2).clip(19.5, 20.5),
    # c=df_plot.with_columns(pl.col("celltype_pred")=='EVL')['celltype_pred'],
    c=df_plot["CCP_closest"],
    # c=df_plot["cycle_corr"].cast(pl.UInt32),
    # cmap=Colormap(cmaps.division_cycle).to_mpl(),
    # cmap=Colormap(Colormap('set1').lut()[:7, :]).to_mpl(),
)

# %%
sns.kdeplot(
    df_plot.with_columns((pl.col("CCP_closest"))),
    x="CCP_closest",
    cut=0,
    hue="cycle_corr",
    common_norm=False,
    legend=False,
)
# %%
sns.kdeplot(
    df_plot,
    x="CCP",
    cut=0,
    hue="cycle_corr",
    common_norm=False,
    legend=False,
)
# %%


def _shift_ccp(df_group, ccp_column, pcna_column, target=np.pi):
    n_sections = 40
    every_string = f"{df_group.height // n_sections}i"
    ccp_value_min = (
        df_group.select(ccp_column, pcna_column)
        .sort(ccp_column)
        .with_row_index()
        .cast(pl.Int64)
        .group_by_dynamic(index_column="index", every=every_string)
        .agg(pl.col(pcna_column).mean(), pl.col(ccp_column).cast(pl.Float64).mean())
        .select(pl.col(ccp_column).get(pl.col(pcna_column).arg_min()))
    ).item()

    shift = target - ccp_value_min

    return df_group.with_columns((pl.col(ccp_column) + shift) % (2 * np.pi))


def cut_normalized_ccp_robust(
    df_all,
    n_sphase_slices=6,
    group_column: str | None = "cycle_corr_pooled",
    ccp_column="NormalizedCCP",
    pcna_column="PCNA.0_Mean",
    lowess_n_samples=200,
    lowess_frac=0.07,
    dy_peak_prominence=0.3,
    ddy_peak_prominence=0.1,
    ddy_smoother=5,
    plot_results=False,
    n_samples=10_000,
    max_attempts=3,
):
    if group_column is None:
        df_all.select(ccp_column, pcna_column).sort(ccp_column)
        group_column = pl.lit(None)
    else:
        df_all.select(group_column, ccp_column, pcna_column).sort(
            group_column, ccp_column
        )
    ress = []
    targets = list(np.linspace(0, 2 * np.pi, max_attempts, endpoint=False))

    for name, df_group in df_all.group_by(group_column, maintain_order=True):
        i = 0
        res = None

        while True:
            if i >= max_attempts:
                print(
                    f"no solution found after {max_attempts} attempts. try changing other parameters."
                )
                break

            try:
                res = cut_normalized_ccp(
                    df_group,
                    n_sphase_slices=n_sphase_slices,
                    group_column=None,
                    ccp_column=ccp_column,
                    pcna_column=pcna_column,
                    lowess_n_samples=lowess_n_samples,
                    lowess_frac=lowess_frac,
                    dy_peak_prominence=dy_peak_prominence,
                    ddy_peak_prominence=ddy_peak_prominence,
                    ddy_smoother=ddy_smoother,
                    plot_results=plot_results,
                    n_samples=n_samples,
                ).select(ccp_column, "ClassCCP")
                print(f"solution found for {name}")
                break
            except (RuntimeError, ValueError) as e:
                print(e)
                df_group = _shift_ccp(
                    df_group,
                    ccp_column=ccp_column,
                    pcna_column=pcna_column,
                    target=targets[i],
                )
            i += 1
        # print(res)
        if res is None:
            ress.append(
                pl.DataFrame(
                    data={
                        ccp_column: [None for _ in range(df_group.height)],
                        "ClassCCP": [None for _ in range(df_group.height)],
                    },
                    schema={ccp_column: pl.Float64, "ClassCCP": pl.Categorical},
                )
            )
        else:
            ress.append(res)
    return ress
