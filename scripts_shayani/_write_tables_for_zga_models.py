# %%
"""Adapted from _Fig4_TransctiptionModelling_First..."""

from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs

from zfish.features.polars_utils import drop_null_columns, drop_null_rows
from zfish.multi_table.features_aggregate import (
    aggregate,
    aggregate_many_to_one,
    subtract_background,
)
from zfish.multi_table.schemas_v2 import sel
from zfish.multi_table.tables_io import (
    Resources,
    join,
    scan_resources_and_features,
    to_wide,
)
from zfish.plot.plot_commons import tpaths
from zfish.polars.utils import (
    plot_df_nulls,
)
from zfish.preprocessing.outlier_ranges import (
    ALIGNMENT_OUTLIER_NAMES,
    INTENSITY_OUTLIER_NAMES,
    LABEL_OUTLIER_NAMES,
    MDL_INTENSITY_OUTLIERS,
    compute_nuclei_outliers,
    mark_outliers_individual,
)
from zfish.preprocessing.sliding_window_samples import stratified_sample

pl.enable_string_cache()
STYLE_SHEET = r"C:\Users\hessm\Documents\Programming\Python\zfish\paper\base.mplstyle"
SAVE_FIGURE = False
OUTPUT_FOLDER = Path(tpaths.transcription)
CCP_QUICKSAVE = (
    r"C:\Users\hessm\Documents\Programming\Python\thesis\Data\ccp1_and_ccp2.parquet"
)
CHANNELS = [
    "Pol-II-S2P.0",
    "FLAG.0",
    "PCNA.0",
    "DAPI.1",
    "H3K27Ac.1",
    "bCatenin.1",
    "pH3.1",
    "H2B.2",
    "ALYREF.2",
    "Pol-II-S5P.2",
    "Nanog.3",
]
CHANNELS_AGGREGATE_FEATURES = ["Mean", "Sum", "Mean_wbg", "Sum_wbg"]

CLIP_10_POW = 0

clip = 10**CLIP_10_POW
# %%
r0 = Resources.from_schemas()
# %%
r_raw, f_raw = scan_resources_and_features(validate_schema=True)


df_intensity_nobg = subtract_background(
    f_raw.filter(pl.col("o").is_in(["nucleiRaw3", "cyto"])),
    r_raw.collect(),
    channels=CHANNELS,
    clip=clip,
)

f_nobg = f_raw.pipe_table(
    lambda x: pl.concat(
        [
            df_intensity_nobg.select(sel.idx, *CHANNELS_AGGREGATE_FEATURES).drop("m"),
            x.filter(pl.col("c").is_in(CHANNELS))
            .select(sel.idx, "Mean", "Sum")
            .filter(pl.col("o") != "nucleiRaw3")
            .collect(),
        ],
        how="diagonal",
    ),
    "intensity",
)
# %%


LABEL_OBJECT_IDX = ["o", "roi", "label"]
OUT_IDX = ["roi", "label"]

from zfish.multi_table.tables_io import safe_collect


def add_morphology_features(r, f):
    return (
        aggregate(
            r.hierarchy,
            r.object_types,
            f.label,
            aggregate_to="cells",
        )
        .pipe(drop_null_columns)
        .drop(cs.matches("BoundingBox"), cs.matches("cells_Centroid"))
        .with_columns(
            (pl.col("cells_PhysicalSize") - pl.col("nucleiRaw3_PhysicalSize")).alias(
                "cyto_PhysicalSize"
            )
        )
        .with_columns(
            (pl.col("nucleiRaw3_PhysicalSize") / pl.col("cyto_PhysicalSize")).alias(
                "nuc/cyto_PhysicalSizeRatio"
            )
        )
    )


def aggregate_intensity_features(r, f):
    return aggregate(
        r.hierarchy,
        r.object_types,
        f.intensity.filter(pl.col("c").is_in(CHANNELS))
        .pipe(safe_collect)
        .pipe(to_wide),
        aggregate_to="cells",
    )


def aggregate_neighborhood_features(r, f):
    return aggregate(
        r.hierarchy,
        r.object_types,
        f.density_count.filter(pl.col("nhood").is_in(["DELAUNAY:1", "TOUCH:1"])),
        aggregate_to="cells",
    )


def aggregate_density_features(r, f):
    return (
        f.density_count.filter(~pl.col("nhood").is_in(["DELAUNAY:1", "TOUCH:1"]))
        .pipe(safe_collect)
        .pipe(to_wide)
        .join(
            f.density_distance.pipe(safe_collect).pipe(to_wide).pipe(drop_null_columns),
            on=LABEL_OBJECT_IDX,
        )
    )


def aggregate_and_join_features(r_raw, f_raw):
    df_density = aggregate_density_features(r_raw, f_raw)
    df_morphology = add_morphology_features(r_raw, f_raw)
    df_intensity = aggregate_intensity_features(r_raw, f_raw)
    df_touching_neighbors = aggregate_neighborhood_features(r_raw, f_raw)

    return (
        df_morphology.join(df_intensity.drop("o"), on=OUT_IDX)
        .join(df_touching_neighbors.drop("o"), on=OUT_IDX)
        .join(df_density.drop("o"), on=OUT_IDX)
        .join(f_raw.classifier.pipe(safe_collect).pipe(to_wide).drop("o"), on=OUT_IDX)
    )


def mark_intensity_outliers(df_features):
    return pl.concat(
        [
            df_features.select(sel.idx),
            df_features.pipe(
                mark_outliers_individual, MDL_INTENSITY_OUTLIERS, plot_results=True
            ),
        ],
        how="horizontal",
    ).filter(pl.col("Any Outliers"))


df_features = aggregate_and_join_features(r_raw, f_nobg)
df_intensity_iqr_outliers = mark_intensity_outliers(df_features)
df_features_clean = df_features.join(
    df_intensity_iqr_outliers, how="anti", on=["roi", "o", "label"]
)
# %%
df_all_outliers = compute_nuclei_outliers(f_raw.collect(), return_all=True)
df_debris_outliers = df_all_outliers.filter(pl.col("debris_proba")).select(sel.idx)
df_nuc_outliers = df_all_outliers.filter(
    pl.any_horizontal(cs.by_dtype(pl.Boolean))
).select(sel.idx)


# %%
r_nodebris = (
    r_raw.collect()
    .filter(pl.col("o") == "nucleiRaw3")
    .pipe_tables(
        lambda x: x.join(df_debris_outliers, on=["o", "roi", "label"], how="anti")
    )
)


df_meta = (
    aggregate_many_to_one(r_nodebris.hierarchy, r_raw.label_objects)
    .drop("m")
    .pipe(to_wide, sep="_")
    .with_columns(
        pl.col("nucleiRaw3__Count").log(base=2).alias("log2_nucleiRaw3__Count")
    )
    .with_columns(
        pl.col("log2_nucleiRaw3__Count").round().cast(pl.UInt32).alias("cycle")
    )
    .with_columns(
        pl.col("log2_nucleiRaw3__Count")
        .cut(np.arange(61, 131, 10) / 10, labels=[str(e) for e in np.arange(6, 14)])
        .cast(pl.String)
        .cast(pl.UInt32)
        .alias("cycle_up")
    )
)

df_meta_nocontrol = df_meta.join(
    r_raw.rois.pipe(safe_collect).join(
        r_raw.wells.pipe(safe_collect), left_on="well", right_on="well"
    ),
    on="roi",
).filter(~pl.col("is_control_well"))

# %%
df_nuc_ccp = (
    pl.scan_parquet(CCP_QUICKSAVE)
    .select(
        pl.col("roi").cast(pl.Categorical),  # .alias("idx.roi"),
        pl.col("o").cast(pl.Categorical).alias("o"),  # .alias("idx.o"),
        pl.col("label"),  # .alias("idx.label"),
        pl.col("CCP"),
        pl.col("NormalizedCCP"),
        # pl.col("SingleCellPseudotime"),
        pl.col("cycle"),
        pl.when(pl.col("cycle") > 9)
        .then(pl.col("cycle"))
        .otherwise(pl.lit(9))
        .alias("cycle_pooled"),
        pl.col("log2_nuc__Count_corr"),
    )
    .collect()
)

df_nuc = (
    df_features.join(df_meta, on="roi")
    .join(df_meta_nocontrol, on="roi", how="semi")
    .drop("cycle")
    # .drop('cycle_pooled')
    .join(df_nuc_ccp.drop("o"), on=["roi", "label"], how="left")
)


df_nuc.select(sel.idx | cs.by_name("celltype_pred"), "CCP").pipe(
    plot_df_nulls, plot_index=("celltype_pred", "o", "roi")
)
# %%

df_nuc.filter(pl.col("CCP").is_not_null()).filter(pl.col("cycle") == 10).hvplot.scatter(
    # "CCP",
    "NormalizedCCP",
    # "nucleiRaw3_Pol-II-S2P.0_Mean",
    "nucleiRaw3_PCNA.0_Mean",
    alpha=0.05,
    color="cellcycle3c_pred",
    size=1,
    logy=True,
)

# %%
df_changepoint = (
    df_nuc.filter(pl.col("CCP").is_not_null())
    .filter(pl.col("cycle") == 10)
    .select("NormalizedCCP", pl.col("nucleiRaw3_PCNA.0_Mean").log())
    .sample(10_000)
    .sort("NormalizedCCP")
)

# sns.scatterplot(df_changepoint, x="NormalizedCCP", y="nucleiRaw3_PCNA.0_Mean", s=1)
# plt.plot(df_changepoint["nucleiRaw3_PCNA.0_Mean"])

# algo = rpt.Dynp(model="l2").fit(df_changepoint[["nucleiRaw3_PCNA.0_Mean"]].to_numpy())
# result = algo.predict(n_bkps=2)
# %%
from functools import partial

from zfish.features.ccp.ccp_normalize_align import cut_normalized_ccp

names_map = {
    "nucleiRaw3": "nuc",
    "cells": "cell",
}


def replace_with_map(s: str, m: dict[str, str] = names_map):
    s_out = s
    for k, v in m.items():
        s_out = s_out.replace(k, v)
    return s_out


old_cols = df_nuc.columns
new_cols = list(map(replace_with_map, df_nuc.columns))

df_nuc.rename(dict(zip(old_cols, new_cols)))
# %%
df_some = df_nuc.filter(pl.col("cycle") != 13)

PCNA_CLIP = 1

df_out = cut_normalized_ccp(
    df_some.with_columns(
        pl.when(pl.col("cycle") < 10)
        .then(pl.lit(9))
        .otherwise(pl.col("cycle"))
        .alias("cycle_pooled")
    )
    .filter(pl.col("cycle") < 13)
    .with_columns(
        pl.col("nucleiRaw3_PCNA.0_Mean")
        .clip(PCNA_CLIP)
        .alias("nucleiRaw3_PCNA.0_Mean_")
    ),
    n_sphase_slices=6,
    group_column="cycle_pooled",
    ccp_column="NormalizedCCP",
    pcna_column="nucleiRaw3_PCNA.0_Mean_",
    lowess_frac=0.05,
)

features = [
    "o",
    "roi",
    "label",
    "nuc_PhysicalSize",
    "nuc_Elongation",
    "nuc_Flatness",
    "nuc_Roundness",
    "nuc_FeretDiameter",
    "nuc_Perimeter",
    "nuc_EquivalentSphericalPerimeter",
    "nuc_EquivalentSphericalRadius",
    "nuc_PerimeterOnBorder",
    "nuc_PerimeterOnBorderRatio",
    "nuc_EquivalentEllipsoidDiameter-a",
    "nuc_EquivalentEllipsoidDiameter-b",
    "nuc_EquivalentEllipsoidDiameter-c",
    "nuc_PrincipalAxes-a-x",
    "nuc_PrincipalAxes-a-y",
    "nuc_PrincipalAxes-a-z",
    "nuc_PrincipalAxes-b-x",
    "nuc_PrincipalAxes-b-y",
    "nuc_PrincipalAxes-b-z",
    "nuc_PrincipalAxes-c-x",
    "nuc_PrincipalAxes-c-y",
    "nuc_PrincipalAxes-c-z",
    "nuc_Centroid-x",
    "nuc_Centroid-y",
    "nuc_Centroid-z",
    "cell_PhysicalSize",
    "cell_Elongation",
    "cell_Flatness",
    "cell_Roundness",
    "cell_Perimeter",
    "cell_EquivalentSphericalPerimeter",
    "cell_EquivalentSphericalRadius",
    "cell_PerimeterOnBorder",
    "cell_PerimeterOnBorderRatio",
    "cell_EquivalentEllipsoidDiameter-a",
    "cell_EquivalentEllipsoidDiameter-b",
    "cell_EquivalentEllipsoidDiameter-c",
    "cell_PrincipalAxes-a-x",
    "cell_PrincipalAxes-a-y",
    "cell_PrincipalAxes-a-z",
    "cell_PrincipalAxes-b-x",
    "cell_PrincipalAxes-b-y",
    "cell_PrincipalAxes-b-z",
    "cell_PrincipalAxes-c-x",
    "cell_PrincipalAxes-c-y",
    "cell_PrincipalAxes-c-z",
    "cyto_PhysicalSize",
    "nuc/cyto_PhysicalSizeRatio",
    "debris_annotation",
    "debris_pred",
    "debris_probas",
    "celltype_annotation",
    "celltype_pred",
    "celltype_probas",
    "cellcycle3c_annotation",
    "cellcycle3c_pred",
    "cellcycle3c_probas",
    "idx.o_right",
    "idx.label_right",
    "nuc__Count",
    "cell__Count",
    "log2_nuc__Count",
    "cycle",
    "cycle_up",
    "CCP",
    "NormalizedCCP",
    "DistanceToCircleCCP",
    "SingleCellPseudotime",
    "cycle_pooled",
    "ClassCCP",
]

selector = (
    cs.by_name(features)
    | cs.ends_with("_Mean_wbg")
    | cs.ends_with("_Mean")
    | cs.ends_with("_Sum_wgb")
    | cs.ends_with("_Sum")
    | cs.ends_with("_Count")
    | cs.starts_with("KNNd:")
)

df_out = (
    df_some.with_columns(df_out["ClassCCP"])
    .rename(
        dict(
            zip(
                df_some.with_columns(df_out["ClassCCP"]).columns,
                list(
                    map(
                        partial(replace_with_map, m=names_map),
                        df_some.with_columns(df_out["ClassCCP"]).columns,
                    )
                ),
            )
        )
    )
    .select(selector)
)

# %%
df_out.write_parquet(
    rf"C:\Users\hessm\Documents\Programming\Python\thesis\Data\for_model_clp_pow(10, {CLIP_10_POW}).parquet"
)

# %%
