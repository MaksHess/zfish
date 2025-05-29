# %%
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from zfish.features.polars_utils import drop_null_columns
from zfish.multi_table.features_aggregate import (
    Count,
    Mean,
    aggregate,
    aggregate_many_to_one,
    aggregate_one_to_one,
)
from zfish.multi_table.schema_migration import features_to_wide, to_wide
from zfish.multi_table.schemas_v2 import sel

# from zfish.features.constants import LabelFeatureQuery, IntensityFeatureQuery, CorrelationFeatureQuery
from zfish.multi_table.tables_io import join, scan_features, scan_resources
from zfish.preprocessing.outlier_ranges import OUTLIERS_NUCLEI, mark_outliers_individual

TABLES_MFRAME_PATH = (
    r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\multi_table\mframe"
)
TABLES_OUTPUT_PATH = r"C:\Users\hessm\Documents\Programming\Python\zfish\scripts_shayani\features\aggregated"

# %% Load Base Features and Resources
f = scan_features(root=TABLES_MFRAME_PATH).collect()
r = scan_resources(root=TABLES_MFRAME_PATH).collect()


# %% Pivot Features to wide format
fw = features_to_wide(f)


# %% Add nuclei features to nuclei (i.e. prepend 'nucleiRaw3_' for consistent naming)
df_nuc = join(
    *[
        aggregate_one_to_one(r.hierarchy, df, aggregate_to="nucleiRaw3").pipe(to_wide)
        for df in fw._tables.values()
    ]
).pipe(drop_null_columns)

# %% Compute 3 sets of outliers (segmentation, ccp & zga)
df_nuc_outliers_seg = mark_outliers_individual(
    df_nuc, OUTLIERS_NUCLEI["segmentation"], plot_results=True, outliers_only=True
)

df_nuc_outliers_ccp = mark_outliers_individual(
    df_nuc,
    OUTLIERS_NUCLEI["segmentation"]
    + OUTLIERS_NUCLEI["alignment"]
    + OUTLIERS_NUCLEI["intensity_ccp"],
    plot_results=True,
    outliers_only=True,
)


df_nuc_outliers_zga = mark_outliers_individual(
    df_nuc,
    OUTLIERS_NUCLEI["segmentation"]
    + OUTLIERS_NUCLEI["alignment"]
    + OUTLIERS_NUCLEI["intensity_zga_iqr"],
    plot_results=True,
    outliers_only=True,
)

# %% Add cell, cyto and nuc features to cells
df_cell = (
    join(
        *[
            aggregate_one_to_one(r.hierarchy, df, aggregate_to="cells").pipe(to_wide)
            for df in fw._tables.values()
        ]
    )
    .pipe(drop_null_columns)
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

df_cell_outliers_seg = aggregate_one_to_one(r.hierarchy, df_nuc_outliers_seg).drop(
    "o.child"
)
# %% Aggregate nuclear and cellular features to embryo (after removing segmentation outliers)

df_emb = (
    aggregate(
        r.hierarchy,
        r.object_types,
        (
            r.label_objects.filter(pl.col("o").is_in(["nucleiRaw3", "cells"]))
            .join(df_nuc_outliers_seg, how="anti", on=["roi", "o", "label"])
            .join(df_cell_outliers_seg, how="anti", on=["roi", "o", "label"])
            .join(
                f.label.select(sel.idx, "PhysicalSize"),
                on=["roi", "o", "label"],
                how="left",
            )
            .select(sel.idx, "PhysicalSize")
        ),
        aggregation=(Count(), Mean("PhysicalSize")),
    )
    .with_columns(
        pl.col("nucleiRaw3__Count").log(base=2).alias("log2_nucleiRaw3__Count")
    )
    .with_columns(
        pl.col("log2_nucleiRaw3__Count").round().cast(pl.UInt32).alias("cycle")
    )
    .with_columns(
        pl.when(pl.col("cycle") < 9).then(pl.lit(9)).otherwise(pl.col("cycle")).alias('cycle_pooled')
    )
    .with_columns(
        pl.col("log2_nucleiRaw3__Count")
        .cut(np.arange(61, 131, 10) / 10, labels=[str(e) for e in np.arange(6, 14)])
        .cast(pl.String)
        .cast(pl.UInt32)
        .alias("cycle_up")
    )
)

# %% Write object tables

df_nuc.write_parquet(Path(TABLES_OUTPUT_PATH) / "nucleiRaw3_CCP.parquet")
df_cell.write_parquet(Path(TABLES_OUTPUT_PATH) / "cells_CCP.parquet")
df_emb.write_parquet(Path(TABLES_OUTPUT_PATH) / "embryoRaw_CCP.parquet")


