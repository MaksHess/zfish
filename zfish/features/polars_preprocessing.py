# %%
import logging
from dataclasses import dataclass
from functools import partial, reduce
from operator import or_
from typing import Any, TypedDict, TypeVar

import polars as pl
import polars.selectors as cs
from polars.type_aliases import SelectorType

from zfish.analysis.hierarchy_aggregate import hierarchy_aggregate_count
from zfish.preprocessing.types import AnyFrameT

logger = logging.getLogger(__name__)

META_COLUMNS = reduce(
    or_,
    [
        cs.matches(f"^{feature}$")
        for feature in [
            "roi",
            "parent.embryoRaw",
            "nucleiRaw3_Count",
            "log2_nucleiRaw3_Count",
            "cycle",
            "well",
            "site",
            "is_control_well",
            "control_well_for_acquisition",
            "embryo",
        ]
    ],
)



def get_metadata(
    df: AnyFrameT,
    objects_to_count: tuple[str] = ("nucleiRaw3",),
    parent_selector: SelectorType = cs.by_name("roi") | cs.matches("parent.embryo"),
    control_wells: list[str] | None = None,
    object_column: SelectorType = cs.by_name("object"),
    drop_objects_with_no_parents: bool = True,
) -> pl.DataFrame:
    df_sub = df.filter(object_column.is_in(objects_to_count))
    df_meta = (
        hierarchy_aggregate_count(df_sub, parent_selector, object_column=object_column)
        .with_columns(cs.matches("_Count").log(base=2).cast(pl.Float32).prefix("log2_"))
        .with_columns(
            cs.matches("log2_nuc.*_Count").round(0).cast(pl.UInt16).alias("cycle")
        )
        .with_columns(pl.col("roi").str.split("_").alias("parts"))
        .with_columns(
            [
                pl.col("parts").list.first().alias("well"),
                pl.col("parts").list.slice(1).list.join("_").alias("site"),
            ]
        )
        .drop("parts")
    )
    if drop_objects_with_no_parents:
        assert (
            df_meta.select(cs.matches("parent\.")).width == 1
        ), "Don't know how to handle multiple parents"

        df_meta = df_meta.filter(
            (cs.matches("parent\.") != 0) & (cs.matches("parent\.").is_not_null())
        )

    if control_wells is not None:
        df_meta = df_meta.with_columns(
            [
                pl.col("well").is_in(control_wells).alias("is_control_well"),
                pl.col("well")
                .apply(lambda x: control_wells.index(x) if x in control_wells else 1000)
                .alias("control_well_for_acquisition"),
            ]
        )
    return df_meta.join(
        df_meta.filter(pl.col("parent.embryoRaw") == 1)
        .with_row_count()
        .with_columns(
            pl.concat_str(pl.lit("emb"), pl.col("row_nr"), separator="_").alias(
                "embryo"
            )
        )
        .select("roi", "parent.embryoRaw", "embryo"),
        on=cs.expand_selector(df_meta, parent_selector),
    )


def get_metadata_old(
    df: AnyFrameT,
    structures={"nucleiRaw3": "nuc_count"},
    control_wells: list[str] | None = None,
) -> pl.DataFrame:
    structure_names = list(structures.keys())
    structures_to_drop = [
        e
        for e in list(df.select(pl.col("object")).to_series().cast(pl.Utf8).unique())
        if e not in structure_names
    ]

    df_meta = (
        df
        # .pipe(show)
        .select(["roi", "object"])
        .filter(pl.col("object").is_in(structure_names))
        .pipe(safe_collect)
        .groupby(["roi", "object"])
        .agg([pl.col("roi").count().alias("_count")])
        # .pipe(show)
        .pivot(values="_count", index="roi", columns="object")
        # .pipe(show)
        .rename(structures)
        .with_columns(pl.col("nuc_count").log(base=2).cast(pl.Float32).prefix("log2_"))
        .with_columns(pl.col("log2_nuc_count").round(0).cast(pl.UInt16).alias("cycle"))
        .with_columns(pl.col("roi").str.split("_").alias("parts"))
        .with_columns(
            [
                pl.col("parts").list.first().alias("well"),
                pl.col("parts").list.slice(1).list.join("_").alias("site"),
            ]
        )
        .drop("parts")
    )
    if control_wells is not None:
        df_meta = df_meta.with_columns(
            [
                pl.col("well").is_in(control_wells).alias("is_control_well"),
                pl.col("well")
                .apply(lambda x: control_wells.index(x) if x in control_wells else 1000)
                .alias("control_well_for_acquisition"),
            ]
        )
    return df_meta


def safe_collect(df: AnyFrameT):
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    return df
