# %%
import re
from functools import reduce
from operator import or_
from typing import TYPE_CHECKING, Literal, TypeAlias

import polars as pl
import polars.selectors as cs
from polars._typing import SelectorType

from zfish.multi_table.object_hierarchy import compute_cardinality_hierarchy
from zfish.multi_table.schemas_v2 import CHILD_IDX, LABEL_OBJECT_IDX, PARENT_IDX, sel
from zfish.multi_table.table_base_mixin import join, safe_shape
from zfish.multi_table.tables_io import (
    read_resources_and_features,
    safe_lazy,
    to_wide,
)
from zfish.polars.column_nesting import nest, unnest
from zfish.preprocessing.types import AnyFrame

if TYPE_CHECKING:
    from polars.type_aliases import SelectorType

    IntoSelectorType: TypeAlias = str | tuple[str, ...] | SelectorType


pl.enable_string_cache()

RENAME_MAP = {"o.parent": "o", "label.parent": "label"}

# Aggregation: TypeAlias = Literal["Count", "Mean", "Max", "Min", "Quantile"]


def _to_selector(maybe_selector: "IntoSelectorType"):
    if cs.is_selector(maybe_selector):
        return maybe_selector
    else:
        return cs.by_name(maybe_selector)


def Count():
    return pl.len().alias("_Count")


def Mean(x: "IntoSelectorType" = (~sel.idx)):
    return _to_selector(x).mean().name.suffix("__Mean")


def Sum(x: "IntoSelectorType" = (~sel.idx)):
    return _to_selector(x).sum().name.suffix("__Sum")


def Min(x: "IntoSelectorType" = (~sel.idx)):
    return _to_selector(x).min().name.suffix("__Min")


def Max(x: "IntoSelectorType" = (~sel.idx)):
    return _to_selector(x).max().name.suffix("__Max")


def Median(x: "IntoSelectorType" = (~sel.idx)):
    return _to_selector(x).median().name.suffix("__Median")


def Quantile(q: float, x: "IntoSelectorType" = (~sel.idx)):
    return _to_selector(x).quantile(q).name.suffix(f"__Q({q:.2f})")


def split_tables_by_factor(tbls, factor="idx.o"):
    return tbls.pipe_tables(lambda x: x.select(pl.col(factor).unique()))


def select_singleton_dims(
    df: pl.DataFrame,
    column_selector: SelectorType = cs.by_dtype(
        pl.UInt8, pl.UInt16, pl.UInt32, pl.String, pl.Categorical
    ),
    # column_selector=cs.starts_with('idx.')
) -> pl.DataFrame:
    columns = cs.expand_selector(df, column_selector)
    # print(columns)
    return (
        df.select(pl.col(columns).unique().len() == 1)
        .transpose(include_header=True, column_names=("is_singleton",))
        .filter(pl.col("is_singleton"))
    )


def squeeze_singleton_dims(
    df: pl.DataFrame,
    column_selector: SelectorType = sel.idx,
):
    return df.pipe(select_singleton_dims, column_selector=column_selector)


def is_singleton(expr: pl.Expr) -> pl.Expr:
    return expr.unique().len() == 1


def is_full_rank(expr: pl.Expr) -> pl.Expr:
    return expr.unique().len() == expr.len()


def safe_rename(df, rename_map=RENAME_MAP):
    return df.rename({k: v for k, v in rename_map.items() if k in df})


def aggregate(
    df_hierarchy: AnyFrame,
    df_object_types: AnyFrame,
    df_features: AnyFrame,
    aggregation: pl.Expr | tuple[pl.Expr, ...] = Count(),
    aggregate_to: str = "embryoRaw",
    aggregate_from: tuple[str, ...] | None = None,
    drop_background_label=True,
    group_by=(
        sel.parent | sel.object_type
    ),  # Valid for wide tables i. e. indexed like label_objects (o, roi, label)
) -> pl.DataFrame:
    df_hierarchy = safe_lazy(df_hierarchy)
    df_object_types = safe_lazy(df_object_types)
    df_features = safe_lazy(df_features)

    df_cardinality = (
        compute_cardinality_hierarchy(df_hierarchy, df_object_types)
        .select("o", "parents", "cardinality")
        .explode("parents", "cardinality")
    )

    df_cardinality_self = pl.concat(
        [
            df_cardinality.filter(pl.col("parents") == aggregate_to),
            pl.DataFrame(
                {"o": aggregate_to, "parents": aggregate_to, "cardinality": "1:1"},
                schema=df_cardinality.collect_schema(),
            ).lazy(),
        ]
    )
    dfs_agg = []
    # print(df_cardinality_self)
    for cardinality, df_o in df_cardinality_self.collect().group_by(
        ("cardinality",), maintain_order=True
    ):
        if aggregate_from is None:
            aggregate_from_group = tuple(df_o["o"].to_list())
        if cardinality[0] == "m:1":
            df_agg_m1 = aggregate_many_to_one(
                df_hierarchy,
                df_features,
                aggregation=aggregation,
                aggregate_to=aggregate_to,
                aggregate_from=aggregate_from_group,
                drop_background_label=drop_background_label,
                group_by=group_by,
            ).pipe(to_wide, sep="_")
            dfs_agg.append(df_agg_m1)
        elif cardinality[0] == "1:1":
            df_agg_11 = aggregate_one_to_one(
                df_hierarchy,
                df_features,
                aggregate_to=aggregate_to,
                aggregate_from=aggregate_from_group,
                drop_background_label=drop_background_label,
            )
            # return df_agg_11
            dfs_agg.append(df_agg_11.collect().pipe(to_wide, sep="_"))
        else:
            raise ValueError(f"Unknown cardinality: {cardinality[0]}")
    # return dfs_agg
    return join(*dfs_agg)


def aggregate_many_to_one(
    df_hierarchy: pl.DataFrame,
    df_features: pl.DataFrame,
    aggregation: pl.Expr | tuple[pl.Expr, ...] = Count(),
    aggregate_to: str = "embryoRaw",
    aggregate_from: tuple[str, ...] | None = ("cells", "nucleiRaw3"),
    drop_background_label=True,
    group_by=(
        sel.idx - cs.by_name("label.child")
    ),  # Valid for wide tables i. e. indexed like label_objects (o, roi, label)
) -> pl.DataFrame:
    if aggregate_from is None:
        aggregate_from = tuple(
            df_hierarchy.select(pl.col("o.child").unique()).collect()
        )

    df_agg = (
        df_hierarchy.filter((pl.col(PARENT_IDX[1]) == aggregate_to)).filter(
            pl.col("o.child").is_in(aggregate_from)
        )
        # .pipe(debug)
    )
    if drop_background_label:
        df_agg = df_agg.filter(pl.col(PARENT_IDX[2]) != 0)
    df_agg = (
        df_agg.lazy()
        .join(
            df_features.lazy(),
            how="inner",
            left_on=CHILD_IDX,
            right_on=["roi", "o", "label"],
        )
        .group_by(group_by)
        .agg(aggregation)
        # .sort("idx.label")
        .collect()
    ).rename(dict(zip(PARENT_IDX, LABEL_OBJECT_IDX)))

    return df_agg


def aggregate_one_to_one(
    df_hierarchy: pl.DataFrame,
    df_features: pl.DataFrame,
    aggregate_to: str = "cells",
    aggregate_from: tuple[str, ...] | None = ("cells", "nucleiRaw3", "cyto"),
    drop_background_label=True,
):
    """'aggregate' 1:1 (i. e. copy features and prepend the child structure to the feature name)."""
    if aggregate_from is None:
        aggregate_from = tuple(df_hierarchy["o.child"].unique().to_list())
    if aggregate_to in aggregate_from:
        df_hierarchy = df_hierarchy.pipe(_add_self_hierarchy)
    df_agg = (
        df_hierarchy.filter(pl.col("o.parent") == aggregate_to)
        .filter(pl.col("o.child").is_in(aggregate_from))
        .join(
            df_features,
            left_on=("roi", "o.child", "label.child"),
            right_on=("roi", "o", "label"),
        )
        .drop("label.child")
        .rename(RENAME_MAP)
    )
    if drop_background_label:
        df_agg = df_agg.filter(pl.col("label") != 0)
    return df_agg


def _add_self_hierarchy(df_hierarchy: pl.DataFrame, include_children_as_parents=True):
    """assure eachy object is it's own child for aggregation."""
    if include_children_as_parents:
        return pl.concat(
            [
                df_hierarchy,
                pl.concat(
                    [
                        df_hierarchy.select(PARENT_IDX),
                        df_hierarchy.select(
                            [
                                "roi",
                                pl.col("o.child").alias("o.parent"),
                                pl.col("label.child").alias("label.parent"),
                            ]
                        ),
                    ]
                )
                .unique(PARENT_IDX)
                .with_columns(
                    pl.col("o.parent").alias("o.child"),
                    pl.col("label.parent").alias("label.child"),
                ),
            ]
        )
    return pl.concat(
        [
            df_hierarchy,
            df_hierarchy.select(PARENT_IDX)
            .unique(PARENT_IDX)
            .with_columns(
                pl.col("o.parent").alias("o.child"),
                pl.col("label.parent").alias("label.child"),
            ),
        ]
    )


def subtract_background(f, r, o="nucleiRaw3", o_bg="cyto", channels=None, clip=1):
    FEATURES = ["Mean", "Sum"]

    channel_filter = pl.lit(True) if channels is None else pl.col("c").is_in(channels)
    df = r.label_objects.filter(pl.col("o") == o).select(sel.idx)

    df = join(
        df,
        (
            f.intensity.filter(pl.col("o") == o)
            .filter(channel_filter)
            .select(sel.idx, cs.by_name(FEATURES))
        ).collect(),
        (
            f.intensity.filter(pl.col("o") == o_bg)
            .filter(channel_filter)
            .select(sel.idx, cs.by_name(FEATURES).name.suffix("_bg"))
            .drop("o")
        ).collect(),
        (f.label.filter(pl.col("o") == o).select(sel.idx, "PhysicalSize")).collect(),
        how="left",
    )
    return (
        df.with_columns(
            (pl.col("Mean") - pl.col("Mean_bg").fill_null(0.0)).clip(lower_bound=clip),
            pl.col("Mean").alias("Mean_wbg"),
            pl.col("Sum").alias("Sum_wbg"),
        )
        .with_columns((pl.col("Mean") * pl.col("PhysicalSize")).alias("Sum"))
        .drop("PhysicalSize")
    )


def test_aggregate_many_to_one():
    import pandera.polars as pa

    df_hierarchy = pl.DataFrame(
        {
            "roi": ["site0"] * 10 + ["site1"] * 10 + ["site0"] * 5 + ["site1"] * 5,
            "o.parent": ["emb"] * 20 + ["cell"] * 10,
            "label.parent": [1] * 20 + (list(range(5))) * 2,
            "o.child": (["cell"] * 5 + ["nuc"] * 5) * 2 + ["nuc"] * 10,
            "label.child": list(range(5)) * 6,
        },
        schema={
            "roi": pl.Categorical,
            "o.parent": pl.Categorical,
            "label.parent": pl.UInt32,
            "o.child": pl.Categorical,
            "label.child": pl.UInt32,
        },
    )
    df_o = pl.DataFrame(
        {
            "o": ["emb", "cell", "nuc"],
            "parents": [[], ["emb"], ["emb"]],
            "hierarchy_level": [0, 1, 2],
        },
        schema={
            "o": pl.Categorical,
            "parents": pl.List(pl.Categorical),
            "hierarchy_level": pl.UInt8,
        },
    )
    df_feat = None
    df_feat = pl.DataFrame(
        {
            "roi": ["site0"] * 5 + ["site1"] * 5 + ["site0"] * 5 + ["site1"] * 5,
            "o": ["cell"] * 10 + ["nuc"] * 10,
            "label": list(range(5)) * 4,
            "f": [0.5] * 10 + [1.5] * 10,
        },
        schema={
            "roi": pl.Categorical,
            "o": pl.Categorical,
            "label": pl.UInt32,
            "f": pl.Float64,
        },
    )

    AGG_TO = "emb"
    AGG_FROM = ("cell", "nuc")
    df_agg = aggregate_many_to_one(
        df_hierarchy,
        df_feat,
        aggregate_from=AGG_FROM,
        aggregate_to=AGG_TO,
        aggregation=(Count(), Mean("f"), Sum("f")),
    )

    result_schema = pa.DataFrameSchema(
        {
            "roi": pa.Column(pl.Categorical, checks=pa.Check.isin(["site0", "site1"])),
            "o": pa.Column(pl.Categorical, checks=pa.Check.isin([AGG_TO])),
            "label": pa.Column(pl.UInt32, checks=pa.Check.isin([1])),
            "o.child": pa.Column(pl.Categorical, checks=pa.Check.isin(list(AGG_FROM))),
            "_Count": pa.Column(pl.UInt32, checks=pa.Check.eq(5)),
            "f__Mean": pa.Column(float, checks=pa.Check.isin([0.5, 1.5])),
            "f__Sum": pa.Column(float, checks=pa.Check.isin([2.5, 7.5])),
        }
    )
    df_agg.pipe(result_schema.validate)

    # automatically converts to wide format
    df_agg2 = aggregate(
        df_hierarchy, df_o, df_feat, (Count(), Sum("f"), Mean("f")), aggregate_to="emb"
    )

    result_schema2 = pa.DataFrameSchema(
        {
            "roi": pa.Column(pl.Categorical, checks=pa.Check.isin(["site0", "site1"])),
            "o": pa.Column(pl.Categorical, checks=pa.Check.isin([AGG_TO])),
            "label": pa.Column(pl.UInt32, checks=pa.Check.isin([1])),
            "nuc__Count": pa.Column(pl.UInt32, checks=pa.Check.eq(5)),
            "nuc_f__Mean": pa.Column(float, checks=pa.Check.isin([1.5])),
            "nuc_f__Sum": pa.Column(float, checks=pa.Check.isin([7.5])),
            "cell__Count": pa.Column(pl.UInt32, checks=pa.Check.eq(5)),
            "cell_f__Mean": pa.Column(float, checks=pa.Check.isin([0.5])),
            "cell_f__Sum": pa.Column(float, checks=pa.Check.isin([2.5])),
        }
    )
    df_agg2.pipe(result_schema2.validate)

    return df_agg2


# %%

# test_aggregate_many_to_one().pipe(to_wide)


# def cardinality_11_aggregation_expr() -> pl.Expr:
#     return (~sel.index).name.prefix('(fidx.o.child)_')

# agg_query = LabelObjectAggregation(aggregation=(~(sel.index | cs.matches('cardinality'))).name.prefix('(fidx.o.child)_'))

# pivot_column = 'fidx.o.child'
# (
# _aggregate_with_hierarchy(df_hierarchy_11, dset.f.label, agg_query)


# )
# # %%
# from resource_index_polars_io import _dataframe_to_tall, _dataframe_to_wide

# df_hierarchy_11.pivot()
# # %%
# aggregation_query = LabelObjectAggregation()
# df_agg = _aggregate_with_hierarchy(
#     r.hierarchy,
#     dset.f.intensity,
#     # aggregation_query=LabelObjectAggregation(),
#     aggregation_query=LabelObjectAggregation(
#         aggregation=(
#             pl.len().alias("{o}_Count"),
#             (~sel.index).mean().name.prefix("Mean__{o}_{c}_"),
#             (~sel.index).max().name.prefix("Max__{o}_{c}_"),
#         )
#     ),
# ).filter(pl.col("idx.label.parent") != 0)

# # r.hierarchy.join(f.label, how='inner', left_on=sel.parent_label_objects, right_on=sel.idx)
# # %%
# # df_agg.filter(pl.col('{o}_Count')!=1).rename({cs.matches('{o}': lambda x: x.format(o=))})
# # %%
# # Mean__nucleiRaw3_DAPI.3_WeightedPrincipalAxes-a.x
# # Max__nucleiRaw3_DAPI.3_WeightedFlatness
# # Mean__nucleiRaw3_DAPI.3_Mean
# # Mean__nucleiRaw3_DAPI.3_Kurtosis
# # Max__nucleiRaw3_DAPI.3_DAPI.1_PearsonR
# # Mean__nucleiRaw3_KNN:10_Max__DAPI.3_DAPI.1_PearsonR
# # Mean__KNN:10_Mean__nucleiRaw3_DAPI.1_Mean
# # nucleiRaw3_DAPI.1_Mean
# # cyto_DAPI.1_Mean
# # Mean__nucleiRaw3_DAPI.1_Mean
# # Mean__nucleiRaw3_embryoRaw-1_DistanceToBorder_Centroid
# # Mean__cyto_DAPI.1_Mean
# # nucleiRaw3-cyto_DAPI.1_Mean
# # nucleiRaw3/cyto_DAPI.1_Mean
# # Mean__cells_KNN:10_Mean__nucleiRaw3/cyto_DAPI.1_Mean

# # %%
# from zfish.features.neighborhood.neighborhoods import NeighborhoodQueryObject
# import zfish.features.neighborhood.aggregation_functions as aggfuncs

# df_nuc_los = r.label_objects.filter(pl.col("idx.o") == "nucleiRaw3")
# sample_roi = r.roi.sample()["idx.roi"].item()
# # %%
# nq = NeighborhoodQueryObject.from_dataframe(
#     df_nuc_los,
#     label_columns=["idx.roi", "idx.label"],
#     centroid_column="^centroid.[xyz]$",
#     region_id_column="idx.roi",
# )
# # %%
# f_roi = f.filter(
#     pl.col("idx.roi").cast(pl.String).str.starts_with("B02_px+0198")
# ).filter(pl.col("fidx.c").is_in(["DAPI.1"]))

# df_nucs = f_roi.intensity.filter(pl.col("idx.o") == "nucleiRaw3")
# df_cyto = f_roi.intensity.filter(pl.col("idx.o") == "cyto")
# df_cells = f_roi.intensity.filter(pl.col("idx.o") == "cells")
# # %%
# r.filter(pl.col("idx.roi").cast(pl.String).str.starts_with("B02_px+0198")).filter(
#     pl.col("idx.c").is_in(["DAPI.1"])
# )
# %%
# [
#     # DIMS
#     "plate"  #         well, roi, m, o, c, z, y, x
#     "wells"  #         well, roi, m, o, c, z, y, x
#     "rois"  # well?          roi, m, o, c, z, y, x
#     "roi",  #                  1, m, o, c, z, y, x
#     "image_pyramids"  #      roi, m,    c, z, y, x
#     "image_pyramid"  #         1, m,    c, z, y, x
#     "images",  # arrays?       1, 1,    c, z, y, x
#     "image",  #                   1,    1, z, y, x
#     "label_pyramids"  #      roi, m, o,    z, y, x
#     "label_pyramid"  #         1, m, o,    z, y, x
#     "label_images",  #            1, o,    z, y, x
#     "label_image",  #             1, 1,    z, y, x
#     "label_objects",  #              1,    z, y, x, lbl
#     "label_object",  #               1,    z, y, x, 1
#     "multiscale_levels",  #       m,
#     "channels",  #                      c,
#     "object_types",  #               o,
# ]
# # %%

# aggregate_with_hierarchy(
#     resources.hierarchy, resources.label_objects, aggregate_to="embryoRaw"
# )  # .filter(pl.col('nucleiRaw3_Count') > 1)
# # %%
# from zfish.features.polars_utils import debug

# for object_type, h_level, parents in resources.o.rows():
#     df_children = resources.hierarchy.filter(pl.col("child.o") == object_type)
#     for parent in parents:
#         df_children.filter(pl.col("parent.o").is_in([parent])).select(sel.o, sel.roi, sel.label).group_by(
#             sel.o | sel.parent_label_objects, maintain_order=True
#         ).len('Count').pipe(debug)
# # %%
# for e in resources.o['idx.o']:
#     print(e)

#     resources.hierarchy.select(sel.o, sel.roi, sel.label).join(
#         resources.o.select(
#             pl.col("idx.o"),
#             pl.col("hierarchy_level"),
#             pl.col("parents").list.len().alias("n_parent_objects"),
#         ),
#         left_on="parent.o",
#         right_on="idx.o",
#     ).sort("hierarchy_level", "n_parent_objects", sel.o, sel.roi, sel.label).group_by(
#         sel.parent_label_objects | sel.o, maintain_order=True
#     ).len('Count').sort('parent.o', 'child.o').select(sel.o, sel.roi, sel.label, 'Count').group_by(sel.o, maintain_order=True).#.agg(sel.child_label_objects)


# # %%
# is_1v1_consisten_labels = resources.hierarchy.group_by(sel.o).agg((pl.col('parent.label') == pl.col('child.label')).all())

# # resources.hierarchy.group_by(sel.o | sel.parent_label_objects, maintain_order=True).len().group_by(sel.o, maintain_order=True).agg((pl.col('len')==1).all())
# resources.hierarchy.group_by(sel.o | sel.child_label_objects, maintain_order=True).len().group_by(sel.o, maintain_order=True).agg((pl.col('len')==1).all())
# # %%
# aggregate_with_hierarchy(resources.hierarchy, features.label, )
# %%
