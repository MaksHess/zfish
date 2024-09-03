# %%
from functools import reduce
from operator import or_
from typing import TYPE_CHECKING, Literal, TypeAlias

import polars as pl
import polars.selectors as cs

from zfish.multi_table.object_hierarchy import compute_cardinality_hierarchy
from zfish.multi_table.schemas import CHILD_IDX, LABEL_OBJECT_IDX, PARENT_IDX, sel
from zfish.multi_table.tables_io import join, read_resources_and_features, to_wide

if TYPE_CHECKING:
    from polars.type_aliases import SelectorType

    IntoSelectorType: TypeAlias = str | tuple[str, ...] | SelectorType


pl.enable_string_cache()

RENAME_MAP = {"idx.o.parent": "idx.o", "idx.label.parent": "idx.label"}

# Aggregation: TypeAlias = Literal["Count", "Mean", "Max", "Min", "Quantile"]


def _to_selector(maybe_selector: "IntoSelectorType"):
    if cs.is_selector(maybe_selector):
        return maybe_selector
    else:
        return cs.by_name(maybe_selector)


def Count():
    return pl.len().alias("Count")


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


# ALLOWED_AGGREGATIONS: dict[tuple[Aggregation, ...], list[str]] = {
#     ("Count",): [],
#     ("Mean", "Max", "Min", "Quantile"): [
#         # label
#         "PhysicalSize",
#         "Elongation",
#         "Flatness",
#         "Roundness",
#         "FeretDiameter",
#         "Perimeter",
#         "EquivalentSphericalPerimeter",
#         "EquivalentSphericalRadius",
#         "PerimeterOnBorder",
#         "PerimeterOnBorderRatio",
#         # intensity
#         "Mean",
#         "Median",
#         "Minimum",
#         "Maximum",
#         "Sum",
#         "Variance",
#         "StandardDeviation",
#         "Skewness",
#         "Kurtosis",
#         "WeightedElongation",
#         "WeightedFlatness",
#         # correlation
#         "PearsonR",
#         "SpearmanR",
#         "KendallTau",
#         # density_count
#         "Count",
#         # density_distance
#         "Max",
#         "Mean",
#         # distance
#         "Centroid",
#         "Maximum",
#         "Minimum",
#     ],
# }


# def _get_allowed_features_selector(agg_func):
#     all_allowed = []
#     for k, v in ALLOWED_AGGREGATIONS.items():
#         if agg_func in k:
#             all_allowed.extend(v)
#     if len(all_allowed) == 0:
#         sel.empty
#     return reduce(or_, [cs.matches(f"^{e}$") for e in set(all_allowed)])


def aggregate(
    df_hierarchy: pl.DataFrame,
    df_object_types: pl.DataFrame,
    df_features: pl.DataFrame,
    aggregation: pl.Expr | tuple[pl.Expr, ...] = Count(),
    aggregate_to: str = "embryoRaw",
    aggregate_from: tuple[str, ...] | None = None,
    drop_background_label=True,
    group_by=(
        sel.parent | sel.object_type
    ),  # Valid for wide tables i. e. indexed like label_objects (o, roi, label)
):
    df_cardinality = (
        compute_cardinality_hierarchy(df_hierarchy, df_object_types)
        .select("idx.o", "parents", "cardinality")
        .explode("parents", "cardinality")
    )

    df_cardinality_self = pl.concat(
        [
            df_cardinality.filter(pl.col("parents") == aggregate_to),
            pl.DataFrame(
                {"idx.o": aggregate_to, "parents": aggregate_to, "cardinality": "1:1"},
                schema=df_cardinality.schema,
            ),
        ]
    )
    dfs_agg = []
    # print(df_cardinality_self)
    for cardinality, df_o in df_cardinality_self.group_by(
        ("cardinality",), maintain_order=True
    ):
        if aggregate_from is None:
            aggregate_from_group = tuple(df_o["idx.o"].to_list())
        if cardinality[0] == "m:1":
            # continue
            # df_hierarchy_group = df_hierarchy.filter(
            #     pl.col("idx.o.child").is_in(df_o["idx.o"])
            # )
            df_agg_m1 = aggregate_many_to_one(
                df_hierarchy,
                df_features,
                aggregation=aggregation,
                aggregate_to=aggregate_to,
                aggregate_from=aggregate_from_group,
                drop_background_label=drop_background_label,
                group_by=group_by,
            ).pipe(to_wide, sep="__")
            dfs_agg.append(df_agg_m1)
        elif cardinality[0] == "1:1":
            # df_hierarchy_group = df_hierarchy.pipe(_add_self_hierarchy).filter(
            #     pl.col("idx.o.child").is_in(df_o["idx.o"])
            # )
            # return df_hierarchy_group, df_features, aggregate_to, df_cardinality_self
            df_agg_11 = aggregate_one_to_one(
                df_hierarchy,
                df_features,
                aggregate_to=aggregate_to,
                aggregate_from=aggregate_from_group,
                drop_background_label=drop_background_label,
            )
            # return df_agg_11
            dfs_agg.append(df_agg_11.pipe(to_wide, sep="_"))
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
        sel.idx - cs.by_name("idx.label.child")
    ),  # Valid for wide tables i. e. indexed like label_objects (o, roi, label)
) -> pl.DataFrame:
    if aggregate_from is None:
        aggregate_from = tuple(df_hierarchy["idx.o.child"].unique().to_list())

    df_agg = df_hierarchy.filter((pl.col(PARENT_IDX[0]) == aggregate_to)).filter(
        pl.col("idx.o.child").is_in(aggregate_from)
    )
    if drop_background_label:
        df_agg = df_agg.filter(pl.col(PARENT_IDX[2]) != 0)
    df_agg = (
        df_agg.lazy()
        .join(
            df_features.lazy(),
            how="inner",
            left_on=CHILD_IDX,
            right_on=["idx.o", "idx.roi", "idx.label"],
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
        aggregate_from = tuple(df_hierarchy["idx.o.child"].unique().to_list())
    if aggregate_to in aggregate_from:
        df_hierarchy = df_hierarchy.pipe(_add_self_hierarchy)
    df_agg = (
        df_hierarchy.filter(pl.col("idx.o.parent") == aggregate_to)
        .filter(pl.col("idx.o.child").is_in(aggregate_from))
        .join(
            df_features,
            left_on=("idx.roi", "idx.o.child", "idx.label.child"),
            right_on=("idx.roi", "idx.o", "idx.label"),
        )
        .drop("idx.label.child")
        .rename(RENAME_MAP)
    )
    if drop_background_label:
        df_agg = df_agg.filter(pl.col("idx.label") != 0)
    return df_agg


def _add_self_hierarchy(df_hierarchy: pl.DataFrame, include_children_as_parents=True):
    """assure eachy object is it's own child for aggregation."""
    if include_children_as_parents:
        return pl.concat(
            [
                df_hierarchy,
                pl.concat(
                    [
                        df_hierarchy.select(
                            ["idx.roi", "idx.o.parent", "idx.label.parent"]
                        ),
                        df_hierarchy.select(
                            [
                                "idx.roi",
                                pl.col("idx.o.child").alias("idx.o.parent"),
                                pl.col("idx.label.child").alias("idx.label.parent"),
                            ]
                        ),
                    ]
                )
                .unique(["idx.roi", "idx.o.parent", "idx.label.parent"])
                .with_columns(
                    pl.col("idx.o.parent").alias("idx.o.child"),
                    pl.col("idx.label.parent").alias("idx.label.child"),
                ),
            ]
        )
    return pl.concat(
        [
            df_hierarchy,
            df_hierarchy.select(["idx.roi", "idx.o.parent", "idx.label.parent"])
            .unique(["idx.roi", "idx.o.parent", "idx.label.parent"])
            .with_columns(
                pl.col("idx.o.parent").alias("idx.o.child"),
                pl.col("idx.label.parent").alias("idx.label.child"),
            ),
        ]
    )


# %%
def example():
    r, f = read_resources_and_features()
    rw = r.pipe_tables(to_wide, exclude_tables=("hierarchy",))

    df_images = rw.images
    df_label_images = rw.label_images
    df_label_objects = rw.label_objects

    df_cardinality = compute_cardinality_hierarchy(r.hierarchy, r.object_types)

    df_hierarchy_m1 = (
        r.hierarchy.join(
            df_cardinality.explode("parents", "cardinality")
            .filter(pl.col("cardinality").is_not_null())
            .select("parents", "idx.o", "cardinality"),
            left_on=["idx.o.parent", "idx.o.child"],
            right_on=["parents", "idx.o"],
            how="full",
            coalesce=True,
        )
        .filter(pl.col("cardinality") == "m:1")
        .filter(pl.col("idx.label.parent") != 0)
    )
    df_hierarchy_11 = (
        r.hierarchy.join(
            df_cardinality.explode("parents", "cardinality")
            .filter(pl.col("cardinality").is_not_null())
            .select("parents", "idx.o", "cardinality"),
            left_on=["idx.o.parent", "idx.o.child"],
            right_on=["parents", "idx.o"],
            how="full",
            coalesce=True,
        )
        .filter(pl.col("cardinality") == "1:1")
        .filter(pl.col("idx.label.parent") != 0)
    )


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
