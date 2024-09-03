# %%
import polars as pl
import polars.selectors as cs

from zfish.multi_table.schemas import sel


def compute_cardinality_hierarchy(df_hierarchy, df_o):
    # Child cardinality (maximum one parent object -> 1 else -> m)
    df_cardinality_child = (
        df_hierarchy.filter(pl.col("idx.label.parent") != 0)
        .group_by(sel.parent | sel.object_type)
        .len()
        .group_by(sel.object_type)
        .agg(
            pl.col("len").min().alias("Min_Count"),
            pl.col("len").max().alias("cardinality_child"),
        )
        .select(sel.object_type, "cardinality_child")
    )
    # parent cardinality -> 1:1, 1:m, m:1, m:m (should be mostly 1:m or 1:1 (parent:child))
    df_cardinality_parent = (
        df_hierarchy.filter(pl.col("idx.label.parent") != 0)
        .group_by(sel.child | sel.object_type)
        .len()
        .group_by(sel.object_type)
        .agg(
            pl.col("len").min().alias("Min_Count"),
            pl.col("len").max().alias("cardinality_parent"),
        )
        .select(sel.object_type, "cardinality_parent")
    )
    # join the result to the o table, where the relationships are initially specified
    # return df_cardinality_child, df_cardinality_parent
    return (
        df_o.explode("parents")
        .with_columns(pl.col("parents").cast(pl.Categorical))
        .join(
            df_cardinality_child.join(
                df_cardinality_parent,
                on=cs.expand_selector(df_cardinality_child, sel.object_type),
            ).select(sel.object_type, "cardinality_child", "cardinality_parent"),
            left_on=["idx.o", "parents"],
            right_on=["idx.o.child", "idx.o.parent"],
            how="full",
            coalesce=True,
        )
        .select(
            "idx.o",
            "hierarchy_level",
            "parents",
            pl.concat_str(
                (pl.col("cardinality_child") > 1).replace({True: "m", False: "1"}),
                (pl.col("cardinality_parent") > 1).replace({True: "m", False: "1"}),
                separator=":",
            ).alias("cardinality"),
        )
        .group_by("idx.o")
        .agg(pl.col("hierarchy_level").first(), pl.col("parents", "cardinality"))
        .with_columns(pl.col("parents", "cardinality").list.drop_nulls())
        .sort(
            "hierarchy_level",
            pl.col("parents").list.len(),
            "idx.o",
            descending=[False, True, False],
        )
    )
