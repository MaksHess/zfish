# %%
import polars as pl
import polars.selectors as cs
from polars.type_aliases import SelectorType

from zfish.features.polars_selector import sel
from zfish.features.polars_utils import join


def hierarchy_aggregate_features(
    df: pl.DataFrame,
    parent_selector: SelectorType = (
        (cs.matches("^roi$") | cs.matches("parent.cell")),
    ),
    feature_selector: SelectorType = ~sel.index,
    aggregator: pl.Expr = pl.mean,
):
    object_column_selector = cs.by_name("object")
    objects_to_aggregate = df.select(object_column_selector).unique().to_series()
    parent_columns = cs.expand_selector(df, parent_selector)
    feature_columns = cs.expand_selector(df, feature_selector)
    print(feature_columns)

    res = []
    for object_ in objects_to_aggregate:
        res.append(
            df.filter(object_column_selector == object_)
            .select(parent_selector | feature_selector)
            .group_by(parent_selector)
            .agg(
                aggregator(feature_columns).prefix(
                    f"{object_}_{aggregator.__name__.title()}."
                )
            )
        )
    return join(res, on=parent_columns).sort(by=parent_columns)


def hierarchy_aggregate_count(
    df: pl.DataFrame,
    parent_selector=cs.matches("^roi$") | cs.matches("parent.emb"),
    object_column=cs.by_name("object"),
) -> pl.DataFrame:
    parent_columns = cs.expand_selector(df, parent_selector)
    objects_to_aggregate = df.select(object_column).unique().to_series()
    res = []
    for object_ in objects_to_aggregate:
        res.append(
            df.select(parent_selector, object_column)
            .filter(object_column == object_)
            .group_by(parent_selector)
            .agg(pl.col("object").count().alias(f"{object_}_Count"))
        )
    res
    return join(res, on=parent_columns).sort(by=parent_columns)
