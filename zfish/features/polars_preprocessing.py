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

logger = logging.getLogger(__name__)

AnyFrameT = TypeVar("AnyFrameT", pl.DataFrame, pl.LazyFrame)

LOG_TRANSFORM_PATTERNS = [
    "^.*_Mean$",
    "^.*_Median$",
    "^.*_Maximum$",
    "^.*_Minimum$",
    "^.*_StandardDeviation$",
    "^.*_Variance$",
]

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

DEFAULT_OUTLIER_RANGES = [
    {
        "feature": "EquivalentSphericalRadius",  # um
        "lower": 4,
        "upper": 8,
    },
    # {
    #     "feature": "pH3.1_Mean",
    #     "lower": 4.0,
    #     "upper": 13.0,
    # },
    {
        "feature": "PCNA.0_Mean",
        "lower": 1.0,
        "upper": 8.0,
    },
    {
        "feature": "pH3.1_Mean",
        "lower": 1.0,
        "upper": 8.0,
    },
    {
        "feature": "pH3.1_Variance",
        "lower": 0.0,
        "upper": 14.0,
    },
    {
        "feature": "PCNA.0_Skewness",
        "upper": 3.0,
    },
    {
        "feature": "PCNA.0_Kurtosis",
        "upper": 5.0,
    },
]


@dataclass
class RangeOutlier:
    feature: str
    lower: str | None = None
    upper: str | None = None

    def compute(self, df: pl.DataFrame) -> pl.Series:
        return df.select(~pl.col(self.feature).is_between(self.lower, self.upper))


@dataclass
class QuantileOutlier:
    feature: str
    q_lower: float
    q_upper: float


@dataclass
class IQROutlier:
    feature: str
    q_lower: float
    q_upper: float
    iqr_mult: float


def mark_range_outlier(
    df: AnyFrameT,
    feature: str,
    lower: float | None = None,
    upper: float | None = None,
) -> pl.DataFrame:
    
    if lower is None and upper is None:
        name = f"{feature}_range_outlier"
        filter_expr = pl.col(feature).is_null()
    elif lower is None:
        name = f"{feature}_u_range_outlier"
        filter_expr = ~pl.col(feature).le(upper)
    elif upper is None:
        name = f"{feature}_l_range_outlier"
        filter_expr = ~pl.col(feature).ge(lower)
    else:
        name = f"{feature}_lu_range_outlier"
        filter_expr = ~pl.col(feature).is_between(lower, upper)
        
    is_outlier = df.select(pl.when(filter_expr).then(True).otherwise(False).alias(name))
    return is_outlier


def mark_range_outliers(
    df: AnyFrameT,
    outlier_ranges: tuple[dict[str, Any]]
) -> pl.DataFrame:
    return (
        pl.concat([mark_range_outlier(df, **rng) for rng in outlier_ranges], how='horizontal')
        .with_columns(pl.any_horizontal(pl.all()).alias('any_range_outlier'), pl.all_horizontal(pl.all()).alias('all_range_outlier'))
    )

def apply_log_transform(df: AnyFrameT, column_patterns = LOG_TRANSFORM_PATTERNS):
    return df.select(
        [
            ~(cs.matches("|".join(LOG_TRANSFORM_PATTERNS))),
            cs.matches("|".join(LOG_TRANSFORM_PATTERNS)).log1p(),
        ]
    )

def discard_bounds(
    df: AnyFrameT,
    feature: str,
    lower: float | None = None,
    upper: float | None = None,
    verbose: bool = True,
    plot_results: bool = True,
    ax=None,
) -> AnyFrameT:
    if lower is None and upper is None:
        out = df
    elif lower is None:
        x = [upper]
        linestyles = ["dashed"]
        out = df.filter(pl.col(feature).lt(upper))
    elif upper is None:
        x = [lower]
        linestyles = ["dotted"]
        out = df.filter(pl.col(feature).gt(lower))
    else:
        x = [lower, upper]
        linestyles = ["dotted", "dashed"]
        out = df.filter(pl.col(feature).is_between(lower, upper))

    if verbose or plot_results:
        if isinstance(df, pl.LazyFrame):
            df: pl.DataFrame = df.collect()
            out: pl.DataFrame = out.collect()
            return_lazy = True
        else:
            return_lazy = False

    if verbose:
        n_obs_before = df.height
        n_obs_after = out.height
        logger.info(
            f"{n_obs_after}/{n_obs_before} ({n_obs_after/n_obs_before:2.2%}) remaining \
objs after {feature!r}"
        )

    if plot_results:
        import matplotlib.pyplot as plt
        import seaborn as sns

        from zfish.features.polars_selector import sel

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        ax = sns.kdeplot(
            df.select(
                [
                    sel.index,
                    pl.col(feature),
                ]
            ).to_pandas(),
            x=feature,
            ax=ax,
        )

        y_lims = ax.get_ylim()
        plt.vlines(
            x=x,
            ymin=y_lims[0],
            ymax=y_lims[1],
            colors=["k"],
            linestyles=linestyles,
        )

    if return_lazy:
        return out.lazy()
    else:
        return out


class OutlierRange(TypedDict):
    feature: str
    lower: float | None
    upper: float | None


def remove_outliers(
    df: AnyFrameT, outliers: list[OutlierRange], verbose=True, plot_results=True
) -> AnyFrameT:
    if verbose:
        if isinstance(df, pl.LazyFrame):
            n_obs_initial = df.collect().height
        else:
            n_obs_initial = df.height

    for outlier in outliers:
        df = partial(
            discard_bounds, **outlier, verbose=verbose, plot_results=plot_results
        )(df)

    if verbose:
        if isinstance(df, pl.LazyFrame):
            n_obs_final = df.collect().height
        else:
            n_obs_final = df.height
        logger.info(
            f"{n_obs_final}/{n_obs_initial} ({n_obs_final/n_obs_initial:2.2%}) remaining \
objs in the end."
        )

    return df


def safe_collect(df: AnyFrameT):
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    return df


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
