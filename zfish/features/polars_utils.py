# %%
import logging
import re
from collections import defaultdict
from collections.abc import Iterable
from enum import Enum
from functools import reduce
from operator import add
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence, TypeGuard, cast
from warnings import warn

import colorcet as cc
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
import tqdm
from polars.type_aliases import JoinStrategy, SelectorType
from strenum import StrEnum
from toolz.dicttoolz import valmap
from typing_extensions import deprecated

from zfish.features.polars_selector import sel as _sel
from zfish.preprocessing.types import AnyFrameT, FrameOrLazy

if TYPE_CHECKING:
    from polars import Expr
    from polars.polars import PyExpr
    from polars.type_aliases import IntoExpr, IntoExprColumn

POLARS_CONFIG_FILE = (
    r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\features\polars.json"
)

logger = logging.Logger(__name__)
# try:
#     pl.Config.load(POLARS_CONFIG_FILE)
# except ValueError:
#     logger.warning(f"polars config file not found in {POLARS_CONFIG_FILE}")


COLUMN_CASTS = {
    cs.matches("^acquisition$"): pl.UInt8,
    cs.matches("Count$|^label$|^parent"): pl.UInt32,
    cs.matches("^BoundingBox$"): pl.Struct(
        {
            **{f"lower-{e}": pl.Int32 for e in ["x", "y", "z"]},
            **{f"upper-{e}": pl.Int32 for e in ["x", "y", "z"]},
        }
    ),
    cs.matches("Index$"): pl.Struct({e: pl.Int32 for e in ["x", "y", "z"]}),
}
FULL_INDEX_COLUMNS = ["roi", "object", "label", "channel", "stain", "acqusition"]
CAT_DTYPE_COLUMNS = cs.matches("^roi$|^object$|^channel$|^stain$")
STRUCT_INT32_INT32_INT32_COLUMNS = cs.by_name("BoundingBox")
STRUCT_INT32_INT32_COLUMNS = cs.matches("Index$")


INDEX = ["roi", "object", "label"]

OBJECT_INDEX_NAMES = ["roi", "object", "label"]
OBJECT_INDEX_COLUMNS = [pl.col(p) for p in OBJECT_INDEX_NAMES]
NOT_OBJECT_INDEX_COLUMNS = [pl.exclude(OBJECT_INDEX_NAMES)]

CORR_FEATURE_NAMES = {
    "PearsonR",
    "SpearmanR",
    "KendallTau",
}

CORR_FEATURE_PATTERNS = {f"^.*{e}$" for e in CORR_FEATURE_NAMES}

INTENSITY_FEATURE_NAMES = {
    "CenterOfGravity",
    "Kurtosis",
    "Maximum",
    "MaximumIndex",
    "Mean",
    "Median",
    "Minimum",
    "MinimumIndex",
    "Skewness",
    "StandardDeviation",
    "Sum",
    "Variance",
    "WeightedElongation",
    "WeightedFlatness",
    "WeightedPrincipalAxes",
    "WeightedPrincipalMoments",
}

INTENSITY_FEATURE_PATTERNS = {
    f"^.*{e}(-lower|-upper)?(-[a-c])?(-[x-z])?$" for e in INTENSITY_FEATURE_NAMES
}
INTENSITY_FEATURE_COLUMNS = [pl.col(e) for e in INTENSITY_FEATURE_PATTERNS]

CORRELATION_BY_ACQUISITION_MAP = {
    "DAPI.0|DAPI.1_PearsonR": 0,
    "DAPI.1|DAPI.1_PearsonR": 1,
    "DAPI.1|DAPI.2_PearsonR": 2,
    "DAPI.1|DAPI.3_PearsonR": 3,
}

BOUNDING_BOX_STRUCT_COLUMN = "BoundingBox"
BOUNDING_BOX_COLUMNS = [
    "lower-x",
    "upper-x",
    "lower-y",
    "upper-y",
    "lower-z",
    "upper-z",
]


DEBUG = True


# def log(df: pl.DataFrame, message: str | None = None) -> pl.DataFrame:
#     if message is not None:
#         logger.info(message)
#     logger.info(f'shape: {df.shape}')
#     return df


def log(
    df: pl.DataFrame,
    message: str | None = "shape:",
    func: Callable[[pl.DataFrame], str] = lambda x: f"{x.shape}",
    **kwargs,
):
    introspection = func(df, **kwargs)
    out_message = f"{message} {introspection}"
    logger.info(out_message)
    return df


def id_(df: pl.DataFrame, **kwargs) -> pl.DataFrame:
    return df


def show(df: pl.DataFrame) -> pl.DataFrame:
    print(df)
    print()
    return df


def debug(df: pl.DataFrame, message: str = "", debug=DEBUG) -> pl.DataFrame:
    if debug:
        if message:
            print(f">>> {message}")
        print(df)
        print()
    return df


def lower_upper_iqr(series, q_lower=0.25, q_upper=0.75):
    ql = series.quantile(q_lower)
    qu = series.quantile(q_upper)
    iqr = qu - ql
    return ql, qu, iqr


def iqr(series: pl.Series, ql=0.25, qu=0.75, r=0.0):
    ql, qu, iqr = lower_upper_iqr(series, q_lower=ql, q_upper=qu)
    return (ql - r * iqr, qu + r * iqr)


def drop_structs(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        [
            column
            for column, dtype in zip(df.columns, df.dtypes)
            if not isinstance(dtype, pl.Struct)
        ]
    )


@deprecated("this function moved to zfish.features.io")
def read_table(
    root: PathLike[str], _object: str = "", use_pyarrow=True
) -> pl.DataFrame:
    fns = list(Path(root).rglob(f"{_object}*.parquet"))
    tables = []
    for fn in tqdm.tqdm(fns):
        table = pl.read_parquet(fn, use_pyarrow=use_pyarrow).with_columns(
            pl.lit(fn.parent.name).alias("roi"), pl.lit(fn.stem).alias("object")
        )
        tables.append(table)
    return pl.concat(tables, how="diagonal").select(sel.index, ~sel.index)


# NOT PERFORMANT WITH SMALL FRAGMENTED TABLES!
def scan_table(root: PathLike[str], _object: str = "") -> pl.LazyFrame:
    fns = list(Path(root).rglob(f"{_object}*.parquet"))
    tables = [
        pl.scan_parquet(fn).with_columns(
            pl.lit(fn.parent.name).alias("roi"), pl.lit(fn.stem).alias("object")
        )
        for fn in tqdm.tqdm(fns)
    ]

    return pl.concat(tables, how="diagonal")


def column_null_count(
    df: pl.DataFrame, drop_zero_columns=False, sort=True
) -> pl.DataFrame:
    df_null_counts = (
        df.with_columns(cs.by_dtype(pl.Float32, pl.Float64))
        .fill_nan(None)
        .describe()
        .filter(pl.col("statistic") == "null_count")
        .select(pl.exclude("statistic").cast(pl.Int64))
        .transpose(
            include_header=True, header_name="feature", column_names=["null_count"]
        )  # , column_names='null_count')
        .with_columns((pl.col("null_count") / df.height).alias("null_percentage"))
    )
    if sort:
        df_null_counts = df_null_counts.sort("null_count", descending=True)
    if drop_zero_columns:
        return df_null_counts.filter(pl.col("null_count") > 0)
    return df_null_counts


def drop_null_columns(
    df: pl.DataFrame,
    columns: SelectorType = cs.all(),
    strategy: Literal["any", "all", "perc"] | float = "all",
    allowed_percentage=0.3,
    include_nan=True,
) -> pl.DataFrame:
    if include_nan:
        df_nan = df.with_columns(cs.by_dtype(pl.Float32, pl.Float64).fill_nan(None))
    else:
        df_nan = df
    if strategy == "all":
        valid_col = df_nan.select(columns.is_null().all().not_())
    elif strategy == "any":
        valid_col = df_nan.select(columns.is_null().any().not_())
    elif strategy == "perc":
        null_percentage = df_nan.select(columns.is_null().sum() / columns.count().sum())
        valid_col = null_percentage <= allowed_percentage
    keep_columns = (
        valid_col.transpose(
            include_header=True, header_name="feature", column_names=["keep"]
        )
        .filter(pl.col("keep"))["feature"]
        .to_list()
    )
    passthrough_columns = df_nan.select(~columns).columns
    return df.select(passthrough_columns + keep_columns)


def drop_null_rows(
    df: pl.DataFrame,
    columns: SelectorType = cs.all(),
    strategy: Literal["any", "all", "perc"] | float = "all",
    allowed_percentage=0.3,
    include_nan=True,
) -> pl.DataFrame:
    if include_nan:
        df_nan = df.with_columns(cs.by_dtype(pl.Float32, pl.Float64).fill_nan(None))
    else:
        df_nan = df
    if strategy == "all":
        # row_filter_pred = ~pl.all_horizontal(columns.is_null())
        row_filter_pred = ~pl.all_horizontal(
            pl.col(cs.expand_selector(df_nan, columns)).is_null()
        )
    elif strategy == "any":
        row_filter_pred = ~pl.any_horizontal(columns.is_null())
    elif strategy == "perc":
        null_percentage = pl.sum_horizontal(columns.is_null()) / len(
            cs.expand_selector(df_nan, columns)
        )
        row_filter_pred = null_percentage <= allowed_percentage
    return df_nan.filter(row_filter_pred)
    # return row_filter_pred


def _unnest_renaming_expression(df: pl.DataFrame, col: str, sep: str = "-") -> pl.Expr:
    return pl.col(col).struct.rename_fields(
        [f"{col}{sep}{f.name}" for f in df.schema.get(col).fields]
    )


def unnest_structs(
    df: pl.DataFrame, cols: str | Sequence[str], sep: str = "-"
) -> pl.DataFrame:
    if isinstance(cols, str):
        cols = (cols,)
    return df.select(
        [
            pl.exclude(cols),
            *[_unnest_renaming_expression(df, col, sep=sep) for col in cols],
        ]
    ).unnest(cols)


def unnest_all_structs(df: pl.DataFrame, sep: str = ".") -> pl.DataFrame:
    cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Struct]
    return unnest_structs(df, cols, sep=sep)


POLARS_CONFIG = {"large": dict(fmt_str_lengths=50, tbl_rows=30)}


class NestingStrategy(StrEnum):
    SPLIT = "split"
    IGNORE_START = "ignore_start"


def _get_table_for_column_nesting(
    df,
    sep=".-?",
    to_level=0,
    ignore_pattern=r".*_[A-Z]",
    separator_in_name: Literal["before", "after", "drop"] = "before",
    allow_singleton_nesting=False,
    nesting_strategy: NestingStrategy = NestingStrategy.IGNORE_START,
):
    if nesting_strategy == NestingStrategy.IGNORE_START:
        SEP = re.escape(sep)
        PATTERN = rf"(?:{ignore_pattern})?([^\n{SEP}]+)?([{SEP}])?"
        # PATTERN = rf"(?:{ignore_pattern})?([^\n({SEP})]+)?([({SEP})])?"
        column_components = (
            pl.Series("columns", df.columns)
            .to_frame()
            .with_columns(
                pl.col("columns").str.extract_all(PATTERN).alias("components")
            )
            # .explode('components')
            # .pipe(debug)
        )
    elif nesting_strategy == NestingStrategy.SPLIT:
        column_components = (
            pl.Series("columns", df.columns)
            .to_frame()
            .with_columns(
                pl.col("columns").str.split(sep, inclusive=True).alias("components")
            )
            .with_columns(pl.col("components").list.slice(0, pl.len() - 1))
        )

    LEN_SEP = len(sep) if nesting_strategy == "split" else 1
    long_table = (
        column_components.with_row_index()
        .explode("components")
        .with_columns(
            # pl.col('components'),
            pl.col("components"),
            pl.int_range(0, pl.len()).over("index").alias("level"),
        )
        .with_columns(
            (pl.col("level") != pl.col("level").max()).over("index").alias("must_split")
        )
        .with_columns(
            pl.when(pl.col("must_split"))
            .then(
                pl.concat_list(
                    pl.col("components").str.slice(
                        0, pl.col("components").str.len_chars() - LEN_SEP
                    ),
                    pl.col("components").str.slice(
                        (pl.col("components").str.len_chars() - LEN_SEP)
                    ),
                )
            )
            .otherwise(pl.concat_list(pl.col("components")))
            .alias("components_with_sep")
        )
        .explode("components_with_sep")
        .with_columns(
            (pl.col("level").diff(n=-1) == -1).fill_null(False).alias("is_separator")
        )
        .with_columns(
            (pl.col("level") + pl.col("is_separator")).alias("level_sep_after")
        )
        .select(
            pl.col("index"),
            pl.col("columns"),
            pl.col("components"),
            pl.col("level"),
            pl.col("level_sep_after"),
            pl.col("components_with_sep"),
            pl.col("is_separator"),
        )
        .group_by(
            "index",
            "columns",
            *(
                (
                    pl.col("level_sep_after").alias("level")
                    if separator_in_name == "after"
                    else "level"
                ),
                "is_separator",
            ),
            maintain_order=True,
        )
        .agg(pl.col("components_with_sep"))
        .with_columns(pl.col("components_with_sep").list.join(""))
        # .filter(~pl.col('is_separator') if separator_in_name == 'drop' else True)
        # .group_by('index', 'columns', 'level', maintain_order=True).agg(pl.col('components_with_sep')).with_columns(pl.col('components_with_sep').list.join(''))
        # .pipe(debug)
    )
    return long_table


def nest_selector(
    df,
    sep=".-?",
    to_level=0,
    method=NestingStrategy.SPLIT,
    ignore_pattern=r".*_[A-Z]",
    separator_in_name: Literal["before", "after", "drop"] = "after",
    allow_singleton_nesting=False,
    column_selector=cs.all(),
):
    columns_to_nest = cs.expand_selector(df, column_selector)
    columns_to_ignore = cs.expand_selector(df, ~column_selector)

    # columns_to_nest_index = [df.get_column_index(c) for c in columns_to_nest]
    # columns_to_ignore_index = [df.get_column_index(c) for c in columns_to_ignore]
    columns_to_nest_index = [df.columns.index(c) for c in columns_to_nest]
    columns_to_ignore_index = [df.columns.index(c) for c in columns_to_ignore]

    long_table = (
        _get_table_for_column_nesting(
            df,
            sep=sep,
            to_level=to_level,
            ignore_pattern=ignore_pattern,
            separator_in_name=separator_in_name,
        )
        # .pipe(debug, 'long_table')
        .filter(pl.col("index").is_in(columns_to_nest_index))
    )
    wide_table = long_table.pivot(
        index=["index", "columns"],
        values="components_with_sep",
        columns=["level", "is_separator"],
    )  # .pipe(debug, "wide_table")
    max_level = long_table["level"].max()
    current_level = max_level
    selectors = []

    while current_level > to_level:
        # print(f"Level: {current_level}")
        group_columns = [cs.matches(r"^\{" + str(c)) for c in range(current_level)]
        field_columns = [cs.matches(r"^\{" + str(current_level))]
        # grop_separator = [cs.matches(r"^\{" + str(current_level) + ',true}$')] # Separator only for grouping
        actual_group_columns = reduce(
            add, [cs.expand_selector(wide_table, e) for e in group_columns]
        )
        # Include the separator in the grouping columns (only relevant for `separator_in_name='after'`)
        group_separator = cs.expand_selector(
            wide_table, cs.matches(r"^\{" + str(current_level) + ",true}$")
        )
        extra_group_columns = ["group_sep"] if len(group_separator) == 1 else []

        agg = (
            wide_table.with_columns(pl.col(group_separator).alias("group_sep"))
            .group_by(group_columns + extra_group_columns, maintain_order=True)
            .agg(
                pl.col("columns").alias("old_columns"),
                pl.concat_str(field_columns).alias("field_names"),
                pl.col("index").min(),
            )
            .with_columns(
                pl.concat_str(group_columns, ignore_nulls=True).alias("columns")
            )
            .with_columns(
                (pl.col("field_names").list.len() <= 1).alias("singleton_struct")
            )
            # .pipe(debug, 'agg')
        )

        struct_expression_table = agg.select(
            ["old_columns", "field_names", "columns", "singleton_struct", "index"]
        )

        selector = []
        for (
            old_columns,
            field_names,
            new_column,
            is_singleton,
            index,
        ) in struct_expression_table.rows():
            if is_singleton and (field_names[0] is None or not allow_singleton_nesting):
                selector_col = (index, pl.col(old_columns))

            else:
                selector_col = (
                    index,
                    (
                        pl.struct(old_columns)
                        .struct.rename_fields(field_names)
                        .alias(new_column)
                    ),
                )
            selector.append(selector_col)
        # add passthrough columns
        selector.extend(
            [
                (index, cs.by_name(col_name))
                for index, col_name in zip(columns_to_ignore_index, columns_to_ignore)
            ]
        )
        # if
        selectors.append([e[1] for e in sorted(selector)])

        if not allow_singleton_nesting:
            agg = agg.with_columns(
                pl.when(pl.col("singleton_struct"))
                .then(
                    pl.concat_str(
                        pl.col(actual_group_columns[-1]),
                        pl.col("field_names").list.join(""),
                    ),
                )
                .otherwise(pl.col(actual_group_columns[-1])),
                pl.when(pl.col("singleton_struct"))
                .then(
                    pl.concat_str(
                        pl.col("columns"), pl.col("field_names").list.join("")
                    ),
                )
                .otherwise(pl.col("columns")),
            )

        wide_table = agg.select("index", "columns", *group_columns)
        current_level -= 1
    return selectors


# TODO: to_level=1 leads to different results depending on the table i. e.
# [idx.o, idx.c] -> [idx];
# [idx.o, idx.c, bbx.x.lower, bbx.x.upper] -> [idx.o, idx.c, bbx.x]
def nest(
    df,
    sep=".",
    to_level=0,
    column_selector=cs.all(),
    # method=NestingStrategy.IGNORE_START,
    ignore_pattern=r".*_[A-Z]",
    separator_in_name="after",
    allow_singleton_nesting=False,
):
    for ns in nest_selector(
        df,
        sep=sep,
        to_level=to_level,
        # method=method,
        ignore_pattern=ignore_pattern,
        separator_in_name=separator_in_name,
        allow_singleton_nesting=allow_singleton_nesting,
        column_selector=column_selector,
    ):
        df = df.select(ns)
    return df


def _nest_selector_old(
    df: pl.DataFrame,
    sep=".",
    to_level=0,
    method=NestingStrategy.SPLIT,
    ignore_pattern=r".*_[A-Z]",
    include_sep="after",
) -> SelectorType:
    # PATTERN_STRATEGY_NEW = fr"(?:{ignore_pattern}[^\n{re.escape(sep)}]+)?(?:[^\n{re.escape(sep)}]+)" # Ignore everything before _[A-Z] -> 'Pol-II-S2P.0_PrincipalAxis-a.x' split on '.' -> ['Pol-II-S2P.0_PrincipalAxis-a', 'x']
    PATTERN_STRATEGY_NEW = (
        rf"(?:{ignore_pattern})?([^\n{re.escape(sep)}]+)?([{re.escape(sep)}])?"
    )

    if to_level < 0:
        raise ValueError("lowest nesting level must be 0 or larger!")

    match method:
        case NestingStrategy.SPLIT:
            columns_selector = pl.col("columns").str.split(sep).alias("columns_split")

        case NestingStrategy.IGNORE_START:
            columns_selector = (
                pl.col("columns")
                .str.extract_all(PATTERN_STRATEGY_NEW)
                .alias("columns_split")
            )

        case _:
            raise ValueError(f"Unknown nesting strategy {method!r}")

    long_table = (
        pl.Series("columns", df.columns)
        .to_frame()
        .pipe(debug)
        .with_columns(columns_selector)
        # .with_columns(pl.col('columns').str.extract_all(pattern).alias('components'))
        .pipe(debug)
        .with_row_index()
        .explode("columns_split")
        .with_columns(pl.int_range(0, pl.len()).over("index").alias("level"))
    )

    wide_table = long_table.pivot(
        index=["index", "columns"], values="columns_split", columns="level"
    )

    max_level = long_table["level"].max()
    current_level = max_level
    selectors = []
    # loop until the appropriate level is reached
    while current_level > to_level:
        group_columns = list(map(str, range(current_level)))
        aggregated = (
            wide_table.group_by(group_columns, maintain_order=True)
            .agg(
                pl.col("columns").alias("old_columns"),
                pl.col(str(current_level)).alias("field_names"),
            )
            .with_columns(
                pl.concat_str(group_columns, ignore_nulls=True, separator=sep).alias(
                    "columns"
                )
            )
        )
        # run struct construction for this round
        structify_pass = aggregated.select(["old_columns", "field_names", "columns"])
        selector = [
            (
                pl.struct(old_columns).struct.rename_fields(field_names).alias(columns)
                if (len(old_columns) > 1)
                else pl.col(old_columns)
            )
            for old_columns, field_names, columns in structify_pass.rows()
        ]
        selectors.append(selector)

        # use the rest of the table for the next round
        wide_table = aggregated.select("columns", *list(map(str, range(current_level))))
        current_level -= 1
    return selectors


def dtype_depth(dtype):
    if isinstance(dtype, pl.Field):
        dtype = dtype.dtype

    if dtype.is_nested():
        if hasattr(dtype, "fields"):
            return 1 + (max(map(dtype_depth, dtype.fields)) if dtype else 0)
        # else:
        #     warn(f'Non-struct nested dtypes like f{dtype} cannot be traversed.')
    return 0


def unnest(df, to_level=-1, sep="."):
    max_depth = max(map(dtype_depth, df.dtypes))
    if to_level < 0:
        to_level = max_depth + to_level + 1
    for _ in range(to_level):
        df = df.pipe(unnest_all_structs, sep=sep)
    return df


def nest_struct_nth_pos(df, n=0, sep=".", from_last=False):
    if not from_last:
        select_nth = pl.col("parts").list.get(n).alias("inside")
        select_pre = pl.col("parts").list.slice(0, n).alias("pre")
        select_post = pl.col("parts").list.slice(n + 1).alias("post")
        join_parts = (
            pl.concat_list(select_pre, select_post).list.join(sep).alias("outside")
        )
    else:
        select_nth = pl.col("parts").list.reverse().list.get(n).alias("inside")
        select_pre = pl.col("parts").list.reverse().list.slice(0, n).alias("pre")
        select_post = pl.col("parts").list.reverse().list.slice(n + 1).alias("post")
        join_parts = (
            pl.concat_list(select_pre, select_post)
            .list.reverse()
            .list.join(sep)
            .alias("outside")
        )

    struct_columns = (
        (
            pl.Series("columns", df.columns)
            .to_frame()
            .with_columns(pl.col("columns").str.split(".").alias("parts"))
        )
        .select(
            pl.col("columns"),
            select_nth,
            join_parts,
        )
        .with_columns(
            pl.when(pl.col("outside") == "")
            .then(pl.col("inside"))
            .otherwise(pl.col("outside"))
            .alias("outside")
        )
        .group_by("outside", maintain_order=True)
        .agg(pl.all())
        .select("outside", "columns", "inside")
    )

    command = []
    for outside, columns, inside in struct_columns.select(
        ["outside", "columns", "inside"]
    ).rows():
        if len(columns) <= 1:
            command.append(pl.col(columns))
        else:
            command.append(
                pl.struct(columns).struct.rename_fields(inside).alias(outside)
            )

    return df.select(command)


def nest_structs(
    df: AnyFrameT,
    sep="\\.",  # regex pattern
    pattern_before_sep: str | Sequence[str] = "[A-Z].*",
    pattern_after_sep: str | Sequence[str] = "[xyz]",
    nest_pattern_after: bool = True,
) -> AnyFrameT:
    struct_columns = (
        (
            pl.Series("columns", df.columns)
            .to_frame()
            .with_columns(
                [
                    pl.col("columns")
                    .str.extract_groups(
                        f"^({pattern_before_sep}){sep}({pattern_after_sep})$"
                    )
                    .alias("parts")
                    .struct.rename_fields(["before", "after"])
                ]
            )
            .unnest("parts")
        )
        # )
        .drop_nulls()
        .group_by("before" if nest_pattern_after else "after", maintain_order=True)
        .agg(pl.all())
        .rows()
    )
    # return struct_columns
    return df.select(
        ~cs.matches(f"^{pattern_before_sep}{sep}{pattern_after_sep}$"),
        *[
            # pl.exclude(
            #     [
            #         f"^{column_name}{sep}{pattern_after_sep}$"
            #         for column_name, field_names in struct_columns
            #     ]
            # ),
            *[
                pl.struct(in_column_names)
                .struct.rename_fields(field_names)
                .alias(out_column_name)
                for out_column_name, in_column_names, field_names in struct_columns
            ],
        ],
    )


def set_index_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    cat_dtype_columns = [c for c in CAT_DTYPE_COLUMNS if c in df.columns]
    uint16_dtype_columns = [c for c in UINT16_DTYPE_COLUMNS if c in df.columns]
    index_columns = [c for c in FULL_INDEX_COLUMNS if c in df.columns]
    return df.with_columns(
        [
            pl.col(cat_dtype_columns).cast(pl.Categorical),
            pl.col(uint16_dtype_columns).cast(pl.UInt16),
        ]
    ).select(
        [
            pl.col(index_columns),
            pl.exclude(index_columns),
        ]
    )


def stack_correlation_metric_by_acquisition(
    df_corr: pl.DataFrame,
    corr_map: dict[str, int] = CORRELATION_BY_ACQUISITION_MAP,
):
    return (
        df_corr.rename(valmap(str, corr_map))
        .melt(["roi", "object", "label"], list(map(str, corr_map.values())))
        .with_columns(
            [
                pl.col("variable").cast(pl.UInt16).alias("acquisition"),
                pl.col("value").alias("alignmentScore"),
            ]
        )
        .drop(["variable", "value"])
    )


def replace_channel_separators(s: pl.Series) -> pl.Series:
    return s.str.replace_all("(\w)-(\d)", "$1.$2").str.replace_all(
        "(\w+.\d+)-(\w+.\d+)", "$1|$2"
    )


def replace_channel_separators_in_columns(df: pl.DataFrame) -> pl.DataFrame:
    rename_map = dict(
        zip(
            *pl.Series("columns", df.columns)
            .to_frame()
            .with_columns(
                pl.col("columns").map(replace_channel_separators).alias("renamed")
            )
            .to_dict(as_series=False)
            .values()
        )
    )
    return df.rename(rename_map)


def join(
    dfs: Sequence[pl.DataFrame],
    on: str | Sequence[str],
    how: JoinStrategy = "outer_coalesce",
) -> pl.DataFrame:
    if len(dfs) == 0:
        return pl.DataFrame()
    df = dfs[0]
    for df_other in dfs[1:]:
        df = df.join(df_other, on=on, how=how)
    return df


def _split_feature_name(columns: list[str], sep="_") -> dict[str, dict[str, str]]:
    res = defaultdict(dict)
    for column in columns:
        res[sep.join(column.split(sep)[:-1])][column] = column.split(sep)[-1]
    return dict(res)


def split_and_melt_column_name(
    df: pl.DataFrame,
    sep: str = "_",
    pattern_before_sep: str | None = None,
    pattern_after_sep: str | None = None,
    new_column_name: str = "resources",
    index: "SelectorType | Iterable[str] | None" = None,
    return_list: bool = False,
):
    if pattern_before_sep is not None or pattern_after_sep is not None:
        raise NotImplementedError("upsi")
    return stack_column_name_to_column(
        df, sep=sep, column_name=new_column_name, index=index, return_list=return_list
    )


def split_and_melt_column_name_expr(
    column: str,
    sep: str = "_",
    pattern_before: str | None = None,
    pattern_after: str | None = None,
    column_before_split: bool = True,
    new_column_name: str = "column",
    drop_non_matching: bool = False,
) -> pl.Expr:
    if pattern_before is None:
        pattern_before = f"(.*[^{sep}])?"
    if pattern_after is None:
        pattern_after = f"([^{sep}].*)?"
    pattern = re.compile(f"{pattern_before}{sep}{pattern_after}")

    mo = pattern.match(column)
    if mo is None:
        return pl.col(column)

    values_column, column_name = mo.groups()
    if not column_before_split:
        values_column, column_name = column_name, values_column

    return pl.struct(
        pl.lit(values_column).alias(new_column_name), pl.col(column).alias(column_name)
    ).alias(column)


def split_and_melt_column_names_on(
    df: AnyFrameT,
    sep: str = "_",
    pattern_before: str | None = None,
    pattern_after: str | None = None,
    column_before_split: bool = False,
    new_column_name: str = "column",
    index_columns: "str | Iterable[str] | SelectorType" = _sel.index
    | cs.starts_with("idx.")
    | cs.starts_with("fidx."),
) -> AnyFrameT:
    # df = df.lazy()
    if pattern_before is None:
        pattern_before = f".*[^{sep}]"
    if pattern_after is None:
        pattern_after = f"[^{sep}].*"
    pattern = f"^({pattern_before}){sep}({pattern_after})$"

    split_columns = (
        (
            pl.Series("columns", cs.expand_selector(df, ~index_columns))
            .to_frame()
            .with_columns(
                [
                    pl.col("columns")
                    .str.extract_groups(pattern)
                    .alias("parts")
                    .struct.rename_fields(["before", "after"])
                ]
            )
            .unnest("parts")
        )
        # )
        .drop_nulls()
        .group_by("after" if column_before_split else "before", maintain_order=True)
        .agg(pl.all())
        .rows()
    )
    out_dfs = []
    for column_value, old_names, new_names in split_columns:
        out_dfs.append(
            df.select(
                index_columns,
                pl.lit(column_value).alias(new_column_name),
                *[pl.col(o).alias(n) for o, n in zip(old_names, new_names)],
            )
        )
    # return out_dfs
    if len(out_dfs) == 0:
        return df
    return pl.concat(out_dfs, how="diagonal")


def is_selector(s: Any) -> TypeGuard[SelectorType]:
    return cs.is_selector(s)


def stack_column_name_to_column(
    df: pl.DataFrame,
    sep: str = "_",
    column_name: str = "resources",
    index: "SelectorType | tuple[str, ...] | None" = None,
    return_list: bool = False,
) -> pl.DataFrame:
    if index is None:
        index = sel.index
    if is_selector(index):
        index = cs.expand_selector(df, index)
    split = _split_feature_name([e for e in df.columns if e not in index], sep=sep)
    static_columns = list(split.get("", dict()).keys())
    static_features = [e for e in static_columns if e not in index]

    dfs = []
    for split_name, rename_map in split.items():
        # if index == tuple():
        #     df_sub = df.select(pl.col(rename_map.keys()))
        # else:
        df_sub = df.select(
            [pl.col(e) for e in index] + [pl.col(e) for e in rename_map.keys()]
        )
        dfs.append(
            df_sub.with_columns(pl.lit(split_name).alias(column_name))
            .rename(rename_map)
            .drop_nulls(list(rename_map.values()))
            .select(
                pl.col(list(index) + [column_name]),
                pl.exclude(list(index) + [column_name]),
            )
        )
    if return_list:
        return dfs
    return pl.concat(dfs, how="diagonal")


def unstack_column_to_column_name(
    df: pl.DataFrame,
    sep: str = "_",
    column_name: str = "resources",
    index: tuple[str, ...] = ("roi", "object", "label"),
) -> pl.DataFrame:
    channel_names = df.select(column_name).unique()[column_name].to_list()
    dfs = []
    for channel_name in channel_names:
        dfs.append(
            df.filter(pl.col(column_name) == channel_name).select(
                [
                    *index,
                    pl.exclude(list(index) + [column_name]).prefix(
                        f"{channel_name}{sep}"
                    ),
                ]
            )
        )
    return join(dfs, on=list(index), how="outer_coalesce")


def split_column(
    df: pl.DataFrame,
    split_column_name="channel",
    out_column_names=("stain", "acquisition"),
    sep=".",
    index=("roi", "object", "label"),
) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col(split_column_name)
            .str.split_exact(by=sep, n=len(out_column_names) - 1)
            .struct.field(f"field_{i}")
            .alias(out_column_names[i])
            for i in range(len(out_column_names))
        ]
    ).select(
        [
            pl.col(list(index) + [split_column_name] + list(out_column_names)),
            pl.exclude(list(index) + [split_column_name] + list(out_column_names)),
        ]
    )


from typing import TypeAlias

Index: TypeAlias = str | Sequence[str] | SelectorType


def split_channel_column(
    df: pl.DataFrame,
    index=("roi", "object", "label"),
) -> pl.DataFrame:
    return split_column(
        df=df,
        split_column_name="channel",
        out_column_names=("stain", "acquisition"),
        sep=".",
        index=index,
    )


def split_channel_pair_column(
    df: pl.DataFrame,
    index: tuple[str, ...] = ("roi", "object", "label"),
    force_reference_column: Literal["auto"] | str | None = "DAPI.1",
) -> pl.DataFrame:
    df_split = split_column(
        df=df,
        split_column_name="channel_pair",
        out_column_names=("channel", "channel_ref"),
        sep="|",
        index=index,
    )
    if force_reference_column is not None:
        df_split = df_split.with_columns(
            [
                pl.when(pl.col("channel") == force_reference_column)
                .then(pl.col("channel_ref"))
                .otherwise(pl.col("channel"))
                .alias("channel"),
                pl.when(pl.col("channel_ref") == force_reference_column)
                .then(pl.col("channel_ref"))
                .otherwise(pl.col("channel"))
                .alias("channel_ref"),
            ]
        ).pipe(split_channel_column, index=index)
    return df_split


def select_numeric_and_nested(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        [
            e
            for e, dtype in zip(df.columns, df.dtypes)
            if dtype in [pl.Struct, pl.List, *pl.NUMERIC_DTYPES]
        ]
    )


def apply_colormap(s: pl.Series, cmap=cc.m_glasbey) -> pl.Series:
    unique_values = s.unique().sort().to_list()
    color_map = {v: cmap(unique_values.index(v)) for v in unique_values}
    return s.map_dict(color_map)


def pipe_df_nulls(df, **kwargs):
    plot_df_nulls(df, **kwargs)
    return df


def plot_null_columns(
    df: pl.DataFrame,
    index: SelectorType = _sel.index | cs.starts_with("idx.") | cs.starts_with("fidx."),
    sort=True,
    drop_zero_columns=False,
    ax=None,
    max_labeled_features: int = 30,
):
    import matplotlib.pyplot as plt

    null_counts_table = column_null_count(
        df, drop_zero_columns=drop_zero_columns, sort=sort
    )
    values = null_counts_table["null_percentage"]

    if ax is None:
        _, ax = plt.subplots(figsize=(2, 6))

    ax.plot(values, np.arange(len(values)))
    if len(values) > max_labeled_features:
        take_every_nth = round(len(values) / max_labeled_features)
        ax.set_yticks(np.arange(len(values)), minor=True)
    else:
        take_every_nth = 1
    ax.set_yticks(np.arange(len(values))[::take_every_nth])
    ax.set_yticklabels(null_counts_table["feature"][::take_every_nth])
    ax.set_xlabel("null percentage")
    ax.set_ylabel(f"feature ({len(values)})")
    # ax.invert_yaxis()


def plot_df_nulls(
    df: pl.DataFrame,
    index: tuple[str, ...] = ("roi", "object", "label"),
    row_color_columns: tuple[str, ...] | None = None,
    subsample_by: int | None = None,
    title: str | None = None,
    **kwargs,
):
    df = df.fill_nan(None)
    if subsample_by is not None:
        df_samp = df.gather_every(subsample_by)
    else:
        df_samp = df

    # pdf_index = df.select(pl.col(e) for e in index).to_pandas()
    pdf_cats = (
        None
        if row_color_columns is None
        else df_samp.select(
            pl.col(e).map(apply_colormap) for e in row_color_columns
        ).to_pandas()
    )

    feature_columns = sorted(
        list(set(df_samp.pipe(select_numeric_and_nested).columns).difference(index)),
        key=df_samp.columns.index,
    )
    pdf_features_is_null = (
        df_samp.select(feature_columns).select(pl.all().is_null()).to_pandas()
    )
    nan_percentage = pdf_features_is_null.sum().sum() / pdf_features_is_null.size

    default_params = dict(
        figsize=(5, 6),
        row_cluster=False,
        col_cluster=False,
        cmap=sns.color_palette(list("gr"), as_cmap=True),
        cbar_pos=(0.02, 0.5, 0.1, 0.05),
        cbar_kws={"boundaries": [0, 0.5, 1], "orientation": "horizontal"},
        row_colors=pdf_cats,
        vmin=0,
        vmax=1,
    )

    g = sns.clustermap(
        pdf_features_is_null,
        **{
            **default_params,
            **kwargs,
        },
    )

    if g.ax_cbar:
        g.ax_cbar.set_xticks([0.25, 0.75])
        g.ax_cbar.xaxis.tick_top()
        g.ax_cbar.set_xticklabels(
            ["not null", f"null ({nan_percentage:.2f})"],
            rotation=90,  # , verticalalignment="center"
        )

    if g.ax_row_colors:
        g.ax_row_colors.xaxis.tick_top()
        xticklabels = g.ax_row_colors.get_xticklabels()
        g.ax_row_colors.set_xticklabels(xticklabels, rotation=90)

    g.ax_heatmap.xaxis.tick_top()
    xticklabels = g.ax_heatmap.get_xticklabels()
    g.ax_heatmap.set_xticklabels(xticklabels, rotation=90)
    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_xlabel(f"{pdf_features_is_null.shape[1]}")
    g.ax_heatmap.set_ylabel(
        f"{pdf_features_is_null.shape[0]:,}{f' out of {df.shape[0]:,}'}"
    )

    return g


def null_percentage(
    df: pl.DataFrame, index: tuple[str, ...] = ("roi", "object", "label")
) -> float:
    df_no_index = df.select(pl.exclude(index))
    return (
        df_no_index.select(pl.all().null_count()).sum(axis=1).item()
        / df_no_index.select(pl.all().count()).sum(axis=1).item()
    )


# TODO: Write tests and fishish those
# # %%
# df_raw = pl.read_parquet(
#     r"C:\Users\hessm\Documents\Programming\Python\zfish\data\features\full.parquet"
# )
# # %%
# pt = ["BoundingBox", "^.*_Mean$", "^.*_Median$", "^.*_PearsonR$", "^.*_CentroidDist.*$"]
# # pt = ['BoundingBox', '^.*_Mean$', '^.*_Median$']
# # pt = ['^.*_Mean$', '^.*_Median$']
# # pt = ["BoundingBox", "^.*_PearsonR$", "^.*_CentroidDist.*$"]

# df_test = (
#     df_raw.select(pl.col(e) for e in INDEX + pt).sample(50, seed=42)
#     # .with_columns(pl.col("roi").str.split("_").arr.first().alias("well"))
# )
# df_tall = split_column_name_to_column(df_test, index=("roi", "object", "label"))
# # %%
# plot_df_nulls(df_test.sort(by=["object", "roi"]), row_color_columns=("object",))
# plot_df_nulls(
#     df_tall.sort(by=["object", "roi", "resources"]),
#     row_color_columns=("object", "resources"),
# )
# # %%
# from enum import Enum

# INDEX_PATTERN = r"(?P<index>[a-z]+(?:_[a-z]+)*)"  # 'snake_case` and `nonumbers` !lowercase
# FEATURE_PATTERN = (
#     r"(?P<feature>(?:[A-Z][a-z0-9]+)+[A-Z]?)"  # 'PascCamelCase' or `PascCamelCaseX` !capitalized
# )
# CHANNEL_PATTERN = (
#     r"(?P<channel>(?P<stain>[a-zA-Z0-9-]*)\.(?P<acquisition>\d+))"  # 'stainName.32' or 'Can-contain-Numb3rs-AND-hyph3ns.0'
# )
# CHANNEL_SET_PATTERN = (
#     f"({CHANNEL_PATTERN})(\\|({CHANNEL_PATTERN}))+"  # 'DAPI.0|DAPI.1|DAPI.2', 'pH3.0|pH3.40'
# )
# OBJECT_PATTERN = r"[a-zA-Z0-9]*-\d+"  # `camelCase-1` and `canHaveNumbers3-14` !no `_` or `-` !lowercase


# LABEL_FEATURE_PATTERN = f"^{FEATURE_PATTERN}$"
# INTENSITY_FEATURE_PATTERN = f"^{CHANNEL_PATTERN}_{FEATURE_PATTERN}$"
# CORR_FEATURE_PATTERN = f"^{CHANNEL_SET_PATTERN}_{FEATURE_PATTERN}$"
# DIST_FEATURE_PATTERN = f"^{OBJECT_PATTERN}_{FEATURE_PATTERN}$"
# DENSITY_FEATURE_PATTERN = "^" # NToughingNeighbors.K1[.THR10], DistanceToClosest.N100[.S['cells']]

# TOUCH_NEIGHBORHOOD_PATTERN = r"^Touch.K\d+(.T\d+)?"
# RADIUS_NEIGHBORHOOD_PATTERN = r"^Radius.R\d+(.T\d+)?"

# class Patterns(str, Enum):
#     pass


# INDEX_SELECTOR = pl.col(INDEX_PATTERN)

# LABEL_SELECTOR = pl.exclude("^.*_.*$")
# INTENSITY_SELECTOR = pl.col(f"^{CHANNEL_PATTERN}_.*$")
# CORR_SELECTOR = pl.col(f"^{CHANNEL_PATTERN}|{CHANNEL_PATTERN}_.*")
# DIST_SELECTOR = pl.col(f"^{OBJECT_PATTERN}_.*$")

# # # %%
# df_raw.columns[3:][0].split("_")

# df_raw.select(pl.col('label'))
# # %%
# pattern = r"(?P<last_name>[A-Za-z]+), (?P<title>[A-Za-z]+)\. (?P<first_name>[\(\)\sA-Za-z]*)*"

# s = pl.Series(
#     name='Name',
#     values=[
#         'Braund, Mr. Owen Harris',
#         'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',
#         'Heikkinen, Miss. Laina',
#         'Futrelle, Mrs. Jacques Heath (Lily May Peel)',
#         'Allen, Mr. William Henry',
#         'Moran, Mr. James',
#         'McCarthy, Mr. Timothy J',
#         'Palsson, Master. Gosta Leonard',
#         'Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)',
#         'Nasser, Mrs. Nicholas (Adele Achem)',
#         ]
#         )

# (
# s.to_frame().with_columns([
#     pl.col('Name'),
#     pl.col('Name').str.extract(pattern, 3).alias('First Name'),
#     pl.col('Name').str.extract(pattern, 1).alias('Last Name'),
#     pl.col('Name').str.extract(pattern, 2).alias('Title'),
# ])

# )
# %%


# import pandas as pd
# import seaborn as sns

# CAT_COLUMNS = INDEX + ['well', 'resources']

# df_plot = df_tall.with_columns(pl.col('roi').str.split('_').arr.first().alias('well')).select(pl.col(CAT_COLUMNS).cast(pl.Utf8).cast(pl.Categorical), pl.exclude(CAT_COLUMNS).is_not_null()).to_pandas()

# df_bools = df_plot.drop(columns=CAT_COLUMNS)
# df_cats = df_plot.loc[:, CAT_COLUMNS]

# object_cmap = {'cells': 'orange', 'nucleiRaw3': 'blue'}
# _object = df_cats['object'].map(object_cmap)

# well_names = sorted(df_cats['well'].unique())
# well_colors = sns.color_palette('Set1', n_colors=len(well_names))
# well_cmap = dict(zip(well_names, well_colors))
# wells = df_cats['well'].astype(str).map(well_cmap)


# fig = sns.clustermap(
#     df_bools,
#     figsize=(5, 6),
#     row_cluster=False,
#     col_cluster=False,
#     cmap=sns.color_palette(list('rg'), as_cmap=True),
#     cbar_pos=(0.02, 0.5, 0.05, 0.1),
#     cbar_kws={'boundaries': [0, 0.5, 1]},
#     row_colors=pd.concat([_object, wells], axis=1),
#     )

# print(fig.cbar_pos)

# fig.ax_cbar.set_yticks([0.25, 0.75])
# fig.ax_cbar.set_yticklabels(['null', 'not null'], rotation=0, verticalalignment='center')

# fig.ax_row_colors.xaxis.tick_top()
# xticklabels = fig.ax_row_colors.get_xticklabels()
# fig.ax_row_colors.set_xticklabels(xticklabels, rotation=90)

# fig.ax_heatmap.xaxis.tick_top()
# xticklabels = fig.ax_heatmap.get_xticklabels()
# fig.ax_heatmap.set_xticklabels(xticklabels, rotation=90)
# fig.ax_heatmap.set_yticks([])

# print(fig.cbar_pos)

# %%

# sns.choose_colorbrewer_palette('qualitative')
# REF_CHANNEL = 'DAPI-1'asdfasdfasdfasdfasdf
# (
# df_tall.select(
#     pl.col('channel').str.extract_all(r'((\w+-)+\d)')
# )

# )
# # %%
# res = df_tall.with_columns(
#     [
#     pl.col('channel')
#     .str.split('-')
#     .arr.lengths()
#     .alias('n_parts_channel'),
#     ]
# ).with_columns(
#     [
#     pl.col('channel')
#     .str.split('-')
#     .arr.slice(0, pl.col('n_parts_channel') / 2)
#     .arr.join('-')
#     .alias('channel0'),

#     pl.col('channel')
#     .str.split('-')
#     .arr.slice(-2, 2)
#     .arr.join('-')
#     .alias('channel1'),

#     ]
# ).pipe(show)

# res.select(pl.col(INDEX), pl.col('channel'), pl.col('n_parts_channel')).filter(pl.col('n_parts_channel')==4).pipe(show)

# ).select(['channel0', 'channel1']).unique()
# ).with_columns(
#     [
#         pl.when(pl.col('channel0') == REF_CHANNEL)
#         .then(pl.col('channel1'))
#         .otherwise(pl.col('channel0'))
#         .alias('channel'),
#         pl.when(pl.col('channel0') == REF_CHANNEL)
#         .then(pl.col('channel0'))
#         .otherwise(pl.col('channel1'))
#         .alias('ref_channel'),
#     ]
# )
# %%
