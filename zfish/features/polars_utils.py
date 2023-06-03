# %%
from collections import defaultdict
from typing import Sequence

import colorcet as cc
import polars as pl
import seaborn as sns
from polars.type_aliases import JoinStrategy
from toolz.dicttoolz import valmap

FULL_INDEX_COLUMNS = ["roi", "object", "label", "channel", "stain", "acqusition"]
CAT_DTYPE_COLUMNS = ["roi", "object", "channel", "stain"]
UINT16_DTYPE_COLUMNS = ["label", "acquisition"]

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

def drop_null_columns(df: pl.DataFrame) -> pl.DataFrame:
    col_is_all_null = df.select(pl.col("*").is_null().all().is_not()).row(0)
    return df.select(pl.col([c for c, filt in zip(df.columns, col_is_all_null) if filt==True]))


def _rename_structs_to_unnest(df: pl.DataFrame, col: str) -> pl.Expr:
    return pl.col(col).struct.rename_fields([f"{col}-{k}" for k in df[col][0].keys()])


def unnest_structs(df: pl.DataFrame, cols: str | Sequence[str]) -> pl.DataFrame:
    if isinstance(cols, str):
        cols = (cols,)
    return df.select(
        [pl.exclude(cols), *[_rename_structs_to_unnest(df, col) for col in cols]]
    ).unnest(cols)


def unnest_all_structs(df: pl.DataFrame) -> pl.DataFrame:
    cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Struct]
    return unnest_structs(df, cols)


def nest_structs(df: pl.DataFrame, sep='-', pattern_after_sep: str | Sequence[str] = '[xyz]', pattern_before_sep: str | Sequence[str] = '[A-Z].*') -> pl.DataFrame:
    struct_columns = (
    pl.Series('columns', df.columns)
    .to_frame()
    .with_columns([
        pl.col('columns').str.extract(f"^({pattern_before_sep})-({pattern_after_sep})$", 1),
        pl.col('columns').str.extract(f"^({pattern_before_sep})-({pattern_after_sep})$", 2).alias('field_names')
        ])
    ).drop_nulls().groupby('columns', maintain_order=True).agg(pl.all()).rows()

    return df.select([
        pl.exclude([f'^{column_name}{sep}{pattern_after_sep}$' for column_name, field_names in struct_columns]),
        *[
        pl.struct([f'^{column_name}{sep}{field_name}$' for field_name in field_names]).struct.rename_fields(field_names).alias(column_name)
        for column_name, field_names in struct_columns
        ]
    ])


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
    df_corr: pl.DataFrame, corr_map: dict[str, int] = CORRELATION_BY_ACQUISITION_MAP
):
    return (
        df_corr.rename(valmap(str, corr_map))
        .melt(["roi", "structure", "label"], list(map(str, corr_map.values())))
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
    dfs: Sequence[pl.DataFrame], on: str | Sequence[str], how: JoinStrategy = "outer"
) -> pl.DataFrame:
    if len(dfs) == 0:
        return pl.DataFrame()
    df = dfs[0]
    for df_other in dfs[1:]:
        df = df.join(df_other, on=on, how=how)
    return df


def get_metadata(
    df: pl.DataFrame,
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
        .filter(pl.col("object").is_in(structure_names))
        .groupby(["roi", "object"])
        .agg([pl.col("roi").count().alias("_count")])
        # .pipe(show)
        .pivot(values="_count", index="roi", columns="object")
        # .pipe(show)
        .rename(structures)
        .with_columns(pl.col("nuc_count").log(base=2).cast(pl.Float32).prefix("log2_"))
        .with_columns(pl.col("log2_nuc_count").round(0).cast(pl.UInt16).alias("cycle"))
        .with_columns(pl.col("roi").cast(pl.Utf8).str.split("_").alias("parts"))
        .with_columns(
            [
                pl.col("parts").arr.first().cast(pl.Categorical).alias("well"),
                pl.col("parts")
                .arr.slice(1)
                .arr.join("_")
                .cast(pl.Categorical)
                .alias("site"),
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


def _split_feature_name(columns: list[str], sep="_") -> dict[str, dict[str, str]]:
    res = defaultdict(dict)
    for column in columns:
        res[sep.join(column.split(sep)[:-1])][column] = column.split(sep)[-1]
    return dict(res)


def stack_column_name_to_column(
    df: pl.DataFrame,
    sep: str = "_",
    column_name: str = "resources",
    index: tuple[str, ...] | None = None,
    return_list: bool = False
) -> pl.DataFrame:
    if index is None:
        index = tuple(df.idx.active)
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
                    pl.exclude(list(index)).prefix(f"{channel_name}{sep}"),
                ]
            )
        )
    return join(dfs, on=list(index))


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
    force_reference_column: str | None = "DAPI.1",
) -> pl.DataFrame:
    df_split = split_column(
        df=df,
        split_column_name="channel_pair",
        out_column_names=("channel", "channel_ref"),
        sep="|",
        index=index,
    )
    if force_reference_column:
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
        ).pipe(split_channel_column)
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


def plot_df_nulls(
    df: pl.DataFrame,
    index: tuple[str, ...] = ("roi", "object", "label"),
    row_color_columns: tuple[str, ...] | None = None,
    **kwargs,
):
    # pdf_index = df.select(pl.col(e) for e in index).to_pandas()
    pdf_cats = (
        None
        if row_color_columns is None
        else df.select(
            pl.col(e).map(apply_colormap) for e in row_color_columns
        ).to_pandas()
    )

    feature_columns = sorted(
        list(set(df.pipe(select_numeric_and_nested).columns).difference(index)),
        key=df.columns.index,
    )
    pdf_features_is_null = (
        df.select(feature_columns).select(pl.all().is_null()).to_pandas()
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
        }
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
    g.ax_heatmap.set_ylabel(f"{pdf_features_is_null.shape[0]}")

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
