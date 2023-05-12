# %%
from collections import defaultdict
from typing import Sequence

import polars as pl
from polars.type_aliases import JoinStrategy
from toolz.dicttoolz import valmap

FULL_INDEX_COLUMNS = ["roi", "object", "label", "channel", "stain", "acqusition"]
CAT_DTYPE_COLUMNS = ["roi", "object", "channel", "stain"]
UINT16_DTYPE_COLUMNS = ["label", "acquisition"]

INDEX = ['roi', 'object', 'label']

OBJECT_INDEX_NAMES = ["roi", "object", "label"]
OBJECT_INDEX_COLUMNS = [pl.col(p) for p in OBJECT_INDEX_NAMES]
NOT_OBJECT_INDEX_COLUMNS = [pl.exclude(OBJECT_INDEX_NAMES)]

CORR_FEATURE_NAMES = {
    "PearsonR",
    "SpearmanR",
    "KendallTau",
}

CORR_FEATURE_PATTERNS = {
    f"^.*{e}$" for e in CORR_FEATURE_NAMES
}

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

BOUNDING_BOX_STRUCT_COLUMN = 'BoundingBox'
BOUNDING_BOX_COLUMNS = ['lower-x', 'upper-x', 'lower-y', 'upper-y', 'lower-z', 'upper-z']



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



def _rename_structs(df: pl.DataFrame, col: str) -> pl.Expr:
    return pl.col(col).struct.rename_fields([f"{col}-{k}" for k in df[col][0].keys()])


def unnest_structs(df: pl.DataFrame, cols: str | Sequence[str]) -> pl.DataFrame:
    if isinstance(cols, str):
        cols = (cols,)
    return df.select(
        [pl.exclude(cols), *[_rename_structs(df, col) for col in cols]]
    ).unnest(cols)


def unnest_all_structs(df: pl.DataFrame) -> pl.DataFrame:
    cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Struct]
    return unnest_structs(df, cols)

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


def _split_feature_name(columns: list[str], sep='_') -> dict[str, dict[str, str]]:
    res = defaultdict(dict)
    for column in columns:
        res[sep.join(column.split(sep)[:-1])][column] = column.split(sep)[-1]
    return dict(res)


def _split_single_channel(df_channel: pl.DataFrame, sep='_', column_name='channel') -> pl.DataFrame:
    channel_name, feature_rename_map = _split_feature_name(
        df_channel.select(NOT_OBJECT_INDEX_COLUMNS).drop_nulls().columns, sep=sep
    )
    return df_channel.with_columns(
        [
            pl.lit(channel_name).alias(column_name),
        ]
    ).rename(feature_rename_map)

def _split_channel_column(df, sep='-', split_column: str = 'channel', column_names: tuple[str, str]=('stain', 'acquisition')):
    return df.with_columns([
            pl.lit(sep.join(split_column.split(sep)[:-1])).alias(column_names[0]),
            pl.lit(split_column.split(sep)[-1]).cast(pl.Int64).alias(column_names[1]),
    ])


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


def stack_channels(df_intensity: pl.DataFrame, patterns: list[str], split_name: str='channel'): #-> pl.DataFrame:
    channel_names = (
        pl.Series(df_intensity.select(patterns).columns)
        .str.split("_")
        .arr.get(0)
        .unique(maintain_order=True)
        .to_list()
    )
    channel_patterns = [f"^{channel_name}_.*$" for channel_name in channel_names]
    print(channel_names)
    print(channel_patterns)
    # channels = [pl.col(e) for e in channel_patterns]
    return pl.concat([
            _split_single_channel(df_intensity.select(pl.col(INDEX), pl.col(channel_pattern)), sep='_', column_name=split_name)
            for channel_pattern in channel_patterns
        ],
        how='diagonal',)

def replace_channel_separators(s: pl.Series) -> pl.Series:
    return (
        s.str.replace_all("(\w)-(\d)", "$1.$2")
        .str.replace_all("(\w+.\d+)-(\w+.\d+)", "$1|$2")
    )

def replace_channel_separators_in_columns(df: pl.DataFrame) -> pl.DataFrame:
    rename_map = dict(zip(
        *pl.Series('columns', df.columns).to_frame()
        .with_columns(pl.col('columns').map(replace_channel_separators).alias('renamed'))
        .to_dict(as_series=False).values()
    ))
    return df.rename(rename_map)




def stack_channels_old(df_intensity: pl.DataFrame) -> pl.DataFrame:
    channel_names = (
        pl.Series(df_intensity.select(INTENSITY_FEATURE_PATTERNS).columns)
        .str.split("_")
        .arr.get(0)
        .unique(maintain_order=True)
        .to_list()
    )
    channel_patterns = [f"^{channel_name}.*$" for channel_name in channel_names]
    channels = [pl.col(e) for e in channel_patterns]
    return pl.concat(
        [
            _split_single_channel(df_intensity.select(OBJECT_INDEX_COLUMNS + [channel]), sep='-')
            for channel in channels
        ]
    )


def unstack_channels(df_intensity_tall: pl.DataFrame) -> pl.DataFrame:
    channel_names = df_intensity_tall.select("channel").unique()["channel"].to_list()
    dfs = []
    for channel_name in channel_names:
        dfs.append(
            df_intensity_tall.filter(pl.col("channel") == channel_name).select(
                [
                    *OBJECT_INDEX_COLUMNS,
                    pl.exclude(OBJECT_INDEX_NAMES).prefix(f"{channel_name}_"),
                ]
            )
        )
    return join(dfs)


def join(dfs: Sequence[pl.DataFrame], on: str, how: JoinStrategy = 'outer') -> pl.DataFrame:
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
                .alias("embryo"),
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

#TODO: Check this out
# df = pl.DataFrame({"x": ["a_1", None, "c", "d_4"]})

# df["x"].str.split_exact("_", 1).alias("fields")
# shape: (4,)
# Series: 'fields' [struct[2]]
# [
#         {"a","1"}
#         {null,null}
#         {"c",null}
#         {"d","4"}
# ]
# %%
df_raw = pl.read_parquet(r'C:\Users\hessm\Documents\Programming\Python\zfish\data\features\full.parquet')
# %%
pt = ['BoundingBox', '^.*_Mean$', '^.*_Median$', '^.*_PearsonR$', '^.*_CentroidDist.*$']
# pt = ['BoundingBox', '^.*_Mean$', '^.*_Median$']
df_test = df_raw.select(pl.col(e) for e in INDEX + pt).sample(20, seed=42)
# %%

# # def split_column_name(sep='_', )
split = _split_feature_name(df_test.columns)
static_columns = list(split.get('', dict()).keys())
static_features = [e for e in static_columns if e not in INDEX]

dfs = []
for split_name, rename_map in split.items():
    if split_name == '':
        # dfs.append(df.select(pl.col(static_columns)))
        continue
    df_sub = df_test.select(pl.col(static_columns), pl.col(rename_map.keys()))
    dfs.append(df_sub.with_columns(
        pl.lit(split_name)
        .alias(SPLIT_NAME))
        .rename(rename_map)
        # .drop_nulls(list(rename_map.values()))
        .select(pl.col(INDEX + [SPLIT_NAME]), pl.exclude(INDEX + [SPLIT_NAME]))
        )
df_tall = pl.concat(dfs, how='diagonal')
df_tall
# %%
def split_column_name_to_column(
        df: pl.DataFrame, 
        sep: str = '_', 
        column_name='resources', 
        index=('roi', 'object', 'label'),
        ) -> pl.DataFrame:
    split = _split_feature_name([e for e in df.columns if e not in index])
    static_columns = list(split.get('', dict()).keys())
    static_features = [e for e in static_columns if e not in index]

    dfs = []
    for split_name, rename_map in split.items():
        df_sub = df_test.select(pl.col(INDEX), pl.col(rename_map.keys()))
        dfs.append(df_sub.with_columns(
            pl.lit(split_name)
            .alias(column_name))
            .rename(rename_map)
            .drop_nulls(list(rename_map.values()))
            .select(pl.col(INDEX + [column_name]), pl.exclude(INDEX + [column_name]))
            )
    return pl.concat(dfs, how='diagonal')
# (
# df_tall
# .with_columns(pl.col('resources').str.split('|').arr.lengths().alias('n_resources'))
# .select(pl.col(INDEX + ['resources', 'n_resources']), pl.exclude(INDEX + ['resources', 'n_resources']))
# )['object'].unique()



# %%
def apply_colormap(s: pl.Series):
    pass
# %%
import colorcet as cc
import seaborn as sns

CAT_COLUMNS = INDEX + ['well', 'resources']

df_plot = df_tall.with_columns(pl.col('roi').str.split('_').arr.first().alias('well')).select(pl.col(CAT_COLUMNS).cast(pl.Utf8).cast(pl.Categorical), pl.exclude(CAT_COLUMNS).is_not_null()).to_pandas()

df_bools = df_plot.drop(columns=CAT_COLUMNS)
df_cats = df_plot.loc[:, CAT_COLUMNS]

object_cmap = {'cells': 'orange', 'nucleiRaw3': 'blue'}
_object = df_cats['object'].map(object_cmap)

well_names = sorted(df_cats['well'].unique())
well_colors = sns.color_palette('Set1', n_colors=len(well_names))
well_cmap = dict(zip(well_names, well_colors))
wells = df_cats['well'].astype(str).map(well_cmap)


fig = sns.clustermap(
    df_bools, 
    figsize=(5, 6),
    row_cluster=False, 
    col_cluster=False, 
    cmap=sns.color_palette(list('rg'), as_cmap=True), 
    # cbar_pos=(0.02, 2.0, 0.05, 0.2),
    cbar_kws={'boundaries': [0, 0.5, 1]},
    row_colors=pd.concat([_object, wells], axis=1),
    )
print(fig.cbar_pos)

# fig.ax_cbar.set_yticks([0.25, 0.75])
# fig.ax_cbar.set_yticklabels(['not null', 'null'], rotation=90, verticalalignment='center')

# fig.ax_row_colors.xaxis.tick_top()
# xticklabels = fig.ax_row_colors.get_xticklabels()
# fig.ax_row_colors.set_xticklabels(xticklabels, rotation=90)


# fig.ax_heatmap.xaxis.tick_top()
# xticklabels = fig.ax_heatmap.get_xticklabels()
# fig.ax_heatmap.set_xticklabels(xticklabels, rotation=90)
# fig.ax_heatmap.set_yticks([])

print(fig.cbar_pos)

# %%

sns.choose_colorbrewer_palette('qualitative')
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