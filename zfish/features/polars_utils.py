from typing import Sequence

import polars as pl
from polars.type_aliases import JoinStrategy
from toolz.dicttoolz import valmap

FULL_INDEX_COLUMNS = ["roi", "structure", "label", "channel", "stain", "acqusition"]
CAT_DTYPE_COLUMNS = ["roi", "structure", "channel", "stain"]
UINT16_DTYPE_COLUMNS = ["label", "acquisition"]

OBJECT_INDEX_NAMES = ["roi", "structure", "label"]
OBJECT_INDEX_COLUMNS = [pl.col(p) for p in OBJECT_INDEX_NAMES]
NOT_OBJECT_INDEX_COLUMNS = [pl.exclude(OBJECT_INDEX_NAMES)]

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
    "DAPI.0*DAPI.1_PearsonR": 0,
    "DAPI.1*DAPI.1_PearsonR": 1,
    "DAPI.1*DAPI.2_PearsonR": 2,
    "DAPI.1*DAPI.3_PearsonR": 3,
}

BOUNDING_BOX_STRUCT_COLUMN = 'BoundingBox'
BOUNDING_BOX_COLUMNS = ['lower-x', 'upper-x', 'lower-y', 'upper-y', 'lower-z', 'upper-z']


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


def _split_feature_name(columns: list[str]) -> tuple[str, dict[str, str]]:
    return "_".join(columns[0].split("_")[:-1]), {
        column: column.split("_")[-1] for column in columns
    }


def _split_single_channel(df_channel: pl.DataFrame) -> pl.DataFrame:
    channel_name, feature_rename_map = _split_feature_name(
        df_channel.select(NOT_OBJECT_INDEX_COLUMNS).drop_nulls().columns
    )
    return df_channel.select(
        [
            *OBJECT_INDEX_COLUMNS,
            pl.lit(channel_name).alias("channel"),
            pl.lit(channel_name.split(".")[0]).alias("stain"),
            pl.lit(channel_name.split(".")[1]).cast(pl.Int64).alias("acquisition"),
            *NOT_OBJECT_INDEX_COLUMNS,
        ]
    ).rename(feature_rename_map)


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


def stack_channels(df_intensity: pl.DataFrame) -> pl.DataFrame:
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
            _split_single_channel(df_intensity.select(OBJECT_INDEX_COLUMNS + [channel]))
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
        for e in list(df["structure"].cast(pl.Utf8).unique())
        if e not in structure_names
    ]
    print(structure_names, structures_to_drop)
    df_meta = (
        df.filter(pl.col("structure").is_in(structure_names))
        .groupby(["roi", "structure"])
        .agg([pl.col("roi").count().alias("_count")])
        .pivot(values="_count", index="roi", columns="structure")
        .rename(structures)
        .drop(structures_to_drop)
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
