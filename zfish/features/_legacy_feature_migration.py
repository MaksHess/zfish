# # %%
# from zfish.io.feature_io import load_features
# df_all = load_features(r"M:\marvwy\20220721_ZE4i2_aligned\imgs\featuresAllRaw3")
# %%
import re
import pandas as pd

INT_FEATURE_PATTERNS = ['^.*NumberOf.*$', "^.*BoundingBox.*$", "^.*Index.*$", "^.*NObjects.*$", "^.*NTouching.*$", "^label$", "^acquisition$"]
STR_COLUMNS = ['roi', 'structure', 'channel', 'stain']
INT_DEFAULT_DTYPE = pd.Int64Dtype()
FLOAT_DEFAULT_DTYPE = pd.Float32Dtype()
STR_DEFAULT_DTYPE = pd.StringDtype()




# def cleanup_columns(columns):
#     c = [re.sub(r"(.*)_([xyz])$", r"\1-\2", n) for n in columns]
#     c = [re.sub(r"(.*)_(\d-[xyz])", r"\1-\2", n) for n in c]
#     c = [re.sub(r"(.*)_(lower|upper)(-[xyz])", r"\1-\2\3", n) for n in c]
#     c = [re.sub(r"(.*Axes-)0(-[xyz])", r"\1a\2", n) for n in c]
#     c = [re.sub(r"(.*Axes-)1(-[xyz])", r"\1b\2", n) for n in c]
#     c = [re.sub(r"(.*Axes-)2(-[xyz])", r"\1c\2", n) for n in c]
#     c = [re.sub(r"(.*)(Moments|Diameter)(-)x", r"\1\2\3a", n) for n in c]
#     c = [re.sub(r"(.*)(Moments|Diameter)(-)y", r"\1\2\3b", n) for n in c]
#     c = [re.sub(r"(.*)(Moments|Diameter)(-)z", r"\1\2\3c", n) for n in c]
#     c = [re.sub(r"(.*)_(\d_.*)", r"\1.\2", n) for n in c]
#     c = [re.sub(r"(.*)_(\d.*)", r"\1-\2", n) for n in c]
#     return c


# # %%
# df_clean_full = df_all.copy()
# df_clean_full.columns = cleanup_columns(df_clean_full.columns)
# df_clean_full = (
#     df_clean_full.reset_index()
#     .rename(columns={"filename_prefix": "roi", "Label": "label"})
#     .set_index(["roi", "structure", "label"])
#     .reset_index()
# )
# new_names = {
#     "borderMaximum": "embryoRaw.1_MaximumDistToBorder",
#     "borderMinimum": "embryoRaw.1_MinimumDistToBorder",
#     "borderCentroid": "embryoRaw.1_CentroidDistToBorder",
#     "topMaximum": "embryoRaw.1_MaximumDistAlongZ",
#     "topMinimum": "embryoRaw.1_MinimumDistAlongZ",
#     "topCentroid": "embryoRaw.1_CentroidDistAlongZ",
# }
# df_clean_full = df_clean_full.rename(columns=new_names)


# int_columns = df_clean_full.filter(regex='|'.join(INT_FEATURE_PATTERNS)).columns
# float_columns = df_clean_full.columns.difference(int_columns)


# int_schema = {int_column: INT_DEFAULT_DTYPE for int_column in int_columns}
# float_schema = {float_column: FLOAT_DEFAULT_DTYPE for float_column in float_columns}
# index_schema = {str_column: STR_DEFAULT_DTYPE for str_column in STR_COLUMNS if str_column in df_clean_full}
# schema = {
#     **int_schema,
#     **float_schema,
#     **index_schema,
#     }

# df_clean_full = df_clean_full.astype(schema)

# # %%
# df_clean_full.to_parquet(
#     r"C:\Users\hessm\Documents\Programming\Python\zfish\data\feature_all_legacy.parquet"
# )
# # %%
# import pandas as pd

# from zfish.io.feature_io import get_cellcounts, load_features, load_features_to_anndata

# # %%

# df_corr = (
#     load_features(r"M:\marvwy\20220721_ZE4i2_aligned\registrationCorrelation_nR3")
#     .rename(
#         columns={
#             "0": "DAPI.0*DAPI.1_PearsonR",
#             "1": "DAPI.1*DAPI.1_PearsonR",
#             "2": "DAPI.1*DAPI.2_PearsonR",
#             "3": "DAPI.1*DAPI.3_PearsonR",
#         }
#     )
#     .assign(structure="nucleiRaw3")
#     .reset_index()
#     .rename(columns={"filename_prefix": "roi", "Label": "label"})
#     # .set_index(["roi", "structure", "label"])
# )

# int_columns = df_corr.filter(regex='|'.join(INT_FEATURE_PATTERNS)).columns
# float_columns = df_corr.columns.difference(int_columns)

# int_schema = {int_column: INT_DEFAULT_DTYPE for int_column in int_columns}
# float_schema = {float_column: FLOAT_DEFAULT_DTYPE for float_column in float_columns}
# index_schema = {str_column: STR_DEFAULT_DTYPE for str_column in STR_COLUMNS if str_column in df_corr}
# schema = {
#     **int_schema,
#     **float_schema,
#     **index_schema,
#     }

# df_corr = df_corr.astype(schema).set_index(['roi', 'structure', 'label']).reset_index()
# df_corr.to_parquet(r"C:\Users\hessm\Documents\Programming\Python\zfish\data\features\coloc.parquet")
### PANDAS
# %%
import pandas as pd
df_clean_full = pd.read_parquet(
    r"C:\Users\hessm\Documents\Programming\Python\zfish\data\feature_all_legacy.parquet"
)
df_corr = pd.read_parquet(r"C:\Users\hessm\Documents\Programming\Python\zfish\data\features\old\coloc.parquet")
df_timepoints = pd.read_parquet(r"C:\Users\hessm\Documents\Programming\Python\zfish\data\acquisition_times.parquet")

# index_schema = {'roi': pd.CategoricalDtype(), 
#                'structure': pd.CategoricalDtype(), 
#                'label': pd.UInt16Dtype(),}



# %%
df_clean_full.dtypes
df_corr.dtypes
df_timepoints.dtypes
# %%

from zfish.features.intensity import DISTRIBUTION_FEATURES, WEIGHTED_SHAPE_FEATURES
from zfish.features.label import ANNDATA_OBSM, ANNDATA_X

# %%
shape_feature_patterns = [f"^{e}" for e in ANNDATA_X]
orientation_feature_patterns = [f"^{e}" for e in ANNDATA_OBSM]
label_feature_patterns = shape_feature_patterns + orientation_feature_patterns
intensity_distribution_feature_patterns = [f"_{e}$" for e in DISTRIBUTION_FEATURES]
intensity_shape_feature_patterns = [f"{e}" for e in WEIGHTED_SHAPE_FEATURES]
intensity_feature_patterns = (
    intensity_distribution_feature_patterns + intensity_shape_feature_patterns
)
density_feature_patterns = [
    "^NTouchingNeighbors",
    "^Density",
    "^ObjectsInDistance",
    "^DistanceTo",
]
distance_feature_patterns = ["DistToBorder$", "DistAlongZ$"]

# %%
obj_id = ["roi", "structure", "label"]

df_clean = df_clean_full[df_clean_full.structure=='nucleiRaw3'].set_index(obj_id)
df_shape = df_clean.filter(regex="|".join(shape_feature_patterns))
df_orientation = df_clean.filter(regex="|".join(orientation_feature_patterns))
df_label = df_clean.filter(regex="|".join(label_feature_patterns))
df_intensity_distribution = df_clean.filter(
    regex="|".join(intensity_distribution_feature_patterns)
)
df_intensity_shape = df_clean.filter(regex="|".join(intensity_shape_feature_patterns))
df_intensity = df_clean.filter(regex="|".join(intensity_feature_patterns))
df_density = df_clean.filter(regex="|".join(density_feature_patterns))
df_distance = df_clean.filter(regex="|".join(distance_feature_patterns))
# %%


def unstack_channel(df: pd.DataFrame) -> pd.DataFrame:
    df_pd = df.copy()
    df_pd.columns = pd.MultiIndex.from_tuples(
        list(map(lambda x: tuple(x.rsplit("_", 1)), df_pd.columns)),
        names=["channel", "feature"],
    )
    df_out = df_pd.T.unstack(level="channel").T
    return df_out.dropna(axis='index', how='all')


# %%
df_intensity_tall = unstack_channel(df_intensity)




### POLARS
# %%
import polars as pl

def set_index_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    all_index_columns = ['roi', 'structure', 'label', 'channel', 'stain', 'acqusition']
    all_cat_dtype_columns = ['roi', 'structure', 'channel', 'stain']
    all_uint16_dtype_columns = ['label', 'acquisition']
    cat_dtype_columns = [c for c in all_cat_dtype_columns if c in df.columns]
    uint16_dtype_columns = [c for c in all_uint16_dtype_columns if c in df.columns]
    index_columns = [c for c in all_index_columns if c in df.columns]
    return (
        df
        .with_columns([
        pl.col(cat_dtype_columns).cast(pl.Categorical),
        pl.col(uint16_dtype_columns).cast(pl.UInt16),
        ])
        .select([
        pl.col(index_columns),
        pl.exclude(index_columns),
        ])
    )
# %%
from zfish.features.intensity import DISTRIBUTION_FEATURES, WEIGHTED_SHAPE_FEATURES
from zfish.features.label import ANNDATA_OBSM, ANNDATA_X

df_clean_full = pl.read_parquet(
    r"C:\Users\hessm\Documents\Programming\Python\zfish\data\feature_all_legacy.parquet"
)

df_clean_full = set_index_dtypes(df_clean_full)

# %%
shape_feature_patterns = [f"^{e}$" for e in ANNDATA_X]
orientation_feature_patterns = [f"^{e}.*$" for e in ANNDATA_OBSM]
label_feature_patterns = shape_feature_patterns + orientation_feature_patterns
intensity_distribution_feature_patterns = [f"^.*_{e}$" for e in DISTRIBUTION_FEATURES]
intensity_shape_feature_patterns = [f"^.*{e}.*$" for e in WEIGHTED_SHAPE_FEATURES]
intensity_feature_patterns = (
    intensity_distribution_feature_patterns + intensity_shape_feature_patterns
)
intensity_feature_names = list(DISTRIBUTION_FEATURES) + list(WEIGHTED_SHAPE_FEATURES)
density_feature_patterns = [
    "^NTouchingNeighbors.*$",
    "^Density.*$",
    "^ObjectsInDistance.*$",
    "^DistanceTo.*$",
]
distance_feature_patterns = ["^.*DistToBorder$", "^.*DistAlongZ$"]
coloc_feature_patterns = ["^.*PearsonR$"]

shape_features = [pl.col(e) for e in shape_feature_patterns]
orientation_features = [pl.col(e) for e in orientation_feature_patterns]
label_features = [pl.col(e) for e in label_feature_patterns]
intensity_features = [pl.col(e) for e in intensity_feature_patterns]
density_features = [pl.col(e) for e in density_feature_patterns]
distance_features = [pl.col(e) for e in distance_feature_patterns]
coloc_features = [pl.col(e) for e in coloc_feature_patterns]

# %%
index_names = ('roi', 'structure', 'label')
index_patterns = index_names
index =[pl.col(p) for p in index_patterns]
not_index = [pl.exclude(index_patterns)]

# %%

df_nuc = df_clean_full.filter(pl.col('structure')=='nucleiRaw3').select(index + not_index)
df_cell = df_clean_full.filter(pl.col('structure')=='cell').select(index + not_index)
df_cyto = df_clean_full.filter(pl.col('structure')=='cyto').select(index + not_index)


# df_shape = df_clean.select(index + shape_features)
# df_orientation = df_clean.select(index + orientation_features)
df_nuc_label = df_nuc.select(index + label_features)
df_nuc_density = df_nuc.select(index + density_features)

df_nuc_distance = df_nuc.select(index + distance_features)

df_nuc_intensity = df_nuc.select(index + intensity_features)

df_nuc_corr = set_index_dtypes(pl.read_parquet(r"C:\Users\hessm\Documents\Programming\Python\zfish\data\features\coloc.parquet"))

df_timepoints = set_index_dtypes(pl.read_parquet(r"C:\Users\hessm\Documents\Programming\Python\zfish\data\acquisition_times.parquet"))



# %%
from typing import Sequence

def _split_feature_name(columns: list[str]) -> tuple[str, dict[str, str]]:
    return '_'.join(columns[0].split('_')[:-1]), {column: column.split('_')[-1] for column in columns}

def _split_single_channel(df_channel: pl.DataFrame) -> pl.DataFrame:
    channel_name, feature_rename_map = _split_feature_name(df_channel.select(not_index).drop_nulls().columns)
    return (
        df_channel
        .select([
        pl.col(index_patterns),
        pl.lit(channel_name).alias('channel'),
        pl.lit(channel_name.split('.')[0]).alias('stain'),
        pl.lit(channel_name.split('.')[1]).cast(pl.Int64).alias('acquisition'),
        pl.exclude(index_patterns),
        ])
        .rename(feature_rename_map)
    )

def stack_channels(df_intensity: pl.DataFrame) -> pl.DataFrame:
    channel_names = pl.Series(df_nuc_intensity.select(intensity_features).columns).str.split('_').arr.get(0).unique(maintain_order=True).to_list()
    channel_patterns = [f"^{channel_name}.*$" for channel_name in channel_names]
    channels = [pl.col(e) for e in channel_patterns]
    return pl.concat([_split_single_channel(df_intensity.select(index + [channel])) for channel in channels])



intensity_index_columns = ['roi', 'structure', 'label', 'channel', 'stain', 'acquisition']
object_index_columns = ['roi', 'structure', 'label']

def join(dfs: Sequence[pl.DataFrame]) -> pl.DataFrame:
    df = dfs[0]
    for df_other in dfs[1:]:
        df = df.join(df_other, on=object_index_columns, how='outer')
    return df

def unstack_channels(df_intensity_tall: pl.DataFrame) -> pl.DataFrame:
    channel_names = df_intensity_tall.select('channel').unique()['channel'].to_list()
    dfs = []
    for channel_name in channel_names:
        dfs.append(df_intensity_tall.filter(pl.col('channel')==channel_name).select([pl.col(object_index_columns), pl.exclude(intensity_index_columns).prefix(f'{channel_name}_')]))
    return join(dfs)
# %%
df_intensity_tall = set_index_dtypes(stack_channels(df_nuc_intensity).drop_nulls())

# %%
df_intensity_tall

# %%


df = pl.read_parquet(r"C:\Users\hessm\Documents\Programming\Python\zfish\data\features\intensity.parquet")
df_timepoints = pl.read_parquet(r"C:\Users\hessm\Documents\Programming\Python\zfish\data\acquisition_times.parquet")
# %%
site_id = ('roi',)
obj_id = ('structure', 'label')
channel_id = ('channel', 'stain', 'acquisition')
timepoints = ('acquisitionTime', 'timeDeltaMinutes')

df2 = (
    df
    .select(
    [
    pl.col('channel').str.split('.').arr.get(0).alias('stain'),
    pl.col('channel').str.split('.').arr.get(1).cast(pl.UInt64).alias('acquisition'),
    pl.all(),
    ]
    )
    .join(df_timepoints, on=('roi', 'acquisition'))
    .select(
    [
    pl.col(site_id),
    pl.col(obj_id),
    pl.col(channel_id),
    pl.col(timepoints),
    pl.exclude(site_id + obj_id + channel_id + timepoints),
    ]
    )
)
df2

# %%
df_timepoints = pl.read_parquet(r"C:\Users\hessm\Documents\Programming\Python\zfish\data\acquisition_times.parquet")
df = pl.read_parquet(r"C:\Users\hessm\Documents\Programming\Python\zfish\data\features\intensity.parquet")
# %%
import pandas as pd
%timeit pd.read_parquet(r"C:\Users\hessm\Documents\Programming\Python\zfish\data\features\label.parquet", columns=['BoundingBox-lower-x'])
# %%
df = pd.read_parquet(r"C:\Users\hessm\Documents\Programming\Python\zfish\data\feature_all_legacy.parquet")
# %%
import polars as pl
pl.scan_parquet(r"C:\Users\hessm\Documents\Programming\Python\zfish\data\feature_all_legacy.parquet").head(5).collect()
# %%
%timeit pl.read_parquet(r"C:\Users\hessm\Documents\Programming\Python\zfish\data\feature_all_legacy.parquet", use_pyarrow=True)

# %%

# df_out = df_out.select(
#     [
#         pl.col("stain_acq").str.split("_").arr.get(0).alias("stain"),
#         pl.col("stain_acq").str.split("_").arr.get(1).alias("acquisition"),
#         pl.exclude("stain_acq"),
#     ]
# ).select([pl.col(obj_id), pl.col(stain_id), pl.exclude(obj_id + stain_id)])
# with pl.StringCache():
#     acq_order = (
#         df_out.select(pl.col("acquisition").unique().sort())
#         .to_series()
#         .cast(pl.Categorical)
#     )
#     stain_order = (
#         df_out.select(pl.col("stain").unique().sort())
#         .to_series()
#         .cast(pl.Categorical)
#     )
#     structure_order = (
#         df_out.select(pl.col("structure").unique().sort())
#         .to_series()
#         .cast(pl.Categorical)
#     )
#     roi_order = (
#         df_out.select(pl.col("roi").unique().sort())
#         .to_series()
#         .cast(pl.Categorical)
#     )
#     return df_out.select(
#         [
#             pl.col(label_image_id),
#             pl.col(label_id).cast(pl.UInt32),
#             pl.col(("stain", "acquisition")).cast(pl.Categorical),
#             pl.exclude(obj_id + stain_id),
#         ]
#     )


# %%
df_nuc = df_all[df_all.structure == "nucleiRaw3"]
df_cells = df_all[df_all.structure == "cells"]
df_cyto = df_all[df_all.structure == "cyto"]
# %%
df_nuc.columns = cleanup_columns(df_nuc.columns)
df_cells.columns = cleanup_columns(df_cells.columns)
df_cyto.columns = cleanup_columns(df_cyto.columns)
# %%
df_cells
