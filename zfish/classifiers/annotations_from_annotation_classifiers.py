# %%
from pathlib import Path

import polars as pl

from zfish.classifiers.clf_io import get_annotations, load_classifier
from zfish.features.polars_utils import (
    unnest_all_structs,
)

# %% Load classifiers & extract annotations
root_path = Path(r"C:\Users\hessm\Documents\Programming\Python\zfish\zfish\classifiers")
fn_clf_cellcycle = (
    root_path / "annotation_classifiers/nucleiRaw3_classifier_cellcycle.clf"
)
fn_clf_celltype = (
    root_path / "annotation_classifiers/nucleiRaw3_classifier_celltype.clf"
)


clf_cellcycle = load_classifier(fn_clf_cellcycle)
clf_celltype = load_classifier(fn_clf_celltype)

df_ann_cellcycle_raw, map_cellcycle = get_annotations(
    clf_cellcycle, return_name_map=True
)
df_ann_celltype_raw, map_celltype = get_annotations(clf_celltype, return_name_map=True)

df_meta_raw = pl.read_parquet(root_path/'meta_raw.parquet')
print("cellcycle map:")
print(map_cellcycle)
print("annotations:")
print(
    df_ann_cellcycle_raw.join(df_meta_raw, on="roi")
    .group_by(["cycle"])
    .agg(pl.col("roi").unique().count(), pl.col("annotations").count())
    .sort(["cycle"])
)

print()

print("celltype map:")
print(map_celltype)
print("annotations:")
print(
    df_ann_celltype_raw.join(df_meta_raw, on="roi")
    .group_by(["cycle"])
    .agg(pl.col("roi").unique().count(), pl.col("annotations").count())
    .sort(["cycle"])
)
# %% Extract annotaions for debris classifier and write to parquet
fn_out_debris = root_path / "annotations/ann_is_debris.parquet"
CELLCYCLE_DEBRIS_ID = 7
CELLTYPE_DEBRIS_ID = 5

df_ann_debris = (
    pl.concat(
        [
            df_ann_cellcycle_raw.with_columns(
                (pl.col("annotations") == CELLCYCLE_DEBRIS_ID).alias("is_debris")
            ),
            df_ann_celltype_raw.with_columns(
                (pl.col("annotations") == CELLTYPE_DEBRIS_ID).alias("is_debris")
            ),
        ]
    )
    .drop("roi_id", "annotations")
    .with_columns(pl.col("label").cast(pl.Int64))
)  # .unique(subset=['roi', 'object', 'label'])

print(df_ann_debris["is_debris"].value_counts())

df_ann_debris.write_parquet(fn_out_debris)
# %% Extract annotaions for cellcycle classifier and write to parquet
fn_out_cellcycle = root_path / "annotations/ann_cellcycle.parquet"
fn_out_cellcycle_clean = root_path / "annotations/ann_cellcycle_clean.parquet"

map_cellcycle1 = {
    **map_cellcycle,
    7: "debris/split/merged",
    8: "debris/split/merged",
    9: "debris/split/merged",
}
map_cellcycle2 = {
    **map_cellcycle1,
    1: "M pro/meta",
    2: "M pro/meta",
    3: "M ana/telo",
    4: "M ana/telo",
}
map_cellcycle3 = {
    **map_cellcycle2,
    5: "S",
    6: "S",
}
map_cellcycle4 = {**map_cellcycle3, **{i: "M" for i in range(1, 5)}}
map_cellcycle5 = {**map_cellcycle4, **{i: "clean" for i in range(1, 7)}}
map_cellcycle6 = {
    **{i: "clean" for i in range(1, 7)},
    7: "debris",
    8: "split/merged",
    9: "split/merged",
}

name_maps = {
    "ann0": map_cellcycle,
    "ann1": map_cellcycle1,
    "ann2": map_cellcycle2,
    "ann3": map_cellcycle3,
    "ann4": map_cellcycle4,
    "ann5": map_cellcycle5,
    "ann6": map_cellcycle6,
}

df_ann_cellcycle = (
    df_ann_cellcycle_raw.with_columns(pl.col("annotations"))
    .with_columns(
        [
            pl.col("annotations").replace(nm).alias(name)
            for name, nm in name_maps.items()
        ]
    )
    .with_columns(
        [
            pl.col(name).map_elements(list(nm.values()).index).name.suffix("int")
            for name, nm in name_maps.items()
        ]
    )
    # .with_columns(pl.when(pl.col('ann_all_int')>6).then(pl.col('ann_all_int')).otherwise(pl.lit(7)).alias('ann_out_int'))
    # .with_columns(pl.when(pl.col('ann_out_int')<=2)))
)
for i in range(len(name_maps.keys())):
    print(
        df_ann_cellcycle.select(pl.col(f"ann{i}").value_counts(sort=True)).pipe(
            unnest_all_structs
        )
    )

df_ann_cellcycle_clean = df_ann_cellcycle.filter(pl.col("annotations") < 7)

df_ann_cellcycle.write_parquet(fn_out_cellcycle)
df_ann_cellcycle_clean.write_parquet(fn_out_cellcycle_clean)

# %% Extract annotaions for celltype classifier and write to parquet
fn_out_celltype = root_path / "annotations/ann_celltype.parquet"
fn_out_celltype_clean = root_path / "annotations/ann_celltype_clean.parquet"
df_ann_celltype = (
    df_ann_celltype_raw.with_columns(pl.col("label").cast(pl.Int64))
    .with_columns(pl.col("annotations").replace(map_celltype).alias("annotation_names"))
    .drop("roi_id")
)  # .unique(subset=['roi', 'object', 'label'])

df_ann_celltype["annotation_names"].value_counts()

df_ann_celltype_clean = df_ann_celltype.filter(pl.col("annotations") < 5)

df_ann_celltype.write_parquet(fn_out_celltype)
df_ann_celltype_clean.write_parquet(fn_out_celltype_clean)
