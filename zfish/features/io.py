# %%
from os import PathLike
from pathlib import Path

import polars as pl
import polars.selectors as cs
import tqdm

from zfish.features.polars_preprocessing import get_metadata
from zfish.features.polars_selector import sel
from zfish.preprocessing.normalize import handle_outliers
from zfish.preprocessing.outlier_ranges import (
    DEBRIS_OUTLIERS,
    INTENSITY_OUTLIERS,
    SEGMENTATION_OUTLIERS,
)

LOAD_REMOTE = False
CONTROL_WELLS = ["B07", "C07", "D07", "E07"]
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

CONSOLIDATED_TABLES_ALL_MODELS = Path(
    r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\features_tcorr_v3_consolidated"
)
CONSOLIDATED_TABLES = CONSOLIDATED_TABLES_ALL_MODELS / "z_model=LogLinear__TwoStep"
CLF_ANNOTATIONS = Path(
    r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\classifier\annotations"
)
CLF_PREDICTIONS = Path(
    r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\classifier\predictions"
)

CONSOLIDATED_TABLES_LOCAL = Path(
    r"C:\Users\hessm\Documents\zfish_local\features_tcorr_v3_consolidated\z_model=LogLinear__TwoStep"
)
CLF_ANNOTATIONS_LOCAL = Path(
    r"C:\Users\hessm\Documents\zfish_local\classifier\annotations"
)
CLF_PREDICTIONS_LOCAL = Path(
    r"C:\Users\hessm\Documents\zfish_local\classifier\predictions"
)

TABLES = {
    "root": CONSOLIDATED_TABLES if LOAD_REMOTE else CONSOLIDATED_TABLES_LOCAL,
    "fld_annotations": CLF_ANNOTATIONS if LOAD_REMOTE else CLF_ANNOTATIONS_LOCAL,
    "fld_predictions": CLF_PREDICTIONS if LOAD_REMOTE else CLF_PREDICTIONS_LOCAL,
}

RESOURCE_TABLES = {
    "fn_images": r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\images.parquet",
    "fn_label_images": r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\label_images.parquet",
    "fn_label_objects": r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\label_objects.parquet",
}


def read_table_fragmented(
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


def scan_tables(
    root: PathLike[str] = TABLES["root"],
    fld_annotations: PathLike[str] | None = TABLES["fld_annotations"],
    fld_predictions: PathLike[str] | None = TABLES["fld_predictions"],
    annotations: tuple[str, ...] = ("debris", "celltype_clean", "cellcycle_clean"),
    predictions: tuple[str, ...] = ("debris", "celltype", "cellcycle3c"),
    index: tuple[str, ...] = ("roi", "object", "label"),
    filters: pl.Expr | tuple[pl.Expr, ...] = ((pl.col("debris_proba") < 0.3),),
    join_metas_on: tuple[str, ...] = tuple(),
) -> dict[str, pl.LazyFrame]:
    dfs = {}
    for fn in Path(root).glob("*.parquet"):
        dfs[fn.stem] = pl.scan_parquet(fn).cast({"roi": pl.Utf8, "object": pl.Utf8})

    CLFS_ON = "nucleiRaw3"

    df_anns = {}
    if fld_annotations is not None:
        for annotation in annotations:
            fn = Path(fld_annotations) / f"ann_{annotation}.parquet"
            df_anns[annotation] = pl.scan_parquet(fn)
        for name, frame in df_anns.items():
            dfs[CLFS_ON] = dfs[CLFS_ON].join(
                frame.cast(dfs[CLFS_ON].select(index).schema), on=index, how="left"
            )

    df_preds = {}
    if fld_predictions is not None:
        for prediction in predictions:
            fn = Path(fld_predictions) / f"{prediction}_pred.parquet"
            df_preds[prediction] = pl.scan_parquet(fn)
        for name, frame in df_preds.items():
            dfs[CLFS_ON] = dfs[CLFS_ON].join(
                frame.cast(dfs[CLFS_ON].select(index).schema), on=index, how="left"
            )
        dfs[CLFS_ON] = dfs[CLFS_ON].filter(filters)
    df_meta = get_metadata(
        dfs[CLFS_ON].select(sel.index).collect(), control_wells=CONTROL_WELLS
    )
    dfs["meta"] = df_meta.lazy()

    for meta_on in join_metas_on:
        dfs[meta_on] = dfs[meta_on].join(df_meta.lazy(), on="roi")
    return dfs


def scan_resources(
    # copy from plot_nuclei_annotations
    fn_images=RESOURCE_TABLES["fn_images"],
    fn_label_images=RESOURCE_TABLES["fn_label_images"],
    fn_label_objects=RESOURCE_TABLES["fn_label_objects"],
):
    dfs = {}
    dfs["images"] = pl.scan_parquet(fn_images).rename({"filename_prefix": "roi"})
    dfs["label_images"] = pl.scan_parquet(fn_label_images).rename(
        {"filename_prefix": "roi"}
    )
    dfs["label_objects"] = pl.scan_parquet(fn_label_objects)
    return dfs
