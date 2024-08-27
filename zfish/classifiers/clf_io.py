import pickle
from pathlib import Path
from typing import Sequence, cast

import polars as pl
from napari_feature_classifier.classifier import Classifier

from zfish.features.polars_index import IndexAccessor


def load_annotations(fld: Path | str, glob_pattern="*annotation.csv") -> pl.DataFrame:
    """
    Don't use this, relies on always saving all the annotations. Better to extract
    annotations from Classifier.
    """
    schema = {"label": pl.Int64, "annotations": pl.Float64, "annotation_names": pl.Utf8}
    dfs = []
    for fn in Path(fld).glob(glob_pattern):
        parts = fn.stem.split("_")
        obj = parts[-2]
        roi = "_".join(parts[:-2])
        dfs.append(
            pl.read_csv(fn, columns=list(schema.keys()), dtypes=schema).with_columns(
                [pl.lit(roi).alias("roi"), pl.lit(obj).alias("object")]
            )
        )
    return pl.concat(dfs)


def load_classifier(fn: Path | str):
    with open(fn, "rb") as f:  # pylint: disable=C0103
        return cast(Classifier, pickle.load(f))


def get_annotations(clf: Classifier, return_name_map=False) -> tuple[pl.DataFrame, dict[int, str]]:
    df_ann = (
        pl.DataFrame(clf._data.reset_index())
        .select([pl.col("roi_id"), pl.col("label"), pl.col("annotations")])
        .with_columns(
            [
                pl.col("roi_id")
                .str.split("_")
                .list.slice(0, 3)
                .list.join("_")
                .alias("roi"),
                pl.col("roi_id").str.split("_").list.get(-1).alias("object"),
            ]
        )
    ).idx.sort()
    if not return_name_map:
        return df_ann
    else:
        name_map = {
            int(i + 1): s
            for i, s in zip(range(len(clf._class_names)), clf._class_names)
        }
        return df_ann, name_map


def new_classifier(
    clf: Classifier,
    feature_names: Sequence[str],
    df: pl.DataFrame,
    index_columns=("roi", "label"),
):
    parsed_feature_names = df.select([pl.col(e) for e in feature_names]).columns
    clf_new = Classifier(
        feature_names=parsed_feature_names, class_names=clf._class_names
    )
    df_ann = get_annotations(clf)
    merge_schema = df_ann.select(list(index_columns)).schema
    clf_new.add_features(
        df_ann.join(
            df.with_columns([pl.col(k).cast(v) for k, v in merge_schema.items()]),
            on=["roi", "label"],
        ).to_pandas()
    )
    return clf_new