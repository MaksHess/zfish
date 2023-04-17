from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Union, Optional, Sequence
import anndata as ad
from functools import reduce


def main():
    fld = Path(r"Z:\hmax\Zebrafish\20211119_cyclerDATA_compressed\measurements_corr")
    df = load_features(fld, glob="*s1.csv")
    df = df.merge(get_cellcounts(df), left_index=True, right_index=True)
    ads = to_anndata(df)


def load_features(
    fld: str,
    label_columns: Sequence[str] = ("filename_prefix", "Label"),
    glob: str = "*.csv",
) -> pd.DataFrame:
    fld = Path(fld)
    dfs = []
    for fn in tqdm(fld.glob(glob)):
        dfs.append(pd.read_csv(fn))
    df = pd.concat(dfs)
    df = df.set_index(list(label_columns))
    return df


def get_cellcounts(df: pd.DataFrame, structure=None) -> pd.DataFrame:
    if structure is not None:
        df = df[df.structure == structure]
    cellcounts = (
        df.groupby("filename_prefix").count()["Centroid_z"].rename("cell_count")
    )
    log2_cellcounts = np.log2(cellcounts).rename("log2_cell_count")
    cycle = np.round(log2_cellcounts).rename("cycle").astype(int)
    well = (
        cycle.reset_index()["filename_prefix"]
        .str.split("_", expand=True)
        .loc[:, 0]
        .rename("well")
    )
    well.index = cycle.index
    embryo = pd.Series(
        cellcounts.reset_index()["filename_prefix"]
        .apply(lambda x: "_".join([x.split("_")[-3], x.split("_")[-1]]))
        .values,
        index=cellcounts.index,
        name="embryo",
    )
    meta = pd.DataFrame([cellcounts, log2_cellcounts, cycle, well, embryo]).T
    return meta


def load_features_to_anndata(fld: str) -> dict[str, ad.AnnData]:
    fld = Path(fld)
    df = load_features(fld)
    return to_anndata(df)


def _flatten(x: Union[None, str, tuple[str], dict[str, tuple[str]]]) -> tuple[str]:
    if x is None:
        return tuple()
    if isinstance(x, str):
        return (x,)
    elif isinstance(x, tuple):
        return x
    elif isinstance(x, dict):
        return _concat(x.values())


def _concat(x) -> tuple:
    return reduce(tuple.__add__, x)


def _validate(df, features, index_cols, layer_col, obsm, obs, **kwargs) -> None:
    column_names = _concat(map(_flatten, (features, index_cols, layer_col, obsm, obs)))
    for p in column_names:
        assert p in df.keys(), f"`{p}` not found in df."


def to_anndata(
    df: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    index_cols: tuple[str] = ("embryo", "Label"),
    layer_col: Optional[str] = "structure",
    layer_category: Optional[str] = "nuclei",
    obsm: dict[str, tuple[str]] = None,
    obs: tuple[str] = (
        "cell_count",
        "log2_cell_count",
        "cycle",
        "well",
        "filename_prefix",
    ),
    index_separator: str = "-",
) -> dict[str, list[ad.AnnData]]:
    df = df.reset_index()
    if obsm is None:
        obsm = {"centroid": ("Centroid_z", "Centroid_y", "Centroid_x")}
    _validate(**locals())

    df = df.set_index(
        _merge_index(df[list(index_cols)], index_separator=index_separator)
    )

    if features is None:
        features = df.columns.drop(
            list(_concat(map(_flatten, (index_cols, layer_col, obs))))
        )

    base_layer_selector = df[layer_col] == layer_category
    df_base = df[base_layer_selector]
    adata = ad.AnnData(
        X=df_base[list(features)],
        dtype=np.float32,
        obs=df_base[[*index_cols, layer_col, *obs]],
        obsm={k: df_base[list(v)] for k, v in obsm.items()},
    )

    df_layers = df[~base_layer_selector]
    for name, df_layer in df_layers.groupby(layer_col):
        _add_layer(name, adata, df_layer[list(features)].astype(np.float32))
    return adata


def _add_layer(name: str, adata: ad.AnnData, df_layer: pd.DataFrame) -> ad.AnnData:
    empty_layer = np.empty(adata.X.shape, dtype=np.float32)
    empty_layer.fill(np.nan)
    indices = adata.obs_names.get_indexer_for(
        adata.obs_names.intersection(df_layer.index)
    )
    empty_layer[indices, :] = df_layer.values.astype(np.float32)
    adata.layers[name] = empty_layer
    return adata


def _merge_index(df: pd.DataFrame, index_separator: str) -> pd.Index:
    df = df.astype(str)
    for col in df.columns:
        assert (
            df[col].str.count(index_separator).sum() == 0
        ), f"Index merger failed because `{col}` contains `{index_separator}`!"
    return pd.Index(df.apply(index_separator.join, axis=1))


if __name__ == "__main__":
    main()
