import polars as pl
from zfish.features.ccp.feature_sets import CYCLER_FEATURE_NAMES


def capped_stratified_sample(by: str, n: int) -> pl.Expr:
    pass

def stratified_sample(by: str, n: int, seed: int | None = None) -> pl.Expr:
    return pl.int_range(0, pl.count()).shuffle().over(by) < n

def subset_selection(df, expr: pl.Expr = pl.col('cycle').is_between(8, 12)) -> pl.DataFrame:
    return df.filter(expr)

def sliding_window_samples(
    df,
    stratify="cycle",
    n_samples=3000,
    n_overlap=1500,
    seed=42,
    annotation_column="annotations",
    enforce_annotation_column=False,
) -> tuple[list[pl.DataFrame], list[str], list[dict[int, int]]]:
    df_no_ann = df.sort(by=stratify)
    if enforce_annotation_column:
        df_ann = df_no_ann.filter(pl.col(annotation_column).is_not_null())
        df_no_ann = df_no_ann.filter(pl.col(annotation_column).is_null())

    dfs_group = []
    original_names = []
    for name, df_group in df.sort(stratify).group_by(stratify, maintain_order=True):
        if enforce_annotation_column:
            df_fixed = df_ann.filter(pl.col("cycle") == name)
            df_sample = pl.concat(
                [df_group.sample(n_samples - df_fixed.height, seed=seed), df_fixed]
            ).sample(fraction=1, seed=seed)
        else:
            df_sample = df_group.sample(n_samples, seed=seed)
        dfs_group.append(df_sample)
        original_names.append(name)
        print(f"sampled {df_sample.height}/{df_group.height} from {name}")

    dfs_overlapping = [dfs_group[0]]
    names_overlapping = [f"{original_names[0]}"]
    for df_group1, df_group2, name1, name2 in zip(
        dfs_group[:-1], dfs_group[1:], original_names[:-1], original_names[1:]
    ):
        df_group1to2 = pl.concat(
            [
                df_group1.with_row_count()
                .filter(pl.col("row_nr") >= (n_samples - n_overlap))
                .drop("row_nr"),
                df_group2.with_row_count()
                .filter(pl.col("row_nr") < n_overlap)
                .drop("row_nr"),
            ]
        )
        dfs_overlapping.append(df_group1to2)
        dfs_overlapping.append(df_group2)
        names_overlapping.append(f"{name1}-{name2}")
        names_overlapping.append(f"{name2}")
        print(f"overlapping {name1}-{name2}")

    relations_dicts = [
        {
            i: j
            for i, j in zip(range(n_samples - n_overlap, n_samples), range(n_overlap))
        }
    ] * (len(dfs_overlapping) - 1)
    return dfs_overlapping, names_overlapping, original_names, relations_dicts


def validate_sliding_window_samples(df):
    import matplotlib.pyplot as plt
    import numpy as np
    dfs = sliding_window_samples(
        df.filter(pl.col("cycle").is_between(8, 12)),
        n_samples=3000,
        n_overlap=1500,
        seed=42,
    )
    plt.plot(np.arange(len(dfs[0])), dfs[0]["PhysicalSize"], alpha=0.7)
    plt.plot(np.arange(len(dfs[0])) + 1500, dfs[1]["PhysicalSize"], alpha=0.7)
    plt.plot(np.arange(len(dfs[0])) + 3000, dfs[2]["PhysicalSize"], alpha=0.7)
# %%
