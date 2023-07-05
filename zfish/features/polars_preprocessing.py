# %%
# TODO: Upgrade to polars 18.x (breaking changes .arr -> .list accessor)
from functools import partial
from typing import Any

import polars as pl

DEBUG = True

if DEBUG:
    import matplotlib.pyplot as plt
    import seaborn as sns
    INDEX = ['roi', 'object', 'label']


def discard_bounds(
    df: pl.LazyFrame,
    feature: str,
    lower: float | None = None,
    upper: float | None = None,
    debug=DEBUG,
    ax=None,
) -> pl.LazyFrame:
    if isinstance(df, pl.DataFrame):
        df = df.lazy()

    if lower is None and upper is None:
        out = df
    elif lower is None:
        x = [upper]
        linestyles = ["dashed"]
        out = df.filter(pl.col(feature).lt(upper))
    elif upper is None:
        x = [lower]
        linestyles = ["dotted"]
        out = df.filter(pl.col(feature).gt(lower))
    else:
        x = [lower, upper]
        linestyles = ["dotted", "dashed"]
        out = df.filter(pl.col(feature).is_between(lower, upper))
 
    if debug:
        print(f"{len(out.collect())}/{len(df.collect())}")
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        ax = sns.kdeplot(
            df.select(
                [
                    pl.col(INDEX),
                    pl.col(feature),
                ]
            )
            .collect()
            .to_pandas(),
            x=feature,
            ax=ax,
        )

        y_lims = ax.get_ylim()
        plt.vlines(
            x=x,
            ymin=y_lims[0],
            ymax=y_lims[1],
            colors=["k"],
            linestyles=linestyles,
        )
    return out


def remove_outliers(df: pl.LazyFrame, outliers: list[dict[str, Any]]) -> pl.LazyFrame:
    for outlier in outliers:
        df = partial(discard_bounds, **outlier)(df)
    return df