import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
import seaborn as sns

from zfish.features.polars_utils import drop_structs


def plot_correlation(df, selector=None, ax=None, figsize=(10, 10)):
    if selector is not None:
        df = df.select(selector)
    corr = drop_structs(df).select(cs.numeric()).corr()
    result = corr.with_columns(pl.Series('index', corr.columns)).to_pandas().set_index('index')
    ax = sns.clustermap(result, vmin=-1, vmax=1, center=0, cmap="RdBu", figsize=figsize)
    return ax


def plot_mutual_information(corr, figsize=(10, 10)):
    ax = sns.clustermap(corr, vmin=-1, vmax=1, center=0, figsize=figsize)
    return ax

