import seaborn as sns


def plot_correlation(corr, figsize=(10, 10)):
    ax = sns.clustermap(corr, vmin=-1, vmax=1, center=0, cmap="RdBu", figsize=figsize)
    return ax


def plot_mutual_information(corr, figsize=(10, 10)):
    ax = sns.clustermap(corr, vmin=-1, vmax=1, center=0, figsize=figsize)
    return ax
