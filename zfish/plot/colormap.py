# %%
import numpy as np
import seaborn as sns
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap


def get_circular_palette(n=6, n_out=2, base_palette='cet_CET_C9s', out_colors=((0.5, 0.5, 0.5, 1.0), (0.3, 0.3, 0.3, 1.0)), as_cmap=False):
    custom_palette = sns.color_palette(
        np.append(
            get_cmap(base_palette)(np.linspace(0, 1, n+1)[:n]),
            np.repeat(np.array(out_colors), int(np.ceil(n_out / len(out_colors))), axis=0)[:n_out, :],
            axis=0,
            ),
        as_cmap=as_cmap,
    )
    if as_cmap:
        return ListedColormap(custom_palette)
    return custom_palette


def get_palette(n=6, n_out=0, rng=(0, 1), base_palette='cividis', out_colors=((0.5, 0.5, 0.5, 1.0), (0.3, 0.3, 0.3, 1.0)), as_cmap=False, circular=False):
    if circular:
        nodes = np.linspace(rng[0], rng[1], n+1)[:n]
    else:
        nodes = np.linspace(rng[0], rng[1], n)
    custom_palette = sns.color_palette(
        np.append(
            get_cmap(base_palette)(nodes),
            np.repeat(np.array(out_colors), int(np.ceil(n_out / len(out_colors))), axis=0)[:n_out, :],
            axis=0,
            ),
        as_cmap=as_cmap,
    )
    if as_cmap:
        return ListedColormap(custom_palette)
    return custom_palette