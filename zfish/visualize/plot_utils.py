# %%
import warnings
from typing import Any, Literal

import arviz as az
from matplotlib import scale as mpl_scale
from numpy.typing import ArrayLike
import numpy as np


def data_range(
    x: ArrayLike,
    interval_range_perc: float | tuple[float, float] = 0.99,
    interval_range_padding: float=0.1,
    kind: Literal["iqr", "hdi"] = "iqr",
    scale: mpl_scale.ScaleBase
    | Literal["linear", "log", "symlog", "asinh", "logit"] = "linear",
    scale_kwargs: dict[str, Any] | None = None,
    warn_perc: float = 0.01,
) -> tuple[float, float]:
    x = np.asarray(x)

    if isinstance(scale, str):
        if scale_kwargs is None:
            scale_kwargs = {}
        mpl_transform = mpl_scale.scale_factory(
            scale, axis=None, **scale_kwargs
        ).get_transform()
    elif isinstance(scale, mpl_scale.ScaleBase):
        mpl_transform = scale.get_transform()
    else:
        raise ValueError(
            f"`scale` must be a `str` or `matplotlib.scale.ScaleBase`, not `{type(scale)}`"
        )

    if kind == "hdi":
        if isinstance(interval_range_perc, tuple):
            raise ValueError("Specifying an explicit range is invalid for kind='hdi'. Provide a float [0, 1.0] or change kind to 'iqr'.")
        if interval_range_perc == 1.0:
            interval_range_perc = 1.0 - 1e-6
        interval_lower, interval_upper = tuple(
            az.hdi(
                mpl_transform.transform(x),
                hdi_prob=interval_range_perc,
            )
        )
    elif kind == 'iqr':
        if isinstance(interval_range_perc, tuple):
            iqr_range = interval_range_perc
        else:
            tail = (1 - interval_range_perc) / 2
            iqr_range = (tail, 1-tail)
        interval_lower, interval_upper = tuple(np.quantile(mpl_transform.transform(x), iqr_range))
    interval_range_mult = (interval_upper - interval_lower) * interval_range_padding
    lim_trans = (
        interval_lower - interval_range_mult,
        interval_upper + interval_range_mult,
    )
    lim = tuple(mpl_transform.inverted().transform(lim_trans))

    n_lower_out_of_bounds = int(np.sum(x < lim[0]))
    n_upper_out_of_bounds = int(np.sum(x > lim[1]))
    percentage_out_of_bounds = (n_lower_out_of_bounds + n_upper_out_of_bounds) / len(x)
    if percentage_out_of_bounds > warn_perc:
        warnings.warn(
            f"""{percentage_out_of_bounds:.02%} of points out of data range.
n_lower: {n_lower_out_of_bounds}, n_upper: {n_upper_out_of_bounds}, n_tot: {len(x)}"""
        )
    return lim