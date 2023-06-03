# %%
import numba as nb
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Mean",
    "Median",
    "Mode",
    "Max",
    "Min",
    "Sum",
    "Std",
    "Var",
    "CircMean",
    "CircR",
    "CircVar",
    "ValueCounts",
    "quantile_factory"
]


def quantile_factory(*qs: float):
    for q in qs:
        @nb.njit
        def quantile(arr):
            return np.quantile(arr, q)
        quantile.__name__ = f'Q{q:.2f}'
        yield(quantile)

@nb.njit
def Mean(arr: NDArray) -> NDArray:
    return np.nanmean(arr)


@nb.njit
def Median(arr: NDArray) -> NDArray:
    return np.nanmedian(arr)


@nb.njit
def Max(arr: NDArray) -> NDArray:
    if arr.size == 0:
        return np.nan
    return np.nanmax(arr)


@nb.njit
def Min(arr: NDArray) -> NDArray:
    if arr.size == 0:
        return np.nan
    return np.nanmin(arr)


@nb.njit
def ValueCounts(arr):
    counter = dict()
    for e in arr.flat:
        if e == np.nan:
            continue
        elif e not in counter:
            counter[e] = 1
        else:
            counter[e] += 1
    return sorted(counter.items(), key=lambda x: (x[1], x[0]), reverse=True)


@nb.njit
def Mode(arr):
    return ValueCounts(arr)[0][0]


@nb.njit
def _Quantile(arr: NDArray, q: float) -> NDArray:
    return np.nanquantile(arr, q)


@nb.njit
def Sum(arr: NDArray) -> NDArray:
    return np.nansum(arr)


@nb.njit
def Std(arr: NDArray) -> NDArray:
    return np.nanstd(arr)


@nb.njit
def Var(arr: NDArray) -> NDArray:
    return np.nanvar(arr)


@nb.njit
def _circfuncs_common(samples, high, low):
    samples = samples.flatten()
    mask = np.isnan(samples)
    sin_samp = np.sin((samples - low) * 2.0 * np.pi / (high - low))
    cos_samp = np.cos((samples - low) * 2.0 * np.pi / (high - low))
    sin_samp[mask] = 0.0
    cos_samp[mask] = 0.0
    return samples, sin_samp, cos_samp, mask


@nb.njit
def CircMean(samples, high=2 * np.pi, low=0.0) -> float:
    samples, sin_samp, cos_samp, nmask = _circfuncs_common(samples, high=high, low=low)

    sin_sum: float = sin_samp.sum()
    cos_sum: float = cos_samp.sum()
    res: float = np.arctan2(sin_sum, cos_sum)

    if res < 0:
        res += 2 * np.pi

    if nmask.all():
        res = np.nan

    return res * (high - low) / 2.0 / np.pi + low


@nb.njit
def CircR(samples, high=2 * np.pi, low=0.0) -> float:
    samples, sin_samp, cos_samp, nmask = _circfuncs_common(samples, high=high, low=low)

    nsum = np.sum(~nmask)
    if nsum == 0:
        nsum = np.nan

    sin_mean: float = sin_samp.sum() / nsum
    cos_mean: float = cos_samp.sum() / nsum

    R = np.minimum(1, np.hypot(sin_mean, cos_mean))
    return R


@nb.njit
def CircVar(samples, high=2 * np.pi, low=0.0) -> float:
    return 1 - CircR(samples=samples, high=high, low=low)

