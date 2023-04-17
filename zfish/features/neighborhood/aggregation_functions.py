import numba as nb
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MEAN",
    "MEDIAN",
    "MAX",
    "MIN",
    "SUM",
    "STD",
    "VAR",
    "CIRCMEAN",
    "CIRCR",
    "CIRCVAR",
    "get_aggregation_functions"
]

# TODO: Implement Mode.

@nb.njit
def MEAN(arr: NDArray) -> NDArray:
    return np.nanmean(arr)


@nb.njit
def MEDIAN(arr: NDArray) -> NDArray:
    return np.nanmedian(arr)


@nb.njit
def MAX(arr: NDArray) -> NDArray:
    if arr.size == 0:
        return np.nan
    return np.nanmax(arr)


@nb.njit
def MIN(arr: NDArray) -> NDArray:
    if arr.size == 0:
        return np.nan
    return np.nanmin(arr)


@nb.njit
def _QUANTILE(arr: NDArray, q: float) -> NDArray:
    return np.nanquantile(arr, q)


@nb.njit
def SUM(arr: NDArray) -> NDArray:
    return np.nansum(arr)


@nb.njit
def STD(arr: NDArray) -> NDArray:
    return np.nanstd(arr)


@nb.njit
def VAR(arr: NDArray) -> NDArray:
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
def CIRCMEAN(samples, high=2 * np.pi, low=0.0) -> float:
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
def CIRCR(samples, high=2 * np.pi, low=0.0) -> float:
    samples, sin_samp, cos_samp, nmask = _circfuncs_common(samples, high=high, low=low)

    nsum = np.sum(~nmask)
    if nsum == 0:
        nsum = np.nan

    sin_mean: float = sin_samp.sum() / nsum
    cos_mean: float = cos_samp.sum() / nsum

    R = np.minimum(1, np.hypot(sin_mean, cos_mean))
    return R


@nb.njit
def CIRCVAR(samples, high=2 * np.pi, low=0.0) -> float:
    return 1 - CIRCR(samples=samples, high=high, low=low)


def get_aggregation_functions():
    return {
        MEAN,
        MEDIAN,
        MAX,
        MIN,
        SUM,
        STD,
        VAR,
        CIRCMEAN,
        CIRCR,
        CIRCVAR,
    }
