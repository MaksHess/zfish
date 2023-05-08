# %%
from typing import TYPE_CHECKING, Callable, Sequence, TypeAlias

import numpy as np
import polars as pl
from scipy.stats import chi2_contingency, kendalltau, pearsonr, spearmanr
from skimage.measure import regionprops_table
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

from zfish.features.types import LabelImage, SpatialImage


def _pearsonr(x: Sequence[float], y: Sequence[float]) -> float:
    statistic, _ = pearsonr(x, y)
    return statistic


def _spearmanr(x: Sequence[float], y: Sequence[float]) -> float:
    statistic, _ = spearmanr(x, y)
    return statistic


def _kendalltau(x: Sequence[float], y: Sequence[float]) -> float:
    statistic, _ = kendalltau(x, y)
    return statistic


def _chi2_contingency(x: Sequence[float], y: Sequence[float], bins: int = 10) -> float:
    c_xy = np.histogram2d(x, y, bins=bins)[0]
    return chi2_contingency(c_xy)[0]


def _mutual_info(x: Sequence[float], y: Sequence[float]) -> float:
    return mutual_info_score(x, y)


def _mutual_info_bins(x: Sequence[float], y: Sequence[float], bins: int = 10) -> float:
    c_xy = np.histogram2d(x, y, bins)[0]
    return mutual_info_score(None, None, contingency=c_xy)


def _normalized_mutual_info(
    x: Sequence[float], y: Sequence[float], bins: int = 10
) -> float:
    return normalized_mutual_info_score(x, y)


ColocalizationFn: TypeAlias = Callable[[Sequence[float], Sequence[float]], float]


COLOC_FUNCTIONS: dict[str, ColocalizationFn] = {
    "PearsonR": _pearsonr,
    "SpearmanR": _spearmanr,
    "KendallTau": _kendalltau,
    # "Chi2": _chi2_contingency,
    # "MI": _mutual_info,
    # "MIbin": _mutual_info_bins,
    # "MIadj": _normalized_mutual_info,
}


def get_colocalization_features(
    lbls_si: LabelImage,
    img1_si: SpatialImage,
    img2_si: SpatialImage,
    coloc_functions: dict[str, ColocalizationFn] = COLOC_FUNCTIONS,
    lbl_dim: str = 'l',
    named_features: bool = True,
    object_column: bool = False,
    struct_index: bool = False,
) -> pl.DataFrame:
    lbls = lbls_si.to_numpy()
    img1 = img1_si.to_numpy()
    img2 = img2_si.to_numpy()
    props = regionprops_table(lbls, properties=("label", "slice"))
    labels = props["label"]
    slices = props["slice"]
    df = pl.DataFrame({"label": labels}).select(pl.col('label').cast(pl.Int64))
    for metric, func in coloc_functions.items():
        corrs = []
        for slc, label in zip(slices, labels):
            lbls_slc = lbls[slc]
            img1_slc = img1[slc]
            img2_slc = img2[slc]
            img1_px = img1_slc[np.where(lbls_slc == label)]
            img2_px = img2_slc[np.where(lbls_slc == label)]
            corrs.append(func(img1_px, img2_px))
        df = df.with_columns(pl.Series(metric, corrs))
    if named_features:
        df = df.select(
            [
                pl.col('label'),
                pl.exclude('label').prefix(
                    f"{img1_si.c.item()}-{img2_si.c.item()}_"
                ),
            ]
        )
    if object_column:
        df = df.with_columns([pl.lit(lbls_si[lbl_dim].item()).alias("object"),]).select(
            [
                pl.col(["object", "label"]),
                pl.exclude(["object", "label"]),
            ]
        )
        if struct_index:
            df = df.select(
                [
                    pl.struct(("object", "label")).alias("index"),
                    pl.exclude(("object", "label")),
                ]
            )
    return df
