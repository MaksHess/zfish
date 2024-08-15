# %%
from enum import auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    NamedTuple,
    Sequence,
    TypeAlias,
)

if TYPE_CHECKING:
    from zfish.roi.spatial_roi import Roi

import numpy as np
import polars as pl
from pydantic import BaseModel
from scipy.stats import chi2_contingency, kendalltau, pearsonr, spearmanr
from skimage.measure import regionprops_table
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

from zfish.features.constants import ColocalizationFeature, DefaultColocalizationFeature
from zfish.features.queries import FeatureQuery
from zfish.features.types import LabelImage, SpatialImage


# TODO: Consistency of error cases between statistics
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


ALL_CORRELATION_FUNCTIONS: dict[str, ColocalizationFn] = {
    ColocalizationFeature.PEARSON_R: _pearsonr,
    ColocalizationFeature.SPEARMAN_R: _spearmanr,
    ColocalizationFeature.KENDALL_TAU: _kendalltau,
    ColocalizationFeature.CHI2_CONTINGENCY: _chi2_contingency,
    ColocalizationFeature.MUTUAL_INFO: _mutual_info,
    ColocalizationFeature.MUTUAL_INFO_BINS: _mutual_info_bins,
    ColocalizationFeature.MUTUAL_INFO_NORMALIZED: _normalized_mutual_info,
}


class ChannelPair(NamedTuple):
    channel0: str
    channel1: str


class ColocalizationQuery(BaseModel, FeatureQuery):
    label_image: str
    channel_pair: ChannelPair
    features: tuple[ColocalizationFeature, ...] = tuple(DefaultColocalizationFeature)

    def load_resources(self, roi: "Roi") -> dict[str, Any]:
        return {
            "label_image": roi.sel(l=self.label_image).drop_dim("c").labels.compute(),
            "channel0": roi.sel(c=self.channel_pair.channel0)
            .drop_dim("l")
            .images.compute(),
            "channel1": roi.sel(c=self.channel_pair.channel1)
            .drop_dim("l")
            .images.compute(),
            "features": self.features,
        }

    def compute(self, roi: "Roi") -> "pl.DataFrame":
        return get_colocalization_features(**self.load_resources(roi))


LABEL_IMAGE_COLUMN = "label_image"
LABEL_ID_COLUMN = "label"
RESOURCE_COLUMNS = ("channel0", "channel1")


def get_colocalization_features(
    label_image: LabelImage,
    channel0: SpatialImage,
    channel1: SpatialImage,
    *,
    features: tuple[ColocalizationFeature, ...] = tuple(DefaultColocalizationFeature),
    index_columns: tuple[Literal[LABEL_IMAGE_COLUMN, LABEL_ID_COLUMN], ...] = (
        # "label_image",
        "label",
    ),
    index_prefix: str | None = None,
    add_resource_column: bool = False,
    resource_prefix: str | None = "resource",
    resource_in_name: bool = True,
    return_metadata: bool = False,
    lbl_dim: str = "l",
) -> pl.DataFrame:
    valid_features = tuple(ColocalizationFeature(e) for e in features)

    coloc_functions = {str(k): ALL_CORRELATION_FUNCTIONS[k] for k in valid_features}
    lbls = label_image.to_numpy()
    img1 = channel0.to_numpy()
    img2 = channel1.to_numpy()

    props = regionprops_table(lbls, properties=("label", "slice"))
    labels = props["label"]
    slices = props["slice"]

    df = pl.DataFrame()

    for metric, func in coloc_functions.items():
        corrs = []
        for slc, label in zip(slices, labels):
            lbls_slc = lbls[slc]
            img1_slc = img1[slc]
            img2_slc = img2[slc]
            img1_px = img1_slc[np.where(lbls_slc == label)]
            img2_px = img2_slc[np.where(lbls_slc == label)]
            try:
                res = func(img1_px, img2_px)
            except ValueError as e:
                res = np.nan
            corrs.append(res)
        df = df.with_columns(pl.Series(metric, corrs))

    df = df.fill_nan(None)

    meta = {
        "feature_type": "correlation",
        "label_image": label_image[lbl_dim].item(),
        "channel0": channel0.c.item(),
        "channel1": channel1.c.item(),
    }

    if resource_in_name:
        df = df.select(
            [
                pl.all().prefix(f"{meta['channel0']}|{meta['channel1']}_"),
            ]
        )

    if LABEL_ID_COLUMN in index_columns:
        df = df.with_columns(
            pl.Series(name=LABEL_ID_COLUMN, values=labels).cast(pl.Int64)
        )

    if LABEL_IMAGE_COLUMN in index_columns:
        df = df.with_columns(pl.lit(meta["label_image"]).alias(LABEL_IMAGE_COLUMN))

    if index_prefix is not None:
        df = df.select(
            pl.col(index_columns).prefix(f"{index_prefix}."), pl.exclude(index_columns)
        )
        index_columns_out = tuple(
            [f"{index_prefix}.{index_column}" for index_column in index_columns]
        )
    else:
        df = df.select(pl.col(index_columns), pl.exclude(index_columns))
        index_columns_out = index_columns

    if add_resource_column:
        df = df.with_columns(
            pl.lit(channel0.c.item()).alias(RESOURCE_COLUMNS[0]),
            pl.lit(channel1.c.item()).alias(RESOURCE_COLUMNS[1]),
        )

        if resource_prefix is not None:
            resource_columns_out = tuple(
                [
                    f"{resource_prefix}.{resource_column}"
                    for resource_column in RESOURCE_COLUMNS
                ]
            )
            df = df.select(
                pl.col(index_columns_out),
                pl.col(RESOURCE_COLUMNS).prefix(f"{resource_prefix}."),
                pl.exclude(index_columns_out + RESOURCE_COLUMNS),
            )
        else:
            resource_columns_out = RESOURCE_COLUMNS
    else:
        resource_columns_out = RESOURCE_COLUMNS

    meta["index_columns"] = index_columns_out
    meta["resource_columns"] = resource_columns_out

    if return_metadata:
        return df, meta
    return df


# %%

# def get_colocalization_features(
#     label_image: LabelImage,
#     channel0: SpatialImage,
#     channel1: SpatialImage,
#     features: tuple[ColocalizationFn, ...] = CORRELATION_FEATURES,
#     lbl_dim: str = "l",
#     named_features: bool = True,
#     object_column: bool = False,
#     struct_index: bool = False,
# ) -> pl.DataFrame:
#     coloc_functions = {k: ALL_CORRELATION_FUNCTIONS[k] for k in features}
#     lbls = label_image.to_numpy()
#     img1 = channel0.to_numpy()
#     img2 = channel1.to_numpy()
#     props = regionprops_table(lbls, properties=("label", "slice"))
#     labels = props["label"]
#     slices = props["slice"]
#     df = pl.DataFrame({"label": labels}).select(pl.col("label").cast(pl.Int64))
#     for metric, func in coloc_functions.items():
#         corrs = []
#         for slc, label in zip(slices, labels):
#             lbls_slc = lbls[slc]
#             img1_slc = img1[slc]
#             img2_slc = img2[slc]
#             img1_px = img1_slc[np.where(lbls_slc == label)]
#             img2_px = img2_slc[np.where(lbls_slc == label)]
#             corrs.append(func(img1_px, img2_px))
#         df = df.with_columns(pl.Series(metric, corrs))
#     if named_features:
#         df = df.select(
#             [
#                 pl.col("label"),
#                 pl.exclude("label").prefix(f"{channel0.c.item()}|{channel1.c.item()}_"),
#             ]
#         )
#     if object_column:
#         df = df.with_columns(
#             [
#                 pl.lit(label_image[lbl_dim].item()).alias("object"),
#             ]
#         ).select(
#             [
#                 pl.col(["object", "label"]),
#                 pl.exclude(["object", "label"]),
#             ]
#         )
#         if struct_index:
#             df = df.select(
#                 [
#                     pl.struct(("object", "label")).alias("index"),
#                     pl.exclude(("object", "label")),
#                 ]
#             )
#     return df.fill_nan(None)


# def get_colocalization_features_v3(
#     label_image: LabelImage,
#     channel0: SpatialImage,
#     channel1: SpatialImage,
#     *,
#     features: tuple[str, ...] = CORRELATION_FEATURES,
# ) -> "pd.DataFrame":
#     import pandas as pd
#     import woodwork as ww

#     df_pl, meta = get_colocalization_features_v2(
#         label_image, channel0, channel1, index_columns=("label",), return_metadata=True
#     )
#     df = df_pl.to_pandas()
#     df.ww.init()
#     df.ww.set_types(
#         logical_types={
#             "label": "Categorical",
#         }
#     )
#     df.ww.set_index("label")

#     return df
#     return df
