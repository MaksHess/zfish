# %%
from enum import auto
from typing import TYPE_CHECKING, Any, TypeAlias

from pydantic import BaseModel, Field
from strenum import PascalCaseStrEnum

from zfish.features._base import get_si_features_df
from zfish.features.constants import IntensityFeature
from zfish.features.queries import FeatureQuery
from zfish.features.types import LabelImage, SpatialImage

if TYPE_CHECKING:
    import polars as pl

    from zfish.roi.spatial_roi import Roi


IntensityFeaturesLike: TypeAlias = tuple[IntensityFeature, ...] | tuple[str, ...]


class IntensityQuery(BaseModel, FeatureQuery):
    label_image: str
    channel: str
    features: tuple[IntensityFeature, ...] = Field(
        default=tuple(IntensityFeature), validate_default=True
    )

    def load_resources(self, roi: "Roi") -> dict[str, Any]:
        return {
            "label_image": roi.sel(l=self.label_image).drop_dim("c").labels.compute(),
            "intensity_image": roi.sel(c=self.channel).drop_dim("l").images.compute(),
            "features": self.features,
        }

    def compute(self, roi: "Roi") -> "pl.DataFrame":
        return get_intensity_features(**self.load_resources(roi))


def get_intensity_features(
    label_image: LabelImage,
    intensity_image: SpatialImage,
    features: IntensityFeaturesLike = tuple(IntensityFeature),
) -> "pl.DataFrame":
    valid_features = tuple(str(IntensityFeature(e)) for e in features)
    return get_si_features_df(
        label_image, intensity_image, props=valid_features, named_features=True
    )


# def get_distribution_features(
#     label_image: LabelImage, intensity_image: SpatialImage
# ) -> pl.DataFrame:
#     return get_si_features_df(
#         label_image, intensity_image, props=DISTRIBUTION_FEATURES, named_features=True
#     )
