# %%
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    import polars as pl

    from zfish.roi.spatial_roi import Roi


from pydantic import BaseModel, Field

from zfish.features._base import get_si_features_df
from zfish.features.constants import (
    DefaultLabelFeature,
    LabelFeature,
    PositionAndOrientationLabelFeature,
)
from zfish.features.queries import FeatureQuery
from zfish.features.types import LabelImage


class LabelQuery(BaseModel, FeatureQuery):
    label_image: str
    features: tuple[LabelFeature, ...] = Field(
        default=DefaultLabelFeature, validate_default=True
    )

    def load_resources(self, roi: "Roi") -> dict[str, Any]:
        return {
            "label_image": roi.sel(l=self.label_image).drop_dim("c").labels.compute(),
            "features": self.features,
        }

    def compute(self, roi: "Roi") -> "pl.DataFrame":
        return get_label_features(**self.load_resources(roi))

LabelFeatureLike: TypeAlias = tuple[LabelFeature, ...] | tuple[str, ...]

def get_label_features(
    label_image: LabelImage,
    features: LabelFeatureLike = tuple(DefaultLabelFeature),
) -> "pl.DataFrame":
    valid_label_features = tuple(str(LabelFeature(e)) for e in features)
    return get_si_features_df(label_image, props=valid_label_features, named_features=True)


def get_centroids(label_image: LabelImage) -> "pl.DataFrame":
    return get_si_features_df(label_image, props=("Centroid",), named_features=True)


def get_position_and_orientation_features(label_image: LabelImage) -> "pl.DataFrame":
    return get_si_features_df(
        label_image,
        props=tuple(PositionAndOrientationLabelFeature),
        named_features=True,
    )

# %%
