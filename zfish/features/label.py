# %%
import warnings

import func_timeout
import polars as pl

from zfish.features._base import get_si_features_df
from zfish.features.types import LabelImage

SHAPE_FEATURES = {
    "PhysicalSize",
    "NumberOfPixels",
    "Elongation",
    "Flatness",
    "Roundness",
    "FeretDiameter",  # SLOW FOR LARGE OBJECTS!!!
    "Perimeter",
    "EquivalentSphericalPerimeter",
    "EquivalentSphericalRadius",
    "NumberOfPixelsOnBorder",
    "PerimeterOnBorder",
    "PerimeterOnBorderRatio",
}

POSITION_AND_ORIENTATION_FEATURES = {
    "Centroid",
    "PrincipalAxes",
    "PrincipalMoments",
    "EquivalentEllipsoidDiameter",
    "BoundingBox",
    "OrientedBoundingBox",
}

CLEAN_SHAPE_FEATURES = {
    "PhysicalSize",
    "Elongation",
    "Flatness",
    "Roundness",
    "FeretDiameter",
    "Perimeter",
    "PerimeterOnBorderRatio",
    "EquivalentSphericalRadius",
}

CLEAN_POSITION_AND_ORIENTATION_FEATURES = {
    "Centroid",
    "PrincipalAxes",
    "EquivalentEllipsoidDiameter",
    "BoundingBox",
    "OrientedBoundingBox",
}

SLOW_FEATURES = {
    "FeretDiameter",
}

LABEL_FEATURES = CLEAN_SHAPE_FEATURES | CLEAN_POSITION_AND_ORIENTATION_FEATURES
# FAST_LABEL_FEATURES = LABEL_FEATURES.difference(SLOW_FEATURES)


def get_centroids(label_image: LabelImage) -> pl.DataFrame:
    return get_si_features_df(label_image, props=["Centroid"], named_features=True)


def get_position_and_orientation_features(label_image: LabelImage) -> pl.DataFrame:
    return get_si_features_df(
        label_image, props=CLEAN_POSITION_AND_ORIENTATION_FEATURES, named_features=True
    )

def get_label_features(label_image: LabelImage, features=LABEL_FEATURES) -> pl.DataFrame:
    return get_si_features_df(
        label_image, props=features, named_features=True
    )