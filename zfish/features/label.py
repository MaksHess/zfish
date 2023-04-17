import polars as pl

from zfish.features._base import get_si_features_df
from zfish.features.types import LabelImage

SHAPE_FEATURES = {
    "PhysicalSize",
    "NumberOfPixels",
    "Elongation",
    "Flatness",
    "Roundness",
    "FeretDiameter",
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
    # "FeretDiameter",
    "Perimeter",
    "PerimeterOnBorderRatio",
    "EquivalentSphericalRadius",
}

CLEAN_POSITION_AND_ORIENTATION_FEATURES = {
    "Centroid",
    "PrincipalAxes",
    "EquivalentEllipsoidDiameter",
    "BoundingBox",
    # "OrientedBoundingBox",
}

LABEL_FEATURES = CLEAN_SHAPE_FEATURES | CLEAN_POSITION_AND_ORIENTATION_FEATURES


def get_label_features(lbl_img: LabelImage) -> pl.DataFrame:
    return get_si_features_df(lbl_img, props=LABEL_FEATURES, named_features=True)
