import polars as pl

from zfish.features._base import get_si_features_df
from zfish.features.types import LabelImage, SpatialImage

DISTRIBUTION_FEATURES = {
    "Mean",
    "Median",
    "Minimum",
    "Maximum",
    "Sum",
    "Variance",
    "StandardDeviation",
    "Skewness",
    "Kurtosis",
}

WEIGHTED_SHAPE_FEATURES = {
    "WeightedElongation",
    "WeightedFlatness",
    "CenterOfGravity",
    "WeightedPrincipalAxes",
    "WeightedPrincipalMoments",
    "MaximumIndex",
    "MinimumIndex",
}

# TODO: Implement if necessary, else delete
HISTOGRAM_FEATURES = {
    "Histogram",
}

INTENSITY_FEATURES = DISTRIBUTION_FEATURES | WEIGHTED_SHAPE_FEATURES


def get_distribution_features(
    label_image: LabelImage, intensity_image: SpatialImage
) -> pl.DataFrame:
    return get_si_features_df(
        label_image, intensity_image, props=DISTRIBUTION_FEATURES, named_features=True
    )


def get_intensity_features(
    label_image: LabelImage, intensity_image: SpatialImage, features=INTENSITY_FEATURES
) -> pl.DataFrame:
    return get_si_features_df(
        label_image, intensity_image, props=features, named_features=True
    )
