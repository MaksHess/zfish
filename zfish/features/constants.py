"""Moved to separate file for fast imports."""
# %%
from dataclasses import dataclass
from enum import auto

from strenum import PascalCaseStrEnum


# For density features in `zfish.features.neighborhood.density`
@dataclass
class DensityParams:
    radius: tuple[float, ...] = tuple([10, 20, 30, 40, 50, 80, 100, 150, 200, 250])
    distance_to_closest_neighbor: bool = True
    knn_distance: tuple[int, ...] = tuple([2, 5, 10, 20, 50, 100, 200])
    delaunay: tuple[int, ...] = tuple([1])
    touch: tuple[int, ...] = tuple([1])
    adjacency_aggfuncs: tuple[str, ...] = tuple(["Count"])
    distance_aggfuncs: tuple[str, ...] = tuple(["Mean", "Max"])

# For label features in `zfish.features.labels`
class LabelFeature(PascalCaseStrEnum):
    ## Shape
    PHYSICAL_SIZE = auto()
    NUMBER_OF_PIXELS = auto()
    ELONGATION = auto()
    FLATNESS = auto()
    ROUNDNESS = auto()
    FERET_DIAMETER = auto()
    PERIMETER = auto()
    EQUIVALENT_SPHERICAL_PERIMETER = auto()
    EQUIVALENT_SPHERICAL_RADIUS = auto()
    NUMBER_OF_PIXELS_ON_BORDER = auto()
    PERIMETER_ON_BORDER = auto()
    PERIMETER_ON_BORDER_RATIO = auto()
    ## Position
    CENTROID = auto()
    ## Orientation
    PRINCIPAL_AXES = auto()
    PRINCIPAL_MOMENTS = auto()
    EQUIVALENT_ELLIPSOID_DIAMETER = auto()
    ## Misc
    BOUNDING_BOX = auto()
    ORIENTED_BOUNDING_BOX = auto()


class DefaultLabelFeature(PascalCaseStrEnum):
    ## Shape
    PHYSICAL_SIZE = auto()
    # NUMBER_OF_PIXELS = auto()
    ELONGATION = auto()
    FLATNESS = auto()
    ROUNDNESS = auto()
    FERET_DIAMETER = auto()
    PERIMETER = auto()
    EQUIVALENT_SPHERICAL_PERIMETER = auto()
    EQUIVALENT_SPHERICAL_RADIUS = auto()
    # NUMBER_OF_PIXELS_ON_BORDER = auto()
    # PERIMETER_ON_BORDER = auto()
    PERIMETER_ON_BORDER_RATIO = auto()
    ## Position
    CENTROID = auto()
    ## Orientation
    PRINCIPAL_AXES = auto()
    # PRINCIPAL_MOMENTS = auto()
    EQUIVALENT_ELLIPSOID_DIAMETER = auto()
    ## Misc
    BOUNDING_BOX = auto()
    ORIENTED_BOUNDING_BOX = auto()


class PositionAndOrientationLabelFeature(PascalCaseStrEnum):
    CENTROID = auto()
    PRINCIPAL_AXES = auto()
    EQUIVALENT_ELLIPSOID_DIAMETER = auto()
    BOUNDING_BOX = auto()
    ORIENTED_BOUNDING_BOX = auto()

# For intensity features in `zfish.features.intensity`
class IntensityFeature(PascalCaseStrEnum):
    ## Intensity Distibution Features
    MEAN = auto()
    MEDIAN = auto()
    MINIMUM = auto()
    MAXIMUM = auto()
    SUM = auto()
    VARIANCE = auto()
    STANDARD_DEVIATION = auto()
    SKEWNESS = auto()
    KURTOSIS = auto()
    ## Weighted Shape and Orientation Features
    WEIGHTED_ELONGATION = auto()
    WEIGHTED_FLATNESS = auto()
    CENTER_OF_GRAVITY = auto()
    WEIGHTED_PRINCIPAL_AXES = auto()
    WEIGHTED_PRINCIPAL_MOMENTS = auto()
    MAXIMUM_INDEX = auto()
    MIMIMUM_INDEX = auto()
    # TODO: Implement if necessary, else delete
    # HISTOGRAM = auto()

# For correlation features in `zfish.features.colocalization`
class ColocalizationFeature(PascalCaseStrEnum):
    PEARSON_R = auto()
    SPEARMAN_R = auto()
    KENDALL_TAU = auto()
    CHI2_CONTINGENCY = auto()
    MUTUAL_INFO = auto()
    MUTUAL_INFO_BINS = auto()
    MUTUAL_INFO_NORMALIZED = auto()

# For distance features in `zfish.features.distance`
class DefaultColocalizationFeature(PascalCaseStrEnum):
    PEARSON_R = auto()
    SPEARMAN_R = auto()
    KENDALL_TAU = auto()
    
class DistanceFeature(PascalCaseStrEnum):
    CENTROID = auto()
    MAXIMUM = auto()
    MINIMUM = auto()
    MEDIAN = auto()
    MAXIMUM_INDEX = auto()
    MINIMUM_INDEX = auto()


class DefaultDistanceFeature(PascalCaseStrEnum):
    CENTROID = auto()
    MAXIMUM = auto()
    MINIMUM = auto()


class DistanceFunction(PascalCaseStrEnum):
    DISTANCE_TO_BORDER = auto()
    DISTANCE_ALONG_Z = auto()
    DISTANCE_ALONG_Y = auto()
    DISTANCE_ALONG_X = auto()


class DefaultDistanceFunction(PascalCaseStrEnum):
    DISTANCE_TO_BORDER = auto()
    DISTANCE_ALONG_Z = auto()