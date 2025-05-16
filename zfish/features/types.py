from typing import TypeAlias

from spatial_image import SpatialImage


Image: TypeAlias = SpatialImage
MultiChannelImage: TypeAlias = SpatialImage

LabelImage: TypeAlias = SpatialImage
MultichannelLabelImage: TypeAlias = SpatialImage
OverlapLabelImage: TypeAlias = MultichannelLabelImage 
# (object_type: 2, z: 100, y: 200, x: 200) -> labels must be unique across both label_images.

BinaryImage: TypeAlias = SpatialImage

DistanceTransform: TypeAlias = SpatialImage
