# %%
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Annotated,
    Generic,
    NamedTuple,
    NewType,
    Optional,
    Protocol,
    Type,
    TypeAlias,
    TypeVar,
    Union,
)

from pydantic.dataclasses import dataclass as pdataclass
from typing_extensions import Self

RoiID = NewType("RoiID", str)
ChannelID = NewType("ChannelID", str)
ObjectTypeID = NewType("ObjectTypeID", str)
ImageID = NewType("ImageID", tuple[RoiID, str])
LabelImageID = NewType("LabelImageID", tuple[RoiID, str])
LabelObjectID = NewType("LabelObjectID", tuple[LabelImageID, int])
PointsID = NewType("PointsID", str)
ShapesID = NewType("ShapesID", str)
TableID = NewType("TableID", str)
FeatureID = NewType("FeatureID", tuple[TableID, str])

RegionID: TypeAlias = Union[LabelImageID, PointsID, ShapesID]
ResourceID: TypeAlias = Union[
    ChannelID, LabelImageID, LabelObjectID, TableID, FeatureID
]


ElementID = Union[
    RoiID,
    ChannelID,
    LabelImageID,
    LabelObjectID,
    PointsID,
    ShapesID,
    TableID,
    FeatureID,
    RegionID,
]


@pdataclass
class Resources:
    label_images: set[LabelImageID] = field(default_factory=set)
    channels: set[ChannelID] = field(default_factory=set)
    channel_pairs: set[tuple[ChannelID, ChannelID]] = field(default_factory=set)
    label_objects: set[LabelObjectID] = field(default_factory=set)
    tables: set[TableID] = field(default_factory=set)
    features: set[FeatureID] = field(default_factory=set)

    def union(self, *others: "Resources") -> "Resources":
        return Resources(
            label_images=self.label_images.union(
                *[other.label_images for other in others]
            ),
            channels=self.channels.union(*[other.channels for other in others]),
            channel_pairs=self.channels.union(
                *[other.channel_pairs for other in others]
            ),
            tables=self.tables.union(*[other.tables for other in others]),
            features=self.features.union(*[other.features for other in others]),
        )

    def validate_available(self, other: "Resources") -> Self:
        _validate_resource_query(self, other)
        return self

    def __lt__(self, other):
        keys = Resources.__dataclass_fields__.keys()
        return tuple(sorted(tuple(getattr(self, key))) for key in keys) < tuple(
            sorted(tuple(getattr(other, key))) for key in keys
        )


def _validate_resource_query(resources: Resources, available_resources: Resources):
    for k in Resources.__dataclass_fields__.keys():
        missing_resources = getattr(resources, k).difference(
            getattr(available_resources, k)
        )
        assert missing_resources == set(), (
            f"Missing resources: `{missing_resources}` not found in {k}"
        )
