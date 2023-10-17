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
ImageID = NewType("ImageID", str)
ChannelID = NewType("ChannelID", str)
LabelImageID = NewType("LabelImageID", str)
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
            channel_pairs=self.channels.union(*[other.channel_pairs for other in others]),
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
        assert (
            missing_resources == set()
        ), f"Missing resources: `{missing_resources}` not found in {k}"


# class FeatureResourceBase(ABC):
#     @property
#     @abstractmethod
#     def region_id(self) -> RegionID:
#         ...

#     @property
#     @abstractmethod
#     def other_ids(self) -> set[ResourceID]:
#         ...

#     # @property
#     # def resources(self) -> Resources:
#     #     [e[0] if isinstance(e, tuple) else e for e in]


# class LabelFeature(FeatureResourceBase):
#     pass


# class ChannelResources(Resources):
#     def validate(self) -> Self:
#         pass


# class ElementModelBase(ABC):
#     @property
#     @abstractmethod
#     def id(self) -> str:
#         ...

#     @property
#     @abstractmethod
#     def resources(self) -> Resources:
#         ...

#     @abstractmethod
#     def resources_valid(self) -> Self:
#         ...

#     def validate(self, available_resources: Resources) -> Self:
#         _validate_resources(self.resources, available_resources)
#         return self

#     def __str__(self) -> str:
#         return str(self.id)


# class Channel(ElementModelBase):
#     def __init__(self, id: str, resources: Resources):
#         pass


# @dataclass_abc
# class LabelImage(ElementBase):
#     id: LabelImageID


# @dataclass_abc
# class LabelObject(ElementBase):
#     id: LabelObjectID


# # %%
# channels = set(["ch0", "ch1"])
# av_channels = set(["ch0", "ch1", "ch2"])

# r = Resources(channels=channels)
# r2 = Resources(channels=av_channels)
# r.union(r2)

# %%
# # RoiID: TypeAlias = str
# ChannelID: TypeAlias = str
# # LabelsImageID: TypeAlias = str
# # LabelsObjectID: TypeAlias = tuple[LabelsImageID, int]
# PointsID: TypeAlias = str
# ShapesID: TypeAlias = str
# TableID: TypeAlias = str
# FeatureID: TypeAlias = tuple[TableID, str]
# RegionID: TypeAlias = Union[LabelsImageID, LabelsObjectID]


# RoiID_T = Annotated[str, "roi"]

# roi1: RoiID = "site1"
# roi2: RoiID_T = "site2"

# # %%
# T = TypeVar("T")


# class Named(Generic[T]):
#     def __init__(self, name: str, value: T):
#         self.name = name
#         self.value = value


# Named("roi_id", str)
# # %%
# @dataclass
# class Named(Generic[T]):
#     name: str
#     value: T


# RoiID: TypeAlias = Named[str]
# LabelsImageID: TypeAlias = Named[str]
# LabelsObjectID: TypeAlias = Named[tuple[LabelsImageID, int]]
# %%


# ID_T = TypeVar("ID_T", bound=ElementID)


# class ID(Generic[ID_T]):
#     value: ID_T


# %%
# try:
#     from typing import _GenericAlias  # type: ignore[attr-defined]
# except ImportError:  # pragma: no cover
#     _GenericAlias = None

# from typing import _type_check


# class BaseID(Generic[ID_T]):
#     default_dtype: Optional[Type] = None

#     def __get__(self, instance: object, owner: Type) -> str:  # pragma: no cover
#         raise AttributeError("Series should resolve to Field-s")


# class ID(BaseID, ElementID, Generic[ID_T]):  # type: ignore
#     """Representation of pandas.Series, only used for type annotation.
#     *new in 0.5.0*
#     """

#     if hasattr(ElementID, "__class_getitem__") and _GenericAlias:

#         def __class_getitem__(cls, item):
#             """Define this to override the patch that pyspark.pandas performs on pandas.
#             https://github.com/apache/spark/blob/master/python/pyspark/pandas/__init__.py#L124-L144
#             """
#             _type_check(item, "Parameters to generic types must be types.")
#             return _GenericAlias(cls, item)
