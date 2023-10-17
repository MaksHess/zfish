# %%
from abc import ABC, abstractmethod
from functools import reduce
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

from collections.abc import Iterable
from itertools import chain, product, repeat
from pathlib import Path
from typing import Any, TypeAlias, TypeVar

from pydantic import BaseModel, Field, field_validator
from typing_extensions import Self

from zfish.features.correlation import CORRELATION_FEATURES, get_colocalization_features
from zfish.features.distance import (
    DISTANCE_ITK_FEATURES,
    DISTANCE_TRANSFORMS,
    get_distance_features,
)
from zfish.features.feature_types import (
    Resources,
)
from zfish.features.intensity import INTENSITY_FEATURES, get_intensity_features
from zfish.features.label import LABEL_FEATURES, get_label_features
from zfish.features.neighborhood.density import get_density_features
from zfish.features.object_hierarchy import get_parent_objects
from zfish.features.types import LabelImage, SpatialImage
from zfish.roi.spatial_roi import Roi


def _get_channels_safe(img: LabelImage | SpatialImage) -> set[str]:
    """
    Custom function to extract channel info from `SpatialImage`. Using the obvious
    `set(img.c.values)` seemed to work initially but lead to unexpected behaviour when
    trying to serialize using yaml.dump. Must be some weird interaction w/ numpy/xarray
    under the hood, explicitly converting to `str` in a comprehension does the trick.
    """
    return set([str(e) for e in img.c.values])


Channel: TypeAlias = str
Labels: TypeAlias = str
ChannelPair: TypeAlias = tuple[Channel, Channel]
ChannelSet: TypeAlias = tuple[Channel, ...]
LabelObject: TypeAlias = tuple[Labels, int]
Feature: TypeAlias = Any


def _parse_star_expression(
    channels: set[str],
    available_channels: set[str],
) -> set[str]:
    channels = set(channels)
    available_channels = set(available_channels)
    if channels == {"*"}:
        return available_channels
    if any([ch.startswith("!") for ch in channels]):
        assert all(
            [ch.startswith("!") for ch in channels]
        ), "All channels have to start with `!` if any channel starts with `!`."
        excluded_channels = [e[1:] for e in channels]
        for ch in excluded_channels:
            assert ch in available_channels, f"`{ch}` not in `{available_channels}`."
        return set([ch for ch in available_channels if ch not in excluded_channels])
    else:
        for ch in channels:
            assert ch in available_channels, f"`{ch}` not in `{available_channels}`."
        return channels


def _parse_pairwise_star_expression(
    channel_pairs: set[tuple[str, str]],
    available_channels: set[str],
) -> set[tuple[str, str]]:
    out_pairs = set()
    for c0, c1 in channel_pairs:
        if (c0 == "*" or c0.startswith("!")) and (c1 == "*" or c1.startswith("!")):
            out_pairs.update(
                [
                    tuple(sorted(e))
                    for e in product(
                        _parse_star_expression(set([c0]), available_channels),
                        _parse_star_expression(set([c1]), available_channels),
                    )
                ]
            )
        elif c0 == "*" or c0.startswith("!"):
            out_pairs.update(
                [
                    tuple(sorted(e))
                    for e in zip(
                        _parse_star_expression(set([c0]), available_channels),
                        repeat(c1),
                    )
                ]
            )
        elif c1 == "*" or c1.startswith("!"):
            out_pairs.update(
                [
                    tuple(sorted(e))
                    for e in zip(
                        repeat(c0),
                        _parse_star_expression(set([c1]), available_channels),
                    )
                ]
            )
        else:
            out_pairs.add(tuple(sorted((c0, c1))))
    return out_pairs


def _parse_channel_pairs(
    channel_pairs: tuple[ChannelPair, ...], available_channels: set[str]
) -> tuple[ChannelPair, ...]:
    channel_pairs_tpl = [(cp.channel0, cp.channel1) for cp in channel_pairs]
    out = _parse_pairwise_star_expression(channel_pairs_tpl, available_channels)
    return tuple(ChannelPair(channel0=ch0, channel1=ch1) for ch0, ch1 in out)


class MockRoi(BaseModel):
    resources: Resources = dict(
        channels=tuple([f"DAPI.{i}" for i in range(4)] + ["PCNA.0", "pH3.1"]),
        label_images=tuple(["nucleiRaw3", "cells", "embryoRaw"]),
    )


def _validate_resources_in_roi(resources: Resources, roi: "Roi") -> None:
    for channel in resources.channels:
        assert (
            channel in roi.resources["channels"]
        ), f"Channel `{channel}` not found in `{roi.resources['channels']}`!"
    for label in resources.label_images:
        assert (
            label in roi.resources["label_images"]
        ), f"Label `{label}` not found in `{roi.resources['label_images']}!"


T = TypeVar("T")


def _make_iterable(a: T | Iterable[T]) -> Iterable[T]:
    """`str` and `LabelObject`, altough iterable, will still get packaged!"""
    if (
        isinstance(a, str)
        or isinstance(a, LabelObject)
        or isinstance(a, ChannelPair)
        or not isinstance(a, Iterable)
    ):
        a = [a]
    return a


class FeatureQuery(ABC):
    @abstractmethod
    def load_resources(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> "pl.DataFrame":
        raise NotImplementedError


class HierarchyQuery(BaseModel, FeatureQuery):
    label_image: str
    parent_label_images: tuple[str, ...]

    @property
    def resources(self) -> Resources:
        return Resources(label_images=(self.label_image,) + self.parent_label_images)

    def load_resources(self, roi: Roi) -> dict[str, Any]:
        return {
            "label_image": roi.sel(l=self.label_image).drop_dim("c").labels.compute(),
            "parent_label_images": roi.sel(l=list(self.parent_label_images))
            .drop_dim("c")
            .labels.compute(),
        }

    def compute(self, roi: Roi) -> "pl.DataFrame":
        return get_parent_objects(**self.load_resources(roi))


class LabelQuery(BaseModel, FeatureQuery):
    label_image: str
    features: tuple[str, ...]

    def load_resources(self, roi: Roi) -> dict[str, Any]:
        return {
            "label_image": roi.sel(l=self.label_image).drop_dim("c").labels.compute(),
            "features": self.features,
        }

    def compute(self, roi: Roi) -> "pl.DataFrame":
        return get_label_features(**self.load_resources(roi))


class LabelQueries(BaseModel):
    labelss: tuple[str, ...] = Field(default="*", validate_default=True)
    props: tuple[str, ...] = Field(default="*", validate_default=True)

    @field_validator("labelss", "props", mode="before")
    def _package_singletons(cls, val):
        return _make_iterable(val)

    @field_validator("props", mode="before")
    def _resolve_star_expression(cls, val):
        val_out = _parse_star_expression(_make_iterable(val), LABEL_FEATURES)
        return val_out

    def _parse_resource_star_expressions(self, roi: Roi) -> "IntensityQueries":
        return LabelQueries(
            labelss=_parse_star_expression(self.labelss, roi.resources["label_images"]),
            props=self.props,
        )

    def _validate_resources(self, roi: Roi) -> Self:
        _validate_resources_in_roi(self.resources, roi)
        return self

    @property
    def resources(self) -> Resources:
        return Resources(label_images=self.labelss)

    def queries(self, roi: Roi) -> list[LabelQuery]:
        parsed_request = self._parse_resource_star_expressions(roi)
        valid_request = parsed_request._validate_resources(roi)
        qs = []
        for labels in valid_request.labelss:
            qs.append(LabelQuery(label_image=labels, features=valid_request.props))
        return qs


class IntensityQuery(BaseModel, FeatureQuery):
    label_image: str
    channel: str
    features: tuple[str, ...]

    def load_resources(self, roi: Roi) -> dict[str, Any]:
        return {
            "label_image": roi.sel(l=self.label_image).drop_dim("c").labels.compute(),
            "intensity_image": roi.sel(c=self.channel).drop_dim("l").images.compute(),
            "features": self.features,
        }

    def compute(self, roi: Roi) -> "pl.DataFrame":
        return get_intensity_features(**self.load_resources(roi))


class IntensityQueries(BaseModel):
    labelss: tuple[str, ...] = Field(default="*", validate_default=True)
    channels: tuple[str, ...] = Field(default="*", validate_default=True)
    props: tuple[str, ...] = Field(default="*", validate_default=True)

    @field_validator("labelss", "channels", "props", mode="before")
    def _package_singletons(cls, val):
        return _make_iterable(val)

    @field_validator("props", mode="before")
    def _resolve_star_expression(cls, val):
        val_out = _parse_star_expression(_make_iterable(val), INTENSITY_FEATURES)
        return val_out

    def _parse_resource_star_expressions(self, roi: Roi) -> "IntensityQueries":
        return IntensityQueries(
            labelss=_parse_star_expression(self.labelss, roi.resources["label_images"]),
            channels=_parse_star_expression(self.channels, roi.resources["channels"]),
            props=self.props,
        )

    def _validate_resources(self, roi: Roi) -> Self:
        _validate_resources_in_roi(self.resources, roi)
        return self

    @property
    def resources(self) -> Resources:
        return Resources(label_images=self.labelss, channels=self.channels)

    def queries(self, roi: Roi) -> list[IntensityQuery]:
        parsed_request = self._parse_resource_star_expressions(roi)
        valid_request = parsed_request._validate_resources(roi)
        qs = []
        for labels, channel in product(valid_request.labelss, valid_request.channels):
            qs.append(
                IntensityQuery(
                    label_image=labels, channel=channel, features=valid_request.props
                )
            )
        return qs


class ChannelPair(BaseModel):
    channel0: str
    channel1: str


class CorrelationQuery(BaseModel, FeatureQuery):
    label_image: str
    channel_pair: ChannelPair
    features: tuple[str, ...]

    def load_resources(self, roi: Roi) -> dict[str, Any]:
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

    def compute(self, roi: Roi) -> "pl.DataFrame":
        return get_colocalization_features(**self.load_resources(roi))


class CorrelationQueries(BaseModel):
    labelss: tuple[str, ...] = Field(default="*", validate_default=True)
    channel_pairs: tuple[ChannelPair, ...] = Field(
        default=(ChannelPair(channel0="*", channel1="*"),), validate_default=True
    )
    props: tuple[str, ...] = Field(default="*", validate_default=True)

    @field_validator("labelss", "channel_pairs", "props", mode="before")
    def _package_singletons(cls, val):
        return _make_iterable(val)

    @field_validator("props", mode="before")
    def _resolve_star_expression(cls, val):
        val_out = _parse_star_expression(_make_iterable(val), CORRELATION_FEATURES)
        return val_out

    def _parse_resource_star_expressions(self, roi: Roi) -> "CorrelationQueries":
        return CorrelationQueries(
            labelss=_parse_star_expression(self.labelss, roi.resources["label_images"]),
            channel_pairs=_parse_channel_pairs(
                self.channel_pairs, roi.resources["channels"]
            ),
            props=self.props,
        )

    def _validate_resources(self, roi: Roi) -> Self:
        _validate_resources_in_roi(self.resources, roi)
        return self

    @property
    def resources(self) -> Resources:
        return Resources(
            label_images=self.labelss,
            channels=set(
                [cp.channel0 for cp in self.channel_pairs]
                + [cp.channel1 for cp in self.channel_pairs]
            ),
        )

    def queries(self, roi: Roi) -> list[CorrelationQuery]:
        parsed_request = self._parse_resource_star_expressions(roi)
        valid_request = parsed_request._validate_resources(roi)
        qs = []
        for labels, channel_pair in product(
            valid_request.labelss, valid_request.channel_pairs
        ):
            qs.append(
                CorrelationQuery(
                    label_image=labels,
                    channel_pair=channel_pair,
                    features=valid_request.props,
                )
            )
        return qs


class LabelObject(BaseModel):
    label_image: str
    label_id: int


class DistanceQuery(BaseModel, FeatureQuery):
    label_image: str
    label_object_to: LabelObject
    features: tuple[str, ...]
    distance_transforms: tuple[str, ...]

    def load_resources(self, roi: Roi) -> dict[str, Any]:
        return {
            "label_image": roi.sel(l=self.label_image).drop_dim("c").labels.compute(),
            "label_image_to": roi.sel(l=self.label_object_to.label_image)
            .drop_dim("c")
            .labels.compute(),
            "label_to": self.label_object_to.label_id,
            "features": self.features,
            "distance_transforms": self.distance_transforms,
        }

    def compute(self, roi: Roi) -> "pl.DataFrame":
        return get_distance_features(**self.load_resources(roi))


class DistanceQueries(BaseModel):
    label_objects_to: tuple[LabelObject, ...]
    labelss: tuple[str, ...] = Field(default="*", validate_default=True)
    props: tuple[str, ...] = Field(default="*", validate_default=True)
    distance_transforms: tuple[str, ...] = Field(default="*", validate_default=True)

    @field_validator(
        "labelss", "label_objects_to", "props", "distance_transforms", mode="before"
    )
    def _package_singletons(cls, val):
        return _make_iterable(val)

    @field_validator("props", mode="before")
    def _resolve_star_expression(cls, val):
        val_out = _parse_star_expression(_make_iterable(val), DISTANCE_ITK_FEATURES)
        return val_out

    @field_validator("distance_transforms", mode="before")
    def _resolve_star_expression2(cls, val):
        val_out = _parse_star_expression(_make_iterable(val), DISTANCE_TRANSFORMS)
        return val_out

    def _parse_resource_star_expressions(self, roi: Roi) -> "DistanceQueries":
        return DistanceQueries(
            labelss=_parse_star_expression(self.labelss, roi.resources["label_images"]),
            label_objects_to=self.label_objects_to,
            props=self.props,
            distance_transforms=self.distance_transforms,
        )

    def _validate_resources(self, roi: Roi) -> Self:
        _validate_resources_in_roi(self.resources, roi)
        return self

    @property
    def resources(self) -> Resources:
        return Resources(
            label_images=self.labelss
            + tuple(
                label_object_to.label_image for label_object_to in self.label_objects_to
            )
        )

    def queries(self, roi: Roi) -> list[LabelQuery]:
        parsed_request = self._parse_resource_star_expressions(roi)
        valid_request = parsed_request._validate_resources(roi)
        qs = []
        for labels, label_object_to in product(
            valid_request.labelss, valid_request.label_objects_to
        ):
            qs.append(
                DistanceQuery(
                    label_image=labels,
                    label_object_to=label_object_to,
                    features=valid_request.props,
                    distance_transforms=valid_request.distance_transforms,
                )
            )
        return qs


DENSITY_RADIUS_NEIGHBORHOODS = tuple([10, 20, 30, 40, 50, 80, 100, 150, 200, 250])

DENSITY_DISTANCE_TO_CLOSEST_NEIGHBOR = True

DENSITY_KNN_DISTANCE_NEIGHBORHOODS = tuple([2, 5, 10, 20, 50, 100, 200])

DENSITY_DELAUNAY_NEIGHBORHOODS = tuple([1])

DENSITY_TOUCH_NEIGHBORHOODS = tuple([1])

DENSITY_ADJACENCY_AGGFUNCS = tuple(["Count"])
DENSITY_DISTANCE_AGGFUNCS = tuple(["Mean", "Max"])


class DensityQuery(BaseModel, FeatureQuery):
    label_image: str
    delaunay_mask_label: str | None = None
    radius: tuple[float, ...] = tuple(DENSITY_RADIUS_NEIGHBORHOODS)
    knn_distance: tuple[int, ...] = tuple(DENSITY_KNN_DISTANCE_NEIGHBORHOODS)
    distance_to_closest_neighbor: bool = DENSITY_DISTANCE_TO_CLOSEST_NEIGHBOR
    delaunay: tuple[int, ...] = tuple(DENSITY_DELAUNAY_NEIGHBORHOODS)
    touch: tuple[int, ...] = tuple(DENSITY_TOUCH_NEIGHBORHOODS)
    adjacency_aggfuncs: tuple[str, ...] = tuple(DENSITY_ADJACENCY_AGGFUNCS)
    distance_aggfuncs: tuple[str, ...] = tuple(DENSITY_DISTANCE_AGGFUNCS)

    def load_resources(self, roi: Roi) -> dict[str, Any]:
        return {
            "label_image": roi.sel(l=self.label_image).drop_dim("c").labels.compute(),
            "delaunay_mask_label": None
            if self.delaunay_mask_label is None
            else roi.sel(l=self.delaunay_mask_label).drop_dim("c").labels.compute(),
        }

    def compute(self, roi: Roi) -> "pl.DataFrame":
        return get_density_features(**self.load_resources(roi))


class DensityQueries(BaseModel):
    labelss: tuple[str, ...]
    delaunay_mask_labels: tuple[str, ...] | None = None

    radius: tuple[float, ...] = tuple(DENSITY_RADIUS_NEIGHBORHOODS)
    knn_distance: tuple[int, ...] = tuple(DENSITY_KNN_DISTANCE_NEIGHBORHOODS)
    distance_to_closest_neighbor: bool = DENSITY_DISTANCE_TO_CLOSEST_NEIGHBOR
    delaunay: tuple[int, ...] = tuple(DENSITY_DELAUNAY_NEIGHBORHOODS)
    touch: tuple[int, ...] = tuple(DENSITY_TOUCH_NEIGHBORHOODS)
    adjacency_aggfuncs: tuple[str, ...] = tuple(DENSITY_ADJACENCY_AGGFUNCS)
    distance_aggfuncs: tuple[str, ...] = tuple(DENSITY_DISTANCE_AGGFUNCS)

    @field_validator("labelss", "delaunay_mask_labels", mode="before")
    def _package_singletons(cls, val):
        return _make_iterable(val)

    def _parse_resource_star_expressions(self, roi: Roi) -> "IntensityQueries":
        return DensityQueries(
            labelss=_parse_star_expression(self.labelss, roi.resources["label_images"]),
            delaunay_mask_labels=_parse_star_expression(
                self.delaunay_mask_labels, roi.resources["label_images"]
            ),
            **{
                k: v
                for k, v in dict(self).items()
                if k not in ["labelss", "delaunay_mask_labels"]
            },
        )

    def _validate_resources(self, roi: Roi) -> Self:
        _validate_resources_in_roi(self.resources, roi)
        return self

    @property
    def resources(self) -> Resources:
        return Resources(label_images=self.labelss + self.delaunay_mask_labels)

    def queries(self, roi: Roi) -> list[DensityQuery]:
        parsed_request = self._parse_resource_star_expressions(roi)
        valid_request = parsed_request._validate_resources(roi)

        params = {
            k: v
            for k, v in dict(self).items()
            if k not in ["labelss", "delaunay_mask_labels"]
        }
        qs = []
        for labels, delaunay_mask_label in product(
            valid_request.labelss, valid_request.delaunay_mask_labels
        ):
            qs.append(
                DensityQuery(
                    label_image=labels,
                    delaunay_mask_label=delaunay_mask_label,
                    **params,
                )
            )
        return qs





class FeatureQueries(BaseModel):
    hierarchy: tuple[HierarchyQuery, ...] = Field(default_factory=tuple)
    label: tuple[LabelQueries, ...] = Field(default_factory=tuple)
    intensity: tuple[IntensityQueries, ...] = Field(default_factory=tuple)
    correlation: tuple[CorrelationQueries, ...] = Field(default_factory=tuple)
    distance: tuple[DistanceQueries, ...] = Field(default_factory=tuple)
    density: tuple[DensityQueries, ...] = Field(default_factory=tuple)

    def parse_star_expressions(self, roi: Roi) -> "FeatureQueries":
        return FeatureQueries(
            hiearchy=self.hierarchy,
            label=(e._parse_resource_star_expressions(roi) for e in self.label),
            intensity=(e._parse_resource_star_expressions(roi) for e in self.intensity),
            correlation=(
                e._parse_resource_star_expressions(roi) for e in self.correlation
            ),
            distance=(e._parse_resource_star_expressions(roi) for e in self.distance),
            density=(e._parse_resource_star_expressions(roi) for e in self.density),
        )

    @property
    def resources(self) -> Resources:
        return Resources().union(
            *[
                e.resources
                for e in chain(
                    self.hierarchy,
                    self.label,
                    self.intensity,
                    self.distance,
                    self.correlation,
                    self.density,
                )
            ]
        )

    def validate_resources(self, roi: Roi) -> Self:
        _validate_resources_in_roi(self.resources, roi)
        return self

    def queries(self, roi: Roi) -> list[FeatureQuery]:
        # TODO: Solve this weird edge case.
        return list(self.hierarchy) + reduce(
            lambda x, y: x + y,
            [
                e.queries(roi)
                for e in chain(
                    self.label,
                    self.intensity,
                    self.distance,
                    self.correlation,
                    self.density,
                )
            ],
        )


class IntensityCorrection(BaseModel):
    t_decay_models: Path | None
    z_decay_models: Path | None
    z_decay_two_step_label: str | None = None
    z_decay_add_model_name_to_feature_path: bool = False
    z_decay_default_models: Path | None = None


# TODO: Fix this
from enum import Enum, auto
from typing import Literal


class IntensityCorrectionV2(BaseModel):
    t: "TIntensityModels" | None = None
    z: "ZIntensityModels" | None = None


class Element(str, Enum):
    channel = "channel"
    wavelength = "wavelength"
    stain = "stain"


class TIntensityModels(BaseModel):
    raise NotImplementedError


class XYIntensityModels(BaseModel):
    raise NotImplementedError


class ZIntensityModels(BaseModel):
    root: Path
    element: Element = "channel"


class SiteFeatureExtractionParams(BaseModel):
    roi_path: Path
    output_path: Path
    level: int
    features: FeatureQueries
    intensity_correction: IntensityCorrection

    def validate_parameters_with(self, roi: Roi) -> "SiteFeatureExtractionParams":
        features = self.features.parse_star_expressions(roi)
        valid_features = features.validate_resources(roi)
        return SiteFeatureExtractionParams(
            **{**dict(self), **{"features": valid_features}}
        )


class FeatureExtractionParams(BaseModel):
    root: Path
    image_dir: str | None
    output_dir: str = "features"
    level: int
    features: FeatureQueries
    intensity_correction: IntensityCorrection
    image_path: Path | None = Field(default=None, validate_default=True)
    output_path: Path | None = Field(default=None, validate_default=True)

    @field_validator("image_path")
    def _resolve_image_path(cls, v, info):
        if v is None and info.data["image_dir"] is None:
            return info.data["root"]
        elif v is None:
            return info.data["root"] / info.data["image_dir"]
        else:
            return v

    @field_validator("output_path")
    def _resolve_output_path(cls, v, info):
        if v is None:
            if info.data["intensity_correction"].z_decay_add_model_name_to_feature_path:
                if info.data["intensity_correction"].z_decay_models is None:
                    name = "NoCorrection"
                else:
                    name = info.data["intensity_correction"].z_decay_models.name
                return info.data["root"] / info.data["output_dir"] / name
            else:
                return info.data["root"] / info.data["output_dir"]
        else:
            return v

    def get_site_params_by_index(self, idx: int) -> SiteFeatureExtractionParams:
        roi_path = list(self.image_path.glob("*.h5"))[idx]
        return SiteFeatureExtractionParams(
            roi_path=roi_path,
            output_path=self.output_path / roi_path.stem,
            level=self.level,
            features=self.features,
            intensity_correction=self.intensity_correction,
        )
