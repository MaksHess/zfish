# %%
from dataclasses import dataclass, field
from itertools import chain, combinations, repeat
from pathlib import Path
from typing import Any, Protocol, TypeAlias

from pydantic import BaseModel, validator
from pydantic_yaml import YamlModel
from typing_extensions import Self

from zfish.features.feature_types import Resources
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
ChannelPair: TypeAlias = tuple[str, str]
LabelObject: TypeAlias = tuple[str, int]
Feature: TypeAlias = Any


@dataclass
class ResourcesBase:
    channels: set[Channel] = field(default_factory=set)
    channel_pairs: set[ChannelPair] = field(default_factory=set)
    label_images: set[LabelImage] = field(default_factory=set)
    label_objects: set[LabelObject] = field(default_factory=set)
    tables: set[str] = field(default_factory=set)
    features: set[Feature] = field(default_factory=set)


def _parse_star_expression(
    channels: set[str],
    available_channels: set[str],
) -> set[str]:
    if channels == {"*"}:
        return available_channels
    return channels


def _parse_pairwise_star_expression(
    channel_pairs: set[tuple[str, str]],
    available_channels: set[str],
) -> set[tuple[str, str]]:
    out_pairs = set()
    for c0, c1 in channel_pairs:
        if c0 == "*" and c1 == "*":
            out_pairs.update(
                [tuple(sorted(e)) for e in combinations(available_channels, 2)]
            )
        elif c0 == "*":
            out_pairs.update(
                [tuple(sorted(e)) for e in zip(available_channels, repeat(c1))]
            )
        elif c1 == "*":
            out_pairs.update(
                [tuple(sorted(e)) for e in zip(repeat(c0), available_channels)]
            )
        else:
            out_pairs.add(tuple(sorted((c0, c1))))
    return out_pairs


def _validate_resources_in_roi(resources: Resources, roi: "Roi") -> None:
    for channel in resources.channels:
        assert (
            channel in roi.resources['channels']
        ), f"Channel `{channel}` not found in `{roi.resources['channels']}`!"
    for label in resources.label_images:
        assert label in roi.resources['label_images'], f"Label `{label}` not found in `{roi.resources['label_images']}!"


# @dataclass
# class RequestedResources(Resources):
#     def validate(self, roi: "ROI") -> "ValidatedResources":
#         pass


# @dataclass
# class ValidatedResources(Resources):
#     def union(self, *others: "ValidatedResources") -> "ValidatedResources":
#         return Resources(
#             channels=self.channels.union(*[other.channels for other in others]),
#             labels=self.labels.union(*[other.labels for other in others]),
#             tables=self.tables.union(*[other.labels for other in others]),
#         )


# @dataclass
# class ROI:
#     channel_images: SpatialImage | None = None
#     label_images: LabelImage | None = None
#     _tables: dict[str, Table] | None = None

#     @property
#     def channels(self) -> set[str]:
#         if self.channel_images is None:
#             return set()
#         return _get_channels_safe(self.channel_images)

#     @property
#     def labels(self) -> set[str]:
#         if self.label_images is None:
#             return set()
#         return _get_channels_safe(self.label_images)

#     @property
#     def tables(self) -> set[str]:
#         if self._tables is None:
#             return set()
#         return set(self._tables.keys())

#     @property
#     def resources(self) -> Resources:
#         return Resources(channels=self.channels, labels=self.labels, tables=self.tables)

#     def compute(self) -> "ROI":
#         if self.channel_images is None:
#             channel_images = None
#         else:
#             channel_images = self.channel_images.compute()

#         if self.label_images is None:
#             label_images = None
#         else:
#             label_images = self.label_images.compute()
#         # TODO: Since tables are not yet lazy they can just be passed on.
#         return ROI(
#             channel_images=channel_images,
#             label_images=label_images,
#             _tables=self._tables,
#         )


# # TODO: implement table loader
# def load_roi(root_path: str, level: int | None = None) -> ROI:
#     return ROI(
#         channel_images=load_channels(root_path=root_path, level=level),
#         label_images=load_labels(root_path=root_path, level=level),
#         _tables=None,
#     )


class FeatureProtocol(Protocol):
    @property
    def resources(self) -> Resources:
        ...

    def parse_star_expression(self, roi: Roi) -> Self:
        ...


class LabelFeature(BaseModel):
    labels: set[str]

    @property
    def resources(self) -> Resources:
        return Resources(label_images=self.labels)

    def parse_star_expression(self, roi: Roi) -> "LabelFeature":
        return LabelFeature(labels=_parse_star_expression(self.labels, roi.resources['label_images']))


class IntensityFeature(BaseModel):
    labels: set[str]
    channels: set[str]

    @property
    def resources(self) -> Resources:
        return Resources(label_images=self.labels, channels=self.channels)

    def parse_star_expression(self, roi: Roi) -> "IntensityFeature":
        return IntensityFeature(
            labels=_parse_star_expression(self.labels, roi.resources['label_images']),
            channels=_parse_star_expression(self.channels, roi.resources['channels']),
        )


class DistanceFeature(BaseModel):
    labels: set[str]
    label_objects: set[tuple[str, int]]

    @property
    def resources(self) -> Resources:
        labels = self.labels.union([e[0] for e in self.label_objects])
        return Resources(label_images=labels)

    def parse_star_expression(self, roi: Roi) -> "DistanceFeature":
        return DistanceFeature(
            labels=_parse_star_expression(self.labels, roi.resources['label_images']),
            label_objects=self.label_objects,
        )


class CorrelationFeature(BaseModel):
    labels: set[str]
    channel_pairs: set[tuple[str, str]]

    @property
    def resources(self) -> Resources:
        return Resources(
            label_images=self.labels,
            channels=set(e for e in chain.from_iterable(self.channel_pairs)),
        )

    def parse_star_expression(self, roi: Roi) -> "CorrelationFeature":
        return CorrelationFeature(
            labels=_parse_star_expression(self.labels, roi.resources['label_images']),
            channel_pairs=_parse_pairwise_star_expression(
                self.channel_pairs, roi.resources['channels']
            ),
        )


class Features(BaseModel):
    label: LabelFeature
    intensity: IntensityFeature
    distance: DistanceFeature
    correlation: CorrelationFeature

    def parse_star_expressions(self, roi: Roi) -> "Features":
        return Features(
            label=self.label.parse_star_expression(roi),
            intensity=self.intensity.parse_star_expression(roi),
            distance=self.distance.parse_star_expression(roi),
            correlation=self.correlation.parse_star_expression(roi),
        )

    @property
    def resources(self) -> Resources:
        return Resources().union(
            *[
                e.resources
                for e in (self.label, self.intensity, self.distance, self.correlation)
            ]
        )

    def validate_resources(self, roi: Roi) -> Self:
        _validate_resources_in_roi(self.resources, roi)
        return self


class IntensityCorrectionDirectories(BaseModel):
    time_decay_models: Path
    z_decay_models: Path


class SiteFeatureExtractionParams(YamlModel):
    roi_path: Path
    output_path: Path
    level: int
    features: Features
    intensity_correction: IntensityCorrectionDirectories

    def validate_roi(self) -> "SiteFeatureExtractionParams":
        roi = Roi.from_file(self.roi_path, level=self.level)
        features = self.features.parse_star_expressions(roi)
        features.validate_resources(roi)
        return SiteFeatureExtractionParams(**{**dict(self), **{"features": features}})


class FeatureExtractionParams(YamlModel):
    root: Path
    image_dir: str | None
    output_dir: str = "features"
    image_path: Path | None = None
    output_path: Path | None = None
    level: int
    features: Features
    intensity_correction: IntensityCorrectionDirectories

    @validator("image_path", always=True)
    def _resolve_image_path(cls, v, values):
        if v is None and values["image_dir"] is None:
            return values["root"]
        elif v is None:
            return values["root"] / values["image_dir"]
        else:
            return v

    @validator("output_path", always=True)
    def _resolve_output_path(cls, v, values):
        if v is None:
            return values["root"] / values["output_dir"]
        else:
            return v

    def get_roi_by_index(self, idx: int) -> SiteFeatureExtractionParams:
        roi_path = list(self.image_path.glob("*.h5"))[idx]
        return SiteFeatureExtractionParams(
            roi_path=roi_path,
            output_path=self.output_path / roi_path.stem,
            level=self.level,
            features=self.features,
            intensity_correction=self.intensity_correction,
        )
