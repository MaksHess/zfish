# %%
from dataclasses import dataclass, field
from itertools import chain, combinations, repeat
from pathlib import Path
from pprint import pformat
from typing import Protocol, TypeAlias

from pydantic import BaseModel, validator
from pydantic.dataclasses import dataclass as pdataclass
from pydantic_yaml import YamlModel
from typing_extensions import Self

from zfish.features.types import LabelImage, SpatialImage
from zfish.image.image import load_channels, load_structures
from zfish.io import h5

# @dataclass
# class Resources:
#     channels: set[str] = field(default_factory=set)
#     labels: set[str] = field(default_factory=set)

#     def union(self, *others: "Resources") -> "Resources":
#         return Resources(
#             channels=self.channels.union(*[other.channels for other in others]),
#             labels=self.labels.union(*[other.labels for other in others]),
#         )


@pdataclass
class Resources:
    channels: set[str] = field(default_factory=set)
    labels: set[str] = field(default_factory=set)

    def union(self, *others: "Resources") -> "Resources":
        return Resources(
            channels=self.channels.union(*[other.channels for other in others]),
            labels=self.labels.union(*[other.labels for other in others]),
        )




class FeatureProtocol(Protocol):
    @property
    def resources(self) -> Resources:
        ...

    def parse_star_expression(self, roi: ROI) -> Self:
        ...


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


def _validate_resources_in_roi(resources: Resources, roi: ROI) -> None:
    for channel in resources.channels:
        assert (
            channel in roi.channels
        ), f"Channel `{channel}` not found in `{roi.channels}`!"
    for label in resources.labels:
        assert label in roi.labels, f"Label `{label}` not found in `{roi.labels}!"


class LabelFeature(BaseModel):
    labels: set[str]

    @property
    def resources(self) -> Resources:
        return Resources(labels=self.labels)

    def parse_star_expression(self, roi: ROI) -> "LabelFeature":
        return LabelFeature(labels=_parse_star_expression(self.labels, roi.labels))


class IntensityFeature(BaseModel):
    labels: set[str]
    channels: set[str]

    @property
    def resources(self) -> Resources:
        return Resources(labels=self.labels, channels=self.channels)

    def parse_star_expression(self, roi: ROI) -> "IntensityFeature":
        return IntensityFeature(
            labels=_parse_star_expression(self.labels, roi.labels),
            channels=_parse_star_expression(self.channels, roi.channels),
        )


class DistanceFeature(BaseModel):
    labels: set[str]
    label_objects: set[tuple[str, int]]

    @property
    def resources(self) -> Resources:
        labels = self.labels.union([e[0] for e in self.label_objects])
        return Resources(labels=labels)

    def parse_star_expression(self, roi: ROI) -> "DistanceFeature":
        return DistanceFeature(
            labels=_parse_star_expression(self.labels, roi.labels),
            label_objects=self.label_objects,
        )


class CorrelationFeature(BaseModel):
    labels: set[str]
    channel_pairs: set[tuple[str, str]]

    @property
    def resources(self) -> Resources:
        return Resources(
            labels=self.labels,
            channels=set(e for e in chain.from_iterable(self.channel_pairs)),
        )

    def parse_star_expression(self, roi: ROI) -> "CorrelationFeature":
        return CorrelationFeature(
            labels=_parse_star_expression(self.labels, roi.labels),
            channel_pairs=_parse_pairwise_star_expression(
                self.channel_pairs, roi.channels
            ),
        )


class Features(BaseModel):
    label: LabelFeature
    intensity: IntensityFeature
    distance: DistanceFeature
    correlation: CorrelationFeature

    def parse_star_expressions(self, roi: ROI) -> "Features":
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

    def validate_resources(self, roi: ROI) -> Self:
        _validate_resources_in_roi(self.resources, roi)


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
        roi = load_roi(self.roi_path)
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
            output_path=self.output_path,
            level=self.level,
            features=self.features,
            intensity_correction=self.intensity_correction,
        )


