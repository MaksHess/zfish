# %%
from collections.abc import Iterable
from functools import partial, reduce
from itertools import chain, product, repeat
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import pydantic
from typing_extensions import Self

from zfish.features.colocalization import ColocalizationQuery
from zfish.features.constants import (
    DefaultColocalizationFeature,
    DefaultDistanceFeature,
    DefaultDistanceFunction,
    DefaultLabelFeature,
    DensityParams,
    IntensityFeature,
)
from zfish.features.distance import DistanceQuery
from zfish.features.feature_types import (
    Resources,
)
from zfish.features.intensity import IntensityQuery
from zfish.features.label import LabelQuery
from zfish.features.neighborhood.density import DensityQuery
from zfish.features.object_hierarchy import HierarchyQuery
from zfish.features.queries import FeatureQuery
from zfish.features.types import LabelImage, SpatialImage
from zfish.roi.spatial_roi import Roi

# FIXME: Nasty fixes scattered through the code to be backwards compatible with pydantic v1.
# Introduced in commit `pydantic v1 compatability`, remove once napari does not pin < 2 anymore.
if pydantic.version.VERSION < "2":
    from pydantic import BaseModel, Field, validator
    before_validator = partial(validator, pre=True, always=True)
else:
    from pydantic import BaseModel, Field, field_validator
    before_validator = partial(field_validator, mode='before')

if TYPE_CHECKING:
    pass

def _get_channels_safe(img: LabelImage | SpatialImage) -> set[str]:
    """
    Custom function to extract channel info from `SpatialImage`. Using the obvious
    `set(img.c.values)` seemed to work initially but lead to unexpected behaviour when
    trying to serialize using yaml.dump. Must be some weird interaction w/ numpy/xarray
    under the hood, explicitly converting to `str` in a comprehension does the trick.
    """
    return set([str(e) for e in img.c.values])


# Channel: TypeAlias = str
# Labels: TypeAlias = str
# ChannelPair: TypeAlias = tuple[Channel, Channel]
# ChannelSet: TypeAlias = tuple[Channel, ...]
# LabelObject: TypeAlias = tuple[Labels, int]
# Feature: TypeAlias = Any
class LabelObject(BaseModel):
    label_image: str
    label: int

class ChannelPair(BaseModel):
    channel_0: str
    channel_1: str
    
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
    channel_pairs_tpl = [(cp.channel_0, cp.channel_1) for cp in channel_pairs]
    out = _parse_pairwise_star_expression(channel_pairs_tpl, available_channels)
    return tuple(ChannelPair(channel_0=ch0, channel_1=ch1) for ch0, ch1 in out)


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

class LabelQueriesParser(BaseModel):
    labelss: tuple[str, ...] = Field(default="*", validate_default=True)
    props: tuple[str, ...] = Field(default="*", validate_default=True)

    @before_validator("labelss", "props")
    def _package_singletons(cls, val):
        return _make_iterable(val)

    @before_validator("props")
    def _resolve_star_expression(cls, val):
        val_out = _parse_star_expression(
            _make_iterable(val), tuple(DefaultLabelFeature)
        )
        return val_out

    def _parse_resource_star_expressions(self, roi: Roi) -> "LabelQueriesParser":
        return LabelQueriesParser(
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

class IntensityQueriesParser(BaseModel):
    labelss: tuple[str, ...] = Field(default="*", validate_default=True)
    channels: tuple[str, ...] = Field(default="*", validate_default=True)
    props: tuple[str, ...] = Field(default="*", validate_default=True)

    @before_validator("labelss", "channels", "props")
    def _package_singletons(cls, val):
        return _make_iterable(val)

    @before_validator("props")
    def _resolve_star_expression(cls, val):
        val_out = _parse_star_expression(_make_iterable(val), tuple(IntensityFeature))
        return val_out

    def _parse_resource_star_expressions(self, roi: Roi) -> "IntensityQueriesParser":
        return IntensityQueriesParser(
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


class ColocalizationQueriesParser(BaseModel):
    labelss: tuple[str, ...] = Field(default="*", validate_default=True)
    channel_pairs: tuple[ChannelPair, ...] = Field(
        default=(ChannelPair(channel_0="*", channel_1="*"),), validate_default=True
    )
    props: tuple[str, ...] = Field(default="*", validate_default=True)

    @before_validator("labelss", "channel_pairs", "props")
    def _package_singletons(cls, val):
        return _make_iterable(val)

    @before_validator("props")
    def _resolve_star_expression(cls, val):
        val_out = _parse_star_expression(
            _make_iterable(val), tuple(DefaultColocalizationFeature)
        )
        return val_out

    def _parse_resource_star_expressions(self, roi: Roi) -> "ColocalizationQueriesParser":
        return ColocalizationQueriesParser(
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
                [cp.channel_0 for cp in self.channel_pairs]
                + [cp.channel_1 for cp in self.channel_pairs]
            ),
        )

    def queries(self, roi: Roi) -> list[ColocalizationQuery]:
        parsed_request = self._parse_resource_star_expressions(roi)
        valid_request = parsed_request._validate_resources(roi)
        qs = []
        for labels, channel_pair in product(
            valid_request.labelss, valid_request.channel_pairs
        ):
            qs.append(
                ColocalizationQuery
                (
                    label_image=labels,
                    channel_pair=tuple(dict(channel_pair).values()),
                    features=valid_request.props,
                )
            )
        return qs


class DistanceQueriesParser(BaseModel):
    label_objects_to: tuple[LabelObject, ...]
    labelss: tuple[str, ...] = Field(default="*", validate_default=True)
    props: tuple[str, ...] = Field(default="*", validate_default=True)
    distance_transforms: tuple[str, ...] = Field(default="*", validate_default=True)

    @before_validator(
        "labelss", "label_objects_to", "props", "distance_transforms"
    )
    def _package_singletons(cls, val):
        return _make_iterable(val)

    @before_validator("props")
    def _resolve_star_expression(cls, val):
        val_out = _parse_star_expression(_make_iterable(val), tuple(DefaultDistanceFeature))
        return val_out

    @before_validator("distance_transforms")
    def _resolve_star_expression2(cls, val):
        val_out = _parse_star_expression(_make_iterable(val), tuple(DefaultDistanceFunction))
        return val_out

    def _parse_resource_star_expressions(self, roi: Roi) -> "DistanceQueriesParser":
        return DistanceQueriesParser(
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
                    #TODO: check label_object_to type
                    label_object_to=tuple(dict(label_object_to).values()),
                    features=valid_request.props,
                    distance_transforms=valid_request.distance_transforms,
                )
            )
        return qs



class DensityQueriesParser(BaseModel):
    labelss: tuple[str, ...]
    delaunay_mask_labels: tuple[str, ...] | None = None
    params: DensityParams = DensityParams()

    @before_validator("labelss", "delaunay_mask_labels")
    def _package_singletons(cls, val):
        return _make_iterable(val)

    def _parse_resource_star_expressions(self, roi: Roi) -> "DensityQueriesParser":
        return DensityQueriesParser(
            labelss=_parse_star_expression(self.labelss, roi.resources["label_images"]),
            delaunay_mask_labels=_parse_star_expression(
                self.delaunay_mask_labels, roi.resources["label_images"]
            ),
            params=self.params
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
        qs = []
        for labels, delaunay_mask_label in product(
            valid_request.labelss, valid_request.delaunay_mask_labels
        ):
            qs.append(
                DensityQuery(
                    label_image=labels,
                    delaunay_mask_label=delaunay_mask_label,
                    params=self.params,
                )
            )
        return qs
    

class FeatureQueriesParser(BaseModel):
    hierarchy: tuple[HierarchyQuery, ...] = Field(default_factory=tuple)
    label: tuple[LabelQueriesParser, ...] = Field(default_factory=tuple)
    intensity: tuple[IntensityQueriesParser, ...] = Field(default_factory=tuple)
    correlation: tuple[ColocalizationQueriesParser, ...] = Field(default_factory=tuple)
    distance: tuple[DistanceQueriesParser, ...] = Field(default_factory=tuple)
    density: tuple[DensityQueriesParser, ...] = Field(default_factory=tuple)

    def parse_star_expressions(self, roi: Roi) -> "FeatureQueriesParser":
        return FeatureQueriesParser(
            hierarchy=self.hierarchy,
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
            []
        )


class IntensityCorrection(BaseModel):
    t_decay_models: Path | None = None
    t_decay_correction_factor_column: str = 'correctionFactor'
    z_decay_models: Path | None = None
    z_decay_add_model_name_to_feature_path: bool = True
    z_decay_two_step_label: str | None = None
    z_decay_default_models: Path | None = None


# TODO: Fix this
# from enum import Enum, auto
# from typing import Literal


# class IntensityCorrectionV2(BaseModel):
#     t: "TIntensityModels" | None = None
#     z: "ZIntensityModels" | None = None


# class Element(str, Enum):
#     channel = "channel"
#     wavelength = "wavelength"
#     stain = "stain"


# class TIntensityModels(BaseModel):
#     raise NotImplementedError


# class XYIntensityModels(BaseModel):
#     raise NotImplementedError


# class ZIntensityModels(BaseModel):
#     root: Path
#     element: Element = "channel"


class SiteFeatureExtractionParams(BaseModel):
    roi_path: Path
    output_path: Path
    level: int
    features: FeatureQueriesParser
    intensity_correction: IntensityCorrection

    def validate_parameters_with(self, roi: Roi) -> "SiteFeatureExtractionParams":
        features = self.features.parse_star_expressions(roi)
        valid_features = features.validate_resources(roi)
        return SiteFeatureExtractionParams(
            **{**dict(self), **{"features": valid_features}}
        )


if pydantic.version.VERSION < "2":
    class FeatureExtractionParams(BaseModel):
        root: Path
        image_dir: str | None
        output_dir: str = "features"
        level: int
        features: FeatureQueriesParser
        intensity_correction: IntensityCorrection
        image_path: Path | None = Field(default=None, validate_default=True)
        output_path: Path | None = Field(default=None, validate_default=True)

        @validator("image_path", always=True)
        def _resolve_image_path(cls, v, values):
            data = values
            if v is None and data["image_dir"] is None:
                return data["root"]
            elif v is None:
                return data["root"] / data["image_dir"]
            else:
                return v

        @validator("output_path", always=True)
        def _resolve_output_path(cls, v, values):
            data = values
            if v is None:
                if data["intensity_correction"].z_decay_add_model_name_to_feature_path:
                    if data["intensity_correction"].z_decay_models is None:
                        name = "NoCorrection"
                    else:
                        name = data["intensity_correction"].z_decay_models.name
                    return data["root"] / data["output_dir"] / name
                else:
                    return data["root"] / data["output_dir"]
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

else:
    class FeatureExtractionParams(BaseModel):
        root: Path
        image_dir: str | None
        output_dir: str = "features"
        level: int
        features: FeatureQueriesParser
        intensity_correction: IntensityCorrection
        image_path: Path | None = Field(default=None, validate_default=True)
        output_path: Path | None = Field(default=None, validate_default=True)

        @field_validator("image_path")
        def _resolve_image_path(cls, v, info):
            # FIXME: See above.
            data = info.data
            if v is None and data["image_dir"] is None:
                return data["root"]
            elif v is None:
                return data["root"] / data["image_dir"]
            else:
                return v

        @field_validator("output_path")
        def _resolve_output_path(cls, v, info):
            # FIXME: See above.
            data = info.data
            if v is None:
                if data["intensity_correction"].z_decay_add_model_name_to_feature_path:
                    if data["intensity_correction"].z_decay_models is None:
                        name = "NoCorrection"
                    else:
                        name = data["intensity_correction"].z_decay_models.name
                    return data["root"] / data["output_dir"] / name
                else:
                    return data["root"] / data["output_dir"]
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