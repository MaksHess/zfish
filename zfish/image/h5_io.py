from typing import Callable

import h5py
from multiscale_spatial_image import MultiscaleSpatialImage

from zfish.features.types import LabelImage, SpatialImage
from zfish.image.conversions import to_si
from zfish.image.meta_accessor import ImageMetaAccessor  # noqa: F401
from zfish.io import h5

# from multiscale_spatial_image import to_multiscale

def load_multiscale_channels(
    root_path: str,
) -> MultiscaleSpatialImage:
    return _load_multiscale(root_path=root_path, channel_loader=load_channels)


def load_multiscale_labels(
    root_path: str,
) -> MultiscaleSpatialImage:
    return _load_multiscale(root_path=root_path, channel_loader=load_labels)


def _load_multiscale(
    root_path: str,
    attrs_select: dict[str, str | int | tuple[str | int, ...]] | None = None,
    channel_loader: Callable = None,
) -> MultiscaleSpatialImage:
    with h5py.File(root_path) as f:
        pyramid_levels = sorted(h5.attrs_set(f, "level"))
    out_dict = {}
    for level in pyramid_levels:
        out_dict[f"scale{level}"] = load_channels(root_path, level=level)
    return MultiscaleSpatialImage.from_dict(out_dict)


def _load_arrays(
    root_path: str,
    attrs_select: dict[str, str | int | tuple[str | int, ...]] | None = None,
) -> SpatialImage:
    if attrs_select is None:
        attrs_select = {"img_type": "intensity", "level": 0}
    f = h5py.File(root_path)
    dsets = [to_si(dset) for dset in h5.select(f, attrs_select)]
    return dsets
    # return xr.concat(dsets, dim="c", combine_attrs="drop")


def load_channels(root_path: str, level: int | None = None) -> SpatialImage:
    f = h5py.File(root_path)
    attrs_select = {"img_type": "intensity"}
    if level is None:
        level = sorted(h5.attrs_set(f, "level", attrs_select=attrs_select))[0]
    attrs_select = {**attrs_select, **{"level": level}}
    return _load_arrays(root_path=root_path, attrs_select=attrs_select)


def load_labels(root_path: str, level: int | None = None) -> LabelImage:
    f = h5py.File(root_path)
    attrs_select = {"img_type": "label"}
    if level is None:
        level = sorted(h5.attrs_set(f, "level", attrs_select=attrs_select))[0]
    attrs_select = {**attrs_select, **{"level": level}}
    return _load_arrays(root_path=root_path, attrs_select=attrs_select)


def load_channel(root_path: str, h5_path: str) -> SpatialImage:
    f = h5py.File(root_path)
    return to_si(f[h5_path])