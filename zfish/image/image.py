# %%
import itk
import numpy as np
import xarray as xr
import xxhash
from spatial_image import to_spatial_image

from zfish.features.types import LabelImage, SpatialImage

SPATIAL_DIMS = ("z", "y", "x")


@xr.register_dataarray_accessor("meta")
class ImageMetaAccessor:
    def __init__(self, dataarray: SpatialImage):
        self._obj = dataarray

    @property
    def scale(self):
        return tuple(
            ImageMetaAccessor._scale_from_coord(self._obj.coords[dim])
            for dim in self._obj.dims
            if dim in SPATIAL_DIMS
        )

    @property
    def scale_dict(self):
        return dict(
            zip((dim for dim in self._obj.dims if dim in SPATIAL_DIMS), self.scale)
        )

    @property
    def spacing(self):
        return self.scale

    @property
    def translate(self):
        return tuple(
            ImageMetaAccessor._translate_from_coord(self._obj.coords[dim])
            for dim in self._obj.dims
        )

    @property
    def origin(self):
        return self.translate

    @property
    def direction(self):
        return self._obj.attrs["direction"]

    # TODO: Can this be cashed while ensuring it stays in sync with `self._obj.data`?
    @property
    def hash(self):
        return xxhash.xxh128(self._obj.data.squeeze()).hexdigest()
        # attrs = getattr(self._obj, "attrs")
        # if attrs.get("_hash") is None:
        #     attrs["_hash"] = xxhash.xxh128(self._obj.data.squeeze()).hexdigest()
        # return attrs["_hash"]

    @staticmethod
    def _scale_from_coord(coord: xr.DataArray):
        return coord[1].item() - coord[0].item()

    @staticmethod
    def _translate_from_coord(coord: xr.DataArray):
        return coord[0].item()


# %%
from typing import Callable

import h5py
import xarray as xr
from multiscale_spatial_image import MultiscaleSpatialImage, to_multiscale

from zfish.features.types import LabelImage, SpatialImage
from zfish.io import h5


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


def _load_roi(
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
    return _load_roi(root_path=root_path, attrs_select=attrs_select)


def load_labels(root_path: str, level: int | None = None) -> LabelImage:
    f = h5py.File(root_path)
    attrs_select = {"img_type": "label"}
    if level is None:
        level = sorted(h5.attrs_set(f, "level", attrs_select=attrs_select))[0]
    attrs_select = {**attrs_select, **{"level": level}}
    return _load_roi(root_path=root_path, attrs_select=attrs_select)


def load_channel(root_path: str, h5_path: str) -> SpatialImage:
    f = h5py.File(root_path)
    return to_si(f[h5_path])


def _get_intensity_channel_selectors(f: h5py.File) -> list[dict[str, str | int]]:
    channel_selectors = h5.attrs_set(
        f,
        ("stain", "cycle", "wavelength"),
        attrs_select={"img_type": "intensity"},
        return_dict=True,
    )
    return sorted(
        channel_selectors, key=lambda x: (x["cycle"], x["wavelength"], x["stain"])
    )


# %%

from functools import singledispatch
from numbers import Number
from typing import Any, Sequence, TypeAlias

import dask.array as da
import h5py
import itk
import numpy as np
import xarray as xr

ALL_DIMS = ("t", "c", "z", "y", "x")
SPATIAL_DIMS = ("z", "y", "x")
H5_DIMS = ("c", "z", "y", "x")
H5_LABEL_DIMS = ("l", "z", "y", "x")

from spatial_image import SpatialImage, to_spatial_image

TimeCoord: TypeAlias = int | float


@singledispatch
def to_si(
    img: Any,
) -> SpatialImage:
    if isinstance(img, itk.Image):
        return itk.xarray_from_image(img)
    raise NotImplementedError(f"No implementation for type {type(img)}.")


@singledispatch
def _(multiscale_dsets: Sequence[h5py.Dataset]) -> MultiscaleSpatialImage:
    pass


@to_si.register
def _(dset: h5py.Dataset) -> SpatialImage:
    data = da.expand_dims(da.array(dset), 0)
    scale = dset.attrs["element_size_um"]
    kwargs = {'scale': dict(zip(SPATIAL_DIMS, scale))}
    if dset.attrs["img_type"] == "intensity":
        channel = f"{dset.attrs['stain']}-{dset.attrs['cycle']}"
        name = "image"
        dims = H5_DIMS
        kwargs = {**kwargs, 'dims': dims, 'c_coords': channel, 'name': name}
    elif dset.attrs["img_type"] == "label":
        channel = f"{dset.attrs['stain']}"
        name = "label"
        # dims = H5_LABEL_DIMS
        dims = H5_DIMS
        # kwargs = {**kwargs, 'dims': dims, 'l_coords': channel, 'name': name}
        kwargs = {**kwargs, 'dims': dims, 'c_coords': channel, 'name': name}

    else:
        channel = f"{dset.name}"
        name = "unknown"
    spi = to_spatial_image(
        data, **kwargs
        # dims=dims,
        # scale=dict(zip(SPATIAL_DIMS, scale)),
        # c_coords=channel,
        # name=name,
    )
    for k, v in dset.attrs.items():
        spi.attrs[k] = v
    return spi


@to_si.register(np.ndarray)
def _(
    img: np.ndarray,
    dims: Sequence[str] | None = None,
    scale: Sequence[float] | None = None,
    c_coords: Sequence[str] | None = None,
    t_coords: Sequence[TimeCoord] | None = None,
    name: str | None = None,
) -> SpatialImage:
    spatial_dims = tuple(e for e in dims if e in SPATIAL_DIMS)
    return to_spatial_image(
        img,
        dims=dims,
        scale=dict(zip(spatial_dims, scale)),
        c_coords=c_coords,
        t_coords=t_coords,
        name=name,
    )
