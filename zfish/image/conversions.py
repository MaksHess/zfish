from functools import singledispatch
from typing import Any, Sequence, TypeAlias

import dask.array as da
import h5py
import numpy as np
from spatial_image import SpatialImage, to_spatial_image

ALL_DIMS = ("t", "c", "z", "y", "x")
SPATIAL_DIMS = ("z", "y", "x")
H5_DIMS = ("c", "z", "y", "x")
H5_LABEL_DIMS = ("l", "z", "y", "x")


TimeCoord: TypeAlias = int | float


@singledispatch
def to_si(
    img: Any,
) -> SpatialImage:
    import itk
    if isinstance(img, itk.Image):
        return itk.xarray_from_image(img)
    raise NotImplementedError(f"No implementation for type {type(img)}.")


# @singledispatch
# def _(multiscale_dsets: Sequence[h5py.Dataset]) -> MultiscaleSpatialImage:
#     pass


@to_si.register
def _(dset: h5py.Dataset) -> SpatialImage:
    data = da.expand_dims(da.array(dset), 0)
    scale = dset.attrs["element_size_um"]
    kwargs = {'scale': dict(zip(SPATIAL_DIMS, scale))}
    if dset.attrs["img_type"] == "intensity":
        channel = f"{dset.attrs['stain']}.{dset.attrs['cycle']}"
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
