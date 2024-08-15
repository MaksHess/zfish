import warnings

import h5py
import itk
import numpy as np
import SimpleITK as sitk
import xarray as xr

# if TYPE_CHECKING:
from itk.support.types import ImageBase as ITKImage
from spatial_image import to_spatial_image

from zfish.features.types import SpatialImage

DTYPE_CONVERSION = {
    np.dtype("uint64"): np.dtype("uint16"),
    np.dtype("uint32"): np.dtype("uint16"),
    np.dtype("uint16"): np.dtype("uint16"),
    np.dtype("uint8"): np.dtype("uint8"),
    np.dtype("int64"): np.dtype("uint16"),
    np.dtype("int32"): np.dtype("uint16"),
    np.dtype("int16"): np.dtype("int16"),
    np.dtype("float64"): np.dtype("float64"),
    np.dtype("float32"): np.dtype("float32"),
    np.dtype("float16"): np.dtype("float16"),
    np.dtype("bool"): np.dtype("uint8"),
}


def to_si(
    img, scale: tuple[float, ...] | None = None, conversion_warning: bool = True
) -> xr.DataArray:
    """Convert to `SpatialImage` (xr.DataArray containing metadata to create an
    itk.Image). Converts to itk.Image internally.

    Args:
        img: Image to convert.
        scale: Image scale in numpy (!) conventions ([z], y, x).
        conversion_warning: Warning when data types are converted. Defaults to True.

    Returns:
        SpatialImage (xr.DataArray)
    """
    return itk.xarray_from_image(
        to_itk(img, scale=scale, conversion_warning=conversion_warning)
    )


def to_sitk(image, scale=None, conversion_warning=True) -> sitk.Image:
    if isinstance(image, itk.Image):
        data = itk.array_view_from_image(image)
        meta = dict(image)
    elif isinstance(image, xr.DataArray):
        data = image.data
        try:
            direction = image.meta.direction
        except KeyError:
            direction = np.eye(data.ndim)
        meta = {
            "origin": image.meta.origin,
            "spacing": image.meta.spacing,
            "direction": direction,
        }
    elif isinstance(image, np.ndarray):
        if scale is None:
            raise ValueError(
                "Can't convert to sitk.Image from numpy.ndarray without providing `scale`!"
            )
        data = image
        meta = {
            "origin": np.array((0,) * data.ndim),
            "spacing": scale,
            "direction": np.eye(data.ndim),
        }
    else:
        raise NotImplementedError(f"Can't converto to sitk.Image from {type(image)}")
    sitk_image = sitk.GetImageFromArray(data)
    sitk_image.SetOrigin(meta["origin"][::-1])
    sitk_image.SetSpacing(meta["spacing"][::-1])
    sitk_image.SetDirection(meta["direction"][::-1][:, ::-1].flatten())
    return sitk_image


def sitk_to_si(image: sitk.Image) -> SpatialImage:
    DIMS = ("z", "y", "x")
    origin = image.GetOrigin()[::-1]
    scale = image.GetSpacing()[::-1]
    # data = sitk.GetArrayViewFromImage(image)
    data = sitk.GetArrayFromImage(image)
    dims = DIMS[-data.ndim :]
    return to_spatial_image(
        data,
        dims=dims,
        scale=dict(zip(dims, scale)),
        translation=dict(zip(dims, origin)),
    )
    
def to_itk(
    img,  #: np.ndarray | ITKImage | h5py.Dataset | xr.DataArray,
    scale: tuple[float, ...] | None = None,
    conversion_warning: bool = True,
) -> ITKImage:
    """Convert something image-like to `itk.Image`.

    Args:
        img: Image to convert.
        scale: Image scale in numpy (!) conventions ([z], y, x).
        conversion_warning: Warning when data types are converted. Defaults to True.

    Raises:
        ValueError: No `scale` provided with np.ndarray.
        TypeError: Unknown image type.

    Returns:
        ITK image.
    """
    if isinstance(img, np.ndarray):
        if scale is None:
            raise ValueError(
                """You need to explicitly specify an image `scale` when converting from
                numpy.ndarray to itk.Image!"""
            )
        new_dtype = DTYPE_CONVERSION[img.dtype]
        if conversion_warning and img.dtype != new_dtype:
            warnings.warn(f"Converting {img.dtype} to {new_dtype}", stacklevel=2)
        img = img.astype(new_dtype)
        trans_img = itk.GetImageFromArray(img)
        trans_img.SetSpacing(scale[::-1])
    elif isinstance(img, itk.Image):
        trans_img = img
        if scale is None:
            scale = tuple(img.GetSpacing())[::1]
        trans_img.SetSpacing(scale[::-1])
    elif isinstance(img, h5py.Dataset):
        img_dset = img
        img = to_numpy(img_dset)
        new_dtype = DTYPE_CONVERSION[img.dtype]
        if conversion_warning:
            warnings.warn(f"Converting {img.dtype} to {new_dtype}", stacklevel=2)

        trans_img = itk.GetImageFromArray(img.astype(new_dtype))
        if scale is None:
            scale = tuple(
                img_dset.attrs.get("element_size_um", img_dset.ndim * [1.0]).astype(
                    np.float64
                )
            )
        trans_img.SetSpacing(scale[::-1])
    elif isinstance(img, xr.DataArray):
        return itk.image_from_xarray(img)
    else:
        raise TypeError(f"Unknown image type: {type(img)}")
    return trans_img


def to_numpy(img) -> np.ndarray:
    """Convert to numpy.

    Args:
        img: Image.

    Raises:
        ValueError: Unknown image type.

    Returns:
        Numpy array.
    """
    if isinstance(img, (itk.Image, itk.VectorImage)):
        trans_img = itk.GetArrayFromImage(img)
    elif isinstance(img, np.ndarray):
        trans_img = img
    elif isinstance(img, h5py.Dataset):
        trans_img = img[...]
    elif isinstance(img, xr.DataArray):
        trans_img = img.to_numpy()
    else:
        raise ValueError(f"Unknown image type: {type(img)}")
    return trans_img
