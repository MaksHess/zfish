# %%
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator, Literal, Sequence

import dask.array as da
import itk
import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from zfish.commons.types_ import RasterMeta

# from zfish.abbott_legacy.conversions import to_itk
from zfish.features.types import LabelImage, SpatialImage
from zfish.image.conversions import to_si

if TYPE_CHECKING:
    from zfish.features.types import LabelImage, SpatialImage

logger = logging.getLogger(__name__)

if itk.__version__ >= "5.4.0":
    # TODO: remove this when ITK 5.4.0 is compatible
    logger.warning(
        f"Some of the functions in {__name__!r} don't work in itk 5.4.0, if you get weird results consider donwgrading to 5.3.0"
    )

ObjectTypeDim = Literal["l", "o", "c"]
ITK_ORDER = ("x", "y", "z")


def lazy_resample_dask_label(
    image: da.array,
    meta: RasterMeta,
    scale_factors: Sequence[float],
    target_level: int | None = None,
    translate_origin: bool = False,
    return_itk_view: bool = True,
):
    out_image = _lazy_resample_dask_label(
        image, scale_factors=scale_factors, return_itk_view=return_itk_view
    )

    # Metadata Processing
    out_scale = tuple(
        e / scale_factor for e, scale_factor in zip(meta.scale, scale_factors)
    )
    if translate_origin:
        out_origin = [
            meta.origin[d] + 0.5 * (out_scale[d] - meta.scale[d])
            for d in range(len(meta.origin))
        ]
    else:
        out_origin = meta.origin
    out_path = None
    out_meta = meta.from_template(
        path=out_path, scale=out_scale, origin=out_origin, level=target_level
    )

    return out_image, out_meta


def _lazy_resample_dask_label(
    image: da.array,
    scale_factors: Sequence[float],
    return_itk_view: bool = True,
):
    in_chunksize = image.chunksize
    out_chunksize = tuple(
        int(in_chsz * scale_factor)
        for in_chsz, scale_factor in zip(in_chunksize, scale_factors)
    )
    out_shape = tuple(
        int(in_sz * scale_factor)
        for in_sz, scale_factor in zip(image.shape, scale_factors)
    )
    # output is a lazy array of the wrong shape, partial chunks not recognized
    # (after calling compute it's correct).
    out_slice = tuple(slice(0, sz) for sz in out_shape)
    out_image = image.map_blocks(
        resample_dask_label,
        scale_factors,
        return_itk_view,
        chunks=out_chunksize,
        dtype=np.uint16,
    )[out_slice]

    return out_image


def resample_dask_label(
    image: ArrayLike, scale_factors: Sequence[float], return_itk_view: bool = True
):
    import itk

    # Compute output shape and spacing (reverse for itk axis convention)
    output_size = tuple(
        int(sz * scale_factor) for sz, scale_factor in zip(image.shape, scale_factors)
    )[::-1]
    output_spacing = tuple(1 / e for e in scale_factors)[::-1]

    itk_image = itk.image_view_from_array(image)
    # itk_image = itk.image_from_array(image)

    interpolator = itk.NearestNeighborInterpolateImageFunction.New(itk_image)

    resample_filter = itk.ResampleImageFilter.New(itk_image)
    resample_filter.SetSize(output_size)
    resample_filter.SetOutputSpacing(output_spacing)
    resample_filter.SetInterpolator(interpolator)

    out_image_itk = resample_filter.GetOutput()
    if return_itk_view:
        return itk.array_view_from_image(out_image_itk)
    return itk.array_from_image(out_image_itk)


def resample_label(
    image: LabelImage,
    scale_factors: float | Sequence[float] | dict[str, float],  # z, y, x
    method: Literal["gaussian", "nearest"] = "nearest",
    interp_sigma_frac: float = 0.7355,  # only used for method=='gaussian'
    translate_origin: bool = True,
    return_itk: bool = False,
) -> LabelImage:
    """ "Adapted from `multiscale_spatial_image.to_multiscale`. 'magic-value' 0.7355 used there."""

    input_image = itk.image_from_xarray(image)
    input_spacing = itk.spacing(input_image)
    input_size = itk.size(input_image)
    input_origin = itk.origin(input_image)

    if isinstance(scale_factors, dict):
        scale_array = tuple(scale_factors[e] for e in ITK_ORDER if e in scale_factors)
    elif isinstance(scale_factors, Sequence):
        scale_array = tuple(reversed(scale_factors))  # flip ([z], y, x) to (x, y, [z])
    else:
        scale_array = tuple(scale_factors for _ in range(len(image.dims)))

    output_spacing = np.array(input_spacing) * np.array(scale_array)
    output_size = [
        int(sz * (input_spacing[i] / output_spacing[i]))
        for i, sz in enumerate(input_size)
    ]
    if translate_origin:
        output_origin = [
            input_origin[d] + 0.5 * (output_spacing[d] - input_spacing[d])
            for d in range(len(input_origin))
        ]
    else:
        output_origin = input_origin

    # Set up interpolator
    if method == "gaussian":
        interp_sigma = [s * interp_sigma_frac for s in output_spacing]
        interpolator = itk.LabelImageGaussianInterpolateImageFunction.New(input_image)
        interpolator.SetSigma(interp_sigma)
        interpolator.SetAlpha(max(interp_sigma) * 2.5)

    elif method == "nearest":
        interpolator = itk.NearestNeighborInterpolateImageFunction.New(input_image)
    else:
        raise TypeError(
            f"Unknown method: {method!r}. Must be either 'gaussian' or 'nearest'."
        )

    # Set up resampler
    resample_filter = itk.ResampleImageFilter.New(input_image)
    resample_filter.SetOutputSpacing(output_spacing)
    resample_filter.SetSize(output_size)
    resample_filter.SetOutputOrigin(output_origin)
    resample_filter.SetInterpolator(interpolator)

    out_image_itk = resample_filter.GetOutput()
    if return_itk:
        return out_image_itk
    # convert to xarray and copy over scalar dimensions
    out_image = to_si(out_image_itk)
    out_image.name = image.name
    return out_image.assign_coords({**dict(image.coords), **dict(out_image.coords)})


def resample_label_to_size(
    label_image: LabelImage,
    target_image: SpatialImage,
    method: Literal["gaussian", "nearest"] = "nearest",
    translate_origin: bool = False,
    object_type_coord: ObjectTypeDim = "l",
) -> LabelImage:
    if "c" in target_image.dims:  # scale only relevant for SpatialDim's
        target_scale = target_image.sel(c=target_image.coords["c"][0].item()).meta.scale
    else:
        target_scale = target_image.meta.scale

    if object_type_coord not in label_image.dims:
        label_scale = label_image.meta.scale
        if all(ls == ts for ls, ts in zip(label_scale, target_scale)):
            return label_image
        scale_factors = tuple(ts / ls for ls, ts in zip(label_scale, target_scale))
        return resample_label(
            label_image,
            scale_factors=scale_factors,
            method=method,
            translate_origin=translate_origin,
        )
    else:
        resampled_label_images = []
        for label_image_one in label_image:
            label_scale = label_image_one.meta.scale
            if all(ls == ts for ls, ts in zip(label_scale, target_scale)):
                resampled_label_images.append(label_image_one)
            scale_factors = tuple(ts / ls for ls, ts in zip(label_scale, target_scale))
            resampled_label_images.append(
                resample_label(
                    label_image_one,
                    scale_factors=scale_factors,
                    method=method,
                    translate_origin=translate_origin,
                )
            )
        return xr.concat(resampled_label_images, dim="l")


def __label_pyramid(
    image: LabelImage, n: int = 4, method: Literal["gaussian", "nearest"] = "gaussian"
):
    spacing = np.array(image.meta.scale)

    sf = tuple([max(int(e), 1) for e in (2**n / (spacing / spacing[1]))])
    shrink_factors = []
    for i in range(n):
        print(sf)
        shrink_factors.append(sf)
        sf = (max(sf[0] // 2, 1), max(sf[1] // 2, 1), max(sf[2] // 2, 1))
    return [
        resample_label(image, scale_factors=sf, method=method)
        for sf in reversed(shrink_factors)
    ]


def __image_pyramid(image: SpatialImage, n: int = 4, start_shrink_factors=None):
    img = itk.image_from_xarray(image)
    spacing = np.array(img.GetSpacing())
    if start_shrink_factors is None:
        start_shrink_factors = tuple(
            [max(int(e), 1) for e in (2**n / (spacing / spacing[1]))]
        )
    filt = itk.MultiResolutionPyramidImageFilter[type(img), type(img)].New()
    filt.SetInput(img)
    filt.SetNumberOfLevels(n)
    sf = start_shrink_factors
    schedule = filt.GetSchedule()
    for i in range(n):
        print(sf)
        for j, e in enumerate(sf):
            schedule.SetElement(i, j, e)
        sf = (max(sf[0] // 2, 1), max(sf[1] // 2, 1), max(sf[2] // 2, 1))
    filt.Update()
    return map(to_si, [filt.GetOutput(i) for i in reversed(range(n))])


def _recursive_shrink_factors(scale, shrink_factor=2):
    current_scale = np.array(scale)
    perfect_factors = (
        current_scale.max() / current_scale
    )  # 'perfect' factors to achieve isotropy
    allowed_factors = np.expand_dims(
        [shrink_factor, 1], 1
    )  #  as column vector for broadcasting
    best_factor_index = np.argmin(np.abs(perfect_factors - allowed_factors), axis=0)
    if np.all(
        best_factor_index == 1
    ):  # reached isotropy, downscale by shrink_factor from now on
        shrink_factors = (shrink_factor, shrink_factor, shrink_factor)
    else:
        shrink_factors = tuple(allowed_factors[best_factor_index].squeeze())
    return shrink_factors


def _pyramid_schedule(scale, n: int, shrink_factor=2):
    schedule = []
    recursive_shrink_factors = _all_recursive_shrink_factors(
        scale, n=n, shrink_factor=shrink_factor
    )
    schedule_factor = recursive_shrink_factors[0]
    schedule.append(schedule_factor)
    for rsfs in recursive_shrink_factors[1:]:
        schedule_factor = tuple([sf * rsf for sf, rsf in zip(schedule_factor, rsfs)])
        schedule.append(schedule_factor)
    return schedule


def _all_recursive_shrink_factors(scale, n: int, shrink_factor=2):
    current_scale = scale
    all_shrink_factors = []
    for _ in range(n - 1):
        shrink_factors = _recursive_shrink_factors(
            current_scale, shrink_factor=shrink_factor
        )
        all_shrink_factors.append(shrink_factors)
        current_scale = tuple(s * f for s, f in zip(current_scale, shrink_factors))
    return all_shrink_factors


def image_pyramid(
    image: SpatialImage, n: int = 4, shrink_factor: int = 2, start_level: int = 0
):
    img = itk.image_from_xarray(image)
    spacing = np.array(img.GetSpacing())
    schedule = _pyramid_schedule(spacing, n=n, shrink_factor=shrink_factor)

    filt = itk.MultiResolutionPyramidImageFilter[type(img), type(img)].New()
    filt.SetInput(img)
    filt.SetNumberOfLevels(n - 1)
    filter_schedule = filt.GetSchedule()
    for i, level_factors in enumerate(schedule):
        print(level_factors)
        for j, e in enumerate(level_factors):
            filter_schedule.SetElement(i, j, int(e))
    yield (start_level, image)
    filt.Update()
    for i in range(n - 1):
        out_image = to_si(filt.GetOutput(i))
        out_image.name = image.name
        yield (
            1 + i + start_level,
            out_image.assign_coords({**dict(image.coords), **dict(out_image.coords)}),
        )


def recursive_label_pyramid(
    image: LabelImage,
    n: int = 4,
    shrink_factor: int = 2,
    start_level: int = 0,
    method: Literal["gaussian", "nearest"] = "gaussian",
    interp_sigma_frac: float = 0.7355,
    translate_origin: bool = True,
) -> Generator[tuple[int, LabelImage], None, None]:
    current_image = image
    yield (start_level, current_image)  # level 0 image is not downsampled
    for i in range(start_level + 1, start_level + n):
        sfs = _recursive_shrink_factors(
            current_image.meta.scale, shrink_factor=shrink_factor
        )
        current_image = resample_label(
            current_image,
            scale_factors=sfs,
            method=method,
            interp_sigma_frac=interp_sigma_frac,
            translate_origin=translate_origin,
        )
        yield (i, current_image)


def recursive_label_pyramid_from_level(
    image: LabelImage,
    level_0_scale: tuple[float, ...] | None = None,
    current_level: int = 0,
    n: int = 4,
    shrink_factor: int = 2,
    method: Literal["gaussian", "nearest"] = "gaussian",
    interp_sigma_frac: float = 0.7355,
    translate_origin=True,
) -> Generator[tuple[int, LabelImage], None, None]:
    """Bug encountered with ITK v 5.4.0 (squired/None images resulting from filters
    using Pyramid/resampling). If persists downgrade to 5.3.0"""
    if current_level == 0:  # regular downsampling from highest resolution
        yield from recursive_label_pyramid(
            image=image,
            n=n,
            scale_factors=shrink_factor,
            method=method,
            interp_sigma_frac=interp_sigma_frac,
            translate_origin=translate_origin,
        )
    else:  # up or downsample to get the final pyramid
        levels = list(range(n))
        if current_level not in levels:
            raise ValueError(
                f"current_level ({current_level}) must be one of {levels:!r}"
            )
        if level_0_scale is None:
            raise ValueError(
                "Passing `level_0_scale=None` is only valid for `current_level=0`."
            )

        all_shrink_factors = _all_recursive_shrink_factors(
            level_0_scale, n=n, shrink_factor=shrink_factor
        )
        # upsampling (starting at current_level-1)
        current_image = image.copy()
        for up_level in range(current_level - 1, -1, -1):
            shrink_factors = all_shrink_factors[up_level]
            upscale_factors = tuple(1 / f for f in shrink_factors)
            current_image = resample_label(
                current_image,
                scale_factors=upscale_factors,
                method=method,
                interp_sigma_frac=interp_sigma_frac,
                translate_origin=translate_origin,
            )
            yield (up_level, current_image)

        # downsampling (including current_level)
        current_image = image.copy()
        yield from recursive_label_pyramid(
            current_image,
            n=n - current_level,
            start_level=current_level,
            method=method,
            interp_sigma_frac=interp_sigma_frac,
            translate_origin=translate_origin,
        )
