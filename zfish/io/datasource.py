# %%
from itertools import accumulate, repeat
from typing import Any, Callable, Literal, Sequence, TypeAlias

import dask.array as da
import itk
import numpy as np
import xarray as xr
from itk.support.types import ImageBase, ImageLike
from numpy.typing import NDArray
from scipy.ndimage import zoom
from skimage.draw import ellipsoid
from skimage.transform import rotate
from skimage.util import view_as_blocks
from spatial_image import SpatialImage, to_spatial_image
from toolz.itertoolz import take

from zfish.features.types import LabelImage, SpatialImage
from zfish.image.h5_io import to_si
from zfish.roi.spatial_roi import Roi

DEFAULT_RNG = np.random.default_rng(42)


UInt: TypeAlias = np.uint8 | np.uint16 | np.uint32 | np.uint64
Int: TypeAlias = np.int8 | np.int16 | np.int32 | np.int64
AllInt: TypeAlias = UInt | Int
AllFloat: TypeAlias = np.float32 | np.float64
Number: TypeAlias = AllInt | AllFloat


def _hierarchical_labels_np_2d(
    shape=(300, 300),
    dtype=np.uint16,
    objects: tuple[str, ...] = ("emb", "cell", "nuc", "cyto", "loc"),
) -> NDArray[Any]:
    lbl = np.zeros(shape, dtype=dtype)
    lbls = []
    if "emb" in objects:
        lbls.append(_gridded_ellipses(lbl, 1, 1, axes_range=(0.8, 1.0)))
    if "cell" in objects:
        lbls.append(_gridded_ellipses(lbl, 4, 4, axes_range=(0.3, 1.0)))
    if "nuc" in objects:
        lbls.append(_gridded_ellipses(lbl, 8, 8))
    if "cyto" in objects and "nuc" in objects and "cell" in objects:
        lbls.append(np.where(lbls[-1] == 0, lbls[-2], 0))
    if "loc" in objects:
        lbls.append(_gridded_ellipses(lbl, 24, 24))
    lbl_stack = np.stack(lbls)
    return lbl_stack


def _hierarchical_labels_np_3d(
    shape=(144, 288, 288),
    dtype=np.uint16,
    objects: tuple[str, ...] = ("emb", "cell", "nuc", "cyto", "loc"),
) -> NDArray[Any]:
    lbls = []
    if "emb" in objects:
        lbls.append(_gridded_ellipsoids(shape, (1, 1, 1), axes_range=(0.8, 1.0)))
    if "cell" in objects:
        lbls.append(_gridded_ellipsoids(shape, (1, 4, 4), axes_range=(0.3, 1.0)))
    if "nuc" in objects:
        lbls.append(_gridded_ellipsoids(shape, (2, 8, 8), axes_range=(0.3, 1.0)))
    if "cyto" in objects and "nuc" in objects and "cell" in objects:
        lbls.append(np.where(lbls[-1] == 0, lbls[-2], 0))
    if "loc" in objects:
        lbls.append(_gridded_ellipsoids(shape, (12, 24, 24), axes_range=(0.2, 1.0)))
    lbl_stack = np.stack(lbls)
    return lbl_stack


def hierarchical_labels(
    shape=(144, 288, 288),
    dtype=np.uint16,
    scale=None,
    objects: tuple[str, ...] = ("emb", "cell", "nuc", "cyto", "loc"),
) -> LabelImage:
    if scale is None:
        scale = (1.0,) * len(shape)
    if not "cell" in objects or not "nuc" in objects:
        objects = tuple(e for e in objects if e != "cyto")
    if len(shape) == 2:
        lbl_stack = _hierarchical_labels_np_2d(
            shape=shape, dtype=dtype, objects=objects
        )
        lbls = to_si(
            lbl_stack,
            dims=("c", "y", "x"),
            scale=scale,
            c_coords=objects,
            name="label",
        )
    elif len(shape) == 3:
        lbl_volume = _hierarchical_labels_np_3d(
            shape=shape, dtype=dtype, objects=objects
        )
        lbls = to_si(
            lbl_volume,
            dims=("c", "z", "y", "x"),
            scale=scale,
            c_coords=objects,
            name="label",
        )
    else:
        raise ValueError(f"Shape can only be 2-D or 3-D, not: {len(shape)}-D\n{shape=}")
    return lbls


rng = np.random.default_rng(seed=42)


def _draw_ellipse(lbl, ellipse):
    import cv2

    return cv2.ellipse(lbl, **ellipse)


def _random_ellipses(lbl, n=10, axes_range=(5, 20)):
    out = np.zeros_like(lbl)
    fg = np.where(lbl > 0)
    idxs = rng.integers(0, len(fg[0]), n)
    for i, center in enumerate(np.array([fg[0][idxs], fg[1][idxs]]).T):
        axes = rng.integers(axes_range[0], axes_range[1], 2)
        ellipse = {
            "center": center,
            "axes": axes,
            "color": i + 1,
            "thickness": -1,
            "angle": rng.uniform(0, 360),
            "startAngle": 0,
            "endAngle": 360,
        }
        _draw_ellipse(out, ellipse)
    return out


def _center_ellips(ellips, shape, label, dtype=np.uint16):
    centered_ellips = np.zeros(shape, dtype=dtype)
    z_orig, y_orig, x_orig = (np.array(shape) - np.array(ellips.shape)) // 2
    dz, dy, dx = ellips.shape
    bbx_view = centered_ellips[
        z_orig : (z_orig + dz), y_orig : (y_orig + dy), x_orig : x_orig + dx
    ]
    bbx_view[np.where(ellips)] = label
    return centered_ellips


def _gridded_ellipsoids(
    shape: tuple[int, int, int] = (64, 256, 256),
    n_zyx: tuple[int, int, int] = (2, 4, 4),
    axes_range: tuple[float, float] = (0.7, 1.0),
):
    assert all(
        i % n == 0 for i, n in zip(shape, n_zyx)
    ), f"{shape=} not evenly divisible by {n_zyx=}"
    out = np.zeros(shape, dtype=np.uint16)
    block_shape = tuple(np.array(shape) // np.array(n_zyx))
    ellipsoid_max_radii = tuple(e // 2 - 2 for e in block_shape)
    blocks = view_as_blocks(out, block_shape=block_shape)
    i = 1
    for z in range(n_zyx[0]):
        for y in range(n_zyx[1]):
            for x in range(n_zyx[2]):
                current_radii = ellipsoid_max_radii * rng.uniform(
                    *axes_range, size=len(ellipsoid_max_radii)
                )
                blocks[z, y, x, ...] = _center_ellips(
                    ellipsoid(*current_radii), block_shape, label=i
                )
                i += 1
    return out


def _gridded_ellipses(lbl, ny=10, nx=10, axes_range=(0.7, 1.0)):
    out = np.zeros_like(lbl)
    ry = out.shape[0] / ny / 2
    rx = out.shape[1] / nx / 2

    cy = np.linspace(ry, out.shape[0] - ry, ny).astype(int)
    cx = np.linspace(rx, out.shape[0] - rx, nx).astype(int)
    yy, xx = np.meshgrid(cy, cx)
    positions = np.array([yy.flatten(), xx.flatten()]).T
    for i, center in enumerate(positions):
        axes = (
            np.array([ry, rx]) * rng.uniform(axes_range[0], axes_range[1], size=2)
        ).astype(int)
        ellipse = {
            "center": center,
            "axes": axes,
            "color": i + 1,
            "thickness": -1,
            "angle": rng.uniform(0, 360),
            "startAngle": 0,
            "endAngle": 360,
        }
        _draw_ellipse(out, ellipse)
    return out


def random_sine_source(sz=256, frequency_range=(0, 2), rng=None):
    if rng is None:
        rng = np.random.default_rng()
    while True:
        angle = rng.uniform(0, 2 * np.pi)
        phase = rng.uniform(0, 2 * np.pi)
        f = rng.uniform(*frequency_range)

        x_off = 256 * np.pi * np.cos(angle) + phase
        y_off = 256 * np.pi * np.sin(angle) + phase

        x_min, x_max = x_off, x_off + 2 * np.pi
        y_min, y_max = y_off, y_off + 2 * np.pi

        n = sz
        x = np.linspace(x_min, x_max, n)
        y = np.linspace(y_min, y_max, n)

        xs, ys = np.meshgrid(x, y, sparse=True)

        yield np.sin(np.sqrt(xs**2 + ys**2) * f)


def summed_sine_patterns(sz, frequency_range, n_patterns, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    sine_generator = random_sine_source(sz=sz, frequency_range=frequency_range, rng=rng)
    canvas = np.zeros((sz, sz), dtype=np.float32)
    for sine, _ in zip(sine_generator, range(n_patterns)):
        canvas = canvas + sine
    return canvas


def resample_to_shape(
    img, output_shape, order, mode="constant", cval=0.0, prefilter=True
):
    """Function resamples image to the desired shape
    Typically used to up or downscale a pyramid image by a potency of 2 (e.g. 0.5, 1, 2 etc.)
    """
    zoom_values = [o / i for i, o in zip(img.shape, output_shape)]
    return zoom(
        img, zoom_values, order=order, mode=mode, cval=cval, prefilter=prefilter
    )


def _get_rotations(lbl, angle_step=10):
    for angle in accumulate(repeat(angle_step), initial=0):
        yield rotate(lbl, angle, order=0)


# def _label_yx(
# shape: tuple[int, int], dtype: type = np.uint16, seed: int | None = 42
# ) -> NDArray[UInt]:
# rng = np.random.default_rng(seed)
# label_grid = rng.integers(0, 2, size=9).cumsum().reshape(3, 3)
# return resample_to_shape(label_grid, shape, order=0).astype(dtype)


def _label_yx(
    shape: tuple[int, int], dtype: type = np.uint16, seed: int | None = 42
) -> NDArray[UInt]:
    label_grid = np.arange(9).reshape(3, 3)
    return resample_to_shape(label_grid, shape, order=0).astype(dtype)


def _label_c_yx(
    shape: tuple[int, int, int], dtype: type = np.uint16, seed: int | None = 42
) -> NDArray[UInt]:
    return np.stack(
        [_label_yx(shape[1:], dtype=dtype, seed=seed) for _ in range(shape[0])]
    )


def _label_zyx(
    shape: tuple[int, int, int], dtype: type = np.uint16, seed: int | None = 42
) -> NDArray[UInt]:
    label_2d = _label_yx(shape[1:], dtype=dtype, seed=seed)
    label_2d = np.expand_dims(label_2d, 0)
    ones = np.ones(shape[0], dtype=dtype).reshape(-1, 1, 1)
    return label_2d * ones


def _label_t_yx(
    shape: tuple[int, int, int], dtype: type = np.uint16, seed: int | None = 42
) -> NDArray[UInt]:
    lbl = _label_yx(shape[1:])
    t = shape[0]
    rotations = _get_rotations(lbl)
    return np.stack(list(take(t, rotations)), axis=0)


def _label_t_zyx(
    shape: tuple[int, int, int, int], dtype: type = np.uint16, seed: int | None = 42
) -> NDArray[UInt]:
    return np.stack(
        list(repeat(_label_t_yx((shape[0], shape[2], shape[3])), shape[1])), axis=1
    )


def _label_c_zyx(
    shape: tuple[int, int, int, int], dtype: type = np.uint16, seed: int | None = 42
) -> NDArray[UInt]:
    return np.stack(list(repeat(_label_zyx((shape[1:])), shape[0])), axis=0)


def _label_t_c_zyx(
    shape: tuple[int, int, int, int, int],
    dtype: type = np.uint16,
    seed: int | None = 42,
) -> NDArray[UInt]:
    _shape = (shape[0], *shape[2:])
    return np.stack(list(repeat(_label_t_zyx(_shape), shape[1])), axis=1)


def _image_yx(
    shape: tuple[int, int], dtype: type = np.uint16, seed: int | None = 42
) -> NDArray[Number]:
    rng = np.random.default_rng(seed)
    dy, dx = shape
    xx, yy = np.meshgrid(np.arange(dx), np.arange(dy))
    n_cycles_x, n_cycles_y = rng.integers(3, 10, size=2)
    x_sin = np.sin(xx / shape[1] * 2 * np.pi * n_cycles_x)
    y_sin = np.sin(yy / shape[0] * 2 * np.pi * n_cycles_y)
    return x_sin * y_sin


def _image_zyx(
    shape: tuple[int, int, int], dtype: type = np.uint16, seed: int | None = 42
) -> NDArray[Number]:
    rng = np.random.default_rng(seed)
    dz, dy, dx = shape
    zz, yy, xx = np.meshgrid(np.arange(dz), np.arange(dy), np.arange(dx), indexing="ij")
    n_cycles_x, n_cycles_y, n_cycles_z = rng.integers(3, 10, size=3)
    z_sin = np.sin(zz / shape[0] * 2 * np.pi * n_cycles_z)
    y_sin = np.sin(yy / shape[1] * 2 * np.pi * n_cycles_y)
    x_sin = np.sin(xx / shape[2] * 2 * np.pi * n_cycles_x)
    return x_sin * y_sin * z_sin


def _image_t_yx(
    shape: tuple[int, int, int], dtype: type = np.uint16, seed: int | None = 42
) -> NDArray[Number]:
    rng = np.random.default_rng(seed)
    dt, dy, dx = shape
    # zz, yy, xx = np.mgrid[0:dz, 0:dy, 0:dx]
    tt, yy, xx = np.meshgrid(np.arange(dt), np.arange(dy), np.arange(dx), indexing="ij")
    n_cycles_x, n_cycles_y = rng.integers(3, 10, size=2)
    n_cycles_t = rng.integers(1, 3)
    t_sin = np.sin(tt / shape[0] * 2 * np.pi * n_cycles_t)
    y_sin = np.sin(yy / shape[1] * 2 * np.pi * n_cycles_y)
    x_sin = np.sin(xx / shape[2] * 2 * np.pi * n_cycles_x)
    return x_sin * y_sin * t_sin


def _image_t_zyx(
    shape: tuple[int, int, int, int], dtype: type = np.uint16, seed: int | None = 42
) -> NDArray[Number]:
    rng = np.random.default_rng(seed)
    dt, dz, dy, dx = shape
    tt, zz, yy, xx = np.meshgrid(
        np.arange(dt), np.arange(dz), np.arange(dy), np.arange(dx), indexing="ij"
    )
    n_cycles_x, n_cycles_y, n_cycles_z = rng.integers(3, 10, size=3)
    n_cycles_t = rng.integers(1, 3)
    t_sin = np.sin(tt / shape[0] * 2 * np.pi * n_cycles_t)
    z_sin = np.sin(zz / shape[0] * 2 * np.pi * n_cycles_z)
    y_sin = np.sin(yy / shape[1] * 2 * np.pi * n_cycles_y)
    x_sin = np.sin(xx / shape[2] * 2 * np.pi * n_cycles_x)
    return x_sin * y_sin * z_sin * t_sin


def _image_c_yx(
    shape: tuple[int, int, int],
    dtype: type = np.uint16,
    seed: int | None = 42,
) -> NDArray[Number]:
    return np.stack(
        [
            _image_yx(shape[1:], dtype=dtype, seed=(seed + idx if seed else seed))
            for idx in range(shape[0])
        ]
    )


def _image_c_zyx(
    shape: tuple[int, int, int, int],
    dtype: type = np.uint16,
    seed: int | None = 42,
) -> NDArray[Number]:
    return np.stack(
        [
            _image_zyx(shape[1:], dtype=dtype, seed=(seed + idx if seed else seed))
            for idx in range(shape[0])
        ]
    )


def _image_c_t_zyx(
    shape: tuple[int, int, int, int, int],
    dtype: type = np.uint16,
    seed: int | None = 42,
) -> NDArray[Number]:
    return np.stack(
        [
            _image_t_zyx(shape[1:], dtype=dtype, seed=(seed + idx if seed else seed))
            for idx in range(shape[0])
        ]
    )


def _image_t_c_zyx(
    shape: tuple[int, int, int, int, int],
    dtype: type = np.uint16,
    seed: int | None = 42,
) -> NDArray[Number]:
    _shape = (shape[0], *shape[2:])
    return np.stack(
        [
            _image_t_zyx(_shape, dtype=dtype, seed=(seed + idx if seed else seed))
            for idx in range(shape[1])
        ],
        axis=1,
    )


ImageGen: dict[
    tuple[str, ...], Callable[[tuple[int, ...], type, int | None], NDArray[Number]]
] = {
    ("y", "x"): _image_yx,
    ("z", "y", "x"): _image_zyx,
    ("t", "y", "x"): _image_t_yx,
    ("t", "z", "y", "x"): _image_t_zyx,
    ("c", "y", "x"): _image_c_yx,
    ("c", "z", "y", "x"): _image_c_zyx,
    # ("c", "t", "z", "y", "x"): _image_c_t_zyx, #TODO: implement in spatial_image
    ("t", "c", "z", "y", "x"): _image_t_c_zyx,
}

LabelGen: dict[
    tuple[str, ...], Callable[[tuple[int, ...], type, int | None], NDArray[UInt]]
] = {
    ("y", "x"): _label_yx,
    ("c", "y", "x"): _label_c_yx,
    ("z", "y", "x"): _label_zyx,
    ("t", "y", "x"): _label_t_yx,
    ("t", "z", "y", "x"): _label_t_zyx,
    ("c", "z", "y", "x"): _label_c_zyx,
    ("t", "c", "z", "y", "x"): _label_t_c_zyx,
}

DIMS = ("t", "c", "z", "y", "x")
SPATIAL_DIMS = ("z", "y", "x")


def get_roi(
    shape: Sequence[int] = (144, 288, 288),
    n_channels: int = 3,
    n_labels: int = 5,
    scale: Sequence[int] | None = None,
):
    labels = hierarchical_labels(
        shape=shape,
        scale=scale,
        objects=("emb", "cell", "nuc", "cyto", "loc")[:n_labels],
    ).rename({"c": "l"})
    channels = get_image_si(shape=(n_channels, *shape), scale=scale)
    return Roi(name="roi", data=dict(labels=labels, images=channels))


def get_image_si(
    shape: Sequence[int],
    dims: Sequence[Literal["t", "c", "z", "y", "x"]] | None = None,
    scale: Sequence[float] | None = None,
    dtype: type = np.uint16,
    seed: int | None = 42,
    c_coords: Sequence[str] | None = None,
) -> SpatialImage:
    ndims = len(shape)
    assert ndims < 6, f"image cannot have > 5 dimensions: {ndims=}"
    if dims is None:
        dims = ("t", "c", "z", "y", "x")[-ndims:]
    if "c" in dims and c_coords is None:
        c_shape = shape[dims.index("c")]
        c_coords = [f"ch{i}" for i in range(c_shape)]
    elif "c" in dims:
        c_shape = shape[dims.index("c")]
        assert c_shape == len(
            c_coords
        ), f"shape dim c ({c_shape}) not equal len(c_coords): ({len(c_coords)})"
    spatial_dims = tuple(e for e in dims if e in SPATIAL_DIMS)
    ndims_spatial = len(spatial_dims)
    if scale is None:
        scale = (1.0,) * ndims_spatial
    data = ImageGen[dims](shape=shape, dtype=dtype, seed=seed)
    return to_spatial_image(
        data,
        dims=dims,
        scale=dict(zip(spatial_dims, scale)),
        c_coords=c_coords,
        name="image",
    )


def expand_channels_si(img: SpatialImage) -> tuple[SpatialImage, ...]:
    if "c" in img.dims:
        return tuple(img.sel(c=c, drop=True) for c in img.c)
    else:
        return (img,)


def label_si(
    shape: Sequence[int],
    dims: Sequence[Literal["c", "t", "z", "y", "x"]] | None = None,
    scale: Sequence[float] | None = None,
    dtype: type = np.uint16,
    seed: int | None = 42,
    c_coords: Sequence[str] | None = None,
) -> SpatialImage:
    ndims = len(shape)
    assert ndims < 6, f"label cannot have > 5 dimensions: {ndims=}"
    if dims is None:
        dims = ("c", "t", "z", "y", "x")[-ndims:]
    if "c" in dims:
        c_shape = shape[dims.index("c")]
        c_coords = [f"obj{i}" for i in range(c_shape)]
    spatial_dims = tuple(e for e in dims if e in SPATIAL_DIMS)
    ndims_spatial = len(spatial_dims)
    if scale is None:
        scale = (1.0,) * ndims_spatial
    data = LabelGen[dims](shape, dtype, seed)
    return to_spatial_image(
        data,
        dims=dims,
        scale=dict(zip(spatial_dims, scale)),
        c_coords=c_coords,
        name="label",
    )


def image_np(
    shape: tuple[int, ...],
    dims: tuple[Literal["c", "t", "z", "y", "x"], ...] | None = None,
    scale: tuple[float, ...] | None = None,
    dtype: type = np.uint16,
    seed: int | None = 42,
) -> NDArray[Number]:
    return get_image_si(
        shape=shape, dims=dims, scale=scale, dtype=dtype, seed=seed
    ).data


def label_np(
    shape: tuple[int, ...],
    dims: tuple[Literal["c", "t", "z", "y", "x"], ...] | None = None,
    scale: tuple[float, ...] | None = None,
    dtype: type = np.uint16,
    seed: int | None = 42,
) -> NDArray[UInt]:
    return label_si(shape=shape, dims=dims, scale=scale, dtype=dtype, seed=seed).data


def image_itk(
    shape: tuple[int, ...],
    dims: tuple[Literal["c", "t", "z", "y", "x"], ...] | None = None,
    scale: tuple[float, ...] | None = None,
    dtype: type = np.uint16,
    seed: int | None = 42,
) -> ImageLike:
    return itk.image_from_xarray(
        get_image_si(shape=shape, dims=dims, scale=scale, dtype=dtype, seed=seed)
    )


def label_itk(
    shape: tuple[int, ...],
    dims: tuple[Literal["c", "t", "z", "y", "x"], ...] | None = None,
    scale: tuple[float, ...] | None = None,
    dtype: type = np.uint16,
    seed: int | None = 42,
) -> ImageLike:
    return itk.image_from_xarray(
        label_si(shape=shape, dims=dims, scale=scale, dtype=dtype, seed=seed)
    )


# %%
