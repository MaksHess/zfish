"""
# 1D line  +          float = point (x) | difference (dx)  or coordinate_system (x) and extent (dx)
# 1D line  + (float, float) = point_pair (x.0, x.1) | interval (x, dx) | half_open_uniformly_spaced_grid (while True: x += dx)


# 2D plane +          float = x/dx along y or y/dy along x
# 2D plane + (float, float) = point (x, y) | vector (dx, dy)
# 2D plane + ((float, float), (float, float)) = (point, point) | (point, vector) | (vector, vector)

# INTENSITY COORDINATES
# channels (c)

# OBJECT_TYPE COORDINATES
# object_types (o)

# SPATIAL COORDINATES
# coordinate_systems (cord[z, y, x])          -> (z.origin, y.origin, x.origin) + (cs,), repr=Point | BasisVectors
# dimension_bounds (bound[z, y, x])            -> (z.extent, y.extent, x.extent) + (bound,), repr=TranslatedOrigin | ScaledBasisVectors
# regions (region[z, y, x]) = (cords, bound)   -> coordinate_system + dimension_bounds, repr=BoundingBoxVertices | BoundingBoxEdges
# uniform_sampling                             -> (z.scale, y.scale, x.scale) repr=TranslatedOrigin | ScaledBasisVectors
# uniform_samplings=ms_sampling (m[z, y, x])   -> uniform_sampling + (m, ), repr=[TranslatedOrigin] | [ScaledBasisVectors] | dict[m, TranslatedOrigin] ...
# uniform_sampling (m:1[z, y, x])              -> ms_sampling + filter(m==level), repr=TranslatedOrigin | ScaledBasisVectors | dict[m=level, TranslatedOrigin] ...
# sampled_regions (sampled_region[m, z, y, x]) -> regions + uniform_samplings
# images (image[m, c, region[z, y, x]])
# images (image[c, sampled_region[m, z, y, x]])
# label_images (label_image[m, o, region[z, y, x]])
# label_images (label_image[o, sampled_region[m, z, y, x]])
# rois (roi[r=c|o, sampled_region[m, z, y, x]])
"""

# %%
import logging
from dataclasses import dataclass, field
from itertools import islice
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, Sequence, TypeAlias

import dask.array as da
import numpy as np
import polars as pl
from numpy.typing import DTypeLike

from zfish.commons.types_ import MISSING, RasterMeta

if TYPE_CHECKING:
    from zfish.features.types import SpatialImage

logger = logging.getLogger(__name__)


DIMS = ["z", "y", "x"]
RASTER_DIMS = ["c", "o"]
MS_DIM = ["m"]
REGION_DIMS = ["plate", "well", "roi", "label_object"]

CAT_DIMS = RASTER_DIMS + MS_DIM + REGION_DIMS
ALL_DIMS = RASTER_DIMS + DIMS + MS_DIM


PixelIncludeCriterion: TypeAlias = Literal["all", "center", "any"]


def plot_coord_ticks(extent=1.0, scale=0.3, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    ax.eventplot([0.0, extent], color="orange", **kwargs)
    n_below = int(extent / scale)
    n_above = n_below + 1
    ax.eventplot([n_below * scale, n_above * scale], **kwargs)
    return ax


# TODO: add a direction: tuple[float, float, float] to represent affine transformations
@dataclass(frozen=True, kw_only=False)
class SpatialCoordinate:
    """One dimensional spatial coordinate with uniform sampling that represents the geometry 
    of a 1D image aka a 1D uniformly sampled grid. Degenerate grids, i.e., grids with zero extent
    represent a point. Grids where the scale is equal to the extent represent a single pixel.
    Grids with infinite sampling (i.e., scale=0) represent a continuous bounded region.
    """

    dim: Literal[*DIMS]
    origin: float = 0.0
    extent: float = 0.0
    scale: float = 0.0
    default_include_criterion: PixelIncludeCriterion = "center"
    _coord: pl.Series = field(compare=False, default=None)

    def __post_init__(self):
        if self.extent is None or self.scale is None:
            raise ValueError("Need to provide `self.extent` and `self.scale`")
        assert self.scale <= self.extent, (
            f"Scale ({self.scale}) must be <= extent ({self.extent})."
        )
        if self.scale > 0.0:
            n_exact = self.extent / self.scale
            if n_exact % 1 != 0:
                n_lower = int(n_exact)
                n_upper = n_lower + 1
                if self.default_include_criterion == "center":
                    n_cutoff = n_lower + 0.5
                elif self.default_include_criterion == "any":
                    n_cutoff = n_lower
                elif self.default_include_criterion == "all":
                    n_cutoff = n_lower + 1
                if (n_cutoff * self.scale) < self.extent:
                    new_extent = self.scale * n_upper
                else:
                    new_extent = self.scale * n_lower
                object.__setattr__(self, "extent", new_extent)

    @classmethod
    def from_bounds(
        cls,
        lower: float,
        upper: float | None = None,
        scale: float | None = None,
        shape: int | None = None,
        dim: str = "z",
    ):
        if upper is None:
            upper = lower
        extent = upper - lower
        if scale is None:
            if shape is None:
                scale = extent
            elif shape == 0:
                scale = 0.0
            else:
                scale = extent / shape
        return cls(dim=dim, origin=lower, extent=extent, scale=scale)

    @classmethod
    def from_shape(cls, shape: int, scale: float, origin: float = None, dim: str = "z"):
        if origin is None:
            origin = 0.0
        extent = shape * scale
        return cls(dim=dim, origin=origin, extent=extent, scale=scale)

    def _get_type(self):
        if self.extent == 0:
            return "point"
        elif self.scale == 0:
            return "bounded"
        elif self.extent == self.scale:
            return "pixel"
        else:
            return "sampled"

    def translate(self, delta):
        return SpatialCoordinate(
            dim=self.dim,
            scale=self.scale,
            extent=self.extent,
            origin=self.origin + delta,
        )

    def center(self):
        return SpatialCoordinate(
            dim=self.dim,
            scale=self.scale,
            extent=self.extent,
            origin=-self.extent / 2,
        )

    @property
    def coord(self):
        "Pixel border coodinates. One more than the number of pixels!"
        if self._coord is None:
            object.__setattr__(
                self,
                "_coord",
                pl.Series(
                    self.dim,
                    np.linspace(self.origin, self.origin + self.extent, self.shape + 1),
                ),
            )
        return self._coord

    @property
    def icoord(self):
        "Pixel center coordinates."
        return self.coord[:-1] + self.iorigin_shift

    @property
    def shape(self):
        if self.extent == 0:
            return 1
        if self.scale == 0:
            return None
        shp = self.extent / self.scale
        return int(round(shp))

    @property
    def iorigin_shift(self):
        return self.scale / 2

    @property
    def iorigin(self):
        return self.origin + self.iorigin_shift

    @property
    def upper_bound(self):
        return self.origin + self.extent

    @property
    def lower_bound(self):
        return self.origin

    def __str__(self):
        return self.dim

    def _slice(
        self,
        bound: tuple[float, float] | None = None,
        fill_zeros: bool = True,
        include_criterion: PixelIncludeCriterion | None = None,
    ):
        if bound is None:
            self, slice(None), (0, 0)

        if include_criterion is None:
            include_criterion = self.default_include_criterion

        if include_criterion == "all":
            lb = pl.col("lower_bound").ge(bound[0])
            ub = pl.col("upper_bound").le(bound[1])
        elif include_criterion == "any":
            lb = pl.col("upper_bound").gt(bound[0])
            ub = pl.col("lower_bound").lt(bound[1])
        elif include_criterion == "center":
            lb = pl.col(self.dim).ge(bound[0])
            ub = pl.col(self.dim).le(bound[1])
        else:
            raise ValueError(f"Unknown slice strategy: {include_criterion}")

        if fill_zeros:
            if bound[0] < self.origin:
                lower_overshoot_px_frac = (self.origin - bound[0]) / self.scale

            else:
                lower_overshoot_px_frac = 0
            if bound[1] > self.upper_bound:
                upper_overshoot_px_frac = (bound[1] - self.upper_bound) / self.scale
            else:
                upper_overshoot_px_frac = 0

            if include_criterion == "all":
                lower_pad = int(np.floor(lower_overshoot_px_frac))
                upper_pad = int(np.floor(upper_overshoot_px_frac))

            elif include_criterion == "any":
                lower_pad = int(np.ceil(lower_overshoot_px_frac))
                upper_pad = int(np.ceil(upper_overshoot_px_frac))

            elif include_criterion == "center":
                lower_pad = int(np.round(lower_overshoot_px_frac))
                upper_pad = int(np.round(upper_overshoot_px_frac))
            else:
                raise ValueError(f"Unknown slice strategy: {include_criterion}")

        else:
            lower_pad, upper_pad = (0, 0)

        any_filter_lb = pl.col("upper_bound").gt(bound[0])
        any_filter_ub = pl.col("lower_bound").lt(bound[1])
        icoords = (  # filter with 'any' to see whether the slice overlaps with the image region
            self.icoord.to_frame()
            .with_columns((self.coord[:-1]).alias("lower_bound"))
            .with_columns((self.coord[1:]).alias("upper_bound"))
            .with_row_index()
            .filter(any_filter_lb, any_filter_ub)
        )
        new_index_coordi = icoords.filter(  # actual filter selected by user
            lb,
            ub,
        )

        print(icoords)
        print(new_index_coordi)
        arr_indices = new_index_coordi["index"]

        if len(arr_indices) == 0:
            # slc = slice(0, 0)
            slc = slice(None)
        else:
            slc = slice(arr_indices.min(), arr_indices.max() + 1)
        # print(new_index_coordi)
        # print(slc)

        if new_index_coordi.height == 0:  # user filter gave no hit
            if icoords.height == 0:  # backup any filter gave no hit
                raise ValueError("Slice seems to be outside image region")
        new_origin = new_index_coordi["lower_bound"][0] - lower_pad * self.scale
        new_extent = (
            new_index_coordi["upper_bound"][-1] + upper_pad * self.scale - new_origin
        )

        if new_extent < self.scale:  # Happend due to floating point math
            new_extent = self.scale

        new_coordinate = self.__class__(
            extent=new_extent,
            scale=self.scale,
            dim=self.dim,
            origin=new_origin,
        )

        return new_coordinate, slc, (lower_pad, upper_pad)

    def slice(
        self,
        bound: tuple[float, float] | None = None,
        return_slice: bool = True,
        fill_zeros: bool = True,
        include_criterion: PixelIncludeCriterion | None = None,
    ) -> tuple[Self, slice | None, tuple[int, int] | None]:
        if bound is None:
            if not return_slice:
                return self
            # if not fill_zeros:
            #     return self, slice(None)
            return self, slice(None), (0, 0)

        if include_criterion is None:
            include_criterion = self.default_include_criterion

        if include_criterion == "all":
            lb = pl.col("lower_bound").ge(bound[0])
            ub = pl.col("upper_bound").le(bound[1])
        elif include_criterion == "any":
            lb = pl.col("upper_bound").gt(bound[0])
            ub = pl.col("lower_bound").lt(bound[1])
        elif include_criterion == "center":
            lb = pl.col(self.dim).ge(bound[0])
            ub = pl.col(self.dim).le(bound[1])
        else:
            raise ValueError(f"Unknown slice strategy: {include_criterion}")

        any_filter_lb = pl.col("upper_bound").gt(bound[0])
        any_filter_ub = pl.col("lower_bound").lt(bound[1])
        icoords = (  # filter with 'any' to see whether the slice overlaps with the image region
            self.icoord.to_frame()
            .with_columns((self.coord[:-1]).alias("lower_bound"))
            .with_columns((self.coord[1:]).alias("upper_bound"))
            .with_row_index()
            .filter(any_filter_lb, any_filter_ub)
        )
        new_index_coordi = icoords.filter(  # actual filter selected by user
            lb,
            ub,
        )

        # print(new_index_coordi)
        # print(new_index_coordi["index"])
        arr_indices = new_index_coordi["index"]
        if len(arr_indices) == 0:
            # slc = slice(0, 0)
            slc = slice(None)
        else:
            slc = slice(arr_indices.min(), arr_indices.max() + 1)

        if fill_zeros:
            if bound[0] < self.origin:
                lower_overshoot_px_frac = (self.origin - bound[0]) / self.scale
                # print(lower_overshoot_px_frac)

            else:
                lower_overshoot_px_frac = 0
            if bound[1] > self.upper_bound:
                upper_overshoot_px_frac = (bound[1] - self.upper_bound) / self.scale
            else:
                upper_overshoot_px_frac = 0

            if include_criterion == "all":
                lower_pad = int(np.floor(lower_overshoot_px_frac))
                upper_pad = int(np.floor(upper_overshoot_px_frac))

            elif include_criterion == "any":
                lower_pad = int(np.ceil(lower_overshoot_px_frac))
                upper_pad = int(np.ceil(upper_overshoot_px_frac))

            elif include_criterion == "center":
                lower_pad = int(np.round(lower_overshoot_px_frac))
                upper_pad = int(np.round(upper_overshoot_px_frac))
            else:
                raise ValueError(f"Unknown slice strategy: {include_criterion}")

        else:
            lower_pad, upper_pad = (0, 0)

        scale = self.scale
        if new_index_coordi.height == 0:  # user filter gave no hit
            if icoords.height == 0:  # backup any filter gave no hit
                # print(lower_overshoot_px_frac, upper_overshoot_px_frac)
                # print(lower_pad, upper_pad)
                raise ValueError("Slice seems to be outside image region")
            else:
                scale = 0
                new_index_coordi = icoords

        new_origin = new_index_coordi["lower_bound"][0] - lower_pad * self.scale
        new_extent = (
            new_index_coordi["upper_bound"][-1]
            + upper_pad * self.scale
            # + lower_pad * self.scale
            - new_origin
        )
        if new_extent < self.scale:  # Happend due to floating point math
            new_extent = self.scale

        new_coordinate = self.__class__(
            extent=new_extent,
            scale=scale,
            dim=self.dim,
            origin=new_origin,
        )
        if not return_slice:
            return new_coordinate
        # if not fill_zeros:
        #     return new_coordinate, slc
        return new_coordinate, slc, (lower_pad, upper_pad)

    def __repr__(self):
        is_degenerate = self.shape == 0
        is_infinite = self.shape is None
        if is_infinite | is_degenerate:
            cardinality = ""
            prefix = " "
        else:
            cardinality = f" :{self.shape}"
            prefix = "*"

        range_repr = f"{self.origin:.2f}"
        if not is_degenerate:
            if not is_infinite:
                range_repr += f"+{self.scale:.2f}"
            range_repr += f":{self.upper_bound:.2f}"
        return f"{prefix} ({self.dim}{cardinality}) [{range_repr}]"


def _coord(lower, upper, shape=None):
    if shape is None:
        shape = upper - lower
    return SpatialCoordinate.from_bounds(lower=lower, upper=upper, shape=shape)


def test_spatial_coordinate_out_of_bounds():
    LOW = 0
    UP = 2
    SHAPE = UP - LOW
    SCALE = (UP - LOW) / SHAPE
    # EPS = 0.000001
    EPS = 0.001

    z_coord = _coord(LOW, UP)
    assert z_coord.shape == SHAPE

    assert _coord(LOW - 1, UP) == _coord(LOW, UP).slice(
        (LOW - EPS, UP), include_criterion="any", return_slice=False
    )
    assert _coord(LOW, UP + 1) == _coord(LOW, UP).slice(
        (LOW, UP + EPS), include_criterion="any", return_slice=False
    )
    assert _coord(LOW, UP) == _coord(LOW, UP).slice(
        (LOW - EPS, UP), include_criterion="center", return_slice=False
    )
    assert _coord(LOW, UP) == _coord(LOW, UP).slice(
        (LOW, UP + EPS), include_criterion="center", return_slice=False
    )
    assert _coord(LOW, UP) == _coord(LOW, UP).slice(
        (LOW - EPS, UP), include_criterion="all", return_slice=False
    )
    assert _coord(LOW, UP) == _coord(LOW, UP).slice(
        (LOW, UP + EPS), include_criterion="all", return_slice=False
    )

    assert _coord(LOW - 1, UP) == _coord(LOW, UP).slice(
        (LOW - EPS - SCALE / 2, UP), include_criterion="any", return_slice=False
    )
    assert _coord(LOW, UP + 1) == _coord(LOW, UP).slice(
        (LOW, UP + EPS + SCALE / 2), include_criterion="any", return_slice=False
    )
    assert _coord(LOW - 1, UP) == _coord(LOW, UP).slice(
        (LOW - EPS - SCALE / 2, UP), include_criterion="center", return_slice=False
    )
    assert _coord(LOW, UP + 1) == _coord(LOW, UP).slice(
        (LOW, UP + EPS + SCALE / 2), include_criterion="center", return_slice=False
    )
    assert _coord(LOW - 1, UP) == _coord(LOW, UP).slice(
        (LOW - EPS - SCALE, UP), include_criterion="all", return_slice=False
    )
    assert _coord(LOW, UP + 1) == _coord(LOW, UP).slice(
        (LOW, UP + EPS + SCALE), include_criterion="all", return_slice=False
    )

    assert _coord(LOW - 2, UP) == _coord(LOW, UP).slice(
        (LOW - EPS - SCALE, UP), include_criterion="any", return_slice=False
    )
    assert _coord(LOW, UP + 2) == _coord(LOW, UP).slice(
        (LOW, UP + EPS + SCALE), include_criterion="any", return_slice=False
    )
    assert _coord(LOW - 1, UP) == _coord(LOW, UP).slice(
        (LOW - EPS - SCALE, UP), include_criterion="center", return_slice=False
    )
    assert _coord(LOW, UP + 1) == _coord(LOW, UP).slice(
        (LOW, UP + EPS + SCALE), include_criterion="center", return_slice=False
    )
    assert _coord(LOW - 1, UP) == _coord(LOW, UP).slice(
        (LOW - EPS - SCALE, UP), include_criterion="all", return_slice=False
    )
    assert _coord(LOW, UP + 1) == _coord(LOW, UP).slice(
        (LOW, UP + EPS + SCALE), include_criterion="all", return_slice=False
    )


def test_spatial_coordinate():
    scl, ext, orig = (1.0, 10.0, 0.0)
    z_coord = SpatialCoordinate("z", scale=scl, extent=ext, origin=orig)
    # basic setup of a coord
    assert np.all(z_coord.coord.to_numpy() == np.linspace(0, 10, 11))
    assert np.all(z_coord.icoord.to_numpy() == np.linspace(0.5, 9.5, 10))
    assert z_coord.shape == 10
    # slicing strategies
    slice_bounds_out = (0.2, 8.7)  # "outside" of px center
    assert z_coord.slice(
        slice_bounds_out, include_criterion="center", return_slice=False
    ) == z_coord.slice(slice_bounds_out, include_criterion="any", return_slice=False)
    assert z_coord.slice(
        slice_bounds_out, include_criterion="center", return_slice=False
    ) != z_coord.slice(slice_bounds_out, include_criterion="all", return_slice=False)
    slice_bounds_in = (0.6, 8.2)  # "inslide" of px center
    assert z_coord.slice(
        slice_bounds_in, include_criterion="center", return_slice=False
    ) == z_coord.slice(slice_bounds_in, include_criterion="all", return_slice=False)
    assert z_coord.slice(
        slice_bounds_in, include_criterion="center", return_slice=False
    ) != z_coord.slice(slice_bounds_in, include_criterion="any", return_slice=False)

    translate = 3.224
    z_coord_trans = z_coord.translate(translate)
    slice_bounds_in_trans = tuple(e + translate for e in slice_bounds_in)
    slice_bounds_out_trans = tuple(e + translate for e in slice_bounds_out)

    # basic setup and translate does not change shape
    assert z_coord_trans.origin == translate
    assert z_coord.shape == z_coord_trans.shape

    # slicing behaves the same in translated coord (with translated slices)
    for crit in ["any", "all", "center"]:
        assert z_coord_trans.slice(
            slice_bounds_in_trans, include_criterion=crit, return_slice=False
        ) == z_coord.slice(
            slice_bounds_in, include_criterion=crit, return_slice=False
        ).translate(translate)
        assert z_coord_trans.slice(
            slice_bounds_out_trans, include_criterion=crit, return_slice=False
        ) == z_coord.slice(
            slice_bounds_out, include_criterion=crit, return_slice=False
        ).translate(translate)

    # Centering two spatial coordinates of same scale and extent leads to identical coords
    assert (
        SpatialCoordinate("z", scale=1.0, extent=4.0, origin=5).center()
        == SpatialCoordinate("z", scale=1.0, extent=4.0, origin=0).center()
    )
    # Centering is idempotent
    assert z_coord.center() == z_coord.center().center().center()
    # Translations are invertible
    assert z_coord == z_coord.translate(5).translate(-5)
    # assert (z_coord.translate(5).translate(-5) == z_coord)

    LOW = 3
    UP = 5
    z_coord = SpatialCoordinate.from_bounds(
        dim="z", lower=LOW, upper=UP, shape=UP - LOW
    )
    # Zero padding above

    _coord(LOW + 1, UP) == z_coord.slice(
        (LOW + 0.0001, UP), return_slice=False, include_criterion="all"
    )


# test_spatial_coordinate()

## -> Quick attempt at auto dispatching to underlying coords.
# @dataclass(frozen=True, slots=True)
# class SpatialCoordinateSystem2:
#     __dispatch_methods__: ClassVar[tuple[str, ...]] = (
#         "coord",
#         "icoord",
#         "origin",
#         "extent",
#         "_get_type",
#         "dim",
#         "slice",
#     )
#     name: str = "spatial"
#     _sub_coords: tuple[SpatialCoordinate, ...] = (
#         SpatialCoordinate("z"),
#         SpatialCoordinate("y"),
#         SpatialCoordinate("x"),
#     )

#     @classmethod
#     def from_bounds(
#         cls,
#         lower: tuple[float, ...],
#         upper: tuple[float, ...],
#         scale: tuple[float, ...] | None = None,
#         shape: tuple[int, ...] | None = None,
#         dims: tuple[str, ...] | None = None,
#     ):
#         if scale is None:
#             scale = tuple([None for _ in range(len(lower))])
#         if shape is None:
#             shape = tuple([None for _ in range(len(lower))])
#         if dims is None:
#             dims = DIMS[-len(lower) :]

#         return cls(
#             _sub_coords=tuple(
#                 [
#                     SpatialCoordinate.from_bounds(
#                         lower=l, upper=u, scale=scl, shape=shp, dim=dim
#                     )
#                     for dim, l, u, scl, shp in zip(dims, lower, upper, scale, shape)
#                 ]
#             )
#         )

#     @classmethod
#     def from_shape(
#         cls,
#         shape: tuple[int, ...],
#         scale: tuple[float, ...],
#         origin: tuple[float, ...] = None,
#         dims: tuple[str, ...] = None,
#     ):
#         if origin is None:
#             origin = (0.0,) * len(shape)
#         if dims is None:
#             dims = DIMS[-len(shape) :]

#         extent = tuple([s * scl for s, scl in zip(shape, scale)])
#         upper_bounds = tuple([o + e for o, e in zip(origin, extent)])
#         return cls.from_bounds(origin, upper_bounds, shape=shape)

#     def __getattr__(self, name):
#         if name in self.__dispatch_methods__:
#             return tuple(getattr(e, name) for e in self._sub_coords)
#         raise AttributeError(f"{self.__class__.__name__!r} has no attribute {name}")


@dataclass(frozen=True, slots=True)
class SpatialCoordinateSystem:
    __dispatch_coords__: ClassVar[tuple[str, ...]] = ("z", "y", "x")

    name: str = "spatial"
    z: SpatialCoordinate = SpatialCoordinate(dim="z")
    y: SpatialCoordinate = SpatialCoordinate(dim="y")
    x: SpatialCoordinate = SpatialCoordinate(dim="x")

    @classmethod
    def from_bounds(
        cls,
        lower: tuple[float, float, float],
        upper: tuple[float, float, float],
        scale: tuple[float, float, float] | None = None,
        shape: tuple[int, int, int] | None = None,
    ):
        if scale is None:
            extent = (up - low for up, low in zip(upper, lower))
            if shape is None:
                scale = extent
            else:
                scale = (ext / shp for ext, shp in zip(extent, shape))
        coords = {}
        for dim, low, up, scl in zip(["z", "y", "x"], lower, upper, scale):
            coords[dim] = SpatialCoordinate(
                dim=dim, origin=low, extent=up - low, scale=scl
            )
        return cls(**coords)

    @classmethod
    def from_shape(
        cls, shape: tuple[int, int, int], scale: tuple[float, float, float], origin=None
    ):
        if origin is None:
            origin = (0.0, 0.0, 0.0)
        extent = (shp * scl for shp, scl in zip(shape, scale))
        coords = {}
        for dim, o, e, scl in zip(["z", "y", "x"], origin, extent, scale):
            coords[dim] = SpatialCoordinate(dim=dim, origin=o, extent=e, scale=scl)
        return cls(**coords)

    def slice(
        self,
        z=None,
        y=None,
        x=None,
        return_slice=True,
        fill_zeros=True,
        include_criterion="any",
    ):
        new_z, slc_z, pad_z = self.z.slice(
            z,
            return_slice=True,
            include_criterion=include_criterion,
            fill_zeros=fill_zeros,
        )
        new_y, slc_y, pad_y = self.y.slice(
            y,
            return_slice=True,
            include_criterion=include_criterion,
            fill_zeros=fill_zeros,
        )
        new_x, slc_x, pad_x = self.x.slice(
            x,
            return_slice=True,
            include_criterion=include_criterion,
            fill_zeros=fill_zeros,
        )
        new_coordinate_system = SpatialCoordinateSystem(
            name=self.name, z=new_z, y=new_y, x=new_x
        )

        return new_coordinate_system, (slc_z, slc_y, slc_x), (pad_z, pad_y, pad_x)
        # if return_slice:
        #     if fill_zeros:
        #         return (
        #             new_coordinate_system,
        #             (slc_z, slc_y, slc_x),
        #             (pad_z, pad_y, pad_x),
        #         )
        #     return new_coordinate_system, (slc_z, slc_y, slc_x)
        # return new_coordinate_system

    def translate(self, z=None, y=None, x=None):
        z_new = self.z.translate(z) if z is not None else self.z
        y_new = self.y.translate(y) if y is not None else self.y
        x_new = self.x.translate(x) if x is not None else self.x
        return SpatialCoordinateSystem(name=self.name, z=z_new, y=y_new, x=x_new)

    def center(self, z=True, y=True, x=True):
        z_new = self.z.center() if z else self.z
        y_new = self.y.center() if y else self.y
        x_new = self.x.center() if x else self.x
        return SpatialCoordinateSystem(name=self.name, z=z_new, y=y_new, x=x_new)

    @property
    def coords(self) -> tuple[pl.Series, ...]:
        return (self.z.coord, self.y.coord, self.x.coord)

    @property
    def icoords(self) -> tuple[pl.Series, ...]:
        return (self.z.icoord, self.y.icoord, self.x.icoord)

    @property
    def dims(self) -> tuple[str, ...]:
        return (self.z.dim, self.y.dim, self.x.dim)

    @property
    def origin(self) -> tuple[float, ...]:
        return tuple([c.origin for c in self._get_components()])

    @property
    def iorigin(self) -> tuple[float, ...]:
        return tuple([c.iorigin for c in self._get_components()])

    @property
    def scale(self) -> tuple[float, ...]:
        return tuple([c.scale for c in self._get_components()])

    @property
    def extent(self) -> tuple[float, ...]:
        return tuple([c.extent for c in self._get_components()])

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple([c.shape for c in self._get_components()])

    @property
    def upper_bound(self) -> tuple[int, ...]:
        return tuple([c.upper_bound for c in self._get_components()])

    def _get_components(self) -> tuple[SpatialCoordinate, ...]:
        return (self.z, self.y, self.x)

    def _get_repr_components(self):
        return (self.shape, *[repr(comp) for comp in self._get_components()])

    def __repr__(self):
        return f"{'   '.join(repr(comp) for comp in self._get_components())}"

    def __bbx_napari__(self):
        return bbx_corners_to_napari_vector_dict(
            self.origin, [o + e for o, e in zip(self.origin, self.extent)]
        )

    def __pixel_centers_napari__(self):
        return {
            "data": np.stack([e.flatten() for e in np.meshgrid(*self.icoords)]).T,
            "face_color": "green",
        }

    def __pixel_bounds_napari__(self):
        return {
            "data": np.stack([e.flatten() for e in np.meshgrid(*self.coords)]).T,
            "face_color": "orange",
        }


@dataclass(frozen=True, slots=True)
class CategoricalCoordinateBase:
    _coord: pl.Series = None
    _squeeze_buffer: str = None

    def __init__(self, coord: Sequence[Any] = None):
        if coord is None:
            valid_coord = pl.Series(self.dim, ())
            object.__setattr__(self, "_squeeze_buffer", None)
        else:
            # degenerate dim (0), retaining coord_label in series name.
            if isinstance(coord, (str, int)):
                valid_coord = pl.Series(f"{self.dim}", (), dtype=self.dtype)
                object.__setattr__(self, "_squeeze_buffer", coord)
            else:
                assert len(coord) == len(set(coord)), (
                    "Coordinate cannot contain duplicate entries."
                )
                valid_coord = pl.Series(self.dim, coord, dtype=self.dtype)
                object.__setattr__(self, "_squeeze_buffer", None)
        object.__setattr__(self, "_coord", valid_coord)

    @property
    def coord(self):
        return self._coord

    @property
    def dim(self):
        return self.__coord_dim__

    @property
    def dtype(self):
        return self.__coord_dtype__

    @property
    def is_dropped(self):
        return len(self) == 0 and self._squeeze_buffer is None

    @property
    def is_degenerate(self):
        return len(self) == 0 and self._squeeze_buffer is not None

    @property
    def is_singleton(self):
        return len(self) == 1

    def squeeze(self):
        if len(self) == 1:
            return self.__class__(self.coord.item())
        return self

    def expand_dims(self):
        if self.is_degenerate:
            return self.__class__((self._squeeze_buffer,))
        return self
        # if len(self) == 0:
        #     coord_dim, coord_label = self.coord.name.split("=", maxsplit=1)
        #     return self.__class__((self.dtype.to_python()(coord_label),))
        # return self

    def slice(
        self,
        elements: str | int | Sequence[str] | Sequence[int] | None,
        strict=True,
        return_slice=True,
    ) -> (
        Self | tuple[Self, dict[str, int | Sequence[int]]]
    ):  # strict = raise for missing elements.
        if elements is None:
            out_self = self
            if self.is_degenerate:
                out_slc = ()
            else:
                out_slc = (slice(None),)
        elif isinstance(elements, (str, int)):
            if elements not in self.coord:
                if strict:
                    raise KeyError(
                        f"Missing element {elements!r} not in {self.coord.to_list()}"
                    )
                else:
                    out_self = self.__class__(())
                    out_slc = ()
            else:
                out_self = self.__class__(elements)
                out_slc = (self.coord.to_list().index(elements),)
        else:
            new_coord_w_indices = (
                self.coord.to_frame()
                .with_row_index()
                .filter(pl.col(self.__coord_dim__).is_in(elements))
            )
            new_coord = new_coord_w_indices[self.__coord_dim__]
            out_slc = (new_coord_w_indices["index"].to_list(),)
            if len(new_coord) != len(elements) and strict:
                missing = [e for e in elements if e not in self.coord]
                raise KeyError(
                    f"Missing elements {missing} not in {self.coord.to_list()}"
                )
            out_self = self.__class__(new_coord)
        if return_slice:
            return out_self, out_slc  # {self.__coord_dim__: out_slc}
        return out_self

    def __len__(self):
        return len(self._coord)

    def __repr__(self):
        if self.is_dropped:
            return f"!{self.coord.name}"
        elif self.is_degenerate:
            return f"{self.coord.name}={self._squeeze_buffer!r}"
        return f"({self.coord.name}:{len(self.coord)}) {list(self._coord)}"


@dataclass(frozen=True, slots=True, repr=False, init=False)
class ChannelCoord(CategoricalCoordinateBase):
    __coord_dim__ = "c"
    __coord_dtype__ = pl.String


@dataclass(frozen=True, slots=True, repr=False, init=False)
class ObjectTypeCoord(CategoricalCoordinateBase):
    __coord_dim__ = "o"
    __coord_dtype__ = pl.String


@dataclass(frozen=True, slots=True, repr=False, init=False)
class LabelCoord(CategoricalCoordinateBase):
    __coord_dim__ = "lbl"
    __coord_dtype__ = pl.UInt32


@dataclass(frozen=True, slots=True, repr=False, init=False)
class MultiscaleCoord(CategoricalCoordinateBase):
    __coord_dim__ = "m"
    __coord_dtype__ = pl.Int8


@dataclass(frozen=True, slots=True, repr=False, init=False)
class LabelObjectCoord(CategoricalCoordinateBase):
    __coord_dim__ = "label_object"
    __coord_dtype__ = pl.Struct({"o": pl.String, "roi": pl.String, "lbl": pl.UInt32})


class CompoundCoord:
    _coord: pl.DataFrame = None
    _squeeze_buffer: str | int = None


# class MultiscaleSpatialCoordinateSystem:
#     __dispatch_coords__: ClassVar[tuple[str, ...]] = ("m", )

#     name: str = "multiscale"
#     m: MultiscaleCoord = MultiscaleCoord()
#     _coords: tuple[SpatialCoordinateSystem, ...]


@dataclass(frozen=True, slots=True)
class MultiscaleCoordinateSystem:
    levels: tuple[SpatialCoordinateSystem, ...] = (SpatialCoordinateSystem(),)

    def __post_init__(self):
        if isinstance(self.levels, SpatialCoordinateSystem):
            object.__setattr__(self, "levels", (SpatialCoordinateSystem(),))

    def __call__(self, level: int = 0) -> SpatialCoordinateSystem:
        return self.get_level(level)

    def get_level(self, level: int = 0) -> SpatialCoordinateSystem:
        return self.levels[level]

    def _get_repr_components(self):
        ms = tuple(range(len(self.levels)))
        child_components = [comps._get_repr_components() for comps in self.levels]
        out = [(m, *cc) for m, cc in zip(ms, child_components)]
        return pl.DataFrame(
            out,
            orient="row",
            strict=False,
            schema=("m", "shape", "z_repr", "y_repr", "x_repr"),
        ).select(
            "m",
            "shape",
            pl.concat_str("z_repr", "y_repr", "x_repr", separator="  ").alias(
                "physical space"
            ),
        )

    def __repr__(self):
        return "\n".join(repr(e) for e in self.levels)

    def slice(self, z=None, y=None, x=None, return_slice=False):
        new_levels = []
        for level in self.levels:
            level_out, level_slc = level.slice(z=z, y=y, x=x, return_slice=True)
            new_levels.append(level_out)
        if return_slice:
            return (MultiscaleCoordinateSystem(levels=tuple(new_levels)),)
        return MultiscaleCoordinateSystem(levels=tuple(new_levels))


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def bbx_corners_to_vertices(c_lower, c_upper):
    c_arr = np.array([c_lower, c_upper])
    vertices = []
    for v_indices in (
        (np.arange(2**3) & (2 ** np.arange(3)).reshape(-1, 1)).clip(0, 1)
    ).T:
        vertices.append(c_arr[v_indices, np.arange(3)])
    return np.array(vertices)


def bbx_vertices_to_edges(vertices):
    ndim = vertices.shape[1]

    edges = []
    for i in range(ndim):
        n = 2**i
        for j in range(n):
            for edge in batched(vertices[j::n], 2):
                edges.append(edge)

    return np.array(edges)


def bbx_corners_to_napari_vector_dict(c_lower, c_upper, **napari_vector_kwargs):
    median_extent = np.median(np.array(c_upper) - np.array(c_lower))
    edges = bbx_vertices_to_edges(bbx_corners_to_vertices(c_lower, c_upper))
    vecs = _lines_to_vecs(edges)
    vec_dict = {
        "data": vecs,
        "vector_style": "line",
        "length": 1,
        "edge_width": median_extent / 40,
        **napari_vector_kwargs,
    }
    return vec_dict


def _lines_to_vecs(bbx_edges):
    bbx_vecs = bbx_edges.copy()
    bbx_vecs[:, 1, :] = bbx_vecs[:, 1, :] - bbx_vecs[:, 0, :]
    return bbx_vecs


COORDINATES = {
    "spatial": SpatialCoordinateSystem,
    "o": ObjectTypeCoord,
    "c": ChannelCoord,
    "m": MultiscaleCoord,
}


@dataclass(frozen=True, slots=True)
class RasterBase:
    __coordinate_key__: ClassVar[Literal["c", "o"]]
    __raster_type__: ClassVar[Literal["image", "label", "mask", "distance"]]
    __raster_default_name__: ClassVar[str]
    __raster_array_types__: ClassVar[tuple[DTypeLike, ...]]

    region_id: tuple[str, ...]
    m: int = None
    spatial: SpatialCoordinateSystem = None
    path: str = None
    _data: da.Array = field(default=None, repr=False)

    def _validate(self):
        if self.shape != self.data.shape:
            raise ValueError(
                f"Shape missmatch between coordinates {self.shape} and data {self.data.shape}"
            )

    def __post_init__(self):
        # self._validate()
        pass

    @classmethod
    def zeros_from_bounds(
        cls,
        lower: tuple[float, ...],
        upper: tuple[float, ...],
        scale: tuple[float, ...] = None,
        shape: tuple[float, ...] = None,
        path: str | None = MISSING,
        m: int = 0,
        value_coord: tuple[str, ...] | str = None,
        region_id: tuple[str, ...] = None,
        dtype: DTypeLike = None,
        array_backend: Literal["dask", "numpy"] = "dask",
    ):
        if value_coord is None:
            value_coord = cls.__raster_default_name__
        if region_id is None:
            region_id = ("site0",)
        if dtype is None:
            dtype = np.uint16

        spatial = SpatialCoordinateSystem.from_bounds(
            lower=lower, upper=upper, scale=scale
        )
        value_coord_ = COORDINATES[cls.__coordinate_key__](value_coord)

        if value_coord_.is_degenerate:
            arr_shape = spatial.shape
            value_coord = value_coord_._squeeze_buffer
        elif value_coord_.is_dropped:
            arr_shape = spatial.shape
            value_coord = cls.__raster_type__
        else:
            arr_shape = (len(value_coord), *spatial.shape)
            value_coord = tuple(value_coord_.coord.to_list())
        if array_backend == "dask":
            arr = da.zeros(arr_shape, dtype=dtype)
        else:
            arr = np.zeros(arr_shape, dtype=dtype)

        return cls.from_array(
            arr,
            scale=scale,
            region_id=region_id,
            value_coord=value_coord,
            path=path,
            m=m,
            origin=lower,
        )

    @classmethod
    def from_spatial_image(
        cls,
        spatial_image: "SpatialImage",
    ):
        if cls.__coordinate_key__ in spatial_image.coords.keys():
            coord = spatial_image.coords[cls.__coordinate_key__].values
        elif "l" in spatial_image.coords.keys():
            assert cls.__coordinate_key__ == "o", (
                "Cannot generate non-LabelImage from label-like coordinate 'l'"
            )
            coord = spatial_image.coords["l"].values
        else:
            value_coord = np.array(cls.__raster_default_name__)

        if coord.ndim == 0:
            value_coord = coord.item()
        else:
            value_coord = tuple(coord)

        spatial_dims_to_expand = [
            DIMS.index(sd) for sd in DIMS if sd not in spatial_image.dims
        ]

        new_dims_to_expand = [-3 + e for e in spatial_dims_to_expand]
        new_arr = np.expand_dims(spatial_image.data, new_dims_to_expand)

        return cls.from_array(
            new_arr,
            scale=tuple([spatial_image.meta.scale_dict.get(e, 0.0) for e in DIMS]),
            region_id=("site",),
            value_coord=value_coord,
            origin=tuple(
                [
                    spatial_image.meta.origin_dict.get(e, 0.0)
                    - spatial_image.meta.scale_dict.get(e, 0.0) / 2
                    for e in DIMS
                ]
            ),
        )

    @classmethod
    def from_array_meta(
        cls,
        arr_meta: tuple[da.Array, RasterMeta],
    ):
        arr, meta = arr_meta
        return cls.from_array(
            arr=arr,
            scale=meta.scale,
            region_id=("site",),
            value_coord=meta.name,
            m=meta.level,
        )

    @classmethod
    def from_array(
        cls,
        arr,
        scale: tuple[float, ...],
        region_id: tuple[str, ...],
        value_coord: tuple[str, ...] | str,
        path: str | None = MISSING,
        m: int | None = None,
        origin: tuple[float, ...] = None,
    ):
        if isinstance(value_coord, str):
            shape = arr.shape
            coord_shape = 0
            # coord_shape = 1
            # arr = np.expand_dims(arr, 0)
        elif len(value_coord) == 0:
            shape = arr.shape
            coord_shape = 0
        else:
            coord_shape, *shape = arr.shape
            assert coord_shape == len(value_coord), (
                f"coordinate mismatch: {coord_shape} {value_coord}"
            )
        assert len(shape) == len(scale), (
            f"Array shape does not match the provided scales: {shape} {scale}"
        )

        if path is MISSING:
            path = None

        spatial_coords = SpatialCoordinateSystem.from_shape(
            shape, scale=scale, origin=origin
        )
        value_coord_ = COORDINATES[cls.__coordinate_key__](value_coord)
        kwargs = {
            "region_id": region_id,
            "spatial": spatial_coords,
            "path": path,
            "_data": arr,
            "m": m,
            cls.__coordinate_key__: value_coord_,
        }
        return cls(**kwargs)

    def _from_template(self, **kwargs):
        """Careful when using this. Easy to get an invalid object."""
        default_kwargs = {
            "region_id": self.region_id,
            "spatial": self.spatial,
            "path": self.path,
            "_data": self._data,
            "m": self.m,
            self.__coordinate_key__: getattr(self, self.__coordinate_key__),
        }
        return self.__class__(**{**default_kwargs, **kwargs})

    def translate(self, z=None, y=None, x=None):
        new_spatial = self.spatial.translate(z=z, y=y, x=x)
        return self._from_template(spatial=new_spatial)

    def center(self, z=True, y=True, x=True):
        new_spatial = self.spatial.center(z=z, y=y, x=x)
        return self._from_template(spatial=new_spatial)

    def pipe(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)

    def slice(
        self,
        value_coord=None,
        z=None,
        y=None,
        x=None,
        include_criterion="any",
        # out_of_bound_slice="zeros",  # drop, zeros
        fill_zeros=True,
        return_view=True,
    ) -> Self:
        new_spatial, spatial_slc, spatial_pad = self.spatial.slice(
            z=z,
            y=y,
            x=x,
            return_slice=True,
            fill_zeros=fill_zeros,
            include_criterion=include_criterion,
        )
        new_cat_coord, cat_coord_slc = getattr(self, self.__coordinate_key__).slice(
            value_coord, return_slice=True
        )
        slc = cat_coord_slc + spatial_slc
        if return_view:
            new_arr = self.data[*slc]
            assert fill_zeros is not True, (
                "padding not available when requesting a view."
            )
        else:
            new_arr = np.pad(self.data[*slc], ((0, 0), *spatial_pad))
        kwargs = {
            "region_id": self.region_id,
            "spatial": new_spatial,
            "path": self.path,
            "_data": new_arr,
            self.__coordinate_key__: new_cat_coord,
        }
        return self.__class__(**kwargs)

    def _spatial_slice_array(self, z=None, y=None, x=None):
        new_spatial, arr_slice = self.spatial.slice(z=z, y=y, x=x, return_slice=True)
        # print(new_spatial)
        # print(arr_slice)
        return new_spatial, self._data[:, *arr_slice["spatial"]]

    @property
    def dims(self):
        value_dim = getattr(self, self.__coordinate_key__).dim
        return (value_dim, *self.spatial.dims)

    @property
    def _value(self):
        return getattr(self, self.__coordinate_key__)

    @property
    def z(self) -> SpatialCoordinate:
        return self.spatial.z

    @property
    def y(self) -> SpatialCoordinate:
        return self.spatial.y

    @property
    def x(self) -> SpatialCoordinate:
        return self.spatial.x

    @property
    def shape(self):
        value_type_coord = getattr(self, self.__coordinate_key__).coord
        coord_shape = len(value_type_coord)
        if coord_shape == 0:
            return self.spatial.shape
        return (coord_shape, *self.spatial.shape)

    @property
    def data(self):
        return self._data

    @property
    def meta(self):
        return RasterMeta(
            name=tuple(getattr(self, self.__coordinate_key__).coord),
            type_=self.__raster_type__,
            dims=self.dims,
            scale=self.spatial.scale,
            level=self.m,
            origin=self.spatial.origin,
            path=self.path,
        )

    def pprint(self):
        print(
            f"{getattr(self, self.__coordinate_key__)!r}   {self.spatial.z!r}  {self.spatial.y!r}  {self.spatial.x!r}"
        )
        return self


# TODO: Check if da.Array can be forced to return a view (compatibility with np.pad)
@dataclass(frozen=True, slots=True)
class Image(RasterBase):
    __coordinate_key__ = "c"
    __raster_type__ = "image"
    __raster_default_name__ = "ch_00"
    __raster_array_types__ = (np.uint8, np.uint16, np.uint32, np.float32, np.float64)
    c: ChannelCoord = ChannelCoord("ch_00")

    def slice(
        self,
        c=None,
        z=None,
        y=None,
        x=None,
        include_criterion="any",
        # out_of_bound_slice="drop",
        fill_zeros=False,
        return_view=True,
    ) -> Self:
        return super(Image, self).slice(
            value_coord=c,
            z=z,
            y=y,
            x=x,
            include_criterion=include_criterion,
            fill_zeros=fill_zeros,
            # out_of_bound_slice=out_of_bound_slice,
            return_view=return_view,
        )


@dataclass(frozen=True, slots=True)
class LabelImage(RasterBase):
    __coordinate_key__ = "o"
    __raster_type__ = "label"
    __raster_default_name__ = "otype_00"
    __raster_array_types__ = (np.uint8, np.uint16, np.uint32)
    o: ChannelCoord = ObjectTypeCoord("otype_00")
    _labels: pl.Series = None

    def slice(
        self,
        o=None,
        z=None,
        y=None,
        x=None,
        include_criterion="any",
        # out_of_bound_slice="drop",
        fill_zeros=False,
        return_view=True,
    ) -> Self:
        return super(LabelImage, self).slice(
            value_coord=o,
            z=z,
            y=y,
            x=x,
            include_criterion=include_criterion,
            fill_zeros=fill_zeros,
            # out_of_bound_slice=out_of_bound_slice,
            return_view=return_view,
        )

    def get_labels(self, include_background=False) -> da.Array:
        if self._labels is None:
            labels = np.unique(self.data)
            if not include_background and labels[0] == 0:
                labels = labels[1:]

            object.__setattr__(self, "_labels", labels)
        return self._labels


def test_image_from_array():
    SHAPE = (2, 50, 100, 100)
    img = Image.from_array(
        np.zeros(SHAPE),
        scale=(0.5, 1.2, 1.2),
        region_id=("site0",),
        value_coord=("DAPI.1", "PCNA.0"),
        m=0,
    )
    assert img.shape == SHAPE
    img2 = img.slice(c=["DAPI.1"])
    assert img2.shape == (1, *SHAPE[1:])
    img3 = img.slice(c="DAPI.1")
    assert img3.shape == SHAPE[1:], f"{img3.shape}, {SHAPE}"


from zfish.preprocessing.types import id_


def get_images(
    n_imgs=10,
    shape=(1, 100, 100),
    scale=(1.0, 0.325, 0.325),
    value_coord=("ch0", "ch1"),
    m=0,
    origin=(0.0, 0.0, 0.0),
    center=False,
) -> tuple[Image, ...]:
    if center:
        pipe_func = RasterBase.center
    else:
        pipe_func = id_
    if not isinstance(value_coord, tuple):
        value_shape = ()
    else:
        value_shape = (len(value_coord),)
    arr_shape = value_shape + shape
    return [
        Image.from_array(
            np.ones(arr_shape) * i + 1,
            scale=scale,
            region_id=("site0",),
            value_coord=value_coord,
            m=m,
            origin=origin,
        ).pipe(pipe_func)
        for i in (range(n_imgs))
    ]


def get_label_images(
    n_imgs=10,
    shape=(1, 100, 100),
    scale=(1.0, 0.325, 0.325),
    value_coord="obj0",
    m=0,
    origin=(0.0, 0.0, 0.0),
    center=False,
) -> tuple[Image, ...]:
    if center:
        pipe_func = RasterBase.center
    else:
        pipe_func = id_
    if isinstance(value_coord, str):
        value_shape = ()
    else:
        value_shape = (len(value_coord),)
    arr_shape = value_shape + shape
    return [
        LabelImage.from_array(
            np.ones(arr_shape, dtype=np.uint16) * i + 1,
            scale=scale,
            region_id=("site0",),
            value_coord=value_coord,
            m=m,
            origin=origin,
        ).pipe(pipe_func)
        for i in (range(n_imgs))
    ]


import matplotlib.pyplot as plt
import numpy as np


def align_coords(img: Image, canvas: Image) -> tuple[float, float, float]:
    pos_shift = np.mod(
        np.array(canvas.spatial.origin) - np.array(img.spatial.origin),
        canvas.spatial.scale,
    )
    pos_or_neg_shifts = np.stack(
        [
            pos_shift - np.array(canvas.spatial.scale),
            pos_shift,
        ]
    )
    return tuple(
        pos_or_neg_shifts[
            np.argmin(np.abs(pos_or_neg_shifts), axis=0),
            np.arange(pos_or_neg_shifts.shape[1]),
        ]
    )


def canvas_from_images(
    imgs: Sequence[Image], return_origins=False
) -> Image | tuple[Image, dict[str, Any]]:
    if len(imgs) == 0:
        return
    canvas = imgs[0].__class__.zeros_from_bounds(
        lower=tuple(
            np.array([i.spatial.origin for i in imgs], dtype=np.float64).min(axis=0)
        ),
        upper=tuple(
            np.array([i.spatial.upper_bound for i in imgs], dtype=np.float64).max(
                axis=0
            )
        ),
        scale=imgs[0].spatial.scale,
        value_coord=imgs[0]._value.coord,
        array_backend="numpy",
    )
    imgs_aligned = [img.translate(*align_coords(img, canvas)) for img in imgs]

    for img in imgs_aligned:
        canvas_slc = canvas.slice(
            **{
                k: (l, u)
                for k, l, u in zip(
                    img.spatial.dims, img.spatial.origin, img.spatial.upper_bound
                )
            },
            include_criterion="center",
            return_view=True,
        )
        canvas_slc._data[...] = canvas_slc._data[...] + img._data

    if return_origins:
        points = [img.spatial.origin for img in imgs_aligned]
        return canvas, points
    return canvas


def translate_images(
    imgs: Sequence[Image],
    translations: Sequence[dict[str, float]],
    # alignment: Literal["origin", "center"] = "center",
) -> tuple[Image, ...]:
    out_imgs = []
    for img, translation in zip(imgs, translations):
        out_imgs.append(img.translate(**translation))
    return out_imgs


def get_spiral(
    n_samples=300, n_turns=6, scale_factor=1000, initial_radius=0.01, final_radius=1.0
):
    b = (final_radius - initial_radius) / (2 * np.pi * n_turns)

    start_theta = np.pi / 2
    end_theta = 2 * np.pi * n_turns

    s = np.linspace(start_theta, end_theta, n_samples)
    x = (initial_radius + b * s) * np.cos(s)
    y = (initial_radius + b * s) * np.sin(s)

    # Calculate the arc length
    L = np.cumsum(np.sqrt(np.gradient(x) ** 2 + np.gradient(y) ** 2))
    Ls = np.linspace(0, L[-1], n_samples)

    # Find s values for equally spaced arc lengths
    new_s = np.interp(Ls, L, s)

    # Calculate x and y for equally spaced arc lengths
    x = (initial_radius + b * new_s) * np.cos(new_s) * scale_factor
    y = (initial_radius + b * new_s) * np.sin(new_s) * scale_factor
    return pl.DataFrame({"phi": np.arctan2(y, x) * -1, "s": s, "y": y, "x": x}).sort(
        "phi"
    )
    # return np.stack([s, x, y], axis=1) * scale_factor


def get_concentric_circles(
    n_samples=300, n_circles=8, scale_factor=500, initial_radius=0.2, final_radius=1.0
):
    radii = np.linspace(initial_radius, final_radius, n_circles)
    circumfs = radii * 2 * np.pi
    percent_path = circumfs / circumfs.sum()
    n_samples_per_r = (percent_path * n_samples).round().astype(np.int_)
    if n_samples_per_r.sum() != n_samples:
        n_samples_per_r[-1] += n_samples - n_samples_per_r.sum()

    results = []
    for i, (r, n) in enumerate(zip(radii, n_samples_per_r)):
        s = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = r * np.cos(s) * scale_factor
        y = r * np.sin(s) * scale_factor
        results.append(np.stack([np.ones_like(s) * (i), s, y, x], axis=1))
    # b = (final_radius - initial_radius) / (2 * np.pi * n_turns)
    # return results
    return (
        pl.DataFrame(np.concatenate(results, axis=0), schema=["circ", "phi", "y", "x"])
        .with_columns(pl.col.circ.cast(pl.UInt16))
        .sort("phi")
    )


def _racecar_path_length(circle_radius, lane_length):
    circle_path_length = circle_radius * 2 * np.pi
    lane_path_length = 2 * lane_length

    total_length = circle_path_length + lane_path_length
    return total_length


def _circle_perc(circle_radius, lane_length):
    circle_path_length = circle_radius * 2 * np.pi
    lane_path_length = 2 * lane_length

    total_length = circle_path_length + lane_path_length
    return circle_path_length / total_length


def get_racecar_lane(
    n_samples,
    circle_radius=1,
    lane_length_perc=0.0,
    lane_length=None,
    circ_perc=None,
    SHIFT=0,
):
    # SHIFT = 0.5  # [0.0, 1.0]
    # r = circle_radius
    # l = lane_length

    circle_path_length = circle_radius * 2 * np.pi
    if lane_length is None:
        lane_length = circle_path_length / (1 - lane_length_perc) * lane_length_perc / 2

    lane_path_length = 2 * lane_length

    total_length = circle_path_length + lane_path_length

    if circ_perc is None:
        circ_perc = circle_path_length / total_length

    dl = total_length / n_samples
    dphi = dl / circle_radius

    linear_positions = np.linspace(0, total_length, n_samples, endpoint=False)
    norm_positions = linear_positions / total_length

    circle0_label = norm_positions[linear_positions < (circle_path_length / 2)]
    lane0_label = norm_positions[
        np.logical_and(
            linear_positions >= (circle_path_length / 2),
            linear_positions < ((circle_path_length / 2) + lane_length),
        )
    ]
    circle1_label = norm_positions[
        np.logical_and(
            linear_positions >= (circle_path_length / 2) + lane_length,
            linear_positions < (circle_path_length + lane_length),
        )
    ]
    lane1_label = norm_positions[linear_positions >= (circle_path_length + lane_length)]

    circle0 = (
        (linear_positions[linear_positions < (circle_path_length / 2)]) * dphi / dl
        - np.pi / 2
    )
    lane0 = (
        linear_positions[
            np.logical_and(
                linear_positions >= (circle_path_length / 2),
                linear_positions < ((circle_path_length / 2) + lane_length),
            )
        ]
        - (circle_path_length / 2)
        - lane_length / 2
    ) * -1
    circle1 = (
        linear_positions[
            np.logical_and(
                linear_positions >= (circle_path_length / 2) + lane_length,
                linear_positions < (circle_path_length + lane_length),
            )
        ]
        - lane_length
    ) * dphi / dl - np.pi / 2
    lane1 = (
        linear_positions[linear_positions >= (circle_path_length + lane_length)]
        - circle_path_length
        - lane_length
        - lane_length / 2
    )

    xc0 = circle_radius * np.cos(circle0) + lane_length / 2
    yc0 = circle_radius * np.sin(circle0)  # + lane_length / 2
    t2_c0 = np.linspace(0, 1, len(xc0), endpoint=False) * (circ_perc / 2) - SHIFT

    xl0 = lane0
    yl0 = np.ones_like(lane0) * circle_radius
    t2_l0 = (
        np.linspace(
            0,  # circle_path_length / 2,
            1,  # circle_path_length / 2 + lane_path_length,
            len(xl0),
            endpoint=False,
        )
        * ((1 - circ_perc) / 2)
        + (circ_perc / 2)
        - SHIFT
    )

    xc1 = circle_radius * np.cos(circle1) - lane_length / 2
    yc1 = circle_radius * np.sin(circle1)  # - lane_length / 2
    t2_c1 = (
        np.linspace(
            0,  # circle_path_length / 2 + lane_path_length,
            1,  # circle_path_length + lane_path_length,
            len(xc1),
            endpoint=False,
        )
        * (circ_perc / 2)
        + 1 / 2
    ) - SHIFT

    xl1 = lane1
    yl1 = np.ones_like(lane1) * circle_radius * -1
    t2_l1 = (
        np.linspace(
            0,  # circle_path_length + lane_path_length,
            1,  # circle_path_length + lane_path_length * 2,
            len(xl1),
            endpoint=False,
        )
        * ((1 - circ_perc) / 2)
        + 1 / 2
        + (circ_perc / 2)
    ) - SHIFT

    x = np.concatenate([xc0, xl0, xc1, xl1])
    y = np.concatenate([yc0, yl0, yc1, yl1])
    t = np.concatenate([circle0_label, lane0_label, circle1_label, lane1_label])
    t2 = np.concatenate([t2_c0, t2_l0, t2_c1, t2_l1]) % 1.0
    return pl.DataFrame({"x": x, "y": y, "t": t, "t2": t2})
    # return x, y, t, t2


def get_racecar_lanes_(
    n_samples,
    n_lanes,
    initial_radius=0.2,
    final_radius=1.0,
    lane_length=4,
    # lane_length_perc=None,
    scale_factor=500,
):
    radii = np.linspace(final_radius, initial_radius, n_lanes)
    path_lengths = np.array([_racecar_path_length(r, lane_length) for r in radii])
    percent_path = path_lengths / path_lengths.sum()
    n_samples_per_r = (percent_path * n_samples).round().astype(np.int_)
    if n_samples_per_r.sum() != n_samples:
        # n_samples_per_r[-1] += n_samples - n_samples_per_r.sum()
        n_samples_per_r[0] += n_samples - n_samples_per_r.sum()

    average_circ_perc = _circle_perc(radii[n_lanes // 2], lane_length)
    res = []
    for i, (r, n_samples_lane) in enumerate(zip(radii, n_samples_per_r)):
        res.append(
            get_racecar_lane(
                n_samples=n_samples_lane,
                circle_radius=r,
                lane_length=lane_length,
                circ_perc=average_circ_perc,
            )
        )

    return pl.concat(res).with_columns(
        pl.col("x") * scale_factor, pl.col("y") * scale_factor
    )


# def get_racecar_lanes(
#     n_samples,
#     circle_perc=0.5,
#     n_lanes=4,
#     scale_factor=500,
# ):
#     initial_radius = 1.0
#     circle_path = 2 * np.pi * initial_radius
#     lanes_path = circle_path * (1 - circle_perc)
#     total_path = circle_path + lanes_path

#     path_lengths = np.array([_racecar_path_length(r, lane_length) for r in radii])
#     percent_path = path_lengths / path_lengths.sum()
#     n_samples_per_r = (percent_path * n_samples).round().astype(np.int_)
#     if n_samples_per_r.sum() != n_samples:
#         # n_samples_per_r[-1] += n_samples - n_samples_per_r.sum()
#         n_samples_per_r[0] += n_samples - n_samples_per_r.sum()

#     average_circ_perc = _circle_perc(radii[n_lanes // 2], lane_length)
#     res = []
#     for i, (r, n_samples_lane) in enumerate(zip(radii, n_samples_per_r)):
#         x, y, t, t2 = get_racecar_lane(
#             n_samples=n_samples_lane,
#             circle_radius=r,
#             lane_length=lane_length,
#             circ_perc=average_circ_perc,
#         )
#         res.append(pl.DataFrame({"x": x, "y": y, "t": t, "t2": t2, "lane": i}))
#     return pl.concat(res).with_columns(
#         pl.col("x") * scale_factor, pl.col("y") * scale_factor
#     )


# # %%
# n_samples = 700
# circle_perc = 0.5
# n_lanes = 5
# scale_factor = 500
# initial_radius = 1.0
# circle_path = 2 * np.pi * initial_radius
# total_path = circle_path / circle_perc
# lanes_path = total_path * (1 - circle_perc)
# n_samples_inner = 100


# def dist_between_nodes(dr, n_lanes=50, r=1.0, lanes_path=2, n_samples=3000):
#     return (
#         np.sum(
#             np.linspace(r, r + ((n_lanes - 1) * dr), n_lanes) * 2 * np.pi + lanes_path,
#             axis=0,
#         )
#         / n_samples
#     )


# cost_func = lambda x: np.abs(x - dist_between_nodes(x))

# from scipy.optimize import minimize

# dr = 0.01

# dp = dist_between_nodes(
#     dr,
#     r=initial_radius,
#     lanes_path=lanes_path,
#     n_lanes=n_lanes,
#     n_samples=n_samples,
# )


# res = minimize(cost_func, 1.0)
# res


# # %%
def plot_racecar_lanes(
    n_samples=100,
    n_lanes=4,
    initial_radius=0.2,
    final_radius=1.0,
    scale_factor=1.0,
    lane_length=2,
    hue="t2",
):
    import seaborn as sns

    sns.scatterplot(
        get_racecar_lanes_(
            n_samples=n_samples,
            n_lanes=n_lanes,
            initial_radius=initial_radius,
            final_radius=final_radius,
            scale_factor=scale_factor,
            lane_length=lane_length,
        ),
        x="x",
        y="y",
        hue=hue,
        palette="turbo",
        legend=False,
    )
    plt.gca().set_aspect("equal")


def fill_concentric_circles(
    n_samples=52, n_inner=12, scale_factor=1.0, initial_radius=1.0
):
    initial_path = 2 * np.pi * initial_radius
    delta = initial_path / n_inner

    next_path = 2 * np.pi * (initial_radius + delta)
    path_length_diff = next_path - initial_path

    total_samples = 0
    i = 0
    radii = []
    n_lane_samples = []
    while total_samples < n_samples:
        lane_samples = int(np.round(n_inner + (path_length_diff * i / delta)))
        total_samples += lane_samples
        radii.append(initial_radius + i * delta)
        n_lane_samples.append(lane_samples)
        i += 1
    n_lane_samples[-1] = n_lane_samples[-1] - (total_samples - n_samples)

    res = []
    for i, (r, n_samples_lane) in enumerate(zip(radii, n_lane_samples)):
        x, y, t, t2 = get_racecar_lane(
            n_samples=n_samples_lane,
            circle_radius=r,
            lane_length=0.0,
            circ_perc=None,
        )
        res.append(pl.DataFrame({"x": x, "y": y, "t": t, "t2": t2, "lane": i}))
    return pl.concat(res).with_columns(
        pl.col("x") * scale_factor, pl.col("y") * scale_factor
    )


def fill_concentric_lanes(
    n_samples=52, n_inner=12, scale_factor=1.0, initial_radius=1.0, circle_perc=0.5
):
    circle_path = 2 * np.pi * initial_radius
    total_path = circle_path / circle_perc
    # lanes_path = total_path - circle_path

    delta = total_path / n_inner

    next_path = 2 * np.pi * (initial_radius + delta) / circle_perc
    path_length_diff = next_path - total_path

    total_samples = 0
    i = 0
    radii = []
    n_lane_samples = []
    while total_samples < n_samples:
        lane_samples = int(np.round(n_inner + (path_length_diff * i / delta)))
        total_samples += lane_samples
        radii.append(initial_radius + i * delta)
        n_lane_samples.append(lane_samples)
        i += 1
    n_lane_samples[-1] = n_lane_samples[-1] - (total_samples - n_samples)

    res = []
    for i, (r, n_samples_lane) in enumerate(zip(radii, n_lane_samples)):
        x, y, t, t2 = get_racecar_lane(
            n_samples=n_samples_lane,
            circle_radius=r,
            lane_length=0.0,
            circ_perc=None,
        )
        res.append(pl.DataFrame({"x": x, "y": y, "t": t, "t2": t2, "lane": i}))
    return pl.concat(res).with_columns(
        pl.col("x") * scale_factor, pl.col("y") * scale_factor
    )


def _total_path_length(r, lane_length):
    return 2 * np.pi * r + 2 * lane_length


def lane_generator(n_samples=12, r=1.0, circle_perc=0.5, lane_length=1.0):
    # path_length = _total_path_length(r=r, circle_perc=circle_perc)
    path_length = 2 * lane_length + 2 * np.pi * r

    delta = path_length / n_samples

    while True:
        yield get_racecar_lane(
            n_samples=n_samples,
            circle_radius=r,
            # lane_length_perc=(1 - circle_perc),
            lane_length=lane_length,
            circ_perc=None,
        )
        r = r + delta
        new_path_length = _total_path_length(r=r, lane_length=lane_length)
        n_samples = int(np.round(new_path_length / delta))


def lane_number_generator(n_samples=12, r=1.0, circle_perc=0.5, lane_length=1.0):
    """circ_perc only for feature mapping, i. e. percentage of colormap range used for circles across all lanes."""
    yield n_samples, _total_path_length(r=r, lane_length=lane_length), r

    path_length = 2 * lane_length + 2 * np.pi * r
    delta = path_length / n_samples

    while True:
        r = r + delta
        new_path_length = _total_path_length(r=r, lane_length=lane_length)
        n_samples = int(np.round(new_path_length / delta))
        yield n_samples, new_path_length, r


# TODO: use for neighborhood plots
def fill_concentric_lanes_redistribute(
    n_samples=52,
    n_inner=12,
    scale_factor=1.0,
    initial_radius=1.0,
    initial_lane_length_perc=0.5,
    lane_length=None,
    shift_circle_units=0,
    shift_lane_units=0,
):
    if lane_length is None:
        # circle_perc = 0.1
        lane_length = 2 * np.pi * initial_radius * initial_lane_length_perc

    rs = []
    path_lengths = []
    n_total = 0
    for i, (n_samples_lane, length_total_lane, r) in enumerate(
        lane_number_generator(
            n_samples=n_inner, r=initial_radius, lane_length=lane_length
        )
    ):
        # print(i, n_samples_lane, length_total_lane)
        n_total += n_samples_lane
        rs.append(r)
        path_lengths.append(length_total_lane)
        if n_total >= n_samples:
            break
    # print(n_total, n_samples)
    total_path = np.sum(path_lengths)
    circle_percentages = [2 * np.pi * r / total for r, total in zip(rs, path_lengths)]
    avg_circle_percentage = circle_percentages[len(circle_percentages) // 2]
    shift = shift_circle_units * avg_circle_percentage + shift_lane_units * (
        1 - avg_circle_percentage
    )

    dt = total_path / n_samples
    new_n_samples = (
        (np.array(path_lengths) / total_path * n_samples).round().astype(np.int_)
    )

    if sum(new_n_samples) != n_samples:
        new_n_samples[-1] += n_samples - sum(new_n_samples)
    res = []
    for i, (r, n_sample_lane) in enumerate(
        zip(
            rs,
            new_n_samples,
        )
    ):
        res.append(
            get_racecar_lane(
                n_samples=n_sample_lane,
                circle_radius=r,
                lane_length=lane_length,
                SHIFT=shift,
                # circ_perc=np.min(circle_percentages),
                circ_perc=avg_circle_percentage,
            )
            .with_columns(pl.lit(i).alias("lane"))
            .with_columns(pl.col("x", "y") / dt)
        )

    return pl.concat(res)


def arrange_in_circle_by_feature(
    data_w_index: pl.DataFrame,
    feature: str = "NormalizedCCP",
    tolerance=0.04,
    feature_range=(0.0, 2 * np.pi),
    return_empty=True,
    space_multiplier=3,
    n_inner=12,
    initial_radius=1.0,
    max_iterations=1000,
):
    data = data_w_index[feature]
    feature_name = data.name
    n_samples = len(data)
    rng_low, rng_high = feature_range
    assert ((data >= rng_low) & (data <= rng_high)).all(), (
        "data has values outside of feature range."
    )

    min_spaces = (
        data.cut(np.linspace(0, 1, 12)[1:-1])
        .to_frame()
        .group_by(feature_name)
        .len("n_samples")
        .with_columns(pl.col("n_samples").max().alias("total"))
        .sum()["total"][0]
    )

    n_spaces = min_spaces * space_multiplier

    spaces = (
        fill_concentric_lanes_redistribute(
            n_spaces, n_inner=n_inner, lane_length=0, initial_radius=initial_radius
        )
        .with_row_index("index_spaces")
        .sort("t2")
        .with_columns(((rng_high - rng_low) * pl.col("t2") + rng_low).alias("t2"))
    )
    # TODO replace with _map_with... function from below.
    remappings = pl.DataFrame(
        schema={"index_spaces": pl.UInt32, "index_samples": pl.UInt32}
    )

    current_data = pl.DataFrame(data).with_row_index("index_samples").clone()
    remaining_spaces = spaces.clone()

    for i in range(max_iterations):
        lowest_lane = remaining_spaces["lane"].min()
        df_lane = remaining_spaces.filter(pl.col("lane") == lowest_lane)
        rem_lane = df_lane.join_asof(
            current_data,
            left_on="t2",
            right_on=feature_name,
            tolerance=tolerance,
            strategy="nearest",
        )

        rem_lane_no_match = rem_lane.filter(pl.col("index_samples").is_null())

        rem_lane_match = rem_lane.filter(pl.col("index_samples").is_not_null())

        rem_lane_duplicates = rem_lane_match.filter(
            pl.col("index_samples").is_duplicated()
        )

        if rem_lane_duplicates.height == 0:
            rem_lane_clean = rem_lane_match
        else:
            closest_matches = (
                rem_lane_match.filter(pl.col("index_samples").is_duplicated())
                .select(
                    "index_spaces",
                    "index_samples",
                    (pl.col("t2") - pl.col(feature_name)).abs(),
                )
                .group_by("index_samples")
                .agg(pl.col("index_spaces").sort_by("t2").first())
            )["index_spaces"]
            rem_lane_clean = pl.concat(
                [
                    rem_lane_match.filter(
                        pl.col("index_samples").is_duplicated().not_()
                    ),
                    rem_lane_duplicates.filter(
                        pl.col("index_spaces").is_in(closest_matches)
                    ),
                ]
            )
        remappings = pl.concat(
            [remappings, rem_lane_clean[["index_spaces", "index_samples"]]]
        )
        remaining_spaces = remaining_spaces.filter(
            pl.col("index_spaces").is_in(rem_lane_no_match["index_spaces"]).not_(),
        ).filter(pl.col("index_spaces").is_in(rem_lane_clean["index_spaces"]).not_())

        current_data = current_data.filter(
            pl.col("index_samples").is_in(remappings["index_samples"]).not_()
        )
        if current_data.height == 0:
            print("breaking cause no data left.")
            print(i)
            break
        if remaining_spaces.height == 0:
            print("breaking cause no space left.")
            print(i)
            break
        if remappings.filter(pl.col("index_samples").is_duplicated()).height > 0:
            print("duplicate remapping.")
            print(i)
            break
    if i == (max_iterations - 1):
        print("maximum number of iterations reached.")
    # return data, remappings, spaces
    print(f"{len(remappings)}/{n_samples} placed")
    return spaces.join(remappings, on="index_spaces", how="left").join(
        data_w_index.with_row_index("index_samples"), on="index_samples", how="left"
    )


def _map_feature_to_spaces(
    data: pl.Series,  # feature values
    spaces: pl.DataFrame,  # positions and "time" axis (0.0, 1.0)
    feature_range: tuple[float, float] = (0.0, 1.0),
    spaces_time_col: str = "t2",
    max_iterations: int = 1000,
    tolerance: float = 0.1,
):
    """Map set of objects via a feature to spaces."""
    feature_name = data.name
    n_samples = len(data)
    rng_low, rng_high = feature_range
    remappings = pl.DataFrame(
        schema={"index_spaces": pl.UInt32, "index_samples": pl.UInt32}
    )

    spaces = (
        spaces.with_row_index("index_spaces")
        .sort(spaces_time_col)
        .with_columns(
            ((rng_high - rng_low) * pl.col(spaces_time_col) + rng_low).alias(
                spaces_time_col
            )
        )
    )

    current_data = pl.DataFrame(data).clone()
    if "index_samples" not in current_data:
        current_data = current_data.with_row_index("index_samples")
    remaining_spaces = spaces.clone()

    for i in range(max_iterations):
        lowest_lane = remaining_spaces["lane"].min()
        df_lane = remaining_spaces.filter(pl.col("lane") == lowest_lane)
        rem_lane = df_lane.join_asof(
            current_data,
            left_on=spaces_time_col,
            right_on=feature_name,
            tolerance=tolerance,
            strategy="nearest",
        )

        rem_lane_no_match = rem_lane.filter(pl.col("index_samples").is_null())

        rem_lane_match = rem_lane.filter(pl.col("index_samples").is_not_null())

        rem_lane_duplicates = rem_lane_match.filter(
            pl.col("index_samples").is_duplicated()
        )

        if rem_lane_duplicates.height == 0:
            rem_lane_clean = rem_lane_match
        else:
            closest_matches = (
                rem_lane_match.filter(pl.col("index_samples").is_duplicated())
                .select(
                    "index_spaces",
                    "index_samples",
                    (pl.col(spaces_time_col) - pl.col(feature_name)).abs(),
                )
                .group_by("index_samples")
                .agg(pl.col("index_spaces").sort_by(spaces_time_col).first())
            )["index_spaces"]
            rem_lane_clean = pl.concat(
                [
                    rem_lane_match.filter(
                        pl.col("index_samples").is_duplicated().not_()
                    ),
                    rem_lane_duplicates.filter(
                        pl.col("index_spaces").is_in(closest_matches)
                    ),
                ]
            )
        remappings = pl.concat(
            [remappings, rem_lane_clean[["index_spaces", "index_samples"]]]
        )
        remaining_spaces = remaining_spaces.filter(
            pl.col("index_spaces").is_in(rem_lane_no_match["index_spaces"]).not_(),
        ).filter(pl.col("index_spaces").is_in(rem_lane_clean["index_spaces"]).not_())

        current_data = current_data.filter(
            pl.col("index_samples").is_in(remappings["index_samples"]).not_()
        )
        if current_data.height == 0:
            exit_message = "breaking cause no data left."
            break
        if remaining_spaces.height == 0:
            exit_message = "breaking cause no space left."
            break
        if remappings.filter(pl.col("index_samples").is_duplicated()).height > 0:
            exit_message = "duplicate remapping."
            break

    if i == (max_iterations - 1):
        logger.info("maximum number of iterations reached.")

    logger.info(f"finished layout after {i} iterations.")
    logger.info(exit_message)
    logger.info(f"{len(remappings)}/{len(spaces)} spaces occupied.")
    logger.info(f"{len(remappings)}/{n_samples} objects placed.")
    return remappings


def plot_fill_lanes_red(
    n_samples=1000, initial_radius=1.0, n_inner=30, initial_lane_length_perc=1.4
):
    import seaborn as sns

    df = fill_concentric_lanes_redistribute(
        n_samples,
        initial_radius=initial_radius,
        n_inner=n_inner,
        initial_lane_length_perc=initial_lane_length_perc,
    )
    return df


def plot_(df, hue="t2"):
    import seaborn as sns

    sns.scatterplot(df, x="x", y="y", hue="t2", legend=False)
    plt.gca().set_aspect("equal")
    return df


# plot_fill_lanes_red(n_samples=50, initial_lane_length_perc=1, n_inner=49)


def get_rect_grid(
    n_samples=100, aspect_ratio=1.0, left_to_right=True, scale_factors=None
):
    n = n_samples
    ny = int(np.ceil(np.sqrt(n * aspect_ratio)))
    nx = int(np.ceil(n / ny))
    cy = np.linspace(0, 1, ny, endpoint=False)
    cx = np.linspace(0, 1, nx, endpoint=False)
    dy = 1 / ny
    dx = 1 / nx
    py, px = np.meshgrid(cy, cx)
    if left_to_right:
        px, py = py, px
    if scale_factors is None:
        scale_factors = (1 / dy, 1 / dx)
    return pl.DataFrame({"y": py.flatten(), "x": px.flatten()}).with_columns(
        pl.col("y") * scale_factors[0], pl.col("x") * scale_factors[1]
    )


def get_column_grid(
    n_samples=100,
    n_columns=10,
    left_to_right=False,
    scale_factors: tuple[float, float] | None = None,
):
    n = n_samples
    n_rows = int(np.ceil(n_samples / n_columns))
    cy = np.linspace(0, 1, n_rows, endpoint=False)
    cx = np.linspace(0, 1, n_columns, endpoint=False)
    dy = 1 / n_rows
    dx = 1 / n_columns
    py, px = np.meshgrid(cy, cx)
    if left_to_right:
        px, py = py, px
    if scale_factors is None:
        scale_factors = (1 / dy, 1 / dx)
    return pl.DataFrame({"y": py.flatten(), "x": px.flatten()}).with_columns(
        pl.col("y") * scale_factors[0], pl.col("x") * scale_factors[1]
    )


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def ccp_nuclei_band(
    lane_length=50,
    n_samples=1000,
    n_inner=300,
    shift_circle_units=0,
    shift_lane_units=0,
):
    points = (
        fill_concentric_lanes_redistribute(
            n_samples,
            initial_radius=0.5,
            n_inner=n_inner,
            lane_length=lane_length,
            shift_circle_units=shift_circle_units,
            shift_lane_units=shift_lane_units,
        )
        # .with_columns(((pl.col("t2") + 0.5) % 2).alias("t3"))
        .pipe(plot_)
        .sort("t2")
        .select("x", "y", "t2")
    )
    return points


def ccp_nuclei_example():
    points = ccp_nuclei_band(lane_length=20)
    imgs = get_images(len(points), center=True)
    extent = np.max(imgs[0].spatial.extent[1:])
    points = points.with_columns(pl.col("x", "y") * extent)

    imgs_trans = translate_images(imgs, points.select("x", "y").rows(named=True))
    canvas = canvas_from_images(imgs_trans)
    return canvas


def circles_example(n_samples=300, extent=32.5, n_circles=9, scale_factor=0.5):
    import napari
    import seaborn as sns

    image_extent = (np.sqrt(n_samples) * extent) * scale_factor
    circs = get_concentric_circles(
        n_samples=n_samples, scale_factor=image_extent, n_circles=n_circles
    )

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    sns.scatterplot(circs, x="x", y="y", hue="phi", palette="turbo", legend=False)

    imgs = get_images(
        n_samples,
        shape=(2, 100, 100),
        scale=(1.0, 0.325, 0.325),
        value_coord=("ch0", "ch1"),
    )
    imgs_trans = translate_images(
        imgs,
        translations=circs.sort("phi").select("y", "x").rows(named=True),
    )

    canvas = canvas_from_images(imgs_trans)
    imshow(canvas)


def imshow(img: Image, viewer=None, is_label=False, **layer_kwargs):
    import napari

    if viewer is None:
        viewer = napari.Viewer()
    if is_label or isinstance(img, LabelImage):
        if len(img._value.coord) == 0:
            name = img.o._squeeze_buffer
        else:
            name = img.o.coord.to_list()
        viewer.add_labels(
            data=img.data,
            scale=img.spatial.scale,
            name=name,
            translate=img.spatial.iorigin,
            **layer_kwargs,
        )
    else:
        viewer.add_image(
            data=img.data,
            scale=img.spatial.scale,
            channel_axis=0,
            name=img._value.coord,
            translate=img.spatial.iorigin,
            blending="additive",
            **layer_kwargs,
        )
    return viewer


def spiral_example():
    import napari
    import seaborn as sns

    n = 700
    pos_ = get_spiral(n, n_turns=13.5, scale_factor=400, initial_radius=0.2)
    sns.scatterplot(
        pos_,
        x="x",
        y="y",
        hue="phi",
        palette="turbo",
        legend=True,
    )
    plt.gca().set_aspect("equal")
    plt.show()
    pos = pos_.sort("phi").select("y", "x")
    # shape = (1, 100, 100)
    shape = (1, 1, 100, 100)
    # value_coord = "ch0"
    value_coord = ["ch0"]

    # plt.scatter(*pos[["y", "x"]].to_numpy().T, label="spiral")

    imgs = translate_images(
        get_images(n, shape=shape, value_coord=value_coord),
        pos.rows(named=True),
    )

    img = imgs[0]
    canvas = canvas_from_images(imgs)

    viewer = imshow(canvas)


def nested_images_example():
    n_imgs = 1600
    n_groups = 16
    n_inner = int(np.ceil(n_imgs / n_groups))

    canvases = []
    for i in range(n_groups):
        lbls = [l.center() for l in get_label_images(n_inner)]
        extent = np.max([lbls[0].x.extent, lbls[0].y.extent])
        dt_path = 0.187  # manually chosen
        scale_factor = extent / dt_path
        arrangement = (
            get_racecar_lanes_(
                n_samples=n_inner,
                n_lanes=5,
                initial_radius=0.2,
                final_radius=1.0,
                scale_factor=scale_factor,
                lane_length=0,
            )
            .sort("t2")
            .select("y", "x")
        ).rows(named=True)
        lbls_trans = translate_images(lbls, arrangement)
        canvas = canvas_from_images(lbls_trans).center()
        canvases.append(canvas)
    super_extent = np.max([canvases[0].x.extent, canvases[0].y.extent])
    super_dt_line = 1 / 4  # fourth of sidelength of square grid
    super_scale_factor = super_extent / super_dt_line
    # super_arrangement = get_rect_grid(n_groups, scale_factor=super_scale_factor).rows(

    super_arrangement = get_rect_grid(n_groups, aspect_ratio=super_scale_factor).rows(
        named=True
    )
    canvases_trans = translate_images(canvases, super_arrangement)
    super_canvas = canvas_from_images(canvases_trans).center()
    return super_canvas


def neighborhood_example(n_neighbors=30):
    neighbor_imgs = get_label_images(n_neighbors)
    img = get_label_images(1)[0]

    extent = np.max([img.spatial.y.extent, img.spatial.x.extent])

    neighbor_arrange = (
        fill_concentric_circles(n_samples=n_neighbors, initial_radius=2.0)
        .sort("t2")
        .select("y", "x")
        * extent
    ).rows(named=True)
    imgs_trans = []
    for (
        im,
        trans,
    ) in zip(
        [img] + list(neighbor_imgs), [{"y": 0.0, "x": 0.0}] + list(neighbor_arrange)
    ):
        imgs_trans.append(im.translate(**trans))

    return canvas_from_images(imgs_trans)


def get_blank(img, m=0):
    blank = Image.from_array(
        np.ones((img.shape[0], *(1, 10, 10))) * 5,
        scale=img.spatial.scale,
        region_id=("blank",),
        value_coord=img.c.coord,
        m=m,
    )
    return blank


def sorted_imgs_with_blanks(df_sorted, imgs):
    blank = get_blank(imgs[0])

    counts_ = (
        df_sorted.select(
            pl.col("NormalizedCCP").cut(np.linspace(0, 2 * np.pi, 10, endpoint=False))
        )
        .group_by("NormalizedCCP", maintain_order=True)
        .agg(pl.len())
        .with_columns((pl.col("len").max() - pl.col("len")).alias("len_diff"))
    )
    counts = counts_.select(
        pl.col("len").sum(),
        pl.col("len_diff").sum(),
        pl.sum_horizontal(pl.col("len").sum(), pl.col("len_diff").sum()).alias("total"),
    ).rows(named=True)[0]

    df_coord = pl.DataFrame(
        {"NormalizedCCP": np.linspace(0, 2 * np.pi, counts["total"], endpoint=False)}
    )

    df_indices = df_sorted.join_asof(
        df_coord.with_row_index("index_filled"), on="NormalizedCCP", strategy="nearest"
    )
    sort_index_filled = df_coord.with_row_index("index_filled").join(
        df_indices, on="index_filled", how="full"
    )[["index", "index_filled"]]

    imgs_sort = [
        imgs[index] if index is not None else blank
        for index in sort_index_filled["index"]
    ]
    return imgs_sort


def remap_close_to_center(df_spaces_with_blanks, tolerance=1 / 24):
    tolerance = 1 / 24
    remappings = pl.DataFrame(schema={"index": pl.UInt32, "outer_index": pl.UInt32})

    for i in df_spaces_with_blanks["lane"].unique().sort()[:-1]:
        df_blanks_inner = df_spaces_with_blanks.filter(pl.col("is_blank")).filter(
            pl.col("lane") == i
        )
        df_spaces_outer = (
            df_spaces_with_blanks.filter(pl.col("is_blank").not_())
            .filter(pl.col("lane") > i)
            .filter(pl.col("index").is_in(remappings["outer_index"]).not_())
        )

        new_remappings = df_blanks_inner.join_asof(
            df_spaces_outer.select(pl.col("index").alias("outer_index"), pl.col("t2")),
            on="t2",
            strategy="nearest",
            tolerance=tolerance,
        ).select("index", "outer_index")

        duplicate_mappings = (
            new_remappings.filter(pl.col("outer_index").is_null().not_())
            .filter(pl.col("outer_index").is_duplicated())
            .group_by("outer_index")
            .agg(pl.col("index"))
            .with_columns(pl.col("index").list.slice(1))
            .explode("index")
        )

        remappings = pl.concat(
            [
                remappings,
                new_remappings.filter(
                    pl.col("index").is_in(duplicate_mappings["index"]).not_()
                ),
            ],
            how="vertical",
        )

    remappings = remappings.filter(pl.col("outer_index").is_null().not_())
    remappings_rev = remappings.select(
        pl.col("outer_index").alias("index"), pl.col("index").alias("outer_index")
    )
    all_remappings = pl.concat([remappings, remappings_rev], how="vertical")

    return df_spaces_with_blanks.join(
        all_remappings, on="index", how="left"
    ).with_columns(
        pl.when(pl.col("outer_index").is_null()).then("index").otherwise("outer_index"),
        pl.when(pl.col("outer_index").is_null())
        .then("is_blank")
        .otherwise(pl.col("is_blank").not_()),
    )


# TODO: should SpatialCoordinate be refactored like this?
# @dataclass(frozen=True)
# class Coordinate:
#     __coord_dim__: ClassVar[Literal[*ALL_DIMS]]
#     __coord_dtype__: ClassVar[pl.DataType]
#     dim: Literal[*ALL_DIMS]
#     # _parent: "Coordinate | None"

#     def __init__(self, dim, parent=None):
#         self._dim = dim
#         self._parent = parent


# @dataclass(frozen=True)
# class CategoricalCoordinate(Coordinate):
#     __coord_dim__: ClassVar[Literal[*CAT_DIMS]]
#     __coord_dtype__: ClassVar[pl.DataType]


# @dataclass(frozen=True)
# class ContinuousCoordinate(Coordinate):
#     __coord_dim__: ClassVar[Literal[*DIMS]]
#     __coord_dtype__: ClassVar[pl.Float64]
#     origin: float


# @dataclass(frozen=True)
# class BoundedCoordinate(ContinuousCoordinate):
#     extent: float

#     @property
#     def upper_bound(self):
#         return self.origin + self.extent


# @dataclass(frozen=True)
# class SampledBoundedCoordinate(BoundedCoordinate):
#     scale: float
#     _coord: pl.Series = None

#     @property
#     def shape(self):
#         if self.extent == 0:
#             return 0
#         if self.scale == 0:
#             return None
#         shp = self.extent / self.scale
#         return int(round(shp))

#     @property
#     def coord(self):
#         if self._coord is None:
#             object.__setattr__(
#                 self,
#                 "_coord",
#                 pl.Series(
#                     self.dim,
#                     np.linspace(self.origin, self.origin + self.extent, self.shape + 1),
#                 ),
#             )
#         return self._coord

#     @property
#     def icoord(self):
#         return self.coord[:-1] + self.iorigin

#     @property
#     def iorigin(self):
#         return self.scale / 2

#     def slice(
#         self, bound: tuple[float, float] | None = None, return_slice: bool = True
#     ) -> Self | tuple[Self, slice]:
#         if bound is None:
#             return self
#         new_index_coordi = (
#             self.icoord.to_frame()
#             .with_row_index()
#             .filter(pl.col(self.dim).is_between(*bound))
#         )
#         new_coordi = new_index_coordi[self.dim]
#         new_origin = new_coordi[0] - self.iorigin
#         new_extent = new_coordi[-1] + self.iorigin - new_origin

#         new_coordinate = self.__class__(
#             extent=new_extent,
#             scale=self.scale,
#             _dim=self.dim,
#             origin=new_origin,
#         )
#         if return_slice:
#             arr_indices = new_index_coordi["index"]
#             slc = slice(arr_indices.min(), arr_indices.max() + 1)
#             return new_coordinate, slc
#         return new_coordinate


# @dataclass(frozen=True, kw_only=False)
# class SpatialCoordinate_(SampledBoundedCoordinate):
#     pass

##################################
# x = MsSampledBoundedAxis(
#     dim="x", scales=(0.325, 0.65, 1.3, 2.6, 5.2, 10.4), extent=650.0
# )
# y = MsSampledBoundedAxis(
#     dim="y", scales=(0.325, 0.65, 1.3, 2.6, 5.2, 10.4), extent=650.0
# )
# z = MsSampledBoundedAxis(dim="z", scales=(1.0, 1.0, 1.0, 2.0, 5.2, 10.4), extent=271.0)
# img = Image(
#     "hi",
#     axes=Axes(
#         (
#             Axis("z", extent=271.0, scale=1.0),
#             Axis("y", extent=650.0, scale=0.65),
#             Axis("x", extent=650.0, scale=0.65),
#         )
#     ),
#     _coord_type=ChannelCoord(("ch_00", "ch_01")),
# )

# msr = MsSampledRegion("ms_r", (z, y, x))
# msr


# rois (roi[o, c, m, z, y, x])


# z_models (z_model)
# t_models (t_model)

# label_objects (o, label)


# rois (roi[cs, bound, m])                   -> coordinate_systems + dimension_bounds + uniform_samplings
# roi_levels (roi[cs, bound m:1])            -> coordinate_systems + dimension_bounds + uniform_sampling = rois + filter(m=level)
# images (roi[])
# sampled_css                                -> coordinate_system + uniform_sampling
# ms_sampled_css                             -> coordinate_system + multiscale_sampling
# samp_bound_css (roi, m:1)                  -> coordinate_system + uniform_sampling + dimension_bounds
# ms_samp_bound_css (roi, m)                 -> coordinate_system + multiscale_sampling + dimension_bounds
# ms_mc_images (roi, m, c)                   -> coordinate_system + multiscale_sampling + dimension_bounds + channels
# ms_images (roi, m, c:1)                    -> ms_mc_images + filter(c==channel)
# mc_images (roi, m:1, c)                    -> ms_mc_images + filter(m==level)
# images (roi, m:1, c:1)                     -> ms_mc_images + filter(m==level) + filter(c==channel)
# ms_mc_label_images (roi, m, o)             -> coordinate_system + multiscale_sampling + dimension_bounds + label_objects
# ... like images
# raster_data_query                  -> roi + raster_coords[c, o] + path_to_specified_arrays (double, lazy, i. e. data not even loaded as dask_array)
# lazy_raster_data                   -> roi + raster_coords[c, o] + dict[raster_coord, da.array]
# mscale_sampled_region -> bounded_region + m * ((shape.m.iz, shape.m.iy, shape.m.ix) | (scale.m.iz, scale.m.iy, scale.m.ix) | (origin.m.iz, origin.m.iy, origin.m.ix))


# region_keys = [
#     "plate",  # coordinate_system ([t], z, y, x)
#     "well",  # coordinate_system ([t], z, y, x)
#     "site",  # coordinate_system ([t], z, y, x)
#     "roi",  # bounded_cs (z, y, x, z.extent, y.extent, x.extent) = roi
#     "ms_lazy_raster",  # roi + multiscale_sampling (can also represent 'sampled_roi' for m:1)
#     "lazy_raster",  # roi + uniform_sampling | ms_lazy_raster + filter(m==level)
#     "ms_lazy_image",  # ms_lazy_raster + (c,) + dict[m, DaskArrayLike[c, z, y, x, dtype]] | dict[m, dict[o, DaskArrayLike[z, y, x, dtype]]]
#     "ms_lazy_label_image",  # ms_lazy_raster + (o, ) + dict[m, DaskArrayLike[o, z, y, x, dtype]] | dict[m, dict[o, DaskArrayLike[z, y, x, dtype]]]
#     "lazy_image",  # (lazy_raster + (c, ) | ms_lazy_image + filter(m==level)) + DaskArrayLike[c, z, y, x, dtype] | dict[c, DaskArrayLike[z, y, x, dytpe]]
#     "lazy_label_image",  # (lazy_label_image + (o, ) | ms_lazy_label_image + filter(m==level)) + DaskArrayLike[o, z, y, x, dtype] | dict[o, DaskArrayLike[z, y, x, dtype]]
#     # ("c.0", "value"), # sample |
#     # ("c.1", "value"), # sample |
#     # ("c.2", "value"), # sample |
#     # "b",     # region + sampled + value=is_part_of_object
#     # "o",     # region + sampled + value=is_part_of_one_of_a_set_of_objects (with convention of one object type -> ...one_of_a_set_of_objects_of_type)
#     # ("o", "label"),      # region + samples + value=is_part_of_a_member_of_object_type
#     # ("o", {"label"}),    # region + samples
# ]


# select:        zeros     background label_object  label_objects     object_type_foreground     identity
#               ("o", {}), ("o", {0}), ("o", {3}), ("o", {1, 2, 5}), ("o", {1, 2, 3, 4, 5, 6}), ("o", {0, 1, 2, 3, 4, 5, 6})
# subtract       identity     o_t_fg


# from zfish.roi.spatial_roi import Roi

# lazy_roi = Roi.from_file(
#     r"Z:\hmax\MARVWY_RESTORED\20220721_ZE4i2_aligned1\imgs\B02_px+0508_py+2478.h5",
#     level=0,
# )

# parent # region # descr.
# ------------------------
# plate  # plate  # region
# plate  # well   # region
# well   # roi    # region
# roi    # b      # region + sampled + masked
# roi    # o      # region + sampled + masked
# o      # label  # region + samples + masked


# RASTER DATA
# raster_data       -> sampled_region + value  # if a value at every location -> dense_array, otherwise -> sparse_array
# image (c)         -> value: float = intensity     (-inf, inf)  # any physical quantity, [0, inf) for concentration measurements as in IF images, [0, 2**16 - 1] effectively for 16-bit images
# binary_mask (b)   -> value: bool  = is_foreground {0, 1}  # partitioning the sampled_region into 2 subregions. 0 is background by convention.
# label_image (o)   -> value: uint  = label         {0, 1, ..., n}  # partitioning the sampled_region in to n+1 subregions. 0 is background by convention.
# dist_transf (dtf) -> value: float = distance      (-inf, inf)  # distance to the mask from which the transform was generated. [0, inf) for unsigned transform.

# RASTER COORDINATES
# channel (c)       -> value: str = useful & UNIQUE identifier. e. g. stain in the case of microscopy data or (stain, acquisition) tuples in a multiplexed setting where the same stain (DAPI) might appear multiple times.
# object_type (o)   -> value: str = useful & UNIQUE identifier of the kind of object segmented in a particular label_image. e. g. cell or nucleus. requires each label_image to only contain one type of object.
# mask_name         -> value: str
# transf_type (dtf) -> value: str

# %%
