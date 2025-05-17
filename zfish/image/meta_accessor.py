# %%
import xarray as xr
import xxhash

from zfish.features.types import SpatialImage

SPATIAL_DIMS = ("z", "y", "x")


@xr.register_dataarray_accessor("meta")
class ImageMetaAccessor:
    def __init__(self, dataarray: SpatialImage):
        self._obj = dataarray

    @property
    def scale(self):
        # return tuple(ImageMetaAccessor._scale_from_coord(self._obj.coords[dim]) if dim_shape > 1 else None for dim, dim_shape in zip(self._obj.dims, self._obj.shape))

        return tuple(
            ImageMetaAccessor._scale_from_coord(self._obj.coords[dim])
            for dim in self._obj.coords.keys()
            if dim in SPATIAL_DIMS
        )

    @property
    def scale_dict(self):
        return dict(
            zip(
                (dim for dim in self._obj.coords.keys() if dim in SPATIAL_DIMS),
                self.scale,
            )
        )

    @property
    def origin_dict(self):
        return dict(
            zip(
                (dim for dim in self._obj.coords.keys() if dim in SPATIAL_DIMS),
                self.origin,
            )
        )

    @property
    def spacing(self):
        return self.scale

    @property
    def translate(self):
        return tuple(
            ImageMetaAccessor._translate_from_coord(self._obj.coords[dim])
            for dim in self._obj.coords.keys()
            if dim not in ["c", "o", "l"]
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
    def _scale_from_coord(coord: xr.DataArray) -> float:
        if coord.ndim == 0 or len(coord) <= 1:
            return 0.0
        return coord[1].item() - coord[0].item()

    @staticmethod
    def _translate_from_coord(coord: xr.DataArray) -> float:
        if coord.ndim == 0:
            return coord.item()
        return coord[0].item()
