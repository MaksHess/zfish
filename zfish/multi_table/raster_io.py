# %%
from dataclasses import astuple, dataclass
from enum import Enum
from typing import Hashable, Literal, Protocol, TypeAlias, runtime_checkable

import dask.array as da
import h5py
import numpy as np
import polars as pl
import polars.selectors as cs
import tqdm
from dask_image.ndfilters import gaussian
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter as gaussian_np

from zfish.features.polars_utils import nest
from zfish.image.pyramid import RasterMeta, lazy_resample_dask_label
from zfish.intensity_normalization.models import Model, lazy_apply_model_to_channel_dask
from zfish.multi_table.schemas import IDX_SEL
from zfish.polars.column_nesting import unnest

ITK_CASTS = {np.dtype("bool"): np.dtype("uint8")}

DaskImage: TypeAlias = tuple[da.Array, RasterMeta]
NumpyImage: TypeAlias = tuple[np.ndarray, RasterMeta]
Image: TypeAlias = tuple[ArrayLike, RasterMeta]


class SpatialDim(Enum):
    Z = -3
    Y = -2
    X = -1


@dataclass(frozen=True, slots=True)
class BoundingBox:
    z: tuple[int | None, int | None] = (0, None)
    y: tuple[int | None, int | None] = (0, None)
    x: tuple[int | None, int | None] = (0, None)
    # m: int | None = None

    @property
    def slice(self) -> tuple[slice, ...]:
        return _bbx_to_slice(self)

    @property
    def origin_index(self) -> tuple[int, ...]:
        return self.z[0], self.y[0], self.x[0]

    @classmethod
    def from_values(
        cls,
        bbx_z_lower: int = 0,
        bbx_z_upper: int | None = None,
        bbx_y_lower: int = 0,
        bbx_y_upper: int | None = None,
        bbx_x_lower: int = 0,
        bbx_x_upper: int | None = None,
    ):
        return cls(
            (bbx_z_lower, bbx_z_upper),
            (bbx_y_lower, bbx_y_upper),
            (bbx_x_lower, bbx_x_upper),
        )

    @classmethod
    def from_flat_tuple(cls, bbx_tuple: tuple[int | None]):
        return cls.from_values(*bbx_tuple)


@dataclass(frozen=True, slots=True)
class MutliscaleBoundingBox:
    levels: dict[int, BoundingBox]

    def slice_level(self, level: int) -> tuple[slice, ...]:
        return self.levels[level].slice


def _bbx_to_slice(bbx: BoundingBox) -> tuple[slice, ...]:
    return tuple(slice(*e) for e in astuple(bbx))


def label_objects_scale_bbx(
    df_m: pl.DataFrame,
    df_label_objects: pl.DataFrame,
    to_level: int | tuple[int, ...],
) -> pl.DataFrame:
    new_m = _bbx_scaler_to_m(df_m)
    if isinstance(to_level, int):
        to_level = (to_level,)
    return pl.concat(
        tuple(
            _join_scaler_to_label_objects(
                df_label_objects=df_label_objects, df_m=new_m, to_level=level
            ).pipe(unnest)
            for level in to_level
        )
    )


def _label_objects_scale_bbx(
    df_m: pl.DataFrame,
    df_label_objects: pl.DataFrame,
    to_level: int,
    from_level: int = 1,
    lowest_level: int | None = None,
) -> pl.DataFrame:
    if lowest_level is None:
        lowest_level = from_level
    new_m = _bbx_scaler_to_m(df_m, from_level=from_level, lowest_level=lowest_level)
    return _join_scaler_to_label_objects(
        df_label_objects=df_label_objects, df_m=new_m, to_level=to_level
    )


def _bbx_scaler_to_m(
    df_m: pl.DataFrame, from_level: int = 1, lowest_level: int = 1
) -> pl.DataFrame:
    new_m = (
        df_m.pipe(nest)
        .with_columns(
            (pl.col("scale").gather(from_level) / pl.col("scale")).alias("bbx.scaler")
        )
        .with_columns(
            (pl.col("scale").gather(lowest_level) / pl.col("scale")).alias(
                "bbx.divisible_by"
            )
        )
        .filter(pl.col("idx.m") <= lowest_level)
        # .select(
        #     sel.idx,
        #     pl.col("scale"),
        #     pl.struct(
        #         pl.lit(1.0).alias("z"), pl.lit(0.65).alias("y"), pl.lit(0.65).alias("x")
        #     ).alias("bbx.scaler")
        #     / pl.col("scale"),
        # )
    )
    return new_m


def _join_scaler_to_label_objects(
    df_label_objects: pl.DataFrame, df_m: pl.DataFrame, to_level: int = 0
) -> pl.DataFrame:
    return (
        df_label_objects.with_columns(
            pl.col("idx.m").replace({1: to_level}).cast(pl.UInt8)
        )
        # .join(df_m, left_on="bbx.m", right_on="idx.m")
        .join(df_m, on="idx.m")
        # .pipe(nest, ".", 1)
        .select(
            IDX_SEL,
            # "bbx.m",
            *[
                (
                    pl.col(f"bbx.{dim}.{bound}").cast(pl.Float64)
                    * pl.col("bbx.scaler").struct.field(f".{dim}")
                )
                .round()
                .cast(pl.Int32)
                for dim in ["z", "y", "x"]
                for bound in ["lower", "upper"]
            ],
        )
        # .pipe(nest, ".", 1)
    )


def get_max_extent(df_bbxs) -> tuple[int, ...]:
    return (
        df_bbxs.select(
            (pl.col(f"bbx.{dim}.upper") - pl.col(f"bbx.{dim}.lower")).alias(
                f"{dim}.extent"
            )
            for dim in ["z", "y", "x"]
        )
        .max()
        .rows()[0]
    )


def get_quantile_extent(df_bbxs, quantile=1.0) -> tuple[int, ...]:
    return (
        df_bbxs.select(
            (pl.col(f"bbx.{dim}.upper") - pl.col(f"bbx.{dim}.lower")).alias(
                f"{dim}.extent"
            )
            for dim in ["z", "y", "x"]
        )
        .quantile(quantile)
        .with_columns(pl.all().ceil().cast(pl.UInt32))
        .rows()[0]
    )


@runtime_checkable
class LazyRasterQueryDask(Protocol):
    def lazy() -> tuple[da.array, RasterMeta]: ...
    def compute() -> tuple[da.array, RasterMeta]: ...


def _meta_from_dset(dset: h5py.Dataset):
    scale = tuple(dset.attrs["element_size_um"])
    origin = tuple([0.0] * len(scale))
    dims = ("z", "y", "x")[-len(scale) :]
    type_ = dset.attrs["img_type"]
    stain = dset.attrs["stain"]
    level = dset.attrs["level"]
    if type_ == "intensity":
        acquisition = dset.attrs["cycle"]
        name = f"{stain}.{acquisition}"
    else:
        name = f"{stain}"
    return RasterMeta(
        name=name,
        path=dset.name,
        type_=type_,
        dims=dims,
        origin=origin,
        scale=scale,
        level=level,
    )


def _safe_to_lazy_dask_image(
    img_or_query: LazyRasterQueryDask | DaskImage | NumpyImage,
) -> DaskImage:
    if isinstance(img_or_query, LazyRasterQueryDask):
        return img_or_query.lazy()
    elif isinstance(img_or_query, tuple) and isinstance(img_or_query[0], da.Array):
        return img_or_query
    elif isinstance(img_or_query, tuple):
        return da.array(img_or_query[0]), img_or_query[1]
    else:
        raise TypeError(f"Cannot handle {type(img_or_query)}")


def _safe_to_numpy_image(
    img_or_query: LazyRasterQueryDask | DaskImage | NumpyImage,
) -> DaskImage:
    if isinstance(img_or_query, LazyRasterQueryDask):
        return img_or_query.compute()
    elif isinstance(img_or_query, tuple) and isinstance(img_or_query[0], da.Array):
        return img_or_query[0].compute(), img_or_query[1]
    elif isinstance(img_or_query, tuple):
        return img_or_query
    else:
        raise TypeError(f"Cannot handle {type(img_or_query)}")


@dataclass
class LabelImageQueryDask(LazyRasterQueryDask):
    root_path: str
    object_type_path: str
    labels: tuple[int, ...] = ()  # empty == select *
    resample_target_size: tuple[int, ...] | None = None
    resample_scale_factors: tuple[float, ...] | None = None
    upcast: bool = True  # don't allow boolean label images (for itk compatibility)
    chunks: int | tuple[int, ...] | str = "auto"

    def lazy(self) -> tuple[da.array, RasterMeta]:
        f = h5py.File(self.root_path)
        dset = f[self.object_type_path]
        da_array = da.from_array(dset, chunks=self.chunks)
        meta = _meta_from_dset(dset)
        if self.upcast:
            if da_array.dtype in ITK_CASTS:
                da_array = da_array.astype(ITK_CASTS[da_array.dtype])
        if self.labels != ():
            da_array = da.isin(da_array, self.labels) * da_array
            meta.from_template(path=None)
        if self.resample_target_size is not None:
            assert (
                self.resample_scale_factors is None
            ), "Only one of `resample_target_size` and `resample_scale_factors` allowed!"
            if not all(
                sz == target_sz
                for sz, target_sz in zip(da_array.shape, self.resample_target_size)
            ):
                da_array, meta = lazy_resample_dask_label(
                    da_array,
                    meta,
                    _resample_scale_factors(da_array.shape, self.resample_target_size),
                )
        if self.resample_scale_factors is not None:
            assert (
                self.resample_target_size is None
            ), "Only one of `resample_target_size` and `resample_scale_factors` allowed!"

            da_array, meta = lazy_resample_dask_label(
                da_array,
                meta,
                self.resample_scale_factors,
            )
        return da_array, meta

    def compute(self) -> tuple[np.ndarray, RasterMeta]:
        da_array, meta = self.lazy()
        return da_array.compute(), meta


@dataclass
class BinaryMaskQueryDask(LazyRasterQueryDask):
    label_image_query: LabelImageQueryDask | DaskImage | NumpyImage
    resample_target_size: tuple[int, ...] | None = None
    upcast: bool = True

    def lazy(self) -> tuple[da.array, RasterMeta]:
        da_array, meta = _safe_to_lazy_dask_image(self.label_image_query)
        da_array = da_array > 0
        meta = meta.from_template(type_="mask")
        if self.upcast:
            if da_array.dtype in ITK_CASTS:
                da_array = da_array.astype(ITK_CASTS[da_array.dtype])
        if self.resample_target_size is not None:
            if not all(
                sz == target_sz
                for sz, target_sz in zip(da_array.shape, self.resample_target_size)
            ):
                da_array, meta = lazy_resample_dask_label(
                    da_array,
                    meta,
                    _resample_scale_factors(da_array.shape, self.resample_target_size),
                )
        return da_array, meta

    def compute(self) -> tuple[np.ndarray, RasterMeta]:
        da_array, meta = self.lazy()
        return da_array.compute(), meta


@dataclass
class FuzzyMaskQueryDask(LazyRasterQueryDask):
    binary_mask_query: BinaryMaskQueryDask | DaskImage | NumpyImage
    gaussian_blur_sigma: int = 1
    order: int = 0

    def lazy(self) -> tuple[da.array, RasterMeta]:
        binary_mask, meta = _safe_to_lazy_dask_image(self.binary_mask_query)
        fuzzy_mask = binary_mask.astype(np.float32)
        fuzzy_mask = gaussian(image=fuzzy_mask, sigma=self.gaussian_blur_sigma)
        meta = meta.from_template(type_="fuzzy_mask")
        return fuzzy_mask, meta

    def compute(self) -> tuple[np.ndarray, RasterMeta]:
        da_array, meta = self.lazy()
        return da_array.compute(), meta


def _resample_scale_factors(da_from_shape, da_to_shape) -> tuple[float, ...]:
    return tuple(
        shp_to / shp_from for shp_from, shp_to in zip(da_from_shape, da_to_shape)
    )


@dataclass
class ImageQueryDask(LazyRasterQueryDask):
    root_path: str
    channel_path: str
    z_model_path: str | None = None
    z_model_label_image_query: LabelImageQueryDask | DaskImage | NumpyImage | None = (
        None
    )
    t_model_correction_factor: float | None = None
    mask_query: BinaryMaskQueryDask | DaskImage | NumpyImage | None = None
    cast_to_original_datatype: bool = True
    chunks: int | tuple[int, ...] | str = "auto"

    def lazy(self) -> tuple[da.array, RasterMeta]:
        f = h5py.File(self.root_path)
        dset = f[self.channel_path]
        channel_da = da.from_array(dset, chunks=self.chunks)
        channel_meta = _meta_from_dset(dset)
        original_datatype = channel_da.dtype

        if self.z_model_path is not None:
            z_model = Model.load(self.z_model_path)

        if self.z_model_path is not None and self.z_model_label_image_query is not None:
            z_model_lbl_da, z_model_lbl_meta = _safe_to_lazy_dask_image(
                self.z_model_label_image_query
            )
            z_model_lbl_da, z_model_lbl_meta = lazy_resample_dask_label(
                z_model_lbl_da,
                z_model_lbl_meta,
                _resample_scale_factors(z_model_lbl_da.shape, channel_da.shape),
            )
            # print(f"{z_model_label_image=}")

        if self.z_model_path is not None:
            channel_da, channel_meta = lazy_apply_model_to_channel_dask(
                z_model, channel_da, channel_meta, z_model_lbl_da
            )
            # print(f"{channel_si=}")

        if self.t_model_correction_factor is not None:
            channel_da = channel_da * self.t_model_correction_factor
            channel_meta = channel_meta.from_template(path=None)
            # print(f"{channel_si=}")

        if self.mask_query is not None:
            mask_da, mask_meta = _safe_to_lazy_dask_image(self.mask_query)
            mask_da, mask_meta = lazy_resample_dask_label(
                mask_da,
                mask_meta,
                _resample_scale_factors(mask_da.shape, channel_da.shape),
            )
            channel_da = channel_da * mask_da

        if self.cast_to_original_datatype:
            channel_da = channel_da.astype(original_datatype)
            # print(f"{channel_si=}")

        return channel_da, channel_meta

    def compute(self) -> tuple[np.ndarray, RasterMeta]:
        channel_da, channel_meta = self.lazy()
        return channel_da.compute(), channel_meta


@dataclass
class MultiChannelQueryDask(LazyRasterQueryDask):
    image_queries: tuple[ImageQueryDask | DaskImage | NumpyImage, ...]

    def lazy(self) -> tuple[da.Array, RasterMeta]:
        imgs_with_metas = [
            _safe_to_lazy_dask_image(img_q) for img_q in self.image_queries
        ]
        channels = da.stack([img_meta[0] for img_meta in imgs_with_metas])
        sample_meta = imgs_with_metas[0][1]
        channels_meta = sample_meta.from_template(
            name=tuple(img_with_meta[1].name for img_with_meta in imgs_with_metas),
            dims=("c",) + tuple(sample_meta.dims),
            type_="multichannel_intensity",
            path=tuple(
                tuple(img_with_meta[1].path for img_with_meta in imgs_with_metas)
            ),
        )
        return channels, channels_meta

    def compute(self) -> tuple[np.ndarray, RasterMeta]:
        channels, channels_meta = self.lazy()
        return channels.compute(), channels_meta


def _apply_bbx(img: Image, bbx: BoundingBox) -> Image:
    arr, meta = img
    spatial_slc = bbx.slice
    full_slc = (
        tuple(slice(None) for _ in range(arr.ndim - len(spatial_slc))) + spatial_slc
    )
    slice_arr = arr[full_slc]
    new_origin = tuple(
        o + scl * bbx_o
        for o, scl, bbx_o in zip(meta.origin, meta.scale, bbx.origin_index)
    )
    slice_meta = meta.from_template(origin=new_origin, path=None)
    return slice_arr, slice_meta


# FIXME: NOT FINISHED BELOW THIS LINE, INCLUDING COMMENTED OUT PART!!!
# FIXME: CONTINUE HERE!!!
@dataclass
class LabelObjectQueryDask(LazyRasterQueryDask):
    object_id: Hashable
    bbx: BoundingBox
    multi_channel_query: (
        MultiChannelQueryDask | DaskImage | NumpyImage
    )  # TODO: check if SpaitalImage can be removed
    channel_to_label: dict[str, int] | None = None
    channel_to_label_image: dict[str, DaskImage | NumpyImage] | None = None
    fuzzy_mask_sigma: int | None = None

    def _process(self, to_image_fn) -> Image:
        channels_arr, channels_meta = to_image_fn(self.multi_channel_query)

        channels_slice_arr, channels_slice_meta = _apply_bbx(
            (channels_arr, channels_meta), self.bbx
        )

        if self.channel_to_label_image is None:
            return channels_slice_arr, channels_slice_meta
        else:
            channel_to_label_image_slice = {
                name: _apply_bbx(label_image, self.bbx)
                for name, label_image in self.channel_to_label_image.items()
            }
            channel_to_mask_slice = {}
            for name, (
                label_image_arr,
                label_image_meta,
            ) in channel_to_label_image_slice.items():
                if self.channel_to_label is None:
                    mask_arr = label_image_arr > 0
                else:
                    if name not in self.channel_to_label:
                        mask_arr = label_image_arr > 0
                    else:
                        mask_arr = label_image_arr == self.channel_to_label[name]
                if self.fuzzy_mask_sigma is not None:
                    mask_arr = mask_arr.astype(np.float32)
                    if isinstance(mask_arr, np.ndarray):
                        mask_arr = gaussian_np(mask_arr, self.fuzzy_mask_sigma)
                    else:
                        mask_arr = gaussian(mask_arr, self.fuzzy_mask_sigma)
                channel_to_mask_slice[name] = mask_arr
            return da.stack(
                (
                    ch * channel_to_mask_slice.get(ch_name, 1)
                    for ch, ch_name in zip(channels_slice_arr, channels_slice_meta.name)
                )
            ), channels_slice_meta

    def lazy(self) -> DaskImage:
        return self._process(_safe_to_lazy_dask_image)

    def compute(self) -> NumpyImage:
        return self._process(_safe_to_numpy_image)


def _aggregate_image_paths(
    r,
    channels: list[str] | None = None,
    multiscale_level: int | None = 0,
    channel_masks: list[str | None] | None = None,
    z_model: str | None = None,
    t_model: str | None = None,
):
    if channels is None:
        channels = r.channels["idx.c"]
    if multiscale_level is None:
        multiscale_levels = r.multiscale_levels["idx.m"]
    else:
        multiscale_levels = [multiscale_level]

    df_imgs = (
        (
            r.images.filter(pl.col("idx.c").is_in(channels))
            .filter(pl.col("idx.m").is_in(multiscale_levels))
            .join(r.rois, on="idx.roi")
        )
        .with_columns(
            pl.col("idx.c")
            .cast(pl.String)
            .replace(dict(zip(channels, range(len(channels)))))
            .cast(pl.UInt16)
            .alias("channel_order")
        )
        .sort(["idx.roi", "channel_order"])
    )
    if channel_masks is None:
        channel_masks = [None for _ in range(len(channels))]
    else:
        if len(channel_masks) != len(channels):
            raise ValueError(
                f"If `channel_masks` are provided, they need to be the same number as channels: {len(channels)=} {len(channel_masks)=}"
            )

    df_paths = (
        df_imgs.with_columns(
            pl.col("idx.c")
            .replace(dict(zip(channels, channel_masks)))
            .cast(pl.Categorical)
            .alias("mask.o")
        )
        # .join(
        #     r.label_images,
        #     on=["idx.roi", "idx.o"],
        #     suffix=".mask",
        #     how="left",
        # )
        .join(
            r.label_images.select(
                pl.col("idx.m"),
                pl.col("idx.roi"),
                pl.col("idx.o"),
                pl.col("o.path").alias("mask.o.path"),
                pl.col("resample_scale_factors").alias("mask.resample_scale_factors"),
            ),
            left_on=["idx.m", "idx.roi", "mask.o"],
            right_on=["idx.m", "idx.roi", "idx.o"],
            how="left",
        )
        .join(
            r.z_models.with_columns(pl.col("idx.c").cast(pl.Categorical)).filter(
                pl.col("idx.z_model").is_null()
                if z_model is None
                else pl.col("idx.z_model") == z_model
            ),
            on="idx.c",
            how="left",
        )
        .join(
            # r.label_images.with_columns(pl.col("idx.o").cast(pl.String)),
            r.label_images.select(
                pl.col("idx.m"),
                pl.col("idx.roi"),
                pl.col("idx.o"),
                pl.col("o.path").alias("z_model.o.path"),
                pl.col("resample_scale_factors").alias(
                    "z_model.resample_scale_factors"
                ),
            ),
            left_on=["idx.m", "idx.roi", "z_model.o"],
            right_on=["idx.m", "idx.roi", "idx.o"],
            how="left",
        )
        .join(
            r.t_models.filter(
                pl.col("idx.t_model").is_null()
                if t_model is None
                else pl.col("idx.t_model") == t_model
            ).select(
                pl.col("idx.c"),
                pl.col("idx.roi"),
                pl.col("idx.t_model"),
                pl.col("t_model.correction_factor"),
            ),
            on=["idx.c", "idx.roi"],
            how="left",
        )
    ).with_columns(pl.col("z_model.path").str.replace("\.json", ".pkl"))

    return df_paths


def _images_to_queries(
    df_imgs: pl.DataFrame,
    group=("idx.roi",),
    paths=(
        "roi.path",
        "c.path",
        "mask.o.path",
        "mask.resample_scale_factors",
        "z_model.path",
        "z_model.o.path",
        "z_model.resample_scale_factors",
        "t_model.correction_factor",
    ),
    dask_chunk_size="auto",
    separator="__",
) -> dict[str, MultiChannelQueryDask]:
    queries = {}
    for name, df in df_imgs.group_by(group, maintain_order=True):
        image_queries = []
        for (
            path_root,
            channel_path,
            mask_object_type_path,
            mask_object_type_resample_scale_factors,
            z_model_path,
            z_model_object_type_path,
            z_model_object_type_resample_scale_factors,
            t_model_correction_factor,
        ) in df.select(paths).rows():
            # path_root = "\\".join(
            #     r"C:\Users\hessm\Documents\zfish_local\imgs".split("\\")
            #     + path_root.split("\\")[-1:]
            # )
            if z_model_object_type_path is not None:
                z_model_labels_q = LabelImageQueryDask(
                    path_root,
                    z_model_object_type_path,
                    chunks=dask_chunk_size,
                    resample_scale_factors=z_model_object_type_resample_scale_factors,
                )
            else:
                z_model_labels_q = None

            if mask_object_type_path is not None:
                mask_labels_q = LabelImageQueryDask(
                    path_root,
                    mask_object_type_path,
                    chunks=dask_chunk_size,
                    resample_scale_factors=mask_object_type_resample_scale_factors,
                )
                mask_q = BinaryMaskQueryDask(mask_labels_q)
            else:
                mask_q = None
            image_queries.append(
                ImageQueryDask(
                    path_root,
                    channel_path,
                    z_model_path=z_model_path,
                    z_model_label_image_query=z_model_labels_q,
                    t_model_correction_factor=t_model_correction_factor,
                    mask_query=mask_q,
                    chunks=dask_chunk_size,
                )
            )
        if separator is not None:
            name = separator.join(map(str, name))
        queries[name] = MultiChannelQueryDask(tuple(image_queries))
    return queries


def _aggregate_label_image_paths(
    r,
    object_types: list[str] | None = None,
    multiscale_level: int | None = 0,
):
    if object_types is None:
        object_types = r.object_types["idx.o"]
    if multiscale_level is None:
        multiscale_levels = r.multiscale_levels["idx.m"]
    else:
        multiscale_levels = [multiscale_level]

    df_label_images = (
        r.label_images.filter(pl.col("idx.o").is_in(object_types))
        .filter(pl.col("idx.m").is_in(multiscale_levels))
        .join(r.rois, on="idx.roi")
    )
    return df_label_images


def _label_images_to_queries(
    df_label_imgs: pl.DataFrame,
    group=("idx.roi", "idx.o"),
    paths=(
        "roi.path",
        "o.path",
        "resample_scale_factors",
    ),
    dask_chunk_size="auto",
    separator="__",
) -> dict[str, LabelImageQueryDask]:
    queries = {}
    for name, df in df_label_imgs.group_by(group, maintain_order=True):
        if separator is not None:
            name = separator.join(map(str, name))
        for (
            path_root,
            object_type_path,
            resample_scale_factors,
        ) in df.select(paths).rows():
            queries[name] = LabelImageQueryDask(
                path_root,
                object_type_path=object_type_path,
                resample_scale_factors=resample_scale_factors,
                chunks=dask_chunk_size,
            )
    return queries


def _query_to_lazy(query):
    return query.lazy()


def _queries_to_lazy_images(
    queries: dict[str, LazyRasterQueryDask], drop_meta: bool = False
):
    lazy_images = {}
    for name, query in tqdm.tqdm(queries.items()):
        lazy_image = query.lazy()
        if isinstance(lazy_image, tuple) and drop_meta:
            lazy_image = lazy_image[0]
        lazy_images[name] = lazy_image
    return lazy_images


def _queries_to_lazy_bag(queries, npartitions=10):
    import dask.bag as db

    bag = db.from_sequence(queries, npartitions=npartitions)
    return bag.map(_query_to_lazy)


def _aggregate_label_object_paths(
    r,
    bbx_object_type: str,
    channels: list[str],
    channel_masks: list[str | None] | None = None,
    isolate_label: list[bool] | bool = True,
    multiscale_level: int = 0,
    z_model: str | None = None,
    t_model: str | None = None,
    rois: list[str] | None = None,
):
    if channel_masks is not None:
        if len(channels) != len(channel_masks):
            raise ValueError(
                f"If channel_masks provided they need to be the same number as channels: {len(channels)=} != {len(channel_masks)=}"
            )
        if isinstance(isolate_label, bool):
            isolate_label = [isolate_label] * len(channel_masks)

        isolate_label_channels = [
            ch for ch, isolate in zip(channels, isolate_label) if isolate
        ]
    if rois is None:
        rois = r.rois["idx.roi"].to_list()

    df_imgs = (
        _aggregate_image_paths(
            r,
            channels,
            multiscale_level=multiscale_level,
            # channel_masks=None,
            channel_masks=channel_masks,
            z_model=z_model,
            t_model=t_model,
        )
        .with_columns(
            pl.col("idx.c")
            .is_in(isolate_label_channels)
            .alias("bbx_mask.isolate_label")
        )
        .filter(pl.col("idx.roi").is_in(rois))
    )

    df_label_objects = r.label_objects.filter(
        pl.col("idx.o") == bbx_object_type
    ).filter(pl.col("idx.roi").is_in(rois))
    df_label_objects = label_objects_scale_bbx(
        r.multiscale_levels, df_label_objects, to_level=multiscale_level
    )
    return df_imgs, df_label_objects


def _label_objects_to_queries(
    df_label_objects: pl.DataFrame,
    df_imgs: pl.DataFrame,
    roi_id_columns: tuple[str, ...] = ("idx.roi", "idx.m"),
    object_id_columns: tuple[str, ...] = ("idx.o", "idx.label"),
    bbx_columns: tuple[str, ...] = (
        "bbx.z.lower",
        "bbx.z.upper",
        "bbx.y.lower",
        "bbx.y.upper",
        "bbx.x.lower",
        "bbx.x.upper",
    ),
    fuzzy_mask_sigma: int | None = None,
    separator: str = "__",
    strategy: Literal["lazy", "memory"] = "lazy",
    dask_chunk_size: tuple[int, ...] | str = "auto",
):
    label_object_queries = []

    for roi_id_parts, df_imgs_roi in df_imgs.group_by(
        roi_id_columns, maintain_order=True
    ):
        roi_id = separator.join(str(e) for e in roi_id_parts)

        channels_q = _images_to_queries(
            df_imgs_roi.with_columns(pl.lit(None).cast(pl.Categorical).alias("mask.o")),
            separator=separator,
            group=roi_id_columns,
            dask_chunk_size=dask_chunk_size,
        )[roi_id]
        if strategy == "lazy":
            channels = channels_q.lazy()
        else:
            channels = channels_q.compute()

        labels_ot_to_q = _label_images_to_queries(
            df_imgs_roi,
            separator=separator,
            group=("mask.o",),
            paths=("roi.path", "mask.o.path", "mask.resample_scale_factors"),
            dask_chunk_size=dask_chunk_size,
        )
        if strategy == "lazy":
            labels_lazy = {k: v.lazy() for k, v in labels_ot_to_q.items()}
        else:
            labels_lazy = {k: v.compute() for k, v in labels_ot_to_q.items()}
        channel_to_mask_object_type = dict(
            zip(df_imgs_roi["idx.c"], df_imgs_roi["mask.o"])
        )
        channel_to_label = {
            k: labels_lazy[v] for k, v in channel_to_mask_object_type.items()
        }

        df_label_objects_roi = df_label_objects.join(
            df_imgs_roi,
            on=roi_id_columns,
        ).select(
            pl.concat_str(object_id_columns, separator=separator).alias("object_id"),
            pl.col(bbx_columns),
            pl.col("idx.c"),
            pl.when(pl.col("bbx_mask.isolate_label"))
            .then(pl.col("idx.label"))
            .otherwise(pl.lit(None))
            .alias("label"),
        )
        for object_id, bbx_tuple, ch_names, labels in (
            df_label_objects_roi.group_by("object_id", maintain_order=True)
            .agg(
                pl.concat_list(cs.starts_with("bbx").first()).alias("bbx"),
                pl.col("idx.c"),
                pl.col("label"),
            )
            .rows()
        ):
            label_object_queries.append(
                LabelObjectQueryDask(
                    object_id=separator.join((roi_id, object_id)),
                    bbx=BoundingBox.from_flat_tuple(bbx_tuple),
                    multi_channel_query=channels,
                    channel_to_label_image=channel_to_label,
                    channel_to_label=dict(zip(ch_names, labels)),
                    fuzzy_mask_sigma=fuzzy_mask_sigma,
                )
            )
    return label_object_queries
