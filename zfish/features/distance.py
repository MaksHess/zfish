# %%
from typing import Callable

import itk
import polars as pl

from zfish.features._base import get_si_features_df
from zfish.features.types import BinaryImage, DistanceTransform, LabelImage

DISTANCE_ITK_FEATURES = {
    "Centroid",  # CentroidDistance
    "Maximum",  # MaximumDistance
    "Minimum",  # MinimumDistance
    "Median",  # MedianDistance
    # "MaximumIndex",  # ClosestPixel
    # "MinimumIndex",  # FurthestPixel
}


def _distance_to_border(mask: BinaryImage) -> DistanceTransform:
    dt = itk.signed_maurer_distance_map_image_filter(mask, inside_is_positive=True)
    dt.coords["c"] = mask.c.item()
    return dt


def _distance_along_axis(mask: BinaryImage, axis: str = "z") -> DistanceTransform:
    sum_along_axis = mask.cumsum(axis) * mask.meta.scale_dict[axis]
    return sum_along_axis


DISTANCE_TRANSFORMS = {
    "DistToBorder": _distance_to_border,
    "DistAlongZ": _distance_along_axis,
    # "DistAlongY": partial(_distance_along_axis, axis="y"),
    # "DistAlongX": partial(_distance_along_axis, axis="x"),
}


def _get_mask(lbl_img: LabelImage, lbl: int) -> BinaryImage:
    mask = (lbl_img == lbl).astype(lbl_img.dtype)
    mask.coords["c"] = f"{lbl_img.c.item()}-{lbl}"
    return mask


def get_distance_features(
    lbl_img: LabelImage,
    lbl_img_to: LabelImage,
    lbl_to: int,
    distance_functions: dict[
        str, Callable[[BinaryImage], DistanceTransform]
    ] = DISTANCE_TRANSFORMS,
    named_features: bool = True,
):
    if named_features:
        index = "index"
    else:
        index = "Label"
    mask = _get_mask(lbl_img_to, lbl_to)

    dfs = []
    for name, distance_function in distance_functions.items():
        try:
            dt = distance_function(mask)
        except ValueError as e:
            print(f"Can't compute {name}")
            print(f"{e}")
            continue
        # return lbl_img, dt
        df = get_si_features_df(
            lbl_img, dt, props=DISTANCE_ITK_FEATURES, named_features=named_features
        )
        df = _get_distance_at_centroid(df, dt)
        dfs.append(df.select([pl.col(index), pl.exclude(index).suffix(name)]))
    return pl.concat(
        [dfs[0].select(index), *[df.drop(index) for df in dfs]], how="horizontal"
    )


def _get_distance_at_centroid(df: pl.DataFrame, distance_transform: DistanceTransform):
    return df.with_columns(
        [
            pl.col("^.*Centroid$").apply(
                lambda x: _lookup_physical_point(x, distance_transform)
            ),
        ]
    )


def _lookup_physical_point(point: pl.Series, image: DistanceTransform):
    return image.sel(method="nearest", **point).item()


# def get_mask_itk(lbl_img: itk.Image, lbl: int) -> itk.Image:
#     mask = itk.image_duplicator(lbl_img)
#     mask_np_view = itk.GetArrayViewFromImage(mask)
#     set_to_1 = np.where(mask_np_view == lbl)
#     set_to_0 = np.where(mask_np_view != lbl)
#     mask_np_view[set_to_1] = 1
#     mask_np_view[set_to_0] = 0
#     return mask


# def distance_to_border_itk(mask: itk.Image) -> itk.Image:
#     return itk.signed_maurer_distance_map_image_filter(mask, inside_is_positive=True)


# def distance_along_axis_itk(mask: itk.Image, axis: str = "z") -> itk.Image:
#     mask_xr = itk.xarray_from_image(mask)
#     sum_along_axis = mask_xr.cumsum(axis) * mask_xr.meta.scale_dict["z"]
#     return itk.image_from_xarray(sum_along_axis)


# def get_distance_features_itk(
#     lbl_img: itk.Image,
#     lbl_img_to: itk.Image,
#     lbl_to: int,
#     distance_function: Callable,
# ):
#     mask = get_mask_itk(lbl_img_to, lbl_to)
#     # for distance_function in distance_functions:
#     dt = distance_function(mask)
#     # return lbl_img, dt
#     df = get_itk_features_df(lbl_img, dt, props=DISTANCE_ITK_FEATURES)
#     df = _get_distance_at_centroid_itk(df, dt)
#     return df


# def _get_distance_at_centroid_itk(df: pl.DataFrame, distance_transform: itk.Image):
#     return df.select(
#         [
#             pl.col("Label"),
#             pl.col("Centroid").apply(
#                 lambda x: _lookup_physical_point_itk(x, distance_transform)
#             ),
#             pl.exclude(["Label", "Centroid"]),
#         ]
#     )


# def _lookup_physical_point_itk(point: pl.Series, image: itk.Image):
#     return image.GetPixel(
#         image.TransformPhysicalPointToIndex(
#             tuple(point[k] for k, _ in zip(("x", "y", "z"), point.keys()))
#         )
#     )
