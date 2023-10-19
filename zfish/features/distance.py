# %%
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import BaseModel

if TYPE_CHECKING:
    from zfish.roi.spatial_roi import Roi

import itk
import polars as pl

from zfish.features._base import get_si_features_df
from zfish.features.constants import (
    DefaultDistanceFeature,
    DefaultDistanceFunction,
    DistanceFeature,
    DistanceFunction,
)
from zfish.features.queries import FeatureQuery
from zfish.features.types import (
    BinaryImage,
    DistanceTransform,
    LabelImage,
    SpatialImage,
)


def _distance_to_border(mask: BinaryImage) -> DistanceTransform:
    lbl_dim = "l"
    dt = itk.signed_maurer_distance_map_image_filter(mask, inside_is_positive=True)
    dt.coords["c"] = mask[lbl_dim].item()
    return dt


def _distance_along_axis(mask: BinaryImage, axis: str = "z") -> DistanceTransform:
    sum_along_axis: SpatialImage = mask.cumsum(axis) * mask.meta.scale_dict[axis]
    return sum_along_axis.rename({"l": "c"})


DISTANCE_FUNCTIONS = {
    "DistanceToBorder": _distance_to_border,
    "DistanceAlongZ": _distance_along_axis,
    "DistanceAlongY": partial(_distance_along_axis, axis="y"),
    "DistanceAlongX": partial(_distance_along_axis, axis="x"),
}


def _get_mask(lbl_img: LabelImage, lbl: int, lbl_dim: str = "l") -> BinaryImage:
    mask = (lbl_img == lbl).astype(lbl_img.dtype)
    label_name = lbl_img[lbl_dim].item()
    mask.coords[lbl_dim] = f"{label_name}-{lbl}"
    return mask


class LabelObject(NamedTuple):
    label_image: str
    label: int


class DistanceQuery(BaseModel, FeatureQuery):
    label_image: str
    label_object_to: LabelObject
    features: tuple[DistanceFeature, ...] = tuple(DefaultDistanceFeature)
    distance_transforms: tuple[DistanceFunction, ...] = tuple(DefaultDistanceFunction)

    def load_resources(self, roi: "Roi") -> dict[str, Any]:
        return {
            "label_image": roi.sel(l=self.label_image).drop_dim("c").labels.compute(),
            "label_image_to": roi.sel(l=self.label_object_to.label_image)
            .drop_dim("c")
            .labels.compute(),
            "label_to": self.label_object_to.label,
            "features": self.features,
            "distance_transforms": self.distance_transforms,
        }

    def compute(self, roi: "Roi") -> "pl.DataFrame":
        return get_distance_features(**self.load_resources(roi))


def get_distance_features(
    label_image: LabelImage,
    label_image_to: LabelImage,
    label_to: int,
    distance_transforms: tuple[DistanceFunction, ...] = tuple(DefaultDistanceFunction),
    features: tuple[DistanceFeature, ...] = tuple(DefaultDistanceFeature),
    lbl_dim: str = "l",
    named_features: bool = True,
    object_column: bool = False,
    struct_index: bool = False,
):
    distance_transforms = {k: DISTANCE_FUNCTIONS[str(k)] for k in distance_transforms}
    if struct_index:
        index = "index"
    elif object_column:
        index = ["object", "label"]
    else:
        index = "label"
    mask = _get_mask(label_image_to, label_to, lbl_dim=lbl_dim)

    dfs = []
    for name, distance_function in distance_transforms.items():
        try:
            dt = distance_function(mask)
        except ValueError as e:
            print(f"Can't compute {name}")
            print(f"{e}")
            continue
        # return label_image, dt
        df = get_si_features_df(
            label_image,
            dt,
            props=features,
            lbl_dim=lbl_dim,
            named_features=named_features,
            object_column=object_column,
            struct_index=struct_index,
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


# get_distance_features()
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
