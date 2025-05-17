# %%
import warnings
from itertools import cycle
from typing import TYPE_CHECKING, TypeAlias

import napari
from spatial_image import SpatialImage

from zfish.features.polars_utils import unnest_all_structs
from zfish.image.meta_accessor import ImageMetaAccessor
from zfish.multi_table.feature_query_schemas import _to_si

if TYPE_CHECKING:
    import dask.array as da
    import polars as pl

    from zfish.features.types import LabelImage
    from zfish.multi_table.raster_io import RasterMeta
    from zfish.roi.spatial_roi import Roi, RoiMap

NAPARI_COMMON_DEFAULTS = {
    "visible": True,
    **{
        k: None
        for k in [
            "name",
            "scale",
            "translate",
            "rotate",
            "shear",
            "affine",
            "multiscale",
        ]
    },
}
CMAP_CYCLE = cycle(["gray"])
NAPARI_IMAGE_DEFAULTS = {
    **NAPARI_COMMON_DEFAULTS,
    "blending": "additive",
    "opacity": 1,
    # "colormap": "gray",
    "gamma": 1,
    "interpolation2d": "nearest",
    "rendering": "mip",
    "iso_threshold": 0.5,
    "attenuation": 0.05,
    "contrast_limits": (0, 1000),
    "rgb": None,
}
NAPARI_LABEL_DEFAULTS = {
    **NAPARI_COMMON_DEFAULTS,
    "blending": "translucent",
    "opacity": 0.6,
    # "num_colors": 500,
    # "seed": 0.5,
    **{k: None for k in ["properties"]},
}
Range: TypeAlias = tuple[float, float]


def imshow_di(
    dask_image: "tuple[da.Array, RasterMeta]", viewer: "napari.Viewer" = None, **kwargs
):
    return imshow_si(_to_si(dask_image), viewer=viewer, **kwargs)


def imshow_map(
    roi_map: "RoiMap",
    viewer: napari.Viewer | None = None,
    object: tuple[str, int] | None = None,
    **kwargs,
) -> napari.Viewer:
    if viewer is None:
        viewer = napari.Viewer()
    for name, roi in roi_map.items():
        viewer = imshow_roi(roi, viewer=viewer, name=name)
    return viewer


def imshow_roi(
    roi: "Roi",
    viewer: napari.Viewer | None = None,
    object: tuple[str, int] | None = None,
    roi_name_as_prefix: bool = False,
    **kwargs,
) -> napari.Viewer:
    if viewer is None:
        viewer = napari.Viewer()
    name_prefix = f"{roi.name}_" if roi_name_as_prefix else ""

    if "c" in roi.coords.keys():
        image_kwargs = {**NAPARI_IMAGE_DEFAULTS, **kwargs}
        viewer = _show_image(
            roi.images, viewer=viewer, name_prefix=name_prefix, **image_kwargs
        )
    if "l" in roi.coords.keys():
        label_kwargs = {**NAPARI_LABEL_DEFAULTS, **kwargs}
        viewer = _show_label(
            roi.labels,
            tables=roi.tables,
            viewer=viewer,
            name_prefix=name_prefix,
            **label_kwargs,
        )
    return viewer


def safe_channel_len(img: "SpatialImage"):
    try:
        return len(img["c"])
    except TypeError as e:
        if e.args[0] != "len() of unsized object":
            raise e
        return 1


def imshow_si(
    img: "SpatialImage",
    is_label: bool = False,
    viewer: napari.Viewer | None = None,
    tables: "dict[str, pl.DataFrame] | None" = None,
    name_prefix: str = "",
    **kwargs,
) -> napari.Viewer:
    if "l" in img.coords.keys() or is_label or img.name == "label":
        label_kwargs = {**NAPARI_LABEL_DEFAULTS, **kwargs}
        return _show_label(
            img,
            viewer=viewer,
            name_prefix=name_prefix,
            tables=tables,
            **label_kwargs,
        )
    if tables is not None:
        warnings.warn("no features can be added to images, only labels.")

    image_kwargs = {**NAPARI_IMAGE_DEFAULTS, **kwargs}
    return _show_image(img, viewer=viewer, name_prefix=name_prefix, **image_kwargs)


def _show_image(
    images: SpatialImage,
    viewer: napari.Viewer | None = None,
    name_prefix: str = "",
    **kwargs,
) -> napari.Viewer:
    if viewer is None:
        viewer = napari.Viewer()
    if "c" not in images.dims:
        images = images.expand_dims("c")
    if safe_channel_len(images) > 1:
        colors = cycle(["blue", "green", "red", "yellow", "magenta", "cyan"])
    else:
        colors = CMAP_CYCLE
    if "c" in images.dims:
        for ch, color in zip(list(images.c), colors):
            print(ch, color)
            channel = images.sel(c=ch.item())
            channel_kwargs = images.attrs.get("napari_channel_kwargs", {}).get(
                ch.item(), {}
            )
            new_kwargs = {
                "name": f"{name_prefix}{ch.item()}",
                "scale": channel.meta.scale,
                "translate": channel.meta.translate,
                "colormap": color,
            }
            # kwargs["name"] = f"{name_prefix}{ch.item()}"
            # kwargs["scale"] = channel.meta.scale
            # kwargs["translate"] = channel.meta.translate
            # if "colormap" not in kwargs:
            #     kwargs["colormap"] = color
            viewer.add_image(
                channel,
                **{
                    **kwargs,
                    **channel_kwargs,
                    **new_kwargs,
                },
            )
    return viewer


def _show_label(
    labels: "LabelImage",
    viewer: napari.Viewer | None = None,
    tables: dict[str, "pl.DataFrame"] | None = None,
    name_prefix: str = "",
    **kwargs,
) -> napari.Viewer:
    if viewer is None:
        viewer = napari.Viewer()

    if "c" in tuple(labels.coords.keys()):
        label_dim = "c"
    else:
        label_dim = "l"
    if label_dim not in labels.dims:
        labels = labels.expand_dims(label_dim)

    for lb in labels[label_dim]:
        lbl_img = labels.sel({label_dim: lb.item()})
        kwargs["name"] = f"{name_prefix}{lb.item()}"
        kwargs["scale"] = lbl_img.meta.scale
        kwargs["translate"] = lbl_img.meta.translate
        if tables is not None and lb.item() in tables:
            kwargs["features"] = (
                tables[lb.item()].pipe(unnest_all_structs).clone().to_pandas()
            )
        viewer.add_labels(lbl_img, **kwargs)
    return viewer
