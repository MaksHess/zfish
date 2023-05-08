from itertools import cycle
from typing import TypeAlias

import napari
from spatial_image import SpatialImage

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
    "colormap": "gray",
    "gamma": 1,
    "interpolation": "nearest",
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
    "num_colors": 500,
    "seed": 0.5,
    **{k: None for k in ["properties", "color"]},
}
Range: TypeAlias = tuple[float, float]

def imshow_map(
        roi_map: "RoiMap",
        viewer: napari.Viewer | None = None,
        object: tuple[str, int] | None = None,
        **kwargs,
) -> napari.Viewer:
    if viewer is None:
        viewer = napari.Viewer()
    for name, roi in roi_map.items():
        viewer = imshow(roi, viewer=viewer, name=name)
    return viewer

def imshow_roi(
    roi: "Roi",
    viewer: napari.Viewer | None = None,
    object: tuple[str, int] | None = None,
    **kwargs,
) -> napari.Viewer:
    if viewer is None:
        viewer = napari.Viewer()
    if "c" in roi.coords.keys():
        image_kwargs = {**NAPARI_IMAGE_DEFAULTS, **kwargs}
        print(image_kwargs)
        viewer = _show_image(roi.images, viewer=viewer, **image_kwargs)
    if "l" in roi.coords.keys():
        label_kwargs = {**NAPARI_LABEL_DEFAULTS, **kwargs}
        viewer = _show_label(roi.labels, viewer=viewer, **label_kwargs)
    return viewer


def _show_image(
    images: SpatialImage, viewer: napari.Viewer | None = None, **kwargs
) -> napari.Viewer:
    if viewer is None:
        viewer = napari.Viewer()
    if images.c.shape == tuple():
        images = images.expand_dims("c")
    if "c" in images.dims:
        for ch in images.c:
            channel = images.sel(c=ch)
            kwargs["name"] = ch.item()
            kwargs["scale"] = channel.meta.scale
            kwargs["translate"] = channel.meta.translate
            viewer.add_image(
                # channel, name=ch.item(), scale=channel.meta.scale, **kwargs
                channel,
                **kwargs,
            )
    return viewer


def _show_label(
    labels: SpatialImage, viewer: napari.Viewer | None = None, **kwargs
) -> napari.Viewer:
    if viewer is None:
        viewer = napari.Viewer()
    if labels.l.shape == tuple():
        labels = labels.expand_dims("l")
    if "l" in labels.dims:
        for ch in labels.l:
            channel = labels.sel(l=ch)
            kwargs["name"] = ch.item()
            kwargs["scale"] = channel.meta.scale
            kwargs["translate"] = channel.meta.translate
            viewer.add_labels(channel, **kwargs)
    return viewer

