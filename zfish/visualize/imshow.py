# %%
from itertools import cycle

import napari
from spatial_image import SpatialImage

from zfish.image.h5_io import ImageMetaAccessor

# def imshow(data: SpatialData, viewer: napari.Viewer = None, **kwargs) -> napari.Viewer:
#     pass
COLORS = cycle(["blue", "green", "red"])


def imshow_spatial_image(
    img: SpatialImage, viewer: napari.Viewer = None, **kwargs
) -> napari.Viewer:
    if viewer is None:
        viewer = napari.Viewer()
    if img.name != "label":
        viewer = _show_image(img, viewer=viewer, **kwargs)
    else:
        viewer = _show_label(img, viewer=viewer, **kwargs)
    return viewer


def _show_image(
    img: SpatialImage, viewer: napari.Viewer = None, colors=COLORS, **kwargs
) -> napari.Viewer:
    if viewer is None:
        viewer = napari.Viewer()
    if "c" in img.dims:
        for ch, colormap in zip(img.c, colors):
            channel = img.sel(c=ch)
            viewer.add_image(
                channel,
                scale=channel.meta.scale,
                blending="additive",
                colormap=colormap,
                **kwargs,
            )
    else:
        viewer.add_image(
            img, scale=img.meta.scale, translate=img.meta.translate, **kwargs
        )
    return viewer


def _show_label(
    lbl: SpatialImage, viewer: napari.Viewer = None, **kwargs
) -> napari.Viewer:
    if viewer is None:
        viewer = napari.Viewer()

    if "c" in lbl.dims:
        for ch in lbl.c:
            objects = lbl.sel(c=ch)
            viewer.add_labels(objects, scale=objects.meta.scale, **kwargs)

    elif "l" in lbl.dims:
        for ch in lbl.l:
            objects = lbl.sel(l=ch)
            viewer.add_labels(objects, scale=objects.meta.scale, **kwargs)

    else:
        viewer.add_labels(
            lbl, scale=lbl.meta.scale, translate=lbl.meta.translate, **kwargs
        )
    return viewer
