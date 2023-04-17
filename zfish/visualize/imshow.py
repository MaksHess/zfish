# %%
import napari
from spatial_image import SpatialImage
from spatialdata import SpatialData

from zfish.image.image import ImageMetaAccessor


def imshow(data: SpatialData, viewer: napari.Viewer = None, **kwargs) -> napari.Viewer:
    pass


def imshow_spatial_image(
    img: SpatialImage, viewer: napari.Viewer = None, **kwargs
) -> napari.Viewer:
    if viewer is None:
        viewer = napari.Viewer()
    if img.name == "image":
        viewer = _show_image(img, viewer=viewer, **kwargs)
    else:
        viewer = _show_label(img, viewer=viewer, **kwargs)
    return viewer


def _show_image(
    img: SpatialImage, viewer: napari.Viewer = None, **kwargs
) -> napari.Viewer:
    if viewer is None:
        viewer = napari.Viewer()
    if "c" in img.dims:
        for ch in img.c:
            channel = img.sel(c=ch)
            viewer.add_image(channel, scale=channel.meta.scale, **kwargs)
    else:
        viewer.add_image(img, scale=img.meta.scale, **kwargs)
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
    else:
        viewer.add_labels(lbl, scale=lbl.meta.scale, **kwargs)
    return viewer
