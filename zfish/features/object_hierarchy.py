# %%
from typing import Literal

import numpy as np
import polars as pl
from skimage.measure import regionprops

from zfish.features.types import LabelImage, MultichannelLabelImage, SpatialImage

# HIERARCHY = {
#     "emb": (),
#     "cell": ("emb",),
#     "nuc": ("cell", "emb"),
#     "mem": ("cell", "emb"),
#     "cyto": ("cell", "emb"),
#     "loc": ("nuc", "cell", "emb"),
# }


def parent_label(lbl: np.array, lbl2: np.array):
    labels, counts = np.unique(lbl2[np.where(lbl)], return_counts=True)
    parent_label = labels[np.argsort(counts)[-1]]
    return int(parent_label)


def get_parent_objects(
    label_image: LabelImage,
    parent_label_images: MultichannelLabelImage,
    lbl_dim: str = "l",
    index_columns: tuple[Literal["label_image", "label"], ...] = ("label",),
    index_prefix: str | None = None,
    parent_prefix: str | None = 'parent',
) -> pl.DataFrame:

    results = {}
    props = regionprops(label_image.to_numpy())
    labels = [prop.label for prop in props]

    for lbl_other in parent_label_images:
        other_obj = lbl_other[lbl_dim].item()
        props = regionprops(
            label_image.to_numpy(), lbl_other.to_numpy(), extra_properties=(parent_label,)
        )
        results[other_obj] = [prop.parent_label for prop in props]
    
    df = pl.DataFrame(results)
    
    if parent_prefix is not None:
        df = df.select(pl.all().prefix(f"{parent_prefix}."))
    
    if "label" in index_columns:
        df = df.with_columns(
            pl.Series(name="label", values=labels).cast(pl.Int64)
        )

    if "label_image" in index_columns:
        df = df.with_columns(pl.lit(label_image[lbl_dim].item()).alias("label_image"))

    if index_prefix is not None:
        df = df.select(
            pl.col(index_columns).prefix(f"{index_prefix}."), pl.exclude(index_columns)
        )
    else:
        df = df.select(pl.col(index_columns), pl.exclude(index_columns))

    return df

# def get_full_object_hierarchy(
#     lbls: MultichannelLabelImage,
#     raw_hierarchy: dict[str, tuple[str, ...]] = HIERARCHY,
#     lbl_dim: str = "l",
# ) -> pl.DataFrame:
#     # assert all(obj in raw_hierarchy for obj in lbls.c), f"Incomplete object hierarchy!"
#     hierarchy = {k: v for k, v in raw_hierarchy.items() if k in lbls[lbl_dim]}
#     full_index = tuple(hierarchy.keys())
#     res = []
#     for obj in hierarchy:
#         parent_objs = list(
#             hierarchy[obj]
#         )  # convert to `list` so it plays well with `xarray.DataArray.sel`
#         parents = get_parents(
#             lbls.sel(**{lbl_dim: obj}), lbls.sel(**{lbl_dim: parent_objs})
#         )
#         polars_dtype = pl.datatypes.numpy_char_code_to_dtype(lbls.dtype.char)
#         df_parents = pl.DataFrame(parents)
#         res.append(
#             df_parents.select(pl.all().cast(polars_dtype))
#             .with_columns(
#                 [
#                     # pl.lit(0).cast(polars_dtype).alias(index_obj)
#                     pl.lit(None).cast(polars_dtype).alias(index_obj)
#                     for index_obj in full_index
#                     if index_obj not in parents
#                 ]
#             )
#             .with_columns([pl.lit(obj).alias("label_image")])
#             .select(("label_image", *full_index))
#         )
#     return pl.concat(res)


# def get_parent(lbl: LabelImage, lbl2: LabelImage, lbl_dim: str = "l") -> pl.DataFrame:
#     obj = lbl[lbl_dim].item()
#     other_obj = lbl2[lbl_dim].item()
#     props = regionprops(
#         lbl.to_numpy(), lbl2.to_numpy(), extra_properties=(parent_label,)
#     )
#     return {
#         obj: [prop.label for prop in props],
#         other_obj: [prop.parent_label for prop in props],
#     }


# def get_parents(
#     lbl: LabelImage, lbls: MultichannelLabelImage, lbl_dim: str = "l"
# ) -> pl.DataFrame:
#     obj = lbl[lbl_dim].item()
#     results = {}
#     props = regionprops(lbl.to_numpy())
#     results[obj] = [prop.label for prop in props]

#     for lbl_other in lbls:
#         other_obj = lbl_other[lbl_dim].item()
#         props = regionprops(
#             lbl.to_numpy(), lbl_other.to_numpy(), extra_properties=(parent_label,)
#         )
#         results[other_obj] = [prop.parent_label for prop in props]

#     return results





# %%

if __name__ == "__main__":
    from zfish.features.label import get_label_features
    from zfish.io.datasource import hierarchical_labels
    from zfish.visualize.imshow import imshow

    lbls = hierarchical_labels((100, 400, 400), scale=(1.2, 0.4, 0.5))
    lbls

    imshow(lbls)

    hierarchies = []
    features = []

    total = []

    full_index = tuple(reversed(LABELS.keys()))

    for obj, parents in LABELS.items():
        print(obj, parents)
        lbl = lbls.sel(c=obj)
        index = (obj, *parents)
        missing_index = tuple(e for e in full_index if e not in index)
        print(index)
        print(missing_index)
        print()
        if parents:
            lbls_parents = lbls.sel(c=list(parents))
            print(lbls_parents.shape)
            hier = pl.dataframe(get_parents(lbl, lbls_parents))
            feat = get_label_features(lbl).unnest("index").rename({"label": obj})
            hierarchies.append(hier)
            features.append(feat)
            total.append(
                hier.join(feat, on=obj)
                .with_columns(
                    [pl.lit(0).cast(pl.int64).alias(e) for e in missing_index]
                )
                .select(
                    [
                        pl.col(("object", *full_index)),
                        pl.exclude(("object", *full_index)),
                    ]
                )
            )
        else:
            total.append(
                get_label_features(lbl)
                .unnest("index")
                .rename({"label": obj})
                .with_columns(
                    [pl.lit(0).cast(pl.int64).alias(e) for e in missing_index]
                )
                .select(
                    [
                        pl.col(("object", *full_index)),
                        pl.exclude(("object", *full_index)),
                    ]
                )
            )

    f = pl.concat(total, how="vertical")

    # %%
    f.filter(pl.col("cyto") != 0)

    from zfish.visualize.napari import (
        napari_centroids,
        napari_principal_axes,
    )

    f.select("centroid").unnest("centroid").select(
        [pl.col("y"), pl.col("x")]
    ).to_numpy()
    # viewer = napari.viewer()
    objects = list(labels.keys())
    objects = ("cell",)

    lbl_show = lbls.sel(c=list(objects))
    viewer = imshow(lbl_show)
    for object in objects:
        e = f.filter(pl.col("object") == object)
        viewer.add_points(**napari_centroids(e))
        for ax in napari_principal_axes(e):
            viewer.add_vectors(**ax)
