from typing import Sequence

import numpy as np
from skimage.measure import regionprops_table

from zfish.features.types import LabelImage


def keep_labels(
    label_image: LabelImage, labels: Sequence[int], inplace=False
) -> LabelImage:
    if inplace:
        keep_labels_np(label_image.data, labels=labels, inplace=True)
        return label_image
    else:
        return label_image.copy(
            data=keep_labels_np(label_image.data, labels=labels, inplace=False)
        )


def delete_labels(
    label_image: LabelImage, labels: Sequence[int], inplace=False
) -> LabelImage:
    if inplace:
        delete_labels_np(label_image.data, labels=labels, inplace=True)
        return label_image
    else:
        return label_image.copy(
            data=delete_labels_np(label_image.data, labels=labels, inplace=False)
        )


def keep_labels_np(
    label_image: np.array, labels: Sequence[int], inplace=True
) -> np.array:
    if not inplace:
        label_image = label_image.copy()
    props = regionprops_table(np.asarray(label_image), properties=("label", "slice"))
    label_ids = props["label"]
    slices = props["slice"]
    labels_to_delete = {
        lbl: s for lbl, s in zip(label_ids, slices) if lbl not in labels
    }

    for label_id, slc in labels_to_delete.items():
        label_image_slice = label_image[slc]  # view, so can be mutated
        label_image_slice[np.where(label_image_slice == label_id)] = 0
    return label_image


def delete_labels_np(
    label_image: LabelImage, labels: Sequence[int], inplace=True
) -> np.array:
    if not inplace:
        label_image = label_image.copy()
    props = regionprops_table(np.asarray(label_image), properties=("label", "slice"))
    label_ids = props["label"]
    slices = props["slice"]
    labels_to_delete = {lbl: s for lbl, s in zip(label_ids, slices) if lbl in labels}

    for label_id, slc in labels_to_delete.items():
        label_image_slice = label_image[slc]  # view, so can be mutated
        label_image_slice[np.where(label_image_slice == label_id)] = 0
    return label_image
