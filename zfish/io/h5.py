import os
import pandas as pd
import numpy as np
import h5py
from typing import Sequence
from numpy.typing import NDArray


SIZE_TO_MB = {
    np.dtype("bool"): 1e-6 * 0.125,
    np.dtype("uint8"): 1e-6,
    np.dtype("int16"): 2 * 1e-6,
    np.dtype("uint16"): 2 * 1e-6,
    np.dtype("int32"): 4 * 1e-6,
    np.dtype("uint32"): 4 * 1e-6,
    np.dtype("float32"): 4 * 1e-6,
    np.dtype("int64"): 8 * 1e-6,
    np.dtype("uint64"): 8 * 1e-6,
    np.dtype("float64"): 8 * 1e-6,
}


def datasets(f: h5py.File, return_names=False, dsets=None):
    if dsets is None:
        dsets = []

    for group_or_dataset_name in f.keys():
        if isinstance(f[group_or_dataset_name], h5py.Group):
            datasets(f[group_or_dataset_name], return_names=return_names, dsets=dsets)
        elif isinstance(f[group_or_dataset_name], h5py.Dataset):
            if return_names:
                dsets.append(f[group_or_dataset_name].name)
            else:
                dsets.append(f[group_or_dataset_name])
    return dsets


def groups(f: h5py.File, return_names=False, groups=None):
    if groups is None:
        groups = []

    for group_or_dataset_name in f.keys():
        if isinstance(f[group_or_dataset_name], h5py.Group):
            if return_names:
                groups.append(f[group_or_dataset_name].name)
            else:
                groups.append(f[group_or_dataset_name])
            groups(f[group_or_dataset_name], return_names=return_names, groups=groups)

    return groups


def select(
    f: h5py.File, attr_select: dict = None, not_attr_select: dict = None
) -> list[h5py.Dataset]:
    dsets = []
    for dset in datasets(f):

        check = []
        if attr_select:
            for a in attr_select:
                if isinstance(attr_select[a], (tuple, list)):
                    check.append(dset.attrs.get(a) in attr_select[a])
                else:
                    check.append(dset.attrs.get(a) == attr_select[a])

        uncheck = []
        if not_attr_select:
            for b in not_attr_select:
                if isinstance(not_attr_select[b], (tuple, list)):
                    uncheck.append(dset.attrs.get(b) in not_attr_select[b])
                else:
                    uncheck.append(dset.attrs.get(b) == not_attr_select[b])

        if all(check) and not any(uncheck):
            dsets.append(dset)
    return dsets


def write_channel(
    f: h5py.File,
    data: NDArray,
    name: str,
    copy_from: h5py.Dataset = None,
    attrs: dict = None,
    compression="gzip",
    overwrite=True,
):
    if overwrite:
        if name in f:
            del f[name]
    dset = f.create_dataset(name=name, data=data, compression=compression)
    if copy_from is not None:
        copy_attributes(copy_from, dset)
    if attrs is not None:
        write_attributes(dset, attrs, overwrite=True)
    return dset


def h5_set(f: h5py.File, attr: str, remove_none=True):
    unique = set([dset.attrs.get(attr) for dset in datasets(f)])
    if None in unique:
        unique.remove(None)
    return unique


def copy_attributes(
    dset_from: h5py.Dataset,
    dset_to: h5py.Dataset,
    *,
    include: Sequence = None,
    exclude: Sequence = None
):
    if exclude is None:
        exclude = []
    if include is None:
        for attr in dset_from.attrs.keys():
            if attr not in exclude:
                dset_to.attrs[attr] = dset_from.attrs.get(attr, "NA")
    else:
        for attr in include:
            dset_to.attrs[attr] = dset_from.attrs.get(attr, "NA")


def write_attributes(dset: h5py.Dataset, attrs: dict = None, overwrite: bool = False):
    if attrs is None:
        attrs = dict()
    if overwrite:
        for key in attrs:
            dset.attrs[key] = attrs.get(key, "NA")
    else:
        for key in attrs:
            if dset.attrs.get(key) in ["NA", None]:
                dset.attrs[key] = attrs[key]


def copy_channel(
    fn_from, fn_to, channel_name_from, channel_name_to=None, overwrite=True
):
    if channel_name_to is None:
        channel_name_to = channel_name_from
    with h5py.File(fn_from) as f_from:
        with h5py.File(fn_to, "a") as f_to:
            dset_from = f_from[channel_name_from]
            if channel_name_to in f_to and not overwrite:
                return
            elif channel_name_to in f_to:
                del f_to[channel_name_to]
            dset_to = f_to.create_dataset(
                name=channel_name_to,
                data=dset_from[...],
                compression=True,
                chunks=True,
            )
            copy_attributes(dset_from, dset_to)


def summary(filename):
    file_attrs = ["condition"]
    file_string_format = "{}:\t{}"
    file_attrs_string = []
    base_attrs = ["shape", "dtype", "size"]
    attrs_attrs = [
        "element_size_um",
        "img_type",
        "stain",
        "cycle",
        "wavelength",
        "level",
    ]

    with h5py.File(filename, "r") as fl:
        file_attrs_string.append(
            file_string_format.format("filename", os.path.basename(fl.filename))
        )
        for attr in file_attrs:
            file_attrs_string.append(
                file_string_format.format(attr, fl.attrs.get(attr, "NA"))
            )
        file_attrs_string.append("")
        dset_names = datasets(fl, return_names=True)
        df = pd.DataFrame(columns=base_attrs + attrs_attrs, index=dset_names)
        df.index.name = "name"
        for k in dset_names:
            for attr in base_attrs:
                if attr == "size":
                    sz = getattr(fl[k], attr, "NA")
                    if sz != "NA":
                        sz = round(
                            sz * SIZE_TO_MB[fl[k].dtype], 2
                        )  # convert size attribute to MB for readability
                    df.loc[k, attr] = sz
                else:
                    df.loc[k, attr] = getattr(fl[k], attr, "NA")

            for attr in attrs_attrs:
                df.loc[k, attr] = fl[k].attrs.get(attr, "NA")

    df.reset_index(level=0, inplace=True)
    return "\n".join(
        [
            *file_attrs_string,
            df.to_string(
                index=False,
                na_rep="",
                max_rows=None,
                max_cols=None,
                line_width=140,
                justify="right",
            ),
            "\n",
        ]
    )
