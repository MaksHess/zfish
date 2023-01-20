from abbott.h5_files import h5_select
from pathlib import Path
import h5py
import warnings
from functools import partial
from typing import Union, Any
import numpy as np
from tqdm import tqdm
from numpy.typing import NDArray
from itertools import product, repeat
from typing import Callable
import pandas as pd

ROW_TO_NUMBER = {k: v for k, v in zip("ABCDEFGH", range(8))}


def get_filenames(*flds: Union[str, Path]) -> list[Path]:
    fns = []
    for fld in flds:
        fns.extend(list(Path(fld).glob("*.h5")))
    return fns


def filenames_from_prefix(fld: Union[str, Path], pfxs: list[str]) -> list[Path]:
    fns = [fn for fn in fld.rglob("*.h5") if fn.stem in pfxs]
    return fns


def load_channels(
    fns: list[Path], selectors: Union[dict[str, Any], list[dict[str, Any]]], level: int
) -> dict[str, np.array]:
    if isinstance(selectors, dict):
        selectors = [selectors]

    imgs = {}
    a = fns[0]
    print(a)
    for fn in tqdm(fns):
        with h5py.File(fn) as f:
            channels = []
            for ci, selector in enumerate(selectors):
                dsets = h5_select(f, {**selector, "level": level})
                if len(dsets) > 1:
                    warnings.warn(f"non-unique selector in {fn}")
                dset = dsets[0]
                channels.append(np.expand_dims(dset[...], axis=(0, 1)))
            imgs[fn.stem] = np.concatenate(channels, axis=1)
    return imgs


def well_to_yx_index(well: str) -> tuple:
    return ROW_TO_NUMBER[well[0]], int(well[1:])


def well_grid(wells: pd.Series):
    raw_indices = np.array([well_to_yx_index(w) for w in wells.index])
    indices = raw_indices - raw_indices.min(axis=0)
    return pd.DataFrame(indices, columns=["iy", "ix"], index=wells.index)


def line_grid(sites: pd.Series, left_to_right=True):
    n = len(sites)
    indices = np.array(list(zip(np.arange(n), repeat(0))))
    if left_to_right:
        indices = np.flip(indices, axis=1)
    return pd.DataFrame(indices, columns=["iy", "ix"], index=sites.sort_values().index)


def rect_grid(sites: pd.Series, left_to_right=True, aspect_ratio=2):
    n = len(sites)
    ny = int(np.ceil(np.sqrt(n * aspect_ratio)))
    nx = int(np.ceil(n / ny))
    indices = np.array(list(product(np.arange(ny), np.arange(nx))))
    if left_to_right:
        indices = np.flip(indices, axis=1)
    return pd.DataFrame(
        indices[: len(sites)], columns=["iy", "ix"], index=sites.sort_values().index
    )


def square_grid(sites: pd.Series, left_to_right=True):
    n = len(sites)
    ny, nx = int(np.ceil(np.sqrt(n))), int(np.ceil(np.sqrt(n)))
    indices = np.array(list(product(np.arange(ny), np.arange(nx))))
    if left_to_right:
        indices = np.flip(indices, axis=1)
    return pd.DataFrame(
        indices[: len(sites)], columns=["iy", "ix"], index=sites.sort_values().index
    )


def arrange_on_grid(
    raw_imgs: dict[str, NDArray],
    order: pd.Series,
    index_function: Callable[[NDArray], pd.DataFrame] = rect_grid,
    margin_px: int = 10,
) -> NDArray:
    embs = list(set(raw_imgs.keys()).intersection(set(order.index)))
    order = order.loc[embs]
    imgs = {k: v for k, v in raw_imgs.items() if k in embs}

    dtype = imgs[list(imgs.keys())[0]].dtype
    max_extent = np.array([img.shape for img in imgs.values()]).max(axis=0)
    print(max_extent)

    dt_site, dc_site, dz_site, dy_site, dx_site = max_extent
    dy_margin = dx_margin = margin_px

    site_indices = index_function(order)

    canvas_extent_yx = (site_indices.max().values + 1) * np.array(
        [dy_site + dy_margin, dx_site + dx_margin]
    ) - np.array([dy_margin, dx_margin])

    canvas = np.zeros((dt_site, dc_site, dz_site, *canvas_extent_yx), dtype=dtype)
    for site, index in site_indices.iterrows():
        _, _, dz_current, dy_current, dx_current = imgs[site].shape
        dy, dx = index.values * np.array([dy_site + dy_margin, dx_site + dx_margin])
        canvas[:, :, :dz_current, dy : dy + dy_current, dx : dx + dx_current] = imgs[
            site
        ]
    return canvas


def arrange_on_dask_grid(
    raw_imgs: dict[str, NDArray],
    order: pd.Series,
    index_function: Callable[[NDArray], pd.DataFrame] = rect_grid,
    margin_px: int = 10,
) -> NDArray:

    import dask.array as da

    embs = list(set(raw_imgs.keys()).intersection(set(order.index)))
    order = order.loc[embs]
    imgs = {k: v for k, v in raw_imgs.items() if k in embs}

    dtype = imgs[list(imgs.keys())[0]].dtype
    max_extent = np.array([img.shape for img in imgs.values()]).max(axis=0)
    print(max_extent)

    dt_site, dc_site, dz_site, dy_site, dx_site = max_extent
    dy_margin = dx_margin = margin_px

    site_indices = index_function(order)

    canvas_extent_yx = (site_indices.max().values + 1) * np.array(
        [dy_site + dy_margin, dx_site + dx_margin]
    ) - np.array([dy_margin, dx_margin])

    canvas = da.zeros((dt_site, dc_site, dz_site, *canvas_extent_yx), dtype=dtype)
    for site, index in site_indices.iterrows():
        _, _, dz_current, dy_current, dx_current = imgs[site].shape
        dy, dx = index.values * np.array([dy_site + dy_margin, dx_site + dx_margin])
        canvas[:, :, :dz_current, dy : dy + dy_current, dx : dx + dx_current] = imgs[
            site
        ]
    return canvas


def arrange_in_buckets(
    raw_imgs: dict[str, NDArray],
    meta: pd.DataFrame,
    group_by: str,
    sort_by: str = None,
    outer_index_function: Callable[[NDArray], pd.DataFrame] = square_grid,
    outer_margin: int = 100,
    inner_index_function: Callable[[NDArray], pd.DataFrame] = square_grid,
    inner_margin: int = 10,
) -> NDArray:
    embs = list(set(raw_imgs.keys()).intersection(set(meta.index)))
    imgs = {k: v for k, v in raw_imgs.items() if k in embs}
    meta = meta.loc[embs]

    canvas_imgs = {}
    for bucket, bucket_meta in meta.groupby(group_by):
        if sort_by is None:
            sorter = pd.Series(index=bucket_meta.index, dtype=float)
        else:
            sorter = bucket_meta[sort_by]
        canvas_imgs[bucket] = arrange_on_grid(
            imgs, sorter, index_function=inner_index_function, margin_px=inner_margin
        )

    canvas = arrange_on_grid(
        canvas_imgs,
        pd.Series(index=np.unique(meta[group_by]), dtype=float),
        index_function=outer_index_function,
        margin_px=outer_margin,
    )
    return canvas


arrange_in_wells = partial(
    arrange_in_buckets,
    group_by="well",
    outer_index_function=well_grid,
    inner_index_function=square_grid,
)


def set_camera_view(viewer, camera):
    viewer.camera.center = camera.center
    viewer.camera.zoom = camera.zoom
    viewer.camera.angles = camera.angles


def get_camrea(viewer):
    return viewer.camera
