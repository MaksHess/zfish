# %%
import argparse
from pathlib import Path
from typing import Literal

import dask.array as da
import polars as pl
import polars.selectors as cs
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from pydantic_yaml import parse_yaml_file_as

from zfish.deploy.thumbnails.run_thumbnail_creation import ThumbnailCreationParams
from zfish.multi_table.features_aggregate import aggregate
from zfish.multi_table.raster_io import (
    NumpyImage,
    _aggregate_label_object_paths,
    _label_objects_to_queries,
    get_quantile_extent,
    label_objects_scale_bbx,
)
from zfish.multi_table.tables_io import read_resources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("idx", type=str)
    parser.add_argument("-p", "--thumbnail_params", type=str)
    args = parser.parse_args()

    params = parse_yaml_file_as(ThumbnailCreationParams, args.thumbnail_params)

    r = read_resources(root=params.resource_fld)

    df_meta = (
        aggregate(
            r.hierarchy,
            r.object_types,
            r.label_objects.select(~(cs.matches("bbx") | cs.matches("centroid"))),
            aggregate_to="embryoRaw",
        )
        .with_columns(
            pl.col("nucleiRaw3__Count").log(2).round().cast(pl.UInt8).alias("cycle")
        )
        .with_columns(pl.col("nucleiRaw3__Count").log(2).name.prefix("log2_"))
    )

    rows = []
    for name, df in (
        label_objects_scale_bbx(
            df_m=r.multiscale_levels,
            df_label_objects=r.label_objects.filter(pl.col("idx.o") == "cells"),
            to_level=params.base_level,
        )
        .join(df_meta, on="idx.roi")
        .sort(params.size_group)
        .group_by((params.size_group,), maintain_order=True)
    ):
        rows.append(
            list(name) + list(get_quantile_extent(df, quantile=params.size_quantile))
        )

    df_extent = (
        pl.DataFrame(rows, schema=[params.size_group, "bbx-e-z", "bbx-e-y", "bbx-e-x"])
        .with_columns(
            ((cs.starts_with("bbx") * (1 + params.size_padding_perc)).ceil()).cast(
                pl.Int32
            )
        )
        .with_columns(((cs.starts_with("bbx") / 2).ceil() * 2).cast(pl.Int32))
        .with_columns(
            pl.max_horizontal(pl.col("bbx-e-y"), pl.col("bbx-e-x")).alias("bbx-e-xy")
        )
        .with_columns(pl.col("cycle").cast(pl.UInt8))
    )

    rows_to_process = (
        df_meta.join(df_extent, on="cycle")
        .sort("log2_nucleiRaw3__Count")
        .select("idx.roi", "cycle", "bbx-e-z", "bbx-e-xy")
        .rows()
    )

    row_to_process = rows_to_process[int(args.idx)]

    roi, cycle, z_extent, xy_extent = row_to_process

    out_shape_full = (len(params.channels), z_extent, xy_extent, xy_extent)
    out_dims = ("c", "z", "y", "x")
    out_shape = tuple(
        e if params.project != dim else 1 for e, dim in zip(out_shape_full, out_dims)
    )

    df_imgs, df_label_objects = _aggregate_label_object_paths(
        r,
        bbx_object_type=params.bbx_object_type,
        channels=params.channels,
        channel_masks=params.channel_masks,
        z_model=params.z_model,
        t_model=params.t_model,
        rois=[roi],
        multiscale_level=params.base_level,
    )

    loqs = _label_objects_to_queries(
        df_label_objects,
        df_imgs,
        strategy="memory",
        dask_chunk_size=params.dask_chunk_size,
        fuzzy_mask_sigma=params.fuzzy_mask_sigma,
    )

    imgs = _pad(
        _project({e.object_id: e.compute() for e in loqs}, params.project), out_shape
    )

    thumbnail_arr = da.stack(list(e[0] for e in imgs.values()))
    thumbnail_idx_df = df_label_objects.select("idx.roi", "idx.m", "idx.o", "idx.label")

    fn_out = Path(params.out_fld) / params.out_fn
    fn_out_idx = Path(params.out_fld) / params.out_fn_idx_fstring.format(roi)

    fn_out.mkdir(exist_ok=True)

    store = parse_url(fn_out, mode="w").store
    root = zarr.group(store=store)
    site_group = root.create_group(roi)
    write_image(
        thumbnail_arr,
        group=site_group,
        axes=["t", "c", "z", "y", "x"],
        storage_options={"chunks": (-1, 1, -1, -1, -1)},
    )
    thumbnail_idx_df.write_parquet(fn_out_idx)


def _project(
    imgs: dict[str, NumpyImage], projection_axis: Literal["x", "y", "z"] | None = None
):
    if projection_axis is None:
        return imgs
    else:
        return {
            name: (
                img[0].max(img[1].dims.index(projection_axis), keepdims=True),
                img[1],
            )
            for name, img in imgs.items()
        }


def _pad(imgs: dict[str, NumpyImage], out_shape: tuple[int]):
    out_imgs = {}
    for name, img in imgs.items():
        in_shape = img[0].shape
        in_out_slices = [_in_out_to_slice(i, o) for i, o in zip(in_shape, out_shape)]
        in_slices = tuple(e[0] for e in in_out_slices)
        out_slices = tuple(e[1] for e in in_out_slices)

        out_arr = da.zeros(out_shape, dtype=img[0].dtype)
        out_arr[out_slices] = img[0][in_slices]
        out_imgs[name] = (out_arr, img[1])
    return out_imgs


def _in_out_to_slice(in_, out_):
    if in_ == out_:
        return slice(None), slice(None)
    if in_ > out_:
        offset = (in_ - out_) // 2
        return slice(0 + offset, out_ + offset), slice(None)
    if out_ > in_:
        offset = (out_ - in_) // 2
        return slice(None), slice(0 + offset, in_ + offset)


if __name__ == "__main__":
    main()
