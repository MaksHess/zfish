import argparse
import os
from pathlib import Path
from typing import Literal

import polars as pl
import polars.selectors as cs
from pydantic import BaseModel
from pydantic_yaml import parse_yaml_file_as

from zfish.multi_table.features_aggregate import aggregate
from zfish.multi_table.raster_io import (
    get_quantile_extent,
    label_objects_scale_bbx,
)
from zfish.multi_table.tables_io import read_resources


class ThumbnailCreationParams(BaseModel):
    out_fld: str
    out_fn: str
    out_fn_idx_fstring: str
    resource_fld: str = (
        "/data/active/hmax/MARVWY_RESTORED/20220721_ZE4i2_aligned1/multi_tables"
    )
    bbx_object_type: str = "cells"
    channels: tuple[str, ...] = (
        "DAPI.1",
        "bCatenin.1",
        "Pol-II-S2P.0",
        "Pol-II-S5P.2",
        "PCNA.0",
    )
    channel_masks: tuple[str, ...] = (
        "nucleiRaw3",
        "cells",
        "nucleiRaw3",
        "nucleiRaw3",
        "nucleiRaw3",
    )
    z_model: str = "ExpPos-2D"
    t_model: str = "Exp"
    size_group: str = "cycle"
    size_quantile: float = 0.99
    size_padding_perc: float = 0.1
    project: Literal["z", "y", "x"] = "z"
    base_level: int = 0
    dask_chunk_size: tuple[int, ...] | None = (10, 2000, 2000)
    fuzzy_mask_sigma: int | None = 1


SLURM_COMMAND = """#!/usr/bin/env bash

#SBATCH --array=0-{0}%200
#SBATCH --mem-per-cpu=15000m
#SBATCH --cpus-per-task=4
#SBATCH --error=./logs/slurm-%A_%a.err
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --time=3-00:00:00

source ~/.bashrc
conda activate zfish

exec python thumbnail_creation.py $SLURM_ARRAY_TASK_ID -p {1}
"""


def main():
    parser = argparse.ArgumentParser()
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

    n_jobs = len(rows_to_process)
    array_task_id = n_jobs - 1

    command = SLURM_COMMAND.format(array_task_id, args.thumbnail_params)
    print(command)

    temp_file_path = f"{Path(__file__).stem}-temp.sh"
    with open(temp_file_path, "w") as f:
        f.write(command)
    os.system(f"sbatch {temp_file_path}")
    os.unlink(temp_file_path)


if __name__ == "__main__":
    main()
