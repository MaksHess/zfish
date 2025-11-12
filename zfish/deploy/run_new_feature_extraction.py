import argparse
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic_yaml import parse_yaml_file_as

from zfish.multi_table.feature_query_schemas import _construct_correlation_queries
from zfish.multi_table.tables_io import read_resources


class CorrelationFeatureExtractionParams(BaseModel):
    resource_path: Path
    output_path: Path
    levels: tuple[int, ...]
    object_types: tuple[str, ...]
    channel_pairs: tuple[tuple[str, str], ...]
    features: tuple[str, ...]
    wells: tuple[str, ...] | None = None
    rois: tuple[str, ...] | None = None
    parallelize_over: tuple[str, ...] = ("idx.m", "idx.roi")
    z_model: str | None = None
    t_model: str | None = None
    process_queries: Literal["invalid", "any_valid", "all_valid"] = "any_valid"


SLURM_COMMAND = """#!/usr/bin/env bash

#SBATCH --array=0-{0}%200
#SBATCH --mem-per-cpu=10000m
#SBATCH --cpus-per-task=2
#SBATCH --error=./logs/slurm-%A_%a.err
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --time=3-00:00:00

source ~/.bashrc
conda activate zfish

exec python new_feature_extraction.py $SLURM_ARRAY_TASK_ID -p {1}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--feature_extraction_params", type=str)
    args = parser.parse_args()

    # params = FeatureExtractionParams.parse_file(args.feature_extraction_params)
    params = parse_yaml_file_as(
        CorrelationFeatureExtractionParams, args.feature_extraction_params
    )

    r = read_resources(params.resource_path, use_pyarrow=False)
    df_all_queries = _construct_correlation_queries(
        r,
        levels=params.levels,
        wells=params.wells,
        rois=params.rois,
        channel_pairs=params.channel_pairs,
        object_types=params.object_types,
        features=params.features,
        return_queries=params.process_queries,
    )

    dfs_sub_queries = list(
        df_all_queries.group_by(params.parallelize_over, maintain_order=True)
    )
    n = len(dfs_sub_queries)
    array_task_id = n - 1

    command = SLURM_COMMAND.format(array_task_id, args.feature_extraction_params)
    print(command)

    temp_file_path = f"{Path(__file__).stem}-temp.sh"
    with open(temp_file_path, "w") as f:
        f.write(command)
    os.system(f"sbatch {temp_file_path}")
    os.unlink(temp_file_path)


if __name__ == "__main__":
    main()
