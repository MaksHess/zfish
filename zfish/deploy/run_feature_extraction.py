import argparse
import os
from pathlib import Path

from pydantic_yaml import parse_yaml_file_as

from zfish.features.feature_extraction_parameters import FeatureExtractionParams

SLURM_COMMAND = """#!/usr/bin/env bash

#SBATCH --array=0-{0}%100
#SBATCH --mem-per-cpu=10000m
#SBATCH --cpus-per-task=2
#SBATCH --error=./logs/slurm-%A_%a.err
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --time=3-00:00:00

source ~/.bashrc
conda activate zfish

exec python feature_extraction.py $SLURM_ARRAY_TASK_ID -p {1}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--feature_extraction_params', type=str)
    args = parser.parse_args()

    # params = FeatureExtractionParams.parse_file(args.feature_extraction_params)
    params = parse_yaml_file_as(FeatureExtractionParams, args.feature_extraction_params)
    if params.image_dir:
        fld = params.root / params.image_dir
    else:
        fld = params.root
    print(fld)
    fns = list(Path(fld).glob('*.h5'))
    n = len(fns)
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
