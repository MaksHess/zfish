import argparse
import os
from pathlib import Path

from zfish.features.feature_extraction_parameters_v2 import FeatureExtractionParams

SLURM_COMMAND = """#!/usr/bin/env bash

#SBATCH --array=0-{0}%100
#SBATCH --mem-per-cpu=10000m
#SBATCH --cpus-per-task=2
#SBATCH --error=./logs/slurm-%a_%A.err
#SBATCH --output=./logs/slurm-%a_%A.out
#SBATCH --time=3-00:00:00

source ~/.bashrc
conda activate zfish

exec python feature_extraction_v2.py $SLURM_ARRAY_TASK_ID -p {1}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--feature_extraction_params', type=str)
    args = parser.parse_args()

    params = FeatureExtractionParams.parse_file(args.feature_extraction_params)
    if params.image_dir:
        fld = params.root / params.image_dir
    else:
        fld = params.root
    print(fld)
    n = len(list(Path(fld).glob('*.h5')))

    command = SLURM_COMMAND.format(n - 1, args.feature_extraction_params)
    print(command)
    with open("temp.sh", "w") as f:
        f.write(command)
    os.system("sbatch temp.sh")
    os.unlink("temp.sh")


if __name__ == "__main__":
    main()
