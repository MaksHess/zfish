import argparse
import os
from pathlib import Path

SLURM_COMMAND = """#!/usr/bin/env bash

#SBATCH --array=0-{0}%50
#SBATCH --mem-per-cpu=20000m
#SBATCH --cpus-per-task=5
#SBATCH -e errors.txt
#SBATCH -o out.txt
#SBATCH --time=3-00:00:00

source ~/.bashrc
conda activate elastix

exec python feature_extraction.py $SLURM_ARRAY_TASK_ID {1} -p {2}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--feature_extraction_params', type=str)
    args = parser.parse_args()

    n = len(list(Path(args.fld).glob('*.h5')))

    command = SLURM_COMMAND.format(n - 1, args.feature_extraction_params)
    print(command)
    with open("temp.sh", "w") as f:
        f.write(command)
    os.system("sbatch temp.sh")
    os.unlink("temp.sh")


if __name__ == "__main__":
    main()
