#!/usr/bin/env bash

#SBATCH --array=0-17%100
#SBATCH --mem-per-cpu=10000m
#SBATCH --cpus-per-task=2
#SBATCH --error=./logs/slurm-%A_%a.err
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --time=3-00:00:00

source ~/.bashrc
conda activate zfish

exec python tracking.py $SLURM_ARRAY_TASK_ID -p /data/active/marvwy/VisiScope/20230329_compressed/20230329-H1-GFP2_s4.parquet