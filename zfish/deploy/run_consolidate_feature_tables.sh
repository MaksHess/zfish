#!/usr/bin/env bash

#SBATCH --array=9-17
#SBATCH --mem-per-cpu=10000m
#SBATCH --cpus-per-task=2
#SBATCH --error=./logs/slurm-%A_%a.err
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --time=3-00:00:00

source ~/.bashrc
conda activate zfish

exec python consolidate_feature_tables.py $SLURM_ARRAY_TASK_ID
