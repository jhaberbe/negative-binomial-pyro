#!/bin/bash
#SBATCH --job-name=negative_binomial_model_fitter
#SBATCH -p owners
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=0-28206
#SBATCH --time=04:00:00
#SBATCH --mem=30000
#SBATCH -c 10

ml python/3.12.1

echo "Processing string: $SLURM_ARRAY_TASK_ID"

source .venv/bin/activate
python3 main.py --adata /oak/stanford/projects/kibr/Reorganizing/Projects/James/negative-binomial-pyro/../choroid-plexus-cell-atlas/data/new_annotations.h5ad --gene $SLURM_ARRAY_TASK_ID