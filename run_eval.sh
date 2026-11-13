#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p edu-medium
#SBATCH -A fabrizio.granelli.edu
#SBATCH --qos=gpuedu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-9
#SBATCH --output=eval_%A_%a.log

# Usage:
#   sbatch run_eval.sh val               # all 10 seeds on the val split
#   sbatch --array=0 run_eval.sh val     # seed 0 only (first test)
#   sbatch run_eval.sh train             # train split (diagnostic only)
SPLIT=${1:-val}

source ~/Sure_Twin/.venv/bin/activate
cd ~/Sure_Twin
python eval_split.py --split "$SPLIT" --seed "$SLURM_ARRAY_TASK_ID"
