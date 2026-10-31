#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p edu-medium
#SBATCH -A fabrizio.granelli.edu
#SBATCH --qos=gpuedu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=out_%j.log

source ~/Sure_Twin/.venv/bin/activate
cd ~/Sure_Twin
python train_STID.py
