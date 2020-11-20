#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:k80:1
##SBATCH --gres=gpu:1
#SBATCH --mem=10GB
#SBATCH --job-name=td3
#SBATCH --mail-type=END
#SBATCH --output=slurm_%j.out

python train.py
