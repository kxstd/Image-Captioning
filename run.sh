#!/bin/bash

#SBATCH -p dgx2
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres gpu:1
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

module load openjdk/1.8.0_222-b10-gcc-8.3.0

python main.py train_evaluate --config_file configs/baseline.yaml

