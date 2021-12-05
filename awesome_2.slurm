#! /bin/sh

#SBATCH --job-name=awesome_2
#SBATCH --output=awesome_2.out
#SBATCH --error=awesome_2.err
#SBATCH --partition=studentbatch
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

python run_david.py
