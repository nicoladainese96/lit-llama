#!/bin/bash
#SBATCH --time=0-01
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -e slurm/slurm-%j.err
#SBATCH -o slurm/slurm-%j.out

hostname
#module load LUMI/22.08 partition/G
echo python $* 
python $* 
