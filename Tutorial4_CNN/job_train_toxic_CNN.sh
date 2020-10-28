#!/bin/bash
#SBATCH -J Train_Toxic_CNN
#SBATCH --nodes=1
#SBATCH --mem=10gb
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=intel

ID=$SLURM_JOB_ID

date
python /home/mvernier/INFO082_2020/Tutorial4_CNN/train_toxic_CNN.py
