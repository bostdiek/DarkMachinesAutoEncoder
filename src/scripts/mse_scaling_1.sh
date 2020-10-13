#!/bin/bash
#SBATCH --job-name=mse1
#SBATCH -n 4               # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-08:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p fas_gpu,gpu,gpu_requeue           # Partition to submit to
#SBATCH --gres=gpu:1           # Partition to submit to
#SBATCH --mem=16G         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o /n/home02/bostdiek/DarkMachines/src/scripts/logs/mse_scaling_1.out  #
#SBATCH -e /n/home02/bostdiek/DarkMachines/src/scripts/logs/mse_scaling_1.err  #

source activate bostdiek_darkmachines
cd /n/home02/bostdiek/DarkMachines/src/models
python train.py --show --batch_size 500 --decoder_width 256 --encoder_width 256 --show_number=5000 --total_masked_weight 1 
