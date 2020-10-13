from subprocess import call
import numpy as np
import time

for LatentSize in [256, 512]:
    fname = f'/n/home02/bostdiek/DarkMachines/src/scripts/latent_size_{LatentSize}.sh'
    with open(fname, 'w') as fout:
        fout.write("#!/bin/bash\n")
        fout.write(f"#SBATCH --job-name=LS{LatentSize}\n")
        fout.write("#SBATCH -n 4               # Number of cores\n")
        fout.write("#SBATCH -N 1                # Ensure that all cores are on one machine\n")
        fout.write("#SBATCH -t 0-08:00          # Runtime in D-HH:MM, minimum of 10 minutes\n")
        fout.write("#SBATCH -p fas_gpu,gpu           # Partition to submit to\n")
        fout.write("#SBATCH --gres=gpu:1           # Partition to submit to\n")
        fout.write("#SBATCH --mem=16G         # Memory pool for all cores (see also --mem-per-cpu)\n")
        fout.write(f"#SBATCH -o /n/home02/bostdiek/DarkMachines/src/scripts/logs/latentsize_{LatentSize}.out  #\n")
        fout.write(f"#SBATCH -e /n/home02/bostdiek/DarkMachines/src/scripts/logs/latentsize_{LatentSize}.err  #\n")
        # fout.write("#SBATCH --requeue\n")
        # fout.write("#SBATCH --array=1-5\n")
        fout.write("\n")
        fout.write("source activate bostdiek_darkmachines\n")
        fout.write("cd /n/home02/bostdiek/DarkMachines/src/models\n")
        fout.write(f"python train.py --show --batch_size 500 --epochs 20 --decoder_width 256 --encoder_width 256 --show_number=5000 --latent_size {LatentSize} --total_masked_weight 0.001 \n")

    print('sbatch ' + fname)
    call('sbatch ' + fname, shell=True)
    time.sleep(1)
