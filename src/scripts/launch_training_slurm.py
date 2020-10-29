from subprocess import call
import numpy as np
import time

LatentSize = 8
for Beta in [1e-6, 1e-3, 1e-1, 0.5, 0.8, 0.999, 1]:
    for ClassPredWeight in [0.1, 1, 10, 100]:
        fname = f'/n/home02/bostdiek/DarkMachines/src/scripts/bash_slurm/betavae_{Beta}_{ClassPredWeight}.sh'
        with open(fname, 'w') as fout:
            fout.write("#!/bin/bash\n")
            fout.write(f"#SBATCH --job-name=bvae{Beta}_{ClassPredWeight}\n")
            fout.write("#SBATCH -n 8               # Number of cores\n")
            fout.write("#SBATCH -N 1                # Ensure that all cores are on one machine\n")
            fout.write("#SBATCH -t 0-08:00          # Runtime in D-HH:MM, minimum of 10 minutes\n")
            fout.write("#SBATCH -p fas_gpu,gpu           # Partition to submit to\n")
            fout.write("#SBATCH --gres=gpu:1           # Partition to submit to\n")
            fout.write("#SBATCH --mem=32G         # Memory pool for all cores (see also --mem-per-cpu)\n")
            fout.write(f"#SBATCH -o /n/home02/bostdiek/DarkMachines/src/scripts/logs/betavae_{Beta}_{ClassPredWeight}.out  #\n")
            fout.write(f"#SBATCH -e /n/home02/bostdiek/DarkMachines/src/scripts/logs/betavae_{Beta}_{ClassPredWeight}.err  #\n")
            # fout.write("#SBATCH --requeue\n")
            # fout.write("#SBATCH --array=1-5\n")
            fout.write("\n")
            fout.write("source activate bostdiek_darkmachines\n")
            fout.write("cd /n/home02/bostdiek/DarkMachines/src/models\n")
            fout.write(f"python train.py --show --num_workers 4 --batch_size 500 --epochs 50 --vae --beta={Beta} --decoder_width 256 --encoder_width 256 --class_pred_weight {ClassPredWeight} --show_number=5000 --latent_size {LatentSize} --total_masked_weight 0.0001 \n")

        print('sbatch ' + fname)
        call('sbatch ' + fname, shell=True)
        time.sleep(1)