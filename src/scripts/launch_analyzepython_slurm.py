from subprocess import call
import numpy as np
import time
import os

channels = ['chan1', 'chan2a', 'chan2b', 'chan3']

for c in channels:
    files = os.listdir(f'/n/home02/bostdiek/DarkMachines/model/logs/{c}/')
    print(files)

    for f in files:
        if 'VariationalAutoEncoderBeta-1.0-8-256' not in f:
            continue
        fname = f'/n/home02/bostdiek/DarkMachines/src/scripts/bash_slurm/{c}_{f[:-4]}.sh'
        print(fname)
        with open(fname, 'w') as fout:
            fout.write("#!/bin/bash\n")
            fout.write(f"#SBATCH --job-name={c}_{f[:-4]}\n")
            fout.write("#SBATCH -n 4               # Number of cores\n")
            fout.write("#SBATCH -N 1                # Ensure that all cores are on one machine\n")
            fout.write("#SBATCH -t 0-04:00          # Runtime in D-HH:MM, minimum of 10 minutes\n")
            fout.write("#SBATCH -p fas_gpu           # Partition to submit to\n")
            fout.write("#SBATCH --gres=gpu:1           # Partition to submit to\n")
            fout.write("#SBATCH --mem=63000         # Memory pool for all cores (see also --mem-per-cpu)\n")
            fout.write(f"#SBATCH -o /n/home02/bostdiek/DarkMachines/src/scripts/logs/{c}_{f[:-4]}.out  #\n")
            fout.write(f"#SBATCH -e /n/home02/bostdiek/DarkMachines/src/scripts/logs/{c}_{f[:-4]}.err  #\n")
            # fout.write("#SBATCH --requeue\n")
            # fout.write("#SBATCH --array=1-5\n")
            fout.write("\n")
            fout.write("source activate bostdiek_darkmachines\n")
            fout.write("cd /n/home02/bostdiek/DarkMachines/src/models\n")
            fout.write(f"python analyze.py --name {c}/{f} \n")

        print('sbatch ' + fname)
        call('sbatch ' + fname, shell=True)
        time.sleep(1)
