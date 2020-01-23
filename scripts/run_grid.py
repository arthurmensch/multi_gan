import os
from os.path import expanduser

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=dopamine
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1 
#SBATCH --mem-per-cpu=20G
#SBATCH --hint=nomultithread
#SBATCH --time=2:00:00
#SBATCH --output=$WORK/output/games_rl/multi_gan/run_%j.out
#SBATCH --error=$WORK/output/games_rl/multi_gan/run_%j.err
#SBATCH --qos=qos_gpu-t3
#SBATCH -A snt@gpu

module load openmpi/4.0.2-cuda
module load cuda/10.1.1
module load nccl/2.4.2-1+cuda10.1
module load cudnn/10.1-v7.5.1.10
module load anaconda-py3/2019.03

conda activate custom

set -x

python $WORK/work/joan/games_rl/code/scripts/multi_gan/run_training.py with {parameters}
"""

job_directory = expanduser('~/output/games_rl/multi_gan/jobs_seed')
if not os.path.exists(job_directory):
    os.makedirs(job_directory)
script_file = expanduser('$WORK/work/joan/games_rl/code/scripts/multi_gan/run_training.py')

i = 0
for n_generators in [1, 3]:
    for n_discriminators in [1, 3]:
        for mirror_lr in [0, 1e-2]:
            for sampling in ['all']:
                job_file = os.path.join(job_directory, f"job_{i}.slurm")
                parameters = 'n_generators={} n_discriminators={} mirror_lr={} sampling={}'.format(
                    n_generators, n_discriminators, mirror_lr, sampling
                )
                template = SLURM_TEMPLATE.format(parameters=parameters)
                with open(job_file, 'w+') as f:
                    f.writelines(template)
                i += 1
                # os.system("sbatch %s" % job_file)