import os
from os.path import join

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=cifar10
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1 
#SBATCH --mem-per-cpu=20G
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output={workdir}/output/multi_gan/run_%j.out
#SBATCH --error={workdir}/output/multi_gan/run_%j.err
#SBATCH --qos=qos_gpu-t3
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --account=glp@gpu

sig_handler_USR1()
{{
        echo "Will died in 120s. Rescheduling"
        sbatch {job_file}
        exit 2
}}

trap 'sig_handler_USR1' SIGUSR1


module load openmpi/4.0.2-cuda
module load cuda/10.1.1
module load nccl/2.5.6-2-cuda
module load cudnn/10.1-v7.5.1.10
module load anaconda-py3/2019.03

set -x

eval "$(conda shell.bash hook)"
conda activate custom

export GIT_PYTHON_REFRESH=quiet

python {workdir}/work/repos/multi_gan/scripts/run_training.py with {parameters} -F {output_dir}
"""

workdir = os.environ['WORK']
script_file = os.path.join(workdir, 'work/repos/multi_gan/run_training.py')
basedir = os.path.join(workdir, 'output/multi_gan/mnist')
grid = 'final_mixed_nash'

if not os.path.exists(basedir):
    os.makedirs(basedir)

if grid == 'final_mixed_nash':
    i = 0
    parameters = []
    for seed in [100]:
        for lr in [5e-5]:
            for nG, nD in [(2, 1), (2, 2)]:
                for mirror_lr in [0., 3e-2]:
                    parameters.append(dict(n_generators=nG,
                                           n_discriminators=nD,
                                           D_lr=10 * lr, G_lr=lr,
                                           mirror_lr=mirror_lr, sampling='all_extra',
                                           seed=seed))
else:
    raise ValueError

for i, parameter in enumerate(parameters):
    output_dir = join(basedir, grid, str(i))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    job_file = os.path.join(output_dir, "job.slurm")
    parameters_str = 'mnist output_dir={} '.format(output_dir)
    for key in parameter:
        parameters_str += "{}={} ".format(key, parameter[key])
    template = SLURM_TEMPLATE.format(parameters=parameters_str, job_file=job_file, output_dir=output_dir,
                                     workdir=workdir)
    with open(job_file, 'w+') as f:
        # print(job_file)
        f.writelines(template)
    i += 1
    os.system("sbatch %s" % job_file)
