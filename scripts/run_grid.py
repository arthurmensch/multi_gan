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
#SBATCH -A snt@gpu

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
basedir = os.path.join(workdir, 'output/multi_gan/cifar10')


i = 0
parameters = []
for n_generators in [1, 3, 5]:
    for n_discriminators in [1, 3]:
        for sampling in ['all', 'pair']:
            parameters.append(dict(n_generators=n_generators,
                                   n_discriminators=n_discriminators,
                                   mirror_lr=0, sampling=sampling))
            if n_generators > 1 or n_generators > 1 and sampling == 'all':
                parameters.append(dict(n_generators=n_generators,
                                       n_discriminators=n_discriminators,
                                       mirror_lr=1e-2, sampling=sampling))
parameters.append(dict(n_generators=4, noise_dim=32,
                       n_discriminators=1,
                       mirror_lr=0, sampling='all'))
parameters.append(dict(n_generators=3,
                       n_discriminators=3,
                       mirror_lr=0, sampling='all', fused_noise=False))

for i, parameter in enumerate(parameters):
    output_dir = join(basedir, str(i))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    job_file = os.path.join(output_dir, "job.slurm")
    parameters_str = 'cifar output_dir={} '.format(output_dir)
    for key in parameter:
        parameters_str += "{}={} ".format(key, parameter[key])
    template = SLURM_TEMPLATE.format(parameters=parameters_str, job_file=job_file, output_dir=output_dir,
                                     workdir=workdir)
    with open(job_file, 'w+') as f:
        f.writelines(template)
    i += 1
    os.system("sbatch %s" % job_file)