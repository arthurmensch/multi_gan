import math
import math
import os
import random
from os.path import expanduser, join

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rc
from sacred import Experiment
from torch.utils.data import DataLoader

from multi_gan.data import make_8gmm, infinite_iter, make_25gmm
from multi_gan.models import GeneratorDCGAN28, DiscriminatorDCGAN28, GeneratorSynthetic, GeneratorResNet32, \
    DiscriminatorSynthetic, DiscriminatorResNet32

matplotlib.rcParams['backend'] = 'pdf'
rc('text', usetex=True)
exp = Experiment('multi_gan')
exp_dir = expanduser('~/output/multi_gan')
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)


@exp.config
def synthetic():
    data_type = 'synthetic'
    data_source = '8gmm'

    n_generators = 3
    n_discriminators = 3

    depth = 3

    ndf = 512
    ngf = 512

    loss_type = 'wgan'
    device = 'cpu'
    n_iter = 30000

    noise_dim = 32

    batch_size = 512
    D_lr = 1e-3
    G_lr = 1e-4
    mirror_lr = 1e-1

    beta1 = 0.
    beta2 = 0.999

    grad_penalty = 10

    sampling = 'all_extra_alternated'
    fused_noise = True

    seed = 100
    eval_every = 500

    print_every = 500
    eval_device = 'cuda:0'
    eval_fid = False
    restart = True
    output_dir = expanduser('~/output/multi_gan/synthetic/final_mixed_nash_5/19')


@exp.main
def train(n_generators, n_discriminators, noise_dim, ngf, ndf, grad_penalty, sampling, mirror_lr, restart,
          output_dir, data_type, data_source, depth, print_every, eval_device, eval_fid, fused_noise,
          beta1, beta2,
          device, batch_size, n_iter, loss_type, eval_every, D_lr, G_lr, _seed, _run):
    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


    if data_source == '8gmm':
        dataset, sampler = make_8gmm()
    elif data_source == '25gmm':
        dataset, sampler = make_25gmm()
    else:
        raise ValueError()
    dataset.tensors = tuple(tensor.to(device) for tensor in dataset.tensors)
    stat_path = None
    fixed_noise = torch.randn(512, noise_dim).to(device)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    data_loader = infinite_iter(data_loader)
    fixed_data = next(data_loader)[0].to('cpu')

    torch.random.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)

    fig, axes = plt.subplots(1, 2, figsize=(4, 2), constrained_layout=False)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0.05, hspace=0, wspace=0    )

    for ax, (output_dir, n_generators, n_discriminators) in zip(axes, [
        (expanduser('~/output/multi_gan/synthetic/final_mixed_nash_5/14'), 5, 5),
        (expanduser('~/output/multi_gan/synthetic/final_mixed_nash_5/18'), 1, 1),
    ]):
        def make_generator():
            if data_type == 'synthetic':
                return GeneratorSynthetic(depth=depth, n_filters=ngf, batch_norm=True)
            else:
                if data_source == 'mnist':
                    return GeneratorDCGAN28(in_features=noise_dim, out_channels=1, n_filters=ngf)
                elif data_source == 'multimnist':
                    return GeneratorDCGAN28(in_features=noise_dim, out_channels=3, n_filters=ngf)
                elif data_source == 'cifar10':
                    return GeneratorResNet32(n_in=noise_dim, n_out=3, num_filters=ngf)

        def make_discriminator():
            batch_norm = loss_type != 'wgan-gp'
            if data_type == 'synthetic':
                return DiscriminatorSynthetic(depth=depth, n_filters=ndf, batch_norm=False)
            else:
                if data_source == 'mnist':
                    return DiscriminatorDCGAN28(in_channels=1, n_filters=ndf, n_targets=1, batch_norm=batch_norm)
                elif data_source == 'multimnist':
                    return DiscriminatorDCGAN28(in_channels=3, n_filters=ndf, n_targets=1, batch_norm=batch_norm)
                elif data_source == 'cifar10':
                    return DiscriminatorResNet32(n_in=3, num_filters=ndf)

        players = {'G': {}, 'D': {}}
        for G in range(n_generators):
            players['G'][G] = make_generator().to(device)
        for D in range(n_discriminators):
            players['D'][D] = make_discriminator().to(device)

        for group, these_players in players.items():
            for player in these_players.values():
                player.train()

        log_weights = {'G': torch.full((n_generators,), fill_value=-math.log(n_generators), device=device),
                       'D': torch.full((n_discriminators,), fill_value=-math.log(n_discriminators), device=device)}
        if 'checkpoint' in os.listdir(output_dir) and restart:
            for name in ['checkpoint', 'last_checkpoint']:
                try:
                    checkpoint = torch.load(join(output_dir, name), map_location='cpu')
                    bundle = joblib.load(join(output_dir, f'{name}_bundle'))
                except (FileNotFoundError, EOFError):
                    print(f'Cannot open {name}')
                else:
                    print(f'Restarting from {join(output_dir, name)}')
                    for group, these_players in players.items():
                        for P, player in these_players.items():
                            players[group][P].load_state_dict(checkpoint['players'][group][P])
                            # averagers[group][P].load_state_dict(checkpoint['averagers'][group][P])
                        log_weights[group] = checkpoint['log_weights'][group]
                    break


        for G, generator in players['G'].items():
            generator.eval()
        if data_type == 'synthetic':
            ax.scatter(fixed_data[:, 0], fixed_data[:, 1], label='True', marker='v',
                       zorder=1, alpha=.8)
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            p = torch.exp(log_weights['G']).cpu().numpy()
            p /= np.sum(p)
            selection = torch.from_numpy(np.random.choice(n_generators, size=fixed_noise.shape[0], p=p))
            fake_points = torch.zeros((fixed_noise.shape[0], 2))
            for G, generator in players['G'].items():
                with torch.no_grad():
                    mask = selection == G
                    if torch.sum(mask) > 0:
                        fake_points[selection == G] = generator(fixed_noise[selection == G]).cpu()
                ax.scatter(fake_points[selection == G][:, 0], fake_points[selection == G][:, 1], label=f'G{G}', zorder=100,
                           alpha=.8)
            ax.axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', frameon=False, ncol=6, bbox_to_anchor=(0, 1.05),columnspacing=0.5, handletextpad=0.2)
    # axes[0].legend(loc='lower left', frameon=False, ncol=6, bbox_to_anchor=(-0.1, -0.1))
    axes[0].annotate('Multi G, multi D', xy=(0.5, 0), ha='center', xycoords='axes fraction', fontsize=12)
    axes[1].annotate('Single G, single D', xy=(0.5, 0), ha='center', xycoords='axes fraction', fontsize=12)
    plt.savefig('synthetic_images.pdf')
    plt.show()


if __name__ == '__main__':
    exp.run_commandline()
