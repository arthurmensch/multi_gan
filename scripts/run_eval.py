import os
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
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from multi_gan.data import make_8gmm, infinite_iter, make_image_data, make_25gmm
from multi_gan.models import GeneratorDCGAN28, DiscriminatorDCGAN28, GeneratorSynthetic, GeneratorResNet32, \
    DiscriminatorSynthetic, DiscriminatorResNet32
from multi_gan.optimizers import ExtraOptimizer, ParamAverager
from multi_gan.training import Scheduler

matplotlib.rcParams['backend'] = 'pdf'
rc('text', usetex=True)

output_dirs = [expanduser('~/output/multi_gan/cifar10'), expanduser('~/output/multi_gan/cifar10_final'),
               expanduser('~/output/multi_gan/cifar10_nplayer')]

exp = Experiment('multi_gan')
exp_dir = expanduser('~/output/multi_gan')
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)


@exp.config
def cifar():
    data_type = 'image'
    data_source = 'cifar10'
    n_generators = 5
    n_discriminators = 1

    depth = 1
    ndf = 128
    ngf = 128

    loss_type = 'wgan-gp'
    device = 'cpu'
    n_iter = int(5e5)

    noise_dim = 128  # For image data

    batch_size = 64
    D_lr = 3e-4
    G_lr = 3e-5
    mirror_lr = 0

    grad_penalty = 10

    sampling = 'all'
    fused_noise = True

    seed = 0

    eval_fid = True
    print_every = 10000
    eval_every = 10000
    eval_device = 'cuda:0'

    restart = True
    output_dir = expanduser('~/output')
    output_dir = expanduser('~/output/multi_gan/cifar10/14/14/')


@exp.main
def train(n_generators, n_discriminators, noise_dim, ngf, ndf, grad_penalty, sampling, mirror_lr, restart,
          output_dir, data_type, data_source, depth, print_every, eval_device, eval_fid, fused_noise,
          device, batch_size, n_iter, loss_type, eval_every, D_lr, G_lr, _seed, _run):
    torch.random.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)

    output_dir = output_dir.replace('$WORK', os.environ['WORK'])

    if not torch.cuda.is_available():
        device = 'cpu'

    if output_dir is None:
        output_dir = join(_run.observers[0].dir, 'artifacts')
    else:
        output_dir = output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    writer = SummaryWriter(log_dir=output_dir)

    if data_type == 'synthetic':
        if data_source == '8gmm':
            dataset, sampler = make_8gmm()
        elif data_source == '25gmm':
            dataset, sampler = make_25gmm()
        else:
            raise ValueError()
        dataset.tensors = dataset.tensors.to(device)
        stat_path = None
    elif data_type == 'image':
        dataset, stat_path = make_image_data(data_source)
        sampler = None
    else:
        raise ValueError()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    data_loader = infinite_iter(data_loader)

    def make_generator():
        if data_type == 'synthetic':
            return GeneratorSynthetic(depth=depth, n_filters=ngf)
        else:
            if data_source == 'mnist':
                return GeneratorDCGAN28(in_features=noise_dim, out_channels=1, n_filters=ngf)
            elif data_source == 'multimnist':
                return GeneratorDCGAN28(in_features=noise_dim, out_channels=3, n_filters=ngf)
            elif data_source == 'cifar10':
                return GeneratorResNet32(n_in=noise_dim, n_out=3, num_filters=ngf)

    def make_discriminator():
        if data_type == 'synthetic':
            return DiscriminatorSynthetic(depth=depth, n_filters=ndf)
        else:
            batch_norm = loss_type != 'wgan-gp'
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

    averagers = {group: {i: ParamAverager(player) for i, player in these_players.items()}
                 for group, these_players in players.items()}
    optimizers = {group: {i: ExtraOptimizer(Adam(player.parameters(),
                                                 lr=D_lr if group == 'D' else G_lr, betas=(0.5, 0.9)))
                          for i, player in these_players.items()}
                  for group, these_players in players.items()}

    log_weights = {'G': torch.full((n_generators,), fill_value=1 / n_generators, device=device),
                   'D': torch.full((n_discriminators,), fill_value=1 / n_discriminators, device=device)}
    losses = {'G': torch.zeros((n_generators, n_discriminators), device=device),
              'D': torch.zeros((n_discriminators, n_generators), device=device)}
    counts = {'G': torch.zeros((n_generators, n_discriminators), device=device),
              'D': torch.zeros((n_discriminators, n_generators), device=device)}
    last_losses = {'G': None, 'D': None}
    fixed_data = next(data_loader)[0].to('cpu')

    if data_type == 'synthetic':
        fixed_noise = torch.randn(512, noise_dim).to(device)
        noise_loader = None
        true_loglike = sampler.log_prob(fixed_data).mean().item()
    else:
        fixed_noise = torch.randn(10000, noise_dim).to(device)
        noise_loader = DataLoader(TensorDataset(fixed_noise), batch_size=batch_size)
        true_loglike = None

    scheduler = Scheduler(n_generators, n_discriminators, sampling=sampling)

    n_gen_upd, n_steps, n_data, n_noise = 0, 0, 0, 0
    next_print_step = 0
    next_eval_step = 0

    if 'checkpoint' in os.listdir(output_dir) and restart:
        for name in ['checkpoint', 'last_checkpoint']:
            try:
                checkpoint = torch.load(join(output_dir, name), map_location=torch.device('cpu'))
                bundle = joblib.load(join(output_dir, f'{name}_bundle'))
            except EOFError:
                print(f'Cannot open {name}')
                continue
            else:
                print(f'Restarting from {join(output_dir, name)}')
                n_gen_upd = checkpoint['n_gen_upd']
                n_steps = checkpoint['n_steps']
                n_data = checkpoint['n_data']
                n_noise = checkpoint['n_noise']
                next_print_step = checkpoint['next_print_step']
                next_eval_step = checkpoint['next_eval_step']
                for group, these_players in players.items():
                    for P, player in these_players.items():
                        players[group][P].load_state_dict(checkpoint['players'][group][P])
                        optimizers[group][P].load_state_dict(checkpoint['optimizers'][group][P])
                        averagers[group][P].load_state_dict(checkpoint['averagers'][group][P])
                    # log_weights = checkpoint['log_weights'][group]
                    losses[group] = checkpoint['past_losses'][group]
                    counts[group] = checkpoint['count_losses'][group]
                scheduler = bundle['scheduler']
                break
    fake_images = {}
    for G, generator in players['G'].items():
        generator.eval()
        with torch.no_grad():
            fake_images[G] = ((generator(fixed_noise[:8]) + 1) / 2).numpy()
    joblib.dump(fake_images, 'fake_images.pkl')


def plot():
    fake_images = joblib.load('fake_images.pkl')
    print(len(fake_images))
    fig = plt.figure(figsize=(4, 2), constrained_layout=False, dpi=600)
    fig.subplots_adjust(left=0.03, right=1, bottom=0.1, top=0.9)
    print(fake_images)
    outer_grid = fig.add_gridspec(2, 3, wspace=.1, hspace=0.1)
    print(outer_grid)
    for i, this_grid in enumerate(outer_grid):
        if i == 5:
            ax = fig.add_subplot(this_grid)
            ax.axis('off')
            ax.annotate(f'Fake CIFAR10\n'
                        f'WFR flow on\n'
                        f'5 gen. 2 discr.', xy=(0.5, .5), xytext=(0, 0), xycoords='axes fraction',
                        ha='center', va='center',
                        textcoords='offset points', fontsize=12)
            fig.add_subplot(ax)
        else:
            inner_grid = this_grid.subgridspec(2, 4, wspace=0.0, hspace=0.0)
            for j, this_inner_grid in enumerate(inner_grid):
                ax = fig.add_subplot(this_inner_grid)
                # ax.plot(range(10), range(10))
                ax.axis('off')
                ax.imshow(fake_images[i][j].transpose(1, 2, 0))
                fig.add_subplot(ax)
            ax = fig.add_subplot(this_grid)
            ax.axis('off')
            if 0 <= i <= 2:
                ax.annotate(f'G{i}', xy=(0.5, 1.0), xytext=(0, 0), xycoords='axes fraction',
                            ha='center', va='bottom',
                            textcoords='offset points', fontsize=12)
            else:
                ax.annotate(f'G{i}', xy=(0.5, 0.0), xytext=(0, 0), xycoords='axes fraction',                        ha='center', va='top',
                            textcoords='offset points', fontsize=12)
            fig.add_subplot(ax)
    plt.savefig('images.pdf')
    plt.show()


if __name__ == '__main__':
    # exp.run_commandline()
    plot()
