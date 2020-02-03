import csv
import math
import os
import random
from os.path import expanduser, join

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from multi_gan.data import make_8gmm, infinite_iter, make_image_data, make_25gmm
from multi_gan.eval.fid_score import calculate_fid_given_paths
from multi_gan.losses import compute_gan_loss, compute_grad_penalty
from multi_gan.models import GeneratorDCGAN28, DiscriminatorDCGAN28, GeneratorSynthetic, GeneratorResNet32, \
    DiscriminatorSynthetic, DiscriminatorResNet32
from multi_gan.optimizers import ExtraOptimizer, ParamAverager
from multi_gan.training import enable_grad_for, Scheduler
from sacred import Experiment
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

exp = Experiment('multi_gan')
exp_dir = expanduser('~/output/multi_gan')
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)


@exp.config
def synthetic():
    data_type = 'synthetic'
    data_source = '8gmm'

    n_generators = 1
    n_discriminators = 1

    depth = 1

    ndf = 512 // n_discriminators
    ngf = 512 // n_generators

    loss_type = 'wgan'
    device = 'cuda:0'
    n_iter = 30000

    noise_dim = 32

    batch_size = 512
    D_lr = 5e-4
    G_lr = 5e-5
    mirror_lr = 0

    grad_penalty = 10

    sampling = 'all'
    fused_noise = True

    seed = 100
    eval_every = 500

    print_every = 500
    eval_device = 'cuda:0'
    eval_fid = False
    restart = False
    output_dir = None


@exp.named_config
def cifar():
    data_type = 'image'
    data_source = 'cifar10'
    n_generators = 3
    n_discriminators = 3

    depth = 1
    ndf = 128
    ngf = 128

    loss_type = 'wgan-gp'
    device = 'cuda:0'
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


@exp.named_config
def mnist():
    data_type = 'image'
    data_source = 'mnist'
    n_generators = 3
    n_discriminators = 3

    depth = 1
    ndf = 128
    ngf = 128

    loss_type = 'wgan-gp'
    device = 'cuda:0'
    n_iter = int(5e4)

    noise_dim = 128  # For image data

    batch_size = 64
    D_lr = 1e-3
    G_lr = 1e-4
    mirror_lr = 0

    grad_penalty = 10

    sampling = 'all'
    fused_noise = True

    seed = 0

    eval_fid = True
    print_every = 100
    eval_every = 100
    eval_device = 'cuda:0'

    restart = True
    output_dir = expanduser('~/output/multi_gan/mnist')


@exp.main
def train(n_generators, n_discriminators, noise_dim, ngf, ndf, grad_penalty, sampling, mirror_lr, restart,
          output_dir, data_type, data_source, depth, print_every, eval_device, eval_fid, fused_noise,
          device, batch_size, n_iter, loss_type, eval_every, D_lr, G_lr, _seed, _run):
    torch.random.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)


    if not torch.cuda.is_available():
        device = 'cpu'

    if output_dir is None:
        output_dir = join(_run.observers[0].dir, 'artifacts')
    else:
        output_dir = output_dir.replace('$WORK', os.environ['WORK'])
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
        dataset.tensors = tuple(tensor.to(device) for tensor in dataset.tensors)
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
        batch_norm = loss_type != 'wgan-gp'
        if data_type == 'synthetic':
            return DiscriminatorSynthetic(depth=depth, n_filters=ndf, batch_norm=batch_norm)
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

    averagers = {group: {i: ParamAverager(player) for i, player in these_players.items()}
                 for group, these_players in players.items()}
    optimizers = {group: {i: ExtraOptimizer(Adam(player.parameters(),
                                                 lr=D_lr if group == 'D' else G_lr, betas=(0.5, 0.9)))
                          for i, player in these_players.items()}
                  for group, these_players in players.items()}

    log_weights = {'G': torch.full((n_generators,), fill_value=-math.log(n_generators), device=device),
                   'D': torch.full((n_discriminators,), fill_value=-math.log(n_discriminators), device=device)}
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
                checkpoint = torch.load(join(output_dir, name))
                bundle = joblib.load(join(output_dir, f'{name}_bundle'))
            except (FileNotFoundError, EOFError):
                print(f'Cannot open {name}')
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
                    log_weights[group] = checkpoint['log_weights'][group]
                    losses[group] = checkpoint['past_losses'][group]
                    counts[group] = checkpoint['count_losses'][group]
                scheduler = bundle['scheduler']
                break

    while n_gen_upd < n_iter:
        for group, these_players in players.items():
            for P, player in these_players.items():
                player.train()
                optimizers[group][P].zero_grad()

        pairs, use, upd, extrapolate = next(scheduler)
        enable_grad_for(players, upd)

        fake_data_dict, true_logits_dict, true_data_dict = {}, {}, {}
        if fused_noise:
            for D, discriminator in players['D'].items():
                if ('D', D) in use:
                    true_data_dict[D] = next(data_loader)[0].to(device)
                    n_data += 1
                    true_logits_dict[D] = discriminator(true_data_dict[D])

            for G, generator in players['G'].items():
                if ('G', G) in use:
                    noise = torch.randn(batch_size, noise_dim, device=device)
                    n_noise += 1
                    fake_data_dict[G] = generator(noise)
        for (G, D) in pairs:
            enable_grad_for(players, {('D', D), ('G', G)})
            if fused_noise:
                true_data = true_data_dict[D]
                true_logits = true_logits_dict[D]
                fake_data = fake_data_dict[G]
            else:
                true_data = next(data_loader)[0].to(device)
                n_data += 1
                true_logits = players['D'][D](true_data)
                noise = torch.randn(batch_size, noise_dim, device=device)
                n_noise += 1
                fake_data = players['G'][G](noise)

            fake_logits = players['D'][D](fake_data)  # the coupling term
            this_loss = compute_gan_loss(true_logits=true_logits,
                                         fake_logits=fake_logits,
                                         loss_type=loss_type)
            if loss_type == 'wgan-gp':
                this_loss['D'] = this_loss['D'] + \
                                 grad_penalty * compute_grad_penalty(players['D'][D], true_data, fake_data)
            # Compute gradients
            for group, adv_group, P, A in [('D', 'G', D, G), ('G', 'D', G, D)]:
                if (group, P) in upd:
                    enable_grad_for(players, {(group, P), })
                    loss = this_loss[group] * torch.exp(log_weights[adv_group])[A]
                    enable_grad_for(players, {(group, P)})
                    # noinspection PyArgumentList
                    loss.backward(retain_graph=True)
                    losses[group][P, A] += loss.item()
                    counts[group][P, A] += 1
        # Steps
        for group, these_players in players.items():
            for P, player in these_players.items():
                if (group, P) in upd:
                    optimizers[group][P].step(extrapolate=extrapolate)
                    n_steps += 1
                    # print(f'{"e" if extrapolate else "u"}{group}{P}')
                    if group == 'D' and loss_type == 'wgan':
                        for p in player.parameters():
                            p.data.clamp_(-5, 5)
                    if not extrapolate and group == 'G':
                        n_gen_upd += 1
                        averagers[group][P].step()

        if not extrapolate:  # deextrapolate everyone in case of update
            for group, these_optimizers in optimizers.items():
                for P, optimizer in these_optimizers.items():
                    optimizer.deextrapolate()

        for group in losses:
            if torch.all(counts[group] >= 1):  # will be true or false for both groups
                sum_losses = torch.sum(losses[group] / counts[group], dim=1)
                log_weights[group] -= mirror_lr * sum_losses
                log_weights[group] -= torch.logsumexp(log_weights[group], dim=0)

                last_losses[group] = {str(P): loss for P, loss in enumerate(sum_losses)}

                losses[group][:] = 0
                counts[group][:] = 0

        if n_gen_upd >= next_print_step:
            next_print_step += print_every
            metrics = {}
            string = f'{n_gen_upd} G_upd / {n_steps} steps / {n_noise} noise / {n_data} data '
            for group, these_losses in last_losses.items():
                if these_losses is not None:
                    metrics[f'training/loss_{group}'] = sum(these_losses.values()).item()
                    for P, weight in enumerate(log_weights[group]):
                        writer.add_scalar(f'log_weights/{group}{P}', weight, n_gen_upd)
                else:
                    metrics[f'training/loss_{group}'] = float('nan')

            if n_gen_upd >= next_eval_step:
                next_eval_step += eval_every

                for G, generator in players['G'].items():
                    generator.eval()
                if data_type == 'synthetic':
                    fig, ax = plt.subplots(1, 1)
                    ax.scatter(fixed_data[:, 0], fixed_data[:, 1], label='True', marker='v',
                               zorder=1000, alpha=.5)
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
                        ax.scatter(fake_points[selection == G][:, 0], fake_points[selection == G][:, 1], label=G,
                                   alpha=.5)
                    ax.legend(loc='lower left')
                    writer.add_figure(f'generated/2d', fig, global_step=n_gen_upd)
                    plt.close(fig)
                    metrics['eval/log_prob'] = true_loglike - sampler.log_prob(fake_points).mean().item()

                else:  # data_type == 'image'
                    for G, generator in players['G'].items():
                        with torch.no_grad():
                            fake_images = (generator(fixed_noise[:64]) + 1) / 2
                        fake_images = fake_images.cpu()
                        if data_source == 'multimnist':
                            fake_images = torch.cat([fake_images[:, [i]] for i in range(fake_images.shape[1])], dim=2)
                        grid = vutils.make_grid(fake_images, normalize=True)
                        writer.add_image(f'generated/{G}', grid, global_step=n_gen_upd)
                    if eval_fid:
                        fake_images = players['G'][0](fixed_noise[:1])  # get shape
                        dataset = torch.zeros((fixed_noise.shape[0], 3, *fake_images.shape[2:]), dtype=torch.float32)
                        p = torch.exp(log_weights['G']).cpu().numpy()
                        p /= np.sum(p)
                        bb, be = 0, 0
                        for (noise,) in noise_loader:
                            G = np.random.choice(n_generators, p=p)
                            be += len(noise)
                            with torch.no_grad():
                                # noinspection PyUnresolvedReferences
                                dataset[bb:be, :].copy_(((players['G'][G](noise) + 1) / 2))
                            bb = be
                        dataset = TensorDataset(dataset)
                        fid, is_m, is_std = calculate_fid_given_paths([dataset, stat_path], batch_size=50,
                                                                      device=eval_device, dims=2048)
                        metrics['eval/fid'] = fid
                        metrics['eval/is'] = is_m
                        metrics['eval/is_std'] = is_std

                for key, value in metrics.items():
                    writer.add_scalar(key, value, n_gen_upd)
                    string += f'{key}: {value:.2f} '

                filename = join(output_dir, 'results.csv')
                if not os.path.exists(filename):
                    with open(filename, 'w+', newline='') as csvfile:
                        csv_writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
                        csv_writer.writeheader()
                with open(filename, 'a', newline='') as csvfile:
                    csv_writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
                    csv_writer.writerow(metrics)

                checkpoint = {}
                checkpoint['n_gen_upd'] = n_gen_upd
                checkpoint['next_print_step'] = next_print_step
                checkpoint['next_eval_step'] = next_eval_step
                checkpoint['n_steps'] = n_steps
                checkpoint['n_data'] = n_data
                checkpoint['n_noise'] = n_noise
                checkpoint['players'] = {}
                checkpoint['optimizers'] = {}
                checkpoint['averagers'] = {}
                checkpoint['log_weights'] = {}
                checkpoint['past_losses'] = {}
                checkpoint['count_losses'] = {}
                for group, these_players in players.items():
                    checkpoint['players'][group] = {}
                    checkpoint['optimizers'][group] = {}
                    checkpoint['averagers'][group] = {}
                    for P, player in these_players.items():
                        checkpoint['players'][group][P] = players[group][P].state_dict()
                        checkpoint['optimizers'][group][P] = optimizers[group][P].state_dict()
                        checkpoint['averagers'][group][P] = averagers[group][P].state_dict()
                    checkpoint['log_weights'][group] = log_weights[group].cpu()
                    checkpoint['past_losses'][group] = losses[group].cpu()
                    checkpoint['count_losses'][group] = counts[group].cpu()
                if os.path.exists('checkpoint'):
                    os.rename(join(output_dir, 'checkpoint'), join(output_dir, 'last_checkpoint'))
                    os.rename(join(output_dir, 'checkpoint_bundle'), join(output_dir, 'last_checkpoint_bundle'))
                torch.save(checkpoint, join(output_dir, 'checkpoint'))
                joblib.dump({'scheduler': scheduler}, join(output_dir, 'checkpoint_bundle'))

            print(string)


if __name__ == '__main__':
    exp.run_commandline()
