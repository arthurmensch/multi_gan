import os
import random
from os.path import expanduser, join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from sacred import Experiment
from sacred.observers import FileStorageObserver
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from multi_gan.data import make_8gmm, infinite_iter, make_image_data, make_25gmm
from multi_gan.eval.fid_score import calculate_fid_given_paths
from multi_gan.losses import compute_gan_loss, compute_grad_penalty
from multi_gan.models import GeneratorDCGAN28, DiscriminatorDCGAN28, GeneratorSynthetic, GeneratorResNet32, \
    DiscriminatorSynthetic, DiscriminatorResNet32
from multi_gan.optimizers import ExtraOptimizer, ParamAverager
from multi_gan.training import enable_grad_for, Scheduler

exp = Experiment('multi_gan')
exp_dir = expanduser('~/output/games_rl/multi_gan')
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

@exp.config
def synthetic():
    data_type = 'synthetic'
    data_source = '8gmm'

    n_generators = 2
    n_discriminators = 2

    ndf = 128 // n_generators
    ngf = 128 // n_discriminators

    loss_type = 'wgan-gp'
    device = 'cuda:0'
    n_iter = 15000

    noise_dim = 32  # For image data

    batch_size = 64
    D_lr = 1e-4
    G_lr = 1e-5
    mirror_lr = 0

    ndf = 512
    ngf = 512
    grad_penalty = 10

    sampling = 'player'

    seed = 100
    eval_every = 500

    print_every = 500
    eval_device = 'cuda:0'
    eval_fid = False

@exp.named_config
def cifar():
    data_type = 'image'
    data_source = 'cifar10'
    n_generators = 5
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

    ndf = 128
    ngf = 128
    grad_penalty = 10

    sampling = 'all'

    seed = 0
    print_every = 10000

    eval_fid = True
    eval_every = 10000
    eval_device = 'cuda:0'


@exp.main
def train(n_generators, n_discriminators, noise_dim, ngf, ndf, grad_penalty, sampling, mirror_lr,
          data_type, data_source, depth, print_every, eval_device, eval_fid,
          device, batch_size, n_iter, loss_type, eval_every, D_lr, G_lr, _seed, _run):
    torch.random.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)

    if not torch.cuda.is_available():
        device = 'cpu'

    artifact_dir = join(_run.observers[0].dir, 'artifacts')
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    writer = SummaryWriter(log_dir=artifact_dir)

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

    weights = {'G': torch.full((n_generators,), fill_value=1 / n_generators, device=device),
               'D': torch.full((n_discriminators,), fill_value=1 / n_discriminators, device=device)}
    past_losses = {'G': torch.zeros((n_generators, n_discriminators), device=device),
                   'D': torch.zeros((n_discriminators, n_generators), device=device)}
    count_losses = {'G': torch.zeros((n_generators, n_discriminators), device=device),
                    'D': torch.zeros((n_discriminators, n_generators), device=device)}
    last_losses = {group: {i: None for i, player in these_players.items()} for group, these_players in players.items()}

    fixed_data = next(data_loader)[0].to('cpu')

    if data_type == 'synthetic':
        fixed_noise = torch.randn(512, noise_dim).to(device)
        noise_loader = None
        true_loglike = sampler.log_prob(fixed_data).mean().item()
        img_folder = None
    else:
        fixed_noise = torch.randn(10000, noise_dim).to(device)
        noise_loader = DataLoader(TensorDataset(fixed_noise), batch_size=batch_size)
        true_loglike = None
        img_folder = join(artifact_dir, 'imgs')
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)

    n_computations = 0

    scheduler = Scheduler(n_generators, n_discriminators, sampling=sampling)
    next_print_step = 0
    next_eval_step = 0

    while n_computations < n_iter:
        for G, generator in players['G'].items():
            generator.train()
        losses = {group: {i: None for i, player in these_players.items()} for group, these_players in players.items()}
        pairs, use, upd, extrapolate = next(scheduler)
        enable_grad_for(players, upd)
        fake_data, true_logits, true_data = {}, {}, {}
        for G, generator in players['G'].items():
            if ('G', G) in use:
                noise = torch.randn(batch_size, noise_dim, device=device)
                fake_data[G] = generator(noise)
        for D, discriminator in players['D'].items():
            if ('D', D) in use:
                true_data[D] = next(data_loader)[0].to(device)
                true_logits[D] = discriminator(true_data[D])

        for (G, D) in pairs:
            fake_logits = players['D'][D](fake_data[G])  # the coup;ing term
            this_loss = compute_gan_loss(true_logits=true_logits[D],
                                         fake_logits=fake_logits,
                                         loss_type=loss_type)
            for group, adv_group, P, A in [('D', 'G', D, G), ('G', 'D', G, D)]:
                if (group, P) in upd:
                    if group == 'D':
                        if loss_type == 'wgan-gp':
                            this_loss[group] += grad_penalty * compute_grad_penalty(players['D'][D], true_data[D],
                                                                                    fake_data[G])
                    loss = this_loss[group] * weights[adv_group][A]
                    if losses[group][P] is not None:
                        losses[group][P] = losses[group][P] + loss
                    else:
                        losses[group][P] = loss
                    past_losses[group][P, A] += loss.item()
                    count_losses[group][P, A] += 1

        for group, these_players in players.items():
            for P, player in these_players.items():
                if losses[group][P] is not None:
                    last_losses[group][P] = losses[group][P].item()
                    enable_grad_for(players, {(group, P), })
                    optimizers[group][P].zero_grad()
                    losses[group][P].backward(retain_graph=True)
                    optimizers[group][P].step(extrapolate=extrapolate)
                    if group == 'D' and loss_type == 'wgan':
                        for p in player.parameters():
                            p.data.clamp_(-5, 5)
                    if not extrapolate:
                        if group == 'G':
                            n_computations += 1
                        averagers[group][P].step()

        if not extrapolate:
            for group, these_optimizers in optimizers.items():
                for P, optimizer in these_optimizers.items():
                    optimizer.deextrapolate()

        for group, count_loss in count_losses.items():
            if torch.all(count_loss >= 1):
                mean_loss = past_losses[group] / count_loss
                weights[group] *= torch.exp(-mirror_lr * torch.sum(mean_loss, dim=1))
                weights[group] /= weights[group].sum()
                past_losses[group][:] = 0
                count_losses[group][:] = 0

        if n_computations >= next_print_step:
            next_print_step += print_every
            string = f'iter:{n_computations} '
            for group, these_losses in last_losses.items():
                for P, loss in these_losses.items():
                    if loss is None:
                        loss = float('nan')
                    weight = weights[group][P]
                    writer.add_scalar(f'loss/loss_{group}{P}', loss, global_step=n_computations)
                    writer.add_scalar(f'weights/{group}{P}', weight, global_step=n_computations)
                    string += f'loss_{group}{P}={loss:.2f} (w={weight:.2f}) '
            if n_computations >= next_eval_step:
                next_eval_step += eval_every

                for G, generator in players['G'].items():
                    generator.eval()
                if data_type == 'synthetic':
                    fig, ax = plt.subplots(1, 1)
                    ax.scatter(fixed_data[:, 0], fixed_data[:, 1], label='True', marker='v',
                               zorder=1000, alpha=.5)
                    ax.set_xlim([-1.5, 1.5])
                    ax.set_ylim([-1.5, 1.5])
                    selection = torch.from_numpy(np.random.choice(n_generators, size=fixed_noise.shape[0],
                                                                  p=weights['G'].cpu().numpy()))
                    fake_points = torch.zeros((fixed_noise.shape[0], 2))
                    for G, generator in players['G'].items():
                        with torch.no_grad():
                            mask = selection == G
                            if torch.sum(mask) > 0:
                                fake_points[selection == G] = generator(fixed_noise[selection == G]).cpu()
                        ax.scatter(fake_points[selection == G][:, 0], fake_points[selection == G][:, 1], label=G, alpha=.5)
                    ax.legend(loc='lower left')
                    writer.add_figure(f'generated/2d', fig, global_step=n_computations)
                    plt.close(fig)
                    log_prob = true_loglike - sampler.log_prob(fake_points).mean().item()
                    writer.add_scalar(f'loss/log_prob', log_prob, global_step=n_computations)
                    string += f'log_prob={log_prob:.2f} '

                else:  # data_type == 'image'
                    for G, generator in players['G'].items():
                        with torch.no_grad():
                            fake_images = (generator(fixed_noise[:64]) + 1) / 2
                        fake_images = fake_images.cpu()
                        if data_source == 'multimnist':
                            fake_images = torch.cat([fake_images[:, [i]] for i in range(fake_images.shape[1])], dim=2)
                        grid = vutils.make_grid(fake_images, normalize=True)
                        writer.add_image(f'generated/{G}', grid, global_step=n_computations)
                    if eval_fid:
                            paths = []
                            p = weights['G'].cpu().numpy()
                            for (noise, ) in noise_loader:
                                with torch.no_grad():
                                    G = np.random.choice(n_generators, p=p)
                                    images = ((players['G'][G](noise) + 1) / 2).cpu()
                                    for image in images:
                                        path = join(img_folder, f'img_{len(paths)}.png')
                                        vutils.save_image(image, path)
                                        paths.append(path)
                            fid, is_m, is_std = calculate_fid_given_paths([img_folder, stat_path], batch_size=50,
                                                                          device=eval_device, dims=2048)
                            writer.add_scalar(f'loss/fid', fid, global_step=n_computations)
                            writer.add_scalar(f'loss/is_m', fid, global_step=n_computations)
                            string += f'fid: {fid:.2f}, is: {is_m:.2f}'
            print(string)


if __name__ == '__main__':
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    exp.observers = [FileStorageObserver(exp_dir)]
    exp.run_commandline()
