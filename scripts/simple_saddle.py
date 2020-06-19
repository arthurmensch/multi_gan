from functools import partial
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from multi_gan.optimizers import ExtraOptimizer
from torch.nn import Parameter
from torch.optim import Adam, SGD


class Game(nn.Module):
    def __init__(self):
        super(Game, self).__init__()

    def forward(self, x, y):
        return x * y.detach(), -x.detach() * y

output_dir = '.'


def optimize(extrapolate=True, optimizer='sgd'):
    x = Parameter(torch.ones(1))
    y = Parameter(torch.ones(1))
    g = Game()

    def make_optimizer(params, optimizer):
        if optimizer == 'sgd':
            return SGD(params, lr=1e-1, )
        else:
            return Adam(params, lr=1e-2, betas=(0.,0.9), amsgrad=False)
    opt_x = ExtraOptimizer(make_optimizer([x], optimizer))
    opt_y = ExtraOptimizer(make_optimizer([y], optimizer))
    trace = []
    for i in range(10000):
        distance = (x ** 2 + y ** 2).item()
        trace.append(dict(x=x.item(), y=y.item(), c=i * 2, distance=distance))
        opt_x.zero_grad()
        opt_y.zero_grad()
        lx, ly = g(x, y)
        lx.backward()
        ly.backward()
        if extrapolate:
            opt_x.step(extrapolate=i % 2)
            opt_y.step(extrapolate=i % 2)
        else:
            opt_x.step()
            opt_y.step()
    return pd.DataFrame(trace)

def compute():
    sgd = optimize(extrapolate=False)
    extra_sgd = optimize(extrapolate=True)
    adam = optimize(extrapolate=False, optimizer='adam')
    extra_adam = optimize(optimizer='adam')
    results = pd.concat([sgd, extra_sgd, adam, extra_adam], keys=['sgd', 'extra_sgd',
                                                                  'adam', 'extra_adam'], names=['method'])
    results.to_pickle(join(output_dir, 'records.pkl'))

def plot():
    df = pd.read_pickle(join(output_dir, 'records.pkl'))

    X, Y = torch.meshgrid(torch.linspace(-2.5, 2.5, 20), torch.linspace(-2.5, 2.5, 20))

    g = Game()

    def get_vector_field(X, Y):
        gX, gY = torch.zeros_like(X), torch.zeros_like(Y)
        for i in range(len(X)):
            for j in range(len(Y)):
                x = Parameter(X[i, [j]])
                y = Parameter(Y[i, [j]])
                lx, ly = g(x, y)
                gX[i, j] = torch.autograd.grad(lx, (x,))[0]
                gY[i, j] = torch.autograd.grad(ly, (y,))[0]
        return gX.detach(), gY.detach()

    gX, gY = get_vector_field(X, Y)

    scale = 50
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(297.48499 / scale, 100 / scale), constrained_layout=False)
    plt.subplots_adjust(left=0., right=0.98, top=0.98, bottom=0.12, wspace=0.33)
    ax1.quiver(X, Y, -gX * 3, -gY * 3, width=3e-3, color='.5')
    for name, rec in df.groupby(['method']):
        ax1.plot(rec['x'], rec['y'], markersize=2, label=name, linewidth=2, alpha=0.8)
        ax2.plot(rec['c'], rec['distance'], markersize=2, label=name, linewidth=2, alpha=0.8)

    ax2.legend(fontsize=10, frameon=False, loc='lower left')
    ax1.set_ylim([-2.5, 2.5])
    ax1.set_xlim([-2.5, 2.5])
    ax2.set_ylabel('Distance to Nash')
    ax2.annotate('Computations', xy=(0, 0), xytext=(0, -7), textcoords='offset points',
                 fontsize=8,
                 xycoords='axes fraction', ha='right', va='top', zorder=10000)
    ax2.tick_params(axis='both', which='major', labelsize=9)
    ax2.tick_params(axis='both', which='minot', labelsize=5)
    ax1.annotate('Nash', xy=(0, 0), xytext=(6, -6), textcoords='offset points', xycoords='data', zorder=10000,
                 fontsize=11)
    ax1.annotate(r'$\theta_0$', xytext=(6, -6), textcoords='offset points', xy=(1.5, 1.2), xycoords='data',
                 fontsize=11)
    ax1.annotate('Trajectory', xy=(0.5, 0), xytext=(-3, -5), textcoords='offset points',
                 xycoords='axes fraction', ha='center', va='top', fontsize=11)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax1.axis('off')
    ax2.set_xlim(1e3, 2e4)
    ax2.set_ylim(1e-7, 2)
    sns.despine(fig, [ax2])
    plt.savefig(join(output_dir, 'figure.pdf'), transparent=True)
    plt.show()


compute()
plot()
