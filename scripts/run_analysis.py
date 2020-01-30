import os
from os.path import expanduser, join
import json
import pandas as pd
from pandas.errors import EmptyDataError
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
output_dir = expanduser('~/output/multi_gan/cifar10')

all = []
for exp in os.listdir(output_dir):
    exp_dir = join(output_dir, exp)
    with open(join(exp_dir, '2', 'config.json'), 'r') as f:
        config = json.load(f)
    try:
        results = pd.read_csv(join(exp_dir, 'results.csv'), usecols=[2, 3, 4], skiprows=2)
    except EmptyDataError:
        continue
    results.columns = ['fid', 'is', 'is_std']
    sampling = config['sampling']
    if sampling == 'all' and config['fused_noise']:
        sampling += '_same_noise'
    if config['n_generators'] == 1 and config['n_discriminators'] == 1:
        # Common baseline
        samplings = [sampling]
        mirror_lrs = [0, 1e-2]
    else:
        samplings = [sampling]
        mirror_lrs = [config['mirror_lr']]
    for i, row in results.iterrows():
        iter = (i + 1) * 10000
        for mirror in mirror_lrs:
            for sampling in samplings:
                for mirror_lr in mirror_lrs:
                    dg = f"G{config['n_generators']}D{config['n_discriminators']}"
                    optim = sampling
                    if mirror:
                         optim += "_mirror"
                    all.append(dict(
                        n_generators=config['n_generators'],
                        n_discriminators=config['n_discriminators'],
                        dg=dg,
                        mirror_lr=mirror_lr,
                        optim=optim,
                        updates=sampling,
                        fid=row['fid'],
                        is_=row['is'],
                        is_std=row['is_std'],
                        total_iter=iter,
                        iter_per_gen=iter / config['n_generators'],
                    ))
all = pd.DataFrame(all)
all.sort_values(by=['dg', 'total_iter'], inplace=True)

grid = sns.FacetGrid(all, hue='dg', col='updates', row='mirror_lr', legend_out=True)
grid.map(plt.plot, "iter_per_gen", "fid", marker="o")
for ax in grid.axes.ravel():
    ax.set_yscale('log')
    ax.set_ylabel('FID')
    ax.set_xlabel('Number of generator update')
grid.add_legend(bbox_to_anchor=(1, 0.3), title='#G#D')
grid.fig.tight_layout(w_pad=1)

plt.show()

grid = sns.FacetGrid(all, hue='dg', col='updates', row='mirror_lr', legend_out=True)
grid.map(plt.plot, "total_iter", "fid", marker="o")
for ax in grid.axes.ravel():
    ax.set_yscale('log')
    ax.set_ylabel('FID')
    ax.set_xlabel('Number of generator update')
grid.add_legend(bbox_to_anchor=(1, 0.3), title='#G#D')
grid.fig.tight_layout(w_pad=1)

plt.show()

all.sort_values(by=['optim', 'total_iter'], inplace=True)

grid = sns.FacetGrid(all, hue='optim', col='n_discriminators', row='n_generators', legend_out=True)
grid.map(plt.plot, "iter_per_gen", "fid", marker="o")
for ax in grid.axes.ravel():
    ax.set_yscale('log')
    ax.set_ylabel('FID')
    ax.set_xlabel('Number of generator update')
grid.add_legend(bbox_to_anchor=(1, 0.1), title='Optimisation')
grid.fig.tight_layout(w_pad=1)

plt.show()