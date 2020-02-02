import json
import os
from os.path import expanduser, join

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.errors import EmptyDataError

output_dir = expanduser('~/output/multi_gan/cifar10_final')

all = []
for exp in os.listdir(output_dir):
    exp_dir = join(output_dir, exp)
    with open(join(exp_dir, '1', 'config.json'), 'r') as f:
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
        mirror_lrs = [0, 2e-2]
    else:
        samplings = [sampling]
        mirror_lrs = [config['mirror_lr']]
    for i, row in results.iterrows():
        iter = (i + 1) * 10000
        for mirror in mirror_lrs:
            for sampling in samplings:
                for mirror_lr in mirror_lrs:
                    all.append(dict(
                        n_generators=config['n_generators'],
                        n_discriminators=config['n_discriminators'],
                        mirror_lr=mirror_lr,
                        updates=sampling,
                        fid=row['fid'],
                        seed=config['seed'],
                        lr=config['G_lr'],
                        is_=row['is'],
                        exp=exp,
                        is_std=row['is_std'],
                        total_iter=iter,
                        iter_per_gen=iter / config['n_generators'],
                    ))
    if mirror_lr > 0 and config['n_generators'] > 1:
        print(mirror_lr, config['n_generators'], config['G_lr'], exp)
all = pd.DataFrame(all)


all = all.groupby(by=['n_generators', 'n_discriminators', 'mirror_lr', 'lr', 'updates', 'total_iter']).aggregate(['mean', 'std'])
all.reset_index(inplace=True)
all = all[all['lr'] == 5e-5]
all['index'] = all[['n_generators', 'n_discriminators', 'lr']].apply(lambda x: f'G{x[0]:.0f}D{x[1]:.0f}_lr{x[2]}',
                                                                     axis=1)
all['fid_mean+std'] = all[("fid", "mean")] + all[("fid", "std")]
all['fid_mean-std'] = all[("fid", "mean")] - all[("fid", "std")]
all['fid_mean'] = all[("fid", "mean")]
all['iter_per_gen_mean'] = all[("iter_per_gen", "mean")]
grid = sns.FacetGrid(all, hue='index', col='mirror_lr', row='updates', legend_out=True)
grid.map(plt.plot, "iter_per_gen_mean", "fid_mean")
grid.map(plt.fill_between, "iter_per_gen_mean", "fid_mean-std", "fid_mean+std", alpha=0.5)
for ax in grid.axes.ravel():
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('FID')
    ax.set_ylim([20, 100])
    ax.set_xlabel('Number of update per generator')
grid.add_legend(bbox_to_anchor=(1, 0.3), title='GD')
grid.fig.tight_layout(w_pad=1)

plt.show()

# grid = sns.FacetGrid(all, hue='dg', col='updates', row='mirror_lr', legend_out=True)
# grid.map(plt.plot, "iter_per_gen", "fid", marker="o")
# for ax in grid.axes.ravel():
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     ax.set_ylabel('FID')
#     ax.set_xlabel('Number of update per generator')
# grid.add_legend(bbox_to_anchor=(1, 0.3), title='#G#D')
# grid.fig.tight_layout(w_pad=1)
#
# plt.show()
#
# all.sort_values(by=['optim', 'total_iter'], inplace=True)
#
# grid = sns.FacetGrid(all, hue='optim', col='n_discriminators', row='n_generators', legend_out=True)
# grid.map(plt.plot, "iter_per_gen", "fid", marker="o")
# for ax in grid.axes.ravel():
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     ax.set_ylabel('FID')
#     ax.set_xlabel('Number of update per generator')
# grid.add_legend(bbox_to_anchor=(1, 0.1), title='Optimisation')
# grid.fig.tight_layout(w_pad=1)

plt.show()