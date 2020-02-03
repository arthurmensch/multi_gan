import json
import os
from os.path import expanduser, join

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.errors import EmptyDataError

output_dirs = [expanduser('~/output/multi_gan/cifar10'), expanduser('~/output/multi_gan/cifar10_final'),
               expanduser('~/output/multi_gan/cifar10_nplayer')]

all_results = []
for output_dir in output_dirs:
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
                        all_results.append(dict(
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
        if config['n_generators'] == 5 and config['n_discriminators'] == 1:
            print(mirror_lr, config['n_generators'], config['G_lr'], sampling, exp, exp_dir)
all_results = pd.DataFrame(all_results)

# all_results = all_results.query("(n_generators == 1 and n_discriminators == 1)"
#                                         "or (n_generators == 3 and n_discriminators == 3)")
all_results = all_results.groupby(by=['n_generators', 'n_discriminators', 'mirror_lr', 'lr', 'updates', 'total_iter']).aggregate(['mean', 'std'])
all_results.reset_index(inplace=True)
all_results['index'] = all_results[['n_generators', 'n_discriminators', 'lr']].apply(lambda x: f'G{x[0]:.0f}D{x[1]:.0f}_lr{x[2]}',
                                                                     axis=1)
all_results['optim_index'] = all_results[['updates', 'mirror_lr', 'lr']].apply(lambda x: f'{x[0]}_{x[1]}_mirror_{x[2]}', axis=1)
all_results['fid_mean+std'] = all_results[("fid", "mean")] + all_results[("fid", "std")]
all_results['fid_mean-std'] = all_results[("fid", "mean")] - all_results[("fid", "std")]
all_results['fid_mean'] = all_results[("fid", "mean")]
all_results['iter_per_gen_mean'] = all_results[("iter_per_gen", "mean")]
grid = sns.FacetGrid(all_results, hue='index', col='mirror_lr', row='updates', legend_out=True)
grid.map(plt.plot, "iter_per_gen_mean", "fid_mean")
grid.map(plt.fill_between, "iter_per_gen_mean", "fid_mean-std", "fid_mean+std", alpha=0.5)
for ax in grid.axes.ravel():
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('FID')
    ax.set_ylim([20, 300])
    ax.set_xlabel('Number of update per generator')
grid.add_legend(bbox_to_anchor=(1, 0.3), title='GD')
grid.fig.tight_layout(w_pad=1)

plt.show()

grid = sns.FacetGrid(all_results, hue='optim_index', col='n_generators', row='n_discriminators', legend_out=True)
grid.map(plt.plot, "iter_per_gen_mean", "fid_mean")
grid.map(plt.fill_between, "iter_per_gen_mean", "fid_mean-std", "fid_mean+std", alpha=0.5)
for ax in grid.axes.ravel():
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('FID')
    ax.set_ylim([20, 300])
    ax.set_xlabel('Number of update per generator')
grid.add_legend(bbox_to_anchor=(0.7, 0.3), title='GD')
grid.fig.tight_layout(w_pad=1)

plt.show()