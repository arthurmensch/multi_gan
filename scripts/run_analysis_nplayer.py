import json
import os
from os.path import expanduser, join

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.errors import EmptyDataError

import matplotlib
from matplotlib import rc
matplotlib.rcParams['backend'] = 'pdf'
rc('text', usetex=True)

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
all_results = pd.DataFrame(all_results)
all_results = all_results.query("n_generators >= 5 and n_discriminators >= 2 and"
                                "(updates == 'pair' or updates == 'all_same_noise')"
                                "and mirror_lr == 0")
all_results = all_results.groupby(
    by=['n_generators', 'n_discriminators', 'mirror_lr', 'lr', 'updates', 'total_iter']).aggregate(['mean', 'std'])
all_results.reset_index(inplace=True)
all_results['index'] = all_results[['n_generators', 'n_discriminators',
                                    'lr']].apply(lambda x: f'G{x[0]:.0f}D{x[1]:.0f}', axis=1)
all_results['fid_mean+std'] = all_results[("fid", "mean")] + all_results[("fid", "std")]
all_results['fid_mean-std'] = all_results[("fid", "mean")] - all_results[("fid", "std")]
all_results['fid_mean'] = all_results[("fid", "mean")]
all_results['iter_per_gen_mean'] = all_results[("iter_per_gen", "mean")]
# grid = sns.FacetGrid(all_results, hue='updates', col='index', row='lr', legend_out=True)
# grid.map(plt.plot, "iter_per_gen_mean", "fid_mean")
# grid.map(plt.fill_between, "iter_per_gen_mean", "fid_mean-std", "fid_mean+std", alpha=0.5)
# for ax in grid.axes.ravel():
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     ax.set_ylabel('FID')
#     ax.set_ylim([20, 300])
#     ax.set_xlabel('Number of update per generator')
# grid.add_legend(bbox_to_anchor=(1, 0.3), title='GD')
# grid.fig.tight_layout(w_pad=1)
fig, ax = plt.subplots(1, 1, figsize=(2, 1.66), constrained_layout=True)

colors_ours = sns.color_palette('Reds', n_colors=1)
colors_theirs = sns.color_palette('Blues', n_colors=1)
colors = {'pair': colors_ours[0], 'all_same_noise': colors_theirs[0]}
labels = {'pair': 'DSEG', 'all_same_noise': 'Full SEG'}
lr_labels_dict = {1e-5: '1e-5', 3e-5: '3e-5', 5e-5: '5e-5'}
linestyles = {1e-5: '-', 3e-5: '--', 5e-5: ':'}
updates_handles, updates_labels, lr_handles, lr_labels = [], [], [], []
for (_, _, lr, updates), sub_df in all_results.groupby(by=['n_generators', 'n_discriminators', 'lr', 'updates']):
    lines, = ax.plot(sub_df['total_iter'], sub_df['fid_mean'], label=labels[updates], color=colors[updates],
            linestyle=linestyles[lr])
    if lr == 1e-5:
        updates_handles.append(lines)
        updates_labels.append(labels[updates])
    if updates == 'pair':
        lr_handles.append(lines)
        lr_labels.append(lr_labels_dict[lr])
    ax.fill_between(sub_df['total_iter'], sub_df['fid_mean-std'], sub_df['fid_mean+std'], color=colors[updates], alpha=0.5)
# ax.set_xlabel('Generator updates')
# ax.set_ylabel('FID (10k)')
ax.set_ylim([25, 150])
ax.set_yticks([25, 50, 100, 125])
sns.despine(fig)
legend1 = plt.legend(updates_handles, updates_labels, frameon=False,
                     bbox_to_anchor=(1.1, 1), loc='upper right')
legend2 = plt.legend(lr_handles, lr_labels, frameon=False, fontsize=7, ncol=1, labelspacing=0., handlelength=1.5,
                     bbox_to_anchor=(1.1, 0.5), loc='center right')
ax.annotate('G. iter', xy=(0, 0), xytext=(-2, -7), textcoords='offset points',
                 xycoords='axes fraction', ha='right', va='top', fontsize=10)
ax.annotate('FID', xy=(0, 1), xytext=(-3, -3), textcoords='offset points',
                 xycoords='axes fraction', ha='right', va='bottom', fontsize=10)
ax.annotate('LR', xy=(.7, .5), xytext=(-10,-1), xycoords='axes fraction', textcoords='offset points',
            fontsize=8, ha='center')
ax.add_artist(legend1)
# ax.add_artist(legend2)
plt.savefig('multi.pdf')
