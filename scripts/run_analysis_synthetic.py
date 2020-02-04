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

output_dirs = [
    # expanduser('~/output/multi_gan/synthetic/final_mixed_nash'),
    #            expanduser('~/output/multi_gan/synthetic/final_mixed_nash_2'),
               expanduser('~/output/multi_gan/synthetic/final_mixed_nash_5')
               ]

all_results = []
for output_dir in output_dirs:
    for exp in os.listdir(output_dir):
        exp_dir = join(output_dir, exp)
        try:
            with open(join(exp_dir, '1', 'config.json'), 'r') as f:
                config = json.load(f)
            results = pd.read_csv(join(exp_dir, 'results.csv'))
        except (EmptyDataError, FileNotFoundError):
            continue
        sampling = config['sampling']
        if sampling == 'all' and config['fused_noise']:
            sampling += '_same_noise'
        samplings = [sampling]
        mirror_lrs = [config['mirror_lr']]
        for i, row in results.iterrows():
            iter = (i + 1) * 500
            for mirror in mirror_lrs:
                for sampling in samplings:
                    for mirror_lr in mirror_lrs:
                        all_results.append(dict(
                            n_generators=config['n_generators'],
                            n_discriminators=config['n_discriminators'],
                            mirror_lr=mirror_lr,
                            updates=sampling,
                            fid=-row['eval/log_prob'],
                            seed=config['seed'],
                            lr=config['G_lr'],
                            exp=exp,
                            total_iter=iter,
                            iter_per_gen=iter,
                            # iter_per_gen=iter,
                        ))
        if config['n_generators'] == config['n_discriminators'] and config['mirror_lr'] != 1e-1 and sampling == "all_extra":
            print(exp_dir, config['n_generators'], config['n_discriminators'])
all_results = pd.DataFrame(all_results)
all_results = all_results.query('n_generators == n_discriminators and'
                                '(mirror_lr == 0)'
                                'and updates =="all_extra"')
all_results = all_results.groupby(
    by=['n_generators', 'n_discriminators', 'mirror_lr', 'lr', 'updates', 'total_iter']).aggregate(['mean', 'std'])
all_results.reset_index(inplace=True)
all_results['index'] = all_results[['n_generators', 'n_discriminators',
                                    'lr']].apply(lambda x: f'G{x[0]:.0f}D{x[1]:.0f}', axis=1)
all_results['optim_index'] = all_results[['updates', 'mirror_lr']].apply(lambda x: f'{x[0]}_{x[1]}', axis=1)
all_results['fid_mean+std'] = all_results[("fid", "mean")] + all_results[("fid", "std")]
all_results['fid_mean-std'] = all_results[("fid", "mean")] - all_results[("fid", "std")]
all_results['fid_mean'] = all_results[("fid", "mean")]
all_results['iter_per_gen_mean'] = all_results[("iter_per_gen", "mean")]
# grid = sns.FacetGrid(all_results, hue="index", row='updates', col='mirror_lr', legend_out=True)
# grid.map(plt.plot, "iter_per_gen_mean", "fid_mean")
# grid.map(plt.fill_between, "iter_per_gen_mean", "fid_mean-std", "fid_mean+std", alpha=0.2)
# for ax in grid.axes.ravel():
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     ax.set_ylabel('Log prob')
#     # ax.set_ylim([20, 300])
#     ax.set_xlabel('Total generator updates')
# grid.add_legend(bbox_to_anchor=(1, 0.3), title='GD')
# plt.show()
#
# grid = sns.FacetGrid(all_results, hue="optim_index", row='n_generators', col='n_discriminators',
#                      legend_out=True)
# grid.map(plt.plot, "iter_per_gen_mean", "fid_mean")
# grid.map(plt.fill_between, "iter_per_gen_mean", "fid_mean-std", "fid_mean+std", alpha=0.5)
# for ax in grid.axes.ravel():
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     ax.set_ylabel('Log prob')
#     # ax.set_ylim([20, 300])
#     ax.set_xlabel('Total updates per generator')
# grid.add_legend(bbox_to_anchor=(1, 0.5), title='GD')
# plt.show()

fig, ax = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)

for (G, D, mirror_lr, _, _), sub_df in all_results.groupby(['n_generators', 'n_discriminators', 'mirror_lr', 'lr',
                                                            'updates']):
    label = f'{G}G {D}D'
    ax.plot(sub_df[('iter_per_gen', 'mean')], sub_df[('fid', 'mean')], label=label)
    ax.fill_between(sub_df[('iter_per_gen', 'mean')], sub_df['fid_mean-std'], sub_df['fid_mean+std'], alpha=0.2)
ax.set_xlim([500, 30000])
ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_ylim([3, 15])
# ax.annotate("FID", xy=(0, 1), ha='left', va='bottom', xycoords='axes fraction', textcoords="offset points",
#             xytext=(0, -3))
ax.set_ylabel("- Log-likelihood")
ax.grid(axis='y')
ax.yaxis.set_label_coords(-0.2,.5)
ax.annotate("Total G. iter", xy=(0, 0), ha='right', va='top', xycoords='axes fraction', textcoords="offset points",
            xytext=(10, -7))
# ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
# plt.ticklabel_format(scilimits=(0, 0), axis='x')
ax.xaxis.offsetText.set_visible(False)
# ax.annotate(r'$\times 10^{5}$', xy=(1, 0), ha='right', va='bottom', xycoords='axes fraction', textcoords="offset points",
#             xytext=(0, 3))
ax.annotate('Deep (overfitting)\nnetworks', xy=(0.05, 0.75), ha='left', va='top', xycoords='axes fraction', textcoords="offset points",
            xytext=(0, 3))
sns.despine(fig)
ax.legend(loc='upper right', frameon=False, bbox_to_anchor=(1, 1.03))
plt.savefig('quantitative_synthetic_deep.pdf')



# fig, ax = plt.subplots(1, 1, figsize=(2, 1.66), constrained_layout=True)
#
# colors_ours = sns.color_palette('Reds', n_colors=1)
# colors_theirs = sns.color_palette('Blues', n_colors=1)
# colors = {'pair': colors_ours[0], 'all_same_noise': colors_theirs[0]}
# labels = {'pair': 'DSEG', 'all_same_noise': 'Full SEG'}
# lr_labels_dict = {1e-5: '1e-5', 3e-5: '3e-5', 5e-5: '5e-5'}
# linestyles = {1e-5: '-', 3e-5: '--', 5e-5: ':'}
# updates_handles, updates_labels, lr_handles, lr_labels = [], [], [], []
# for (_, _, lr, updates), sub_df in all_results.groupby(by=['n_generators', 'n_discriminators', 'lr', 'updates']):
#     lines, = ax.plot(sub_df['total_iter'], sub_df['fid_mean'], label=labels[updates], color=colors[updates],
#             linestyle=linestyles[lr])
#     if lr == 1e-5:
#         updates_handles.append(lines)
#         updates_labels.append(labels[updates])
#     if updates == 'pair':
#         lr_handles.append(lines)
#         lr_labels.append(lr_labels_dict[lr])
#     ax.fill_between(sub_df['total_iter'], sub_df['fid_mean-std'], sub_df['fid_mean+std'], color=colors[updates], alpha=0.5)
# # ax.set_xlabel('Generator updates')
# # ax.set_ylabel('FID (10k)')
# ax.set_ylim([25, 150])
# ax.set_yticks([25, 50, 100, 125])
# sns.despine(fig)
# legend1 = plt.legend(updates_handles, updates_labels, frameon=False,
#                      bbox_to_anchor=(1.1, 1), loc='upper right')
# legend2 = plt.legend(lr_handles, lr_labels, frameon=False, fontsize=7, ncol=1, labelspacing=0., handlelength=1.5,
#                      bbox_to_anchor=(1.1, 0.5), loc='center right')
# ax.annotate('G. iter', xy=(0, 0), xytext=(-2, -7), textcoords='offset points',
#                  xycoords='axes fraction', ha='right', va='top', fontsize=10)
# ax.annotate('FID', xy=(0, 1), xytext=(-3, -3), textcoords='offset points',
#                  xycoords='axes fraction', ha='right', va='bottom', fontsize=10)
# ax.annotate('LR', xy=(.7, .5), xytext=(-10,-1), xycoords='axes fraction', textcoords='offset points',
#             fontsize=8, ha='center')
# ax.add_artist(legend1)
# # ax.add_artist(legend2)
# plt.savefig('multi.pdf')
