import numpy as np
from joblib import Parallel, delayed

import numpy.linalg as linalg
from sklearn.model_selection import ParameterGrid

import pandas as pd
from numba import jit

import seaborn as sns

import matplotlib.pyplot as plt

def make_sheared_rotation(theta, shear):
    theta *= 2 * np.pi
    R = np.array(
        [[np.cos(theta), np.sin(theta)],
         [- np.sin(theta), np.cos(theta)]],
        dtype=np.float64
    )
    D = np.array(
        [[1., shear],
         [0, 1]], dtype=np.float64
    )
    res = D @ R @ linalg.inv(D)
    return res

@jit
def transform_field(f, method, groups=None):
    if method == 'scaled_signed':
        return np.abs(f).sum() * np.sign(f).astype(np.float64)
    elif method == 'signed':
        return np.sign(f).astype(np.float64)
    elif method in ['group', 'scaled_group']:
        scale = 0.
        for group in groups:
            this_scale = np.sqrt(np.sum(f[group] ** 2))
            f[group] /= this_scale
            scale += this_scale
        if method == 'scaled_group':
            f *= scale
        return f
    else:
        return f

@jit
def make_field(weight, bias, x):
    return x @ weight - bias

@jit
def optimise(x0, optimum, weight, bias, n_iter, eta, method='gradient', groups=None, extrapolate=False):
    xs = []
    ds = []
    x = x0.copy()
    xs.append(x.copy())
    d = np.sqrt(np.sum((x - optimum) ** 2))
    ds.append(d)
    for i in range(n_iter):
        f = make_field(weight, bias, x)
        f = transform_field(f, method, groups=groups)
        if extrapolate:
            f = make_field(weight, bias, x - eta * f)
            f = transform_field(f, method, groups=groups)
        x -= eta * f
        d = np.sqrt(np.sum((x - optimum) ** 2))
        ds.append(d)
        xs.append(x.copy())
    return xs, ds


extrapolate = True

optimum = np.array([0, 0, 0, 0])
x0 = np.array([2., 1, 2, 1])
groups = np.array([[0, 1], [2, 3]])
methods = ['scaled_signed', 'signed', 'gradient', 'group', 'scaled_group']
shears = np.linspace(0, 1, 11)
lrs = np.logspace(-4, 0, 10)
scales = [1., 5, 10]
params = ParameterGrid(dict(method=methods, shear=shears, scale=scales, lr=lrs))


def single_run(p):
    weight1 = make_sheared_rotation(0.23, p['shear'])
    weight2 = make_sheared_rotation(0.23, p['shear']) * p['scale']
    weight = np.block([[weight1, np.zeros((2, 2))], [np.zeros((2, 2)), weight2]])
    bias = optimum @ weight
    xs, ds = optimise(x0, optimum, weight, bias, int(1e4), p['lr'], method=p['method'], groups=groups,
                      extrapolate=extrapolate)
    result = pd.DataFrame(data=xs[::10], index=range(len(xs[::10])), columns=['x1', 'x2', 'x3', 'x4'])
    result['distance'] = ds[::10]
    for k, v in p.items():
        result[k] = v
    return result

#
results = Parallel(n_jobs=8, verbose=True)(delayed(single_run)(p) for p in params)
results = pd.concat(results, axis=0)
results.to_pickle('results_lr.pkl')

results = pd.read_pickle('results_lr.pkl')

def find_iteration(df):
    a = np.where(df['distance'] < 1e-2)[0]
    if len(a) > 0:
        return a[0]
    else:
        return np.inf


iterations = results.groupby(by=['scale', 'method', 'shear', 'lr']).apply(find_iteration)
iterations = iterations.groupby(by=['scale', 'method', 'shear']).min()
iterations.name = 'iterations'
iterations = iterations.reset_index('shear')

fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=True)

p = sns.color_palette("Paired")
colors = {'gradient': p[0],
          'signed': p[2],
          'scaled_signed': p[3],
          'group': p[4],
          'scaled_group': p[5],
          }
labels = {'gradient': 'F',
          'signed': 'sgn(F)',
          'scaled_signed': '||F||_1 sgn(F)',
          'group': 'sgn_B(F)',
          'scaled_group': '||F||_{B,1} sgn_B(F)',}

for i, (scale, df1) in enumerate(iterations.groupby('scale')):
    for method, df2 in df1.groupby('method'):
        axes[i].plot(df2['shear'], df2['iterations'], label=labels[method], color=colors[method],
                     marker='o')
    axes[i].set_xlabel('Shear')
    axes[i].set_ylabel('#Iterations @1e-2 precision')
    axes[i].set_title(f'|l_1|/|l_2|: {scale:.1f}')
    axes[i].set_ylim([1, 1000])
    axes[i].set_xlim([0, 1])
axes[2].set_yscale('log')
axes[2].legend()
plt.savefig('shear_scale.png')



#
#
# x, y = np.meshgrid(np.linspace(-4, 4, 20), np.linspace(-4, 4, 20))
# z = np.concatenate([x[:, :, None], y[:, :, None]], axis=2)
# a = - make_field(weight[:2, :2], bias[:2], z.reshape(-1, 2)).reshape((20, 20, 2))
# b = - make_field(weight[2:, 2:], bias[2:], z.reshape(-1, 2)).reshape((20, 20, 2))
#
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
#
# ax1.quiver(x, y, a[:, :, 0], a[:, :, 1], width=3e-3, color='.5')
# ax2.quiver(x, y, b[:, :, 0], b[:, :, 1], width=3e-3, color='.5')
# for method in methods:
#     xs, ds = results[method]
#     xs = np.array(xs)
#     ds = np.array(ds)
#     ax1.plot(xs[::10, 0], xs[::10, 1], marker='+', label='method')
#     ax2.plot(xs[::10, 2], xs[::10, 3], marker='+', label='method')
#     ax3.plot(np.arange(len(ds)), ds, label=method)
# ax1.set_ylim([-4, 4])
# ax1.set_xlim([-4, 4])
# ax2.set_ylim([-4, 4])
# ax2.set_xlim([-4, 4])
# ax3.set_yscale('log')
# ax3.set_xscale('log')
# ax3.legend()
# ax3.set_ylim([1e-10, 100])
# plt.show()
