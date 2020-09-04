import matplotlib.pyplot as plt
import numpy as np

from numba import jit
import numpy.linalg as linalg


def make_skewed_rotation(theta, shear):
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



optimum = np.array([0, 0, 0, 0])
x0 = np.array([2., -1.9, 2, 1])
weight1 = make_skewed_rotation(0.25, 1)

weight2 = make_skewed_rotation(0., 10) * 10

weight = np.block([[weight1, np.zeros((2, 2))], [np.zeros((2, 2)), weight2]])
groups = np.array([[0, 1], [2, 3]])

bias = optimum @ weight

extrapolate = True
results = {}
methods = ['scaled_signed', 'signed', 'gradient', 'group', 'scaled_group']
for method in methods:
    results[method] = optimise(x0, optimum, weight, bias, int(1e4), 5e-2, method=method, groups=groups,
                               extrapolate=extrapolate)


x, y = np.meshgrid(np.linspace(-4, 4, 20), np.linspace(-4, 4, 20))
z = np.concatenate([x[:, :, None], y[:, :, None]], axis=2)
a = - make_field(weight[:2, :2], bias[:2], z.reshape(-1, 2)).reshape((20, 20, 2))
b = - make_field(weight[2:, 2:], bias[2:], z.reshape(-1, 2)).reshape((20, 20, 2))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

ax1.quiver(x, y, a[:, :, 0], a[:, :, 1], width=3e-3, color='.5')
ax2.quiver(x, y, b[:, :, 0], b[:, :, 1], width=3e-3, color='.5')
for method in methods:
    xs, ds = results[method]
    xs = np.array(xs)
    ds = np.array(ds)
    ax1.plot(xs[::10, 0], xs[::10, 1], marker='+', label='method')
    ax2.plot(xs[::10, 2], xs[::10, 3], marker='+', label='method')
    ax3.plot(np.arange(len(ds)), ds, label=method)
ax1.set_ylim([-4, 4])
ax1.set_xlim([-4, 4])
ax2.set_ylim([-4, 4])
ax2.set_xlim([-4, 4])
ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.legend()
ax3.set_ylim([1e-10, 100])
plt.show()