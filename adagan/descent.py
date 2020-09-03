from scipy.stats import skew

from adagan.generate_matrix import make_skewed_rotation
import matplotlib.pyplot as plt

import numpy as np

class ConstantField():
    def __init__(self, theta, skew, optimum):
        self.weight = make_skewed_rotation(theta, skew)
        self.bias = optimum @ self.weight

    def __call__(self, x):
        return x @ self.weight - self.bias


def transform_field(f, method):
    if method == 'scaled_signed':
        return np.abs(f).sum() * np.sign(f).astype('float')
    elif method == 'signed':
        return np.sign(f).astype('float')
    else:
        return f

def optimise(x0, optimum, field, n_iter, eta, method='gradient', extrapolate=False):
    xs = []
    ds = []
    x = x0.copy()
    xs.append(x.copy())
    for i in range(n_iter):
        f = field(x)
        f = transform_field(f, method)
        if extrapolate:
            f = field(x - eta * f)
            f = transform_field(f, method)
        x -= eta * f
        d = np.sqrt(np.sum((x - optimum) ** 2))
        ds.append(d)
        xs.append(x.copy())
    xs = np.array(xs)
    ds = np.array(ds)
    return xs, ds


optimum = np.array([0, 0])
x0 = np.array([2., 2])
field = ConstantField(0.25, 1, optimum, )

extrapolate = True
results = {}
methods = ['scaled_signed', 'signed', 'gradient']
for method in methods:
    results[method] = optimise(x0, optimum, field, int(1e4), 0.1, method=method,
                               extrapolate=extrapolate)


x, y = np.meshgrid(np.linspace(-4, 4, 20), np.linspace(-4, 4, 20))
z = np.concatenate([x[:, :, None], y[:, :, None]], axis=2)
a = - field(z)

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.quiver(x, y, a[:, :, 0], a[:, :, 1], width=3e-3, color='.5')
for method in methods:
    xs, ds = results[method]
    ax1.plot(xs[:, 0], xs[:, 1], marker='+', label='method')
    ax2.plot(np.arange(len(ds)), ds, label=method)
ax1.set_ylim([-4, 4])
ax1.set_xlim([-4, 4])
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.legend()
plt.show()