import numpy as np
from sklearn.utils import check_random_state


def quadratic(r, theta, x):
    return 1 - (1 + r ** 2 * x ** 2 - 2 * r * x * np.cos(theta))


x = np.linspace(0, 1, 1000)

rng = check_random_state(43001)
thetas = rng.uniform(-np.pi / 2, np.pi / 2, 10)
rs = rng.uniform(1, 2, 10)

ys = np.array([quadratic(r, theta, x) for (r, theta) in zip(rs, thetas)]).T

ys = np.min(ys, axis=1)
print(ys)

import matplotlib.pyplot as plt

plt.plot(x, ys)
plt.ylim([0, 1])
plt.show()