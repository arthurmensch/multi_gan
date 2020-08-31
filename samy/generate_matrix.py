import math

import numpy as np
import numpy.linalg as linalg
from scipy.sparse import diags


def rotation(theta):
    theta *= 2 * np.pi
    return np.array(
        [[np.cos(theta), np.sin(theta)],
         [- np.sin(theta), np.cos(theta)]]
    )


def generate_matrix(theta1, theta2, tau):
    A = np.zeros((5, 5))
    A[:2, :2] = rotation(theta1)
    A[2:4, 2:4] = rotation(theta2)
    A[-1, -1] = 1
    V = np.eye(5)
    V[:2, 2:4] = np.eye(2) * tau
    A = V @ A @ linalg.inv(V)

    return A

def block_diagonalise(A):
    vs, ws = linalg.eig(A)
    plans = []
    angles = []
    lines = []
    values = []
    for i, (v, w) in enumerate(zip(vs, ws.T)):
        if v.imag != 0 and i % 2 == 0:
            real = w.real
            real /= np.sqrt(np.sum(real ** 2))
            imag = w.imag
            w.imag -= np.dot(w.imag, real) * real # OS ortho
            plans.append((real, imag))
            angles.append((np.absolute(v), np.angle(v)))
        else:
            break
    lines = ws[i:]
    values = vs[i:]
    return angles, values, plans, lines


def block_diagonalise(A):
    vs, ws = linalg.eig(A)
    cut = np.where(vs.imag == 0)[0][0]
    c_vs = np.zeros_like(ws.real)
    c_vs[cut:, cut:] = np.diag(vs[cut:].real)
    for i in range(0, cut, 2):
        c_vs[i:i+2, i:i+2] = np.array([[vs[i].real, - vs[i].imag], [vs[i].imag, vs[i].real]])

    c_ws = ws.copy()
    c_ws[:, :cut:2] = c_ws[:, :cut:2].real
    c_ws[:, 1:cut:2] = c_ws[:, 1:cut:2].imag
    return c_vs, c_ws.real


A = generate_matrix(0.3, 0.6, 1)
V, W = block_diagonalise(A)
Ap = W @ V @ linalg.inv(W)
print(W)