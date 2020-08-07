import numpy as np
import numpy.linalg as linalg

def rotation(theta):
    theta *= 2 * np.pi
    return np.array(
        [[np.cos(theta), np.sin(theta)],
         [- np.sin(theta), np.cos(theta)]]
    )


def generate_matrix(theta1, theta2, tau):
    A = np.zeros((4, 4))
    A[:2, :2] = rotation(theta1)
    A[2:, 2:] = rotation(theta2)

    gram = np.eye(4) * (1 - tau)
    gram += tau
    V = linalg.cholesky(gram)
    A = V @ A @ linalg.inv(V)

    return A

A = generate_matrix(0.7, 0.3, 0.0)
v, w = linalg.eig(A)
print(v)
print(w * np.sqrt(2))
print((w @ w.conj().T).real)