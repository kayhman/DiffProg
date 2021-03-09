import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import grad, jacfwd

from lagrangian_polynomial import LagrangianPolynome

# program inputs
Ts = [0.0, 1.0, 2.0, 3.0]
real_traj = jnp.array([0, 1, 1, 0])
# program params
Xs = np.array([0.0, 1.0, 2.0, 3.0])


def candidate_trajectory(Ts, Xs, t):
    poly = LagrangianPolynome(Ts, Xs)
    return poly.eval(t)


def squared_error(true, pred):
    return jnp.dot(true - pred, true - pred)


def traj_error(Xx):
    pred = jnp.array([candidate_trajectory(Ts, Xx, t) for t in [0.0, 1.0, 2.0, 3.0]])
    return squared_error(real_traj, pred)

error_grad = grad(traj_error)

max_iter = 150
delta = 2e-2
for i in range(0, max_iter):
    gradient = error_grad(Xs)
    Xs = Xs - gradient * delta
    if i % 10 == 0:
        times = np.linspace(0.0, 3.0, 40)
        traj = [candidate_trajectory(Ts, Xs, t) for t in times]
        plt.plot(times, traj, label=f'Iteration {i}')
print(Xs)


times = np.linspace(0.0, 3.0, 40)
traj = [candidate_trajectory(Ts, Xs, t) for t in times]
plt.plot([0, 1, 2, 3], real_traj, 'o')
plt.plot(times, traj)
plt.legend()
plt.show()
