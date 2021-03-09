import numpy as np
import matplotlib.pyplot as plt
from jax import grad

from lagrangian_polynomial import LagrangianPolynome

Ts = [0.0, 1.0, 2.0, 3.0]
Xs = [0.0, 1.0, 1.0, 0.0]

poly = LagrangianPolynome(Ts, Xs)
poly_der = grad(poly.eval)

# Check key values
print(poly_der(1.5))

times = np.linspace(0.0, 3.0, 40)
traj = [poly.eval(t) for t in times]
derivative = [poly_der(t) for t in times]
plt.plot(Ts, Xs, 'o')
plt.plot(times, traj)
plt.plot([1.5], [0.0], 'o')
plt.plot(times, derivative)
plt.show()
