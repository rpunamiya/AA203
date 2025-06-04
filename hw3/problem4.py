import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[0.9, 0.6], [0.0, 0.8]])
B = np.array([[0, 1]])
N = 4
rx = 5
ru = 1
Q = np.eye(2)
P = np.eye(2)

M = cp.Variable((2, 2), symmetric=True)
block_matrix = cp.vstack([cp.hstack([M, A@M]), cp.hstack([M.T @ A.T, M])])

constraints = [
    block_matrix >> 0,
    M >> 0,
    M << rx**2 * np.eye(2)
]

objective = cp.Maximize(cp.log_det(M))
problem = cp.Problem(objective, constraints)
problem.solve()

print("Matrix M:")
print(M.value)
print("Matrix W:")
W = np.linalg.inv(M.value)
print("Matrix W:", np.round(W, 3))

def generate_ellipsoid_points(M, num_points=100):
    """Generate points on a 2-D ellipsoid."""
    L = np.linalg.cholesky(M)
    θ = np.linspace(0, 2*np.pi, num_points)
    u = np.column_stack((np.cos(θ), np.sin(θ)))
    x = u @ L.T
    return x

# Generate points
X_T_points = generate_ellipsoid_points(M.value)
A = np.array([[0.9, 0.6], [0, 0.8]])
AX_T_points = X_T_points @ A.T
X_points = generate_ellipsoid_points(np.eye(2) * rx**2)

# Plotting
plt.figure(figsize=(8, 8))
plt.plot(X_T_points[:, 0], X_T_points[:, 1], 'r', label='X_T')
plt.plot(AX_T_points[:, 0], AX_T_points[:, 1], 'g', label='AX_T')
plt.plot(X_points[:, 0], X_points[:, 1], 'b', label='X')
plt.legend()
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Ellipsoids for X_T, AX_T, and X')
plt.grid(True)
plt.show()
