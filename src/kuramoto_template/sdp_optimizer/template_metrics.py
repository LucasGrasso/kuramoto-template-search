from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space


@dataclass
class TemplateMetrics:
    mu: float
    degrees: np.ndarray
    forces: np.ndarray
    restoring: np.ndarray
    max_force_error: float
    min_restoring: float
    min_cluster_eigenvalue: float


def template_metrics(x, phi, P):
    """Evaluate a template independently of CVXPY."""

    m = len(x)

    diff = phi[:, None] - phi[None, :]
    sine = np.sin(diff)
    cosine = np.cos(diff)

    degrees = P @ x

    forces = np.sum(P * sine * x[None, :], axis=1)

    restoring = np.sum(P * cosine * x[None, :], axis=1)

    # Weighted cluster Laplacian.
    W = P * np.outer(x, x) * cosine
    np.fill_diagonal(W, 0.0)

    L = np.diag(W.sum(axis=1)) - W

    # Orthonormal basis for vectors perpendicular to 1.
    B = null_space(np.ones((1, m)))

    restricted = B.T @ L @ B
    restricted = 0.5 * (restricted + restricted.T)

    eigenvalues = np.linalg.eigvalsh(restricted)

    return TemplateMetrics(
        mu=float(degrees.min()),
        degrees=degrees,
        forces=forces,
        restoring=restoring,
        max_force_error=float(np.abs(forces).max()),
        min_restoring=float(restoring.min()),
        min_cluster_eigenvalue=float(eigenvalues.min()),
    )
