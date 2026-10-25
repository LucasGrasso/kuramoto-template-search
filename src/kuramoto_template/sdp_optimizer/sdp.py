"""
Optimize inter-cluster connection densities P for fixed
cluster sizes x and phases phi in the homogeneous
Kuramoto model.

All quantities are asymptotic (large graph limit).
"""

import cvxpy as cp
import numpy as np
from scipy.linalg import null_space

from .sdp_result import SDPResult
from .validate_input import validate_input
from .template_metrics import template_metrics


def optimize_template(
    x,
    phi,
    *,
    allowed=None,
    internal_margin=1e-7,
    cluster_margin=1e-8,
    force_cliques=True,
    solver="CLARABEL",
    solver_options=None,
    residual_tolerance=1e-7,
    initial_P=None,
    verbose=False,
):
    """
    Maximize the asymptotic minimum-degree density.

    Parameters
    ----------
    x : array of shape (m,)
        Cluster size fractions, summing to one.

    phi : array of shape (m,)
        Cluster phases in radians.

    allowed : optional bool array (m, m)
        If allowed[a, b] is False, force P[a, b] = 0.

    internal_margin : float
        Lower bound on each within-cluster restoring
        coefficient c_a.

    cluster_margin : float
        Lower bound on the restricted cluster Hessian
        eigenvalues.

    force_cliques : bool
        Force P[a, a] = 1 (complete internal wiring).

    solver : str
        CVXPY conic solver, default CLARABEL.

    initial_P : optional array
        Starting density matrix for solver warm start.

    Returns
    -------
    TemplateSDPResult

    Notes
    -----
    Only x and phi are fixed. P and mu are optimized.

    Results are numerical candidates, not proofs.
    """

    x, phi, allowed = validate_input(x, phi, allowed)
    m = len(x)

    if internal_margin <= 0 or cluster_margin <= 0:
        raise ValueError("Stability margins must be positive")

    if force_cliques and allowed is not None:
        if not np.all(np.diag(allowed)):
            raise ValueError("force_cliques conflicts with forbidden diagonal")

    diff = phi[:, None] - phi[None, :]
    sine = np.sin(diff)
    cosine = np.cos(diff)

    # Decision variables
    P = cp.Variable((m, m), symmetric=True)
    mu = cp.Variable()

    # Normalized degree of each cluster
    degrees = P @ x

    # Equilibrium: F_a = 0
    forces = cp.sum(cp.multiply(P, sine * x[None, :]), axis=1)

    # Stability against splitting individual clusters
    restoring = cp.sum(cp.multiply(P, cosine * x[None, :]), axis=1)

    # Weighted cluster Laplacian
    coeff = np.outer(x, x) * cosine
    np.fill_diagonal(coeff, 0.0)

    W = cp.multiply(P, coeff)
    L = cp.diag(cp.sum(W, axis=1)) - W

    # Remove global phase-rotation zero mode
    B = null_space(np.ones((1, m)))
    restricted_L = B.T @ L @ B

    constraints = [
        P >= 0,
        P <= 1,
        # Max-min degree
        degrees >= mu,
        # Kuramoto equilibrium
        forces == 0,
        # Individual cluster stability
        restoring >= internal_margin,
        # Stability of cluster phase modes
        restricted_L >> cluster_margin * np.eye(m - 1),
    ]

    # Optional structural restrictions
    if allowed is not None:
        forbidden = (~allowed).astype(float)
        constraints.append(cp.multiply(P, forbidden) == 0)

    if force_cliques:
        constraints.append(cp.diag(P) == 1)

    problem = cp.Problem(cp.Maximize(mu), constraints)

    if initial_P is not None:
        initial_P = np.asarray(initial_P, dtype=float)

        if initial_P.shape != (m, m):
            raise ValueError("Invalid initial_P shape")

        P.value = np.clip(0.5 * (initial_P + initial_P.T), 0, 1)

    options = solver_options or {}

    try:
        problem.solve(
            solver=solver,
            warm_start=True,
            verbose=verbose,
            **options,
        )
    except cp.SolverError as exc:
        return SDPResult(
            status="solver_error",
            message=str(exc),
        )

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return SDPResult(
            status=problem.status,
            message="No feasible solution returned",
        )

    if P.value is None:
        return SDPResult(
            status=problem.status,
            message="Solver returned no density matrix",
        )

    # Do not round/clip before checking feasibility.
    P_result = np.asarray(P.value, dtype=float)

    metrics = template_metrics(x, phi, P_result)

    density_ok = (
        P_result.min() >= -residual_tolerance
        and P_result.max() <= 1 + residual_tolerance
    )

    symmetry_ok = np.max(np.abs(P_result - P_result.T)) <= residual_tolerance

    pattern_ok = True
    if allowed is not None:
        pattern_ok = (
            np.max(np.abs(P_result[~allowed]), initial=0.0) <= residual_tolerance
        )

    clique_ok = True
    if force_cliques:
        clique_ok = np.max(np.abs(np.diag(P_result) - 1)) <= residual_tolerance

    # Strictly positive numerical eigenvalues.
    # This does NOT constitute rigorous certification.
    numerically_feasible = bool(
        density_ok
        and symmetry_ok
        and pattern_ok
        and clique_ok
        and metrics.max_force_error <= residual_tolerance
        and metrics.min_restoring > 0
        and metrics.min_cluster_eigenvalue > 0
    )

    stats = problem.solver_stats

    return SDPResult(
        status=problem.status,
        mu=metrics.mu,
        P=P_result,
        degrees=metrics.degrees,
        forces=metrics.forces,
        restoring=metrics.restoring,
        max_force_error=metrics.max_force_error,
        min_restoring=metrics.min_restoring,
        min_cluster_eigenvalue=metrics.min_cluster_eigenvalue,
        numerically_feasible=numerically_feasible,
        solve_time=stats.solve_time,
    )


if __name__ == "__main__":
    # Five-cluster twisted state.
    # Each cluster connects to itself and its neighbors.

    m = 5
    x = np.full(m, 1.0 / m)
    phi = np.arange(m) * (2 * np.pi / m)

    allowed = np.eye(m, dtype=bool)

    for i in range(m):
        j = (i + 1) % m
        allowed[i, j] = True
        allowed[j, i] = True

    result = optimize_template(
        x,
        phi,
        allowed=allowed,
    )

    print("Status:", result.status)
    print("Mu:", result.mu)
    print("Numerically feasible:", result.numerically_feasible)
    print("Minimum cluster eigenvalue:", result.min_cluster_eigenvalue)

    if result.P is not None:
        print("P:")
        print(np.round(result.P, 5))
