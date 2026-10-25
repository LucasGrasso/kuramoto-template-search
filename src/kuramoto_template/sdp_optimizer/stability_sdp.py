"""Maximize a common stability margin at fixed asymptotic degree target."""

from typing import TypedDict

import cvxpy as cp
import numpy as np
from scipy.linalg import null_space

from .stability_sdp_result import StabilitySDPResult
from .template_metrics import template_metrics
from .validate_input import validate_input


class _ResultMetadata(TypedDict):
    status: str
    target_mu: float
    solve_time: float | None


def maximize_stability_margin(
    x,
    phi,
    target_mu: float,
    *,
    allowed=None,
    force_cliques=True,
    solver="CLARABEL",
    solver_options=None,
    residual_tolerance=1e-7,
    stability_tolerance=1e-10,
    verbose=False,
) -> StabilitySDPResult:
    """Optimize P and unrestricted gamma with P @ x >= target_mu.

    Equilibrium is exact in the optimization model. Each restoring coefficient
    and every eigenvalue of the cluster Laplacian restricted to 1-perp must
    be at least gamma. These quantities have different scales: their common
    lower bound is a search surrogate, not an intrinsic stability measure.

    Negative gamma is allowed; equilibrium/degree constraints can still be
    infeasible. ``mu`` in the result is the measured minimum degree, whereas
    ``gamma`` is the optimized objective. All residuals are nonnegative
    absolute constraint violations computed on the unmodified solver output.

    ``numerically_valid`` checks constraints, including for OPTIMAL_INACCURATE.
    ``numerically_feasible`` additionally screens for strict stability above
    stability_tolerance, independently of residual_tolerance. Set
    stability_tolerance to zero to inspect tiny positive float64 margins; this still
    does not certify stability. Solver accuracy is set via solver_options.
    """
    x, phi, allowed = validate_input(x, phi, allowed)
    # validate_input uses NumPy's default relative tolerance; this API promises
    # the same absolute mass tolerance as PhaseTemplate, without normalization.
    if not np.isclose(x.sum(), 1.0, atol=1e-10, rtol=0):
        raise ValueError("Cluster sizes must sum to one (atol=1e-10)")
    target_mu = float(target_mu)
    if not np.isfinite(target_mu) or not 0 <= target_mu <= 1:
        raise ValueError("target_mu must be finite and between zero and one")
    for name, value in (("residual_tolerance", residual_tolerance),
                        ("stability_tolerance", stability_tolerance)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if force_cliques and allowed is not None and not np.all(np.diag(allowed)):
        raise ValueError("force_cliques conflicts with forbidden diagonal")

    m = len(x)
    diff = phi[:, None] - phi[None, :]
    sine, cosine = np.sin(diff), np.cos(diff)
    P = cp.Variable((m, m), symmetric=True, name="P")
    gamma = cp.Variable(name="gamma")
    forces = cp.sum(cp.multiply(P, sine * x[None, :]), axis=1)
    restoring = cp.sum(cp.multiply(P, cosine * x[None, :]), axis=1)
    coeff = np.outer(x, x) * cosine
    np.fill_diagonal(coeff, 0.0)
    W = cp.multiply(P, coeff)
    L = cp.diag(cp.sum(W, axis=1)) - W
    B = null_space(np.ones((1, m)))
    constraints = [
        P >= 0, P <= 1, P @ x >= target_mu, forces == 0,
        restoring >= gamma, B.T @ L @ B >> gamma * np.eye(m - 1),
    ]
    if allowed is not None:
        constraints.append(cp.multiply(P, (~allowed).astype(float)) == 0)
    if force_cliques:
        constraints.append(cp.diag(P) == 1)
    problem = cp.Problem(cp.Maximize(gamma), constraints)
    try:
        problem.solve(solver=solver, verbose=verbose, **(solver_options or {}))
    except cp.SolverError as exc:
        return StabilitySDPResult(
            status="solver_error", target_mu=target_mu, message=str(exc),
        )

    solve_time = getattr(problem.solver_stats, "solve_time", None)
    metadata: _ResultMetadata = {
        "status": problem.status or "unknown",
        "target_mu": target_mu,
        "solve_time": solve_time,
    }
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return StabilitySDPResult(**metadata, message="No feasible solution returned")
    if P.value is None or gamma.value is None:
        return StabilitySDPResult(**metadata, message="Solver returned no P or gamma")
    density = np.array(P.value, dtype=float, copy=True)
    margin = float(gamma.value)
    if not np.all(np.isfinite(density)) or not np.isfinite(margin):
        return StabilitySDPResult(**metadata, message="Solver returned nonfinite P or gamma")

    metrics = template_metrics(x, phi, density)
    residuals = {
        "symmetry": float(np.max(np.abs(density - density.T))),
        "lower_bound": float(max(0.0, -density.min())),
        "upper_bound": float(max(0.0, density.max() - 1)),
        "forbidden": (float(np.max(np.abs(density[~allowed]), initial=0.0))
                      if allowed is not None else 0.0),
        "cliques": (float(np.max(np.abs(np.diag(density) - 1)))
                    if force_cliques else 0.0),
        "degree": float(max(0.0, target_mu - metrics.mu)),
        "equilibrium": metrics.max_force_error,
        "restoring": float(max(0.0, margin - metrics.min_restoring)),
        "cluster_eigenvalue": float(max(0.0, margin - metrics.min_cluster_eigenvalue)),
    }
    valid = all(np.isfinite(r) and r <= residual_tolerance for r in residuals.values())
    stable = bool(valid and margin > stability_tolerance
                  and metrics.min_restoring > stability_tolerance
                  and metrics.min_cluster_eigenvalue > stability_tolerance)
    return StabilitySDPResult(
        **metadata, gamma=margin, mu=metrics.mu, P=density,
        degrees=metrics.degrees, forces=metrics.forces, restoring=metrics.restoring,
        max_force_error=metrics.max_force_error, min_restoring=metrics.min_restoring,
        min_cluster_eigenvalue=metrics.min_cluster_eigenvalue,
        residuals=residuals, numerically_valid=bool(valid), numerically_feasible=stable,
        message=("" if valid else "Solver output violates constraints beyond residual_tolerance"),
    )
