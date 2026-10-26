"""First-order force correction and the complete effective Hessian LMI."""

import cvxpy as cp
import numpy as np

from .expansions import (U4, coefficients, cut_matrix, edges, effective_matrix,
                         full_rescaled_matrix, reference_rescaled_matrix)
from .models import Numerics, Stage2Result
from .solvers import solve
from .stage1 import screen_candidate


def correction_violation(c, r, n):
    """Check the unmodified R, including forbidden/internal entries and trial box."""
    if r.shape != c.p0.shape or not np.all(np.isfinite(r)):
        return float("inf")
    neighbor = (c.groups[:,None]-c.groups[None,:]) % 2 == 1
    permitted = neighbor & c.allowed_mask
    if np.any(r[~permitted] != 0) or not np.array_equal(r,r.T):
        return float("inf")
    low = permitted & (c.p0 <= n.residual_tolerance)
    high = permitted & (c.p0 >= 1-n.residual_tolerance)
    trial = c.p0+n.trial_h*r
    return float(max(0., np.max(np.abs(r-r.T)), np.max(np.abs(r[~permitted]), initial=0),
                     -np.min(r[low], initial=0), np.max(r[high], initial=0),
                     np.max(np.abs(r))-n.r_max, -trial.min(), trial.max()-1))


def validate_correction(c, r, *, numerics=None, solver_status="analytic", reported_gamma=None):
    n = numerics or Numerics()
    output = Stage2Result(c, None if r is None else np.array(r, copy=True), "failed", solver_status,
                          reported_gamma=reported_gamma)
    if r is None:
        return output
    r = np.asarray(r, dtype=float)
    violation = correction_violation(c, r, n)
    output.cone_violation = violation
    if not np.isfinite(violation):
        return output
    screened = screen_candidate(c, n)
    if screened.status not in ("stage1_screened", "uncertain"):
        output.message = "Stage-1 assumptions failed: " + screened.message
        return output
    e = coefficients(c,r)
    output.force1_inf = float(np.max(np.abs(e.f1)))
    output.budget_identity_error = float(abs(e.k.sum()-e.energy))
    try:
        Q = effective_matrix(c, r, n.rank_tolerance)
        full = full_rescaled_matrix(c,r)
        independent = reference_rescaled_matrix(c,r)
        hard = full[3:,3:]
        output.hard_gap = float(np.linalg.eigvalsh(hard).min()) if len(hard) else float("inf")
        schur = full[:3,:3] - full[:3,3:]@np.linalg.solve(hard,full[3:,:3])
        output.schur_error = float(max(np.max(np.abs(full-independent)),np.max(np.abs(schur-U4.T@Q@U4))))
        output.effective_gap = float(np.linalg.eigvalsh(U4.T@Q@U4).min())
        output.full_rescaled_gap = float(np.linalg.eigvalsh(full).min())
    except (ValueError, np.linalg.LinAlgError) as exc:
        output.status, output.message = "asymptotic_degenerate", str(exc)
        return output
    valid = (max(violation, output.force1_inf, output.budget_identity_error, output.schur_error,
                 screened.force0_inf, screened.cut0_inf) <= n.residual_tolerance)
    if not valid:
        output.status, output.message = "failed", "Independent equation/cone/Schur validation failed"
    elif solver_status not in ("optimal", "analytic"):
        output.status, output.message = "uncertain", "Solver status cannot support strict classification"
    elif reported_gamma is not None and (not np.isfinite(reported_gamma)
              or reported_gamma > output.effective_gap + n.residual_tolerance):
        output.status, output.message = "uncertain", "Reported margin disagrees with measured margin"
    elif output.effective_gap > n.strict_gap and output.full_rescaled_gap > n.strict_gap and output.hard_gap > n.rank_tolerance:
        output.status = "asymptotic_full_second_order_strict"
    elif output.effective_gap < -n.slack_tolerance:
        output.status = "asymptotic_unstable"
    else:
        output.status = "asymptotic_degenerate"
    return output


def solve_correction(c, *, numerics=None, ledger=None, secondary=True):
    n = numerics or Numerics()
    metrics = screen_candidate(c,n)
    if metrics.status != "stage1_screened":
        return Stage2Result(c, None, metrics.status, message=metrics.message)
    cuts = [(a,b,j) for a,b,j in edges(c)[1] if c.allowed_mask[a,b]]
    v, gamma = cp.Variable(len(cuts)), cp.Variable()
    e = coefficients(c)
    f = cp.Constant(e.f1)
    Q = cp.Constant(effective_matrix(c,np.zeros_like(c.p0)))
    constraints = [v >= -n.r_max, v <= n.r_max]
    for i,(a,b,j) in enumerate(cuts):
        vector = np.zeros(c.m)
        vector[a], vector[b] = c.x[b], -c.x[a]
        f = f + v[i]*vector
        k = np.zeros(4)
        k[j] = -c.x[a]*c.x[b]*(c.s[b]-c.s[a])
        Q = Q + v[i]*cut_matrix(k)
        p = c.p0[a,b]
        constraints += [p+n.trial_h*v[i] >= 0, p+n.trial_h*v[i] <= 1]
        if p <= n.residual_tolerance:
            constraints.append(v[i] >= n.inward_slack)
        elif p >= 1-n.residual_tolerance:
            constraints.append(v[i] <= -n.inward_slack)
    constraints += [f == 0, U4.T@Q@U4 >> gamma*np.eye(3)]
    problem = cp.Problem(cp.Maximize(gamma), constraints)
    result = solve(problem, "stage2_correction_sdp", n, ledger)
    if result.status not in ("optimal", "optimal_inaccurate") or v.value is None:
        return Stage2Result(c,None,"infeasible" if result.status == "infeasible" else "uncertain",
                            result.status, message=result.message)
    values, reported = v.value.copy(), float(gamma.value)
    # Uniform thinning is an equilibrium null direction. Minimize its arbitrary
    # size while preserving 95% of the positive margin; this is a second PAID SDP.
    if (secondary and result.status == "optimal" and reported > 2*n.strict_gap
            and (ledger is None or ledger.remaining("stage2_correction_sdp") > 0)):
        secondary_problem = cp.Problem(cp.Minimize(cp.sum_squares(v)),
                                       constraints+[gamma >= .95*reported])
        secondary_result = solve(secondary_problem,"stage2_correction_sdp",n,ledger)
        if secondary_result.status == "optimal" and v.value is not None:
            values, reported = v.value.copy(), float(gamma.value)
    r = np.zeros_like(c.p0)
    for (a,b,_),value in zip(cuts,values):
        r[a,b] = r[b,a] = value
    return validate_correction(c,r,numerics=n,solver_status=result.status,reported_gamma=reported)
