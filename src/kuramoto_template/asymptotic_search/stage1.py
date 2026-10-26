"""Direct necessary dipole tests and fixed-(x,s) LP / Schur-LMI screens."""

from dataclasses import dataclass, replace

import cvxpy as cp
import numpy as np
from scipy.linalg import null_space

from .expansions import DIPOLES, coefficients, internal_solve
from .families import winding
from .models import LimitingCandidate, Numerics, Stage1Metrics
from .solvers import solve


def screen_candidate(c, numerics=None):
    n = numerics or Numerics()
    e = coefficients(c)
    box = max(0., -c.p0.min(), c.p0.max()-1,
              np.max(np.abs(c.p0-c.p0.T)), np.max(np.abs(c.p0[~c.allowed_mask]), initial=0),
              np.max(np.abs(np.diag(c.p0)-1)))
    force, cut = float(np.max(np.abs(e.f0))), float(np.max(np.abs(e.cuts)))
    status, message = "stage1_screened", ""
    solve_res, gap, slacks = float("inf"), 0., (float("nan"),)*2
    range_res = max(float(abs(e.g[c.groups == j].sum(axis=0)).max()) for j in range(4))
    try:
        z, gap, solve_res = internal_solve(c, e.lin, e.g, n.rank_tolerance)
        energy = e.g.T @ z
        slacks = tuple(float(e.energy-v@energy@v) for v in DIPOLES)
        if np.linalg.eigvalsh((energy+energy.T)/2).min() < -n.slack_tolerance:
            status, message = "uncertain", "Negative inverse-Laplacian energy"
    except ValueError as exc:
        status, message = "asymptotic_degenerate", str(exc)
    if e.energy < n.energy_floor or np.ptp(c.s) <= n.rank_tolerance:
        status, message = "asymptotic_degenerate", "Trivial or unresolved splitting energy"
    elif box > n.residual_tolerance or force > n.residual_tolerance or cut > n.residual_tolerance:
        status, message = "screen_rejected", "Box, F0, or K residual exceeds tolerance"
    elif c.x.min() < n.min_mass or not winding(c, c.p0, n.trial_h, min_density=n.backbone_density):
        status, message = "screen_rejected", "Mass floor or winding backbone failed"
    elif status == "stage1_screened":
        if max(solve_res, range_res) > n.residual_tolerance:
            status, message = "uncertain", "Internal solve/range residual unresolved"
        elif min(slacks) < -n.slack_tolerance:
            status, message = "screen_rejected", "Necessary dipole inequality violated"
        elif min(slacks) <= n.slack_tolerance:
            status, message = "uncertain", "Dipole slack near zero"
    return Stage1Metrics(float((c.p0@c.x).min()), force, cut, gap, e.energy,
                         slacks, solve_res, range_res, float(box), status, message)


@dataclass
class Stage1Solve:
    candidate: LimitingCandidate | None
    metrics: Stage1Metrics | None
    status: str
    solver_status: str
    method: str
    message: str = ""


def optimize_limiting(c, *, method="lp", numerics=None, ledger=None):
    """One solver invocation. LP failures do not exclude other p at this outer point.

    Only an accurate infeasible status from the LMI model is labeled
    necessary_screen_infeasible, and only for the configured fixed outer point,
    connectivity/energy/backbone floors. This is numerical, not a certificate.
    """
    if method not in ("lp", "sdp"):
        raise ValueError("method must be lp or sdp")
    n = numerics or Numerics()
    pairs = [(a, b) for a in range(c.m) for b in range(a+1, c.m) if c.allowed_mask[a, b]]
    v = cp.Variable(len(pairs), name="p_edges")
    # Compile affine maps once. Nesting a whole matrix expression inside every
    # edge term otherwise makes CVXPY canonicalization dominate small screens.
    p_map = np.zeros((c.m*c.m,len(pairs)))
    f_map = np.zeros((c.m,len(pairs)))
    k_map = np.zeros((4,len(pairs)))
    l_map = np.zeros_like(p_map)
    g_map = np.zeros((c.m*4,len(pairs)))
    i_map = np.zeros(len(pairs))
    eye, eye4 = np.eye(c.m), np.eye(4)
    for i, (a, b) in enumerate(pairs):
        p_map[a*c.m+b,i] = p_map[b*c.m+a,i] = 1
        if c.groups[a] == c.groups[b]:
            w = c.x[a]*c.x[b]
            vec = eye[a]-eye[b]
            ds = c.s[b]-c.s[a]
            l_map[:,i] = (w*np.outer(vec,vec)).ravel()
            i_map[i] = w*ds**2
        else:
            if (c.groups[b]-c.groups[a])%4 != 1:
                a,b = b,a
            j = c.groups[a]
            ds,w = c.s[b]-c.s[a],c.x[a]*c.x[b]
            f_map[:,i] = c.x[b]*eye[a]-c.x[a]*eye[b]
            k_map[j,i] = w*ds
            g_map[:,i] = (w*ds*np.outer(eye[a]-eye[b],eye4[j]-eye4[(j+1)%4])).ravel()
    p = cp.reshape(np.eye(c.m).ravel()+p_map@v,(c.m,c.m),order="C")
    F,K = f_map@v,k_map@v
    L = cp.reshape(l_map@v,(c.m,c.m),order="C")
    G = cp.reshape(g_map@v,(c.m,4),order="C")
    I = i_map@v
    mu = cp.Variable(name="mu0")
    constraints = [v >= 0, v <= 1, p@c.x >= mu, F == 0, K == 0, I >= n.energy_floor]
    cycle = c.backbone_cycle
    for a,b in zip(cycle, cycle[1:]+cycle[:1]):
        constraints.append(p[a,b] >= n.backbone_density)
    if method == "sdp":
        for eta in DIPOLES:
            g = cp.reshape(G@eta, (c.m,1), order="C")
            constraints.append(cp.bmat([[L, g], [g.T, cp.reshape(I,(1,1),order="C")]]) >> 0)
        for j in range(4):
            idx = np.flatnonzero(c.groups == j)
            if len(idx)>1:
                B = null_space(np.ones((1,len(idx))))
                constraints.append(B.T@L[np.ix_(idx,idx)]@B >> n.connectivity_margin*np.eye(len(idx)-1))
    problem = cp.Problem(cp.Maximize(mu), constraints)
    result = solve(problem, "stage1_lp" if method == "lp" else "stage1_screen_sdp", n, ledger)
    if result.status != "optimal" or v.value is None:
        status = ("necessary_screen_infeasible" if method == "sdp" and result.status == "infeasible"
                  else "lp_screen_fail" if method == "lp" and result.status == "infeasible" else "uncertain")
        return Stage1Solve(None, None, status, result.status, method, result.message)
    # Fixed diagonal/forbidden entries are algebraic constants, not rounded output.
    density = np.eye(c.m)
    for (a,b), value in zip(pairs, v.value):
        density[a,b] = density[b,a] = value
    try:
        candidate = replace(c, p0=density)
    except ValueError as exc:
        return Stage1Solve(None, None, "uncertain", result.status, method, str(exc))
    metrics = screen_candidate(candidate, n)
    status = metrics.status
    if method == "lp" and status != "stage1_screened":
        status = "lp_screen_fail"
    return Stage1Solve(candidate, metrics, status, result.status, method)
