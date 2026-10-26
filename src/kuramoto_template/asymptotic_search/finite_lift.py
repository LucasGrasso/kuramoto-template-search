"""Local tree force repair and independent exact finite-angle diagnostics."""

from dataclasses import dataclass, field

import numpy as np

from kuramoto_template.sdp_optimizer.template_metrics import template_metrics
from kuramoto_template.sdp_optimizer.validate_input import in_open_semicircle
from .expansions import edges, finite_rescaled_matrix
from .families import winding
from .models import LimitingCandidate, Numerics
from .stage2 import validate_correction


@dataclass(eq=False)
class LiftResult:
    candidate: LimitingCandidate
    r: np.ndarray
    h: float
    status: str
    P: np.ndarray | None = None
    phi: np.ndarray | None = None
    mu: float | None = None
    base_force_inf: float | None = None
    repair_inf: float | None = None
    tree: list[tuple[int,int]] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    tree_attempts: int = 0
    message: str = ""


def exact_forces(c, P, h):
    phi = c.template(h).phi
    return np.sum(P*np.sin(phi[None,:]-phi[:,None])*c.x[None,:],axis=1)


def validate_finite(c, P, h, *, numerics=None):
    """Reuse the project validator and add scaled and support/backbone checks.

    The positive eigenvalue must resolve a conservative float64 roundoff floor.
    This does not bound every source of numerical error and is not certification.
    """
    n = numerics or Numerics()
    P = np.asarray(P,dtype=float)
    template = c.template(h)
    if P.shape != c.p0.shape or not np.all(np.isfinite(P)):
        return "finite_invalid", {"message":"Invalid matrix shape or nonfinite entries"}
    metrics = template_metrics(template.x,template.phi,P)
    scaled = finite_rescaled_matrix(c,P,h)
    scaled_gap = float(np.linalg.eigvalsh(scaled).min())
    cosine = np.cos(template.phi[:,None]-template.phi[None,:])
    w = np.outer(c.x,c.x)*P*cosine
    np.fill_diagonal(w,0)
    lap = np.diag(w.sum(axis=1))-w
    noise = float(256*np.finfo(float).eps*c.m*max(np.linalg.norm(lap,2),1e-3))
    box = float(max(0.,-P.min(),P.max()-1))
    structural = float(max(np.max(np.abs(P-P.T)),np.max(np.abs(P[~c.allowed_mask]),initial=0),
                           np.max(np.abs(np.diag(P)-1))))
    diagnostics = dict(mu=metrics.mu, force_inf=metrics.max_force_error,
                       c_min=metrics.min_restoring, cluster_gap=metrics.min_cluster_eigenvalue,
                       rescaled_gap=scaled_gap, roundoff_floor=noise, box_violation=box,
                       structural_violation=structural,
                       winding_one=winding(c,P,h,min_density=n.backbone_density),
                       nonsynchronous=not in_open_semicircle(template.phi),
                       force_tolerance=n.residual_tolerance, proof_available=False)
    if box > 0 or structural > n.residual_tolerance or metrics.max_force_error > n.residual_tolerance:
        status = "finite_invalid"
    elif not diagnostics["winding_one"] or not diagnostics["nonsynchronous"]:
        status = "finite_invalid"
    elif metrics.min_cluster_eigenvalue < -noise or metrics.min_restoring < -noise or scaled_gap < -n.slack_tolerance:
        status = "finite_unstable"
    elif (metrics.min_cluster_eigenvalue > noise and metrics.min_restoring > noise
          and scaled_gap > max(n.strict_gap,noise/h**2)):
        status = "finite_h_numeric_strict"
    else:
        status = "finite_uncertain"
    return status, diagnostics


def _tree(c, P, phi, variant):
    """Maximum-slack spanning trees, with deterministic alternative tie orders."""
    eligible = []
    for a,b,_ in edges(c)[1]:
        if not c.allowed_mask[a,b] or abs(np.sin(phi[b]-phi[a])) < .5:
            continue
        slack = min(P[a,b],1-P[a,b])
        if slack <= 0:
            continue
        # Rotate priority to try alternate trees when a correction hits a bound.
        eligible.append((slack,a,b))
    eligible.sort(key=lambda item:(-item[0],item[1],item[2]))
    if eligible and variant:
        shift = variant % len(eligible)
        eligible = eligible[shift:]+eligible[:shift]
    parents = list(range(c.m))

    def root(a):
        while parents[a] != a:
            a = parents[a]
        return a

    tree = []
    for _,a,b in eligible:
        ra,rb = root(a),root(b)
        if ra != rb:
            parents[ra] = rb
            tree.append((a,b))
        if len(tree)==c.m-1:
            return tree
    return []


def lift_candidate(stage2, h, *, numerics=None, ledger=None, max_trees=12):
    n = numerics or Numerics()
    c, r = stage2.candidate, stage2.r
    c.template(h)  # Validate h before arithmetic.
    result = LiftResult(c,r,h,"correction_rejected")
    validated = validate_correction(c,r,numerics=n,solver_status=stage2.solver_status,
                                    reported_gamma=stage2.reported_gamma)
    if validated.status != "asymptotic_full_second_order_strict":
        result.message = "Correction failed independent strict validation"
        return result
    base = c.p0+h*r
    result.phi = c.template(h).phi
    result.base_force_inf = float(np.max(np.abs(exact_forces(c,base,h))))
    if base.min() < 0 or base.max() > 1:
        result.status, result.message = "box_rejected", "P0+hR is outside [0,1]; no clipping"
        return result
    result.status = "force_repair_failed"
    weighted = c.x*exact_forces(c,base,h)
    seen = set()
    for variant in range(max_trees):
        tree = _tree(c,base,result.phi,variant)
        key = tuple(sorted(tuple(sorted(pair)) for pair in tree))
        if not tree or key in seen:
            continue
        seen.add(key)
        A = np.zeros((c.m,c.m-1))
        for i,(a,b) in enumerate(tree):
            w = c.x[a]*c.x[b]*np.sin(result.phi[b]-result.phi[a])
            A[a,i], A[b,i] = w,-w

        def linear_solve():
            return np.linalg.solve(A[:-1],-weighted[:-1])

        result.tree_attempts += 1
        delta = ledger.invoke("finite_tree_linear",linear_solve) if ledger is not None else linear_solve()
        P = base.copy()
        for value,(a,b) in zip(delta,tree):
            P[a,b] += value
            P[b,a] += value
        if P.min() < 0 or P.max() > 1:
            continue
        result.P, result.tree = P,tree
        result.repair_inf = float(np.max(np.abs(delta)))
        result.status, result.diagnostics = validate_finite(c,P,h,numerics=n)
        result.mu = result.diagnostics.get("mu")
        if result.repair_inf > n.max_repair_coefficient*h*h:
            result.status, result.message = "nonlocal_repair", "Tree change exceeds configured O(h^2) tube"
        return result
    result.message = "No usable spanning tree with a box-feasible force correction"
    return result


def scaling_slopes(lifts):
    """Measured log slopes; report unresolved data rather than fitting zeros."""
    def value(result,name):
        return result.get(name) if isinstance(result,dict) else getattr(result,name)

    out = {}
    for name in ("base_force_inf","repair_inf"):
        pairs = [(value(r,"h"),value(r,name)) for r in lifts if value(r,name) is not None
                 and value(r,name)>1e-14]
        out[name] = (float(np.polyfit(np.log([a for a,b in pairs]),np.log([b for a,b in pairs]),1)[0])
                     if len(pairs)>=2 else None)
    pairs = [(value(r,"h"),value(r,"diagnostics")["cluster_gap"]) for r in lifts
             if value(r,"diagnostics").get("cluster_gap",0)>value(r,"diagnostics").get("roundoff_floor",float("inf"))]
    out["cluster_gap"] = (float(np.polyfit(np.log([a for a,b in pairs]),np.log([b for a,b in pairs]),1)[0])
                           if len(pairs)>=2 else None)
    return out
