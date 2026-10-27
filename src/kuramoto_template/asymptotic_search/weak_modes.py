"""Fixed-gauge weak-mode rankings. Every proposed state must be re-solved."""

from dataclasses import dataclass, replace
from decimal import Decimal, localcontext

import numpy as np

from .expansions import U4, coefficients, internal_basis, internal_solve
from .families import normalize_offsets, parent_mask
from .joint_limit import build_affine_model


def weakest_mode(hlim):
    H = np.asarray(hlim, dtype=float)
    if H.ndim != 2 or H.shape[0] != H.shape[1] or not np.all(np.isfinite(H)):
        raise ValueError("A finite square Hessian is required")
    values, vectors = np.linalg.eigh((H+H.T)/2)
    v = vectors[:,0]
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    return float(values[0]), v


def internal_partial_sensitivities(candidate, eta):
    """Derivative with respect to physical edge weight, holding G and cut fixed.

    This is not the derivative of the reoptimized equilibrium problem. For an
    edge density multiply by x_a*x_b. Its sign is nonnegative even when the
    total change after changing masses, G, cuts and feasible P is negative.
    """
    e = coefficients(candidate)
    z, _, residual = internal_solve(candidate, e.lin, e.g@np.asarray(eta))
    if residual > 1e-9:
        raise ValueError("Unresolved range residual")
    return {(a,b): float((z[a]-z[b])**2) for a in range(candidate.m)
            for b in range(a+1,candidate.m) if candidate.groups[a] == candidate.groups[b]}


@dataclass(frozen=True)
class TopologyMove:
    edge: tuple[int,int]
    operation: str
    predicted_gain: float
    delta_p: float
    delta_r: float
    interpretation: str = "fixed weak vector before reoptimization; not a feasibility decision"


def rank_topology_moves(candidate, r, hlim, *, opening_density=.05):
    _, v = weakest_mode(hlim)
    full = build_affine_model(candidate.x,candidate.s,candidate.groups,parent_mask(candidate.groups),
                              backbone_cycle=candidate.backbone_cycle)
    r_index = {pair:i for i,pair in enumerate(full.r_pairs)}
    protected = {tuple(sorted((a,b))) for a,b in zip(candidate.backbone_cycle,
                                                   candidate.backbone_cycle[1:]+candidate.backbone_cycle[:1])}
    moves = []
    for i,(a,b) in enumerate(full.p_pairs):
        if (a,b) in protected:
            continue
        if candidate.allowed_mask[a,b]:
            operation, dp, dr = "close", -candidate.p0[a,b], -r[a,b]
        else:
            operation, dp, dr = "open", opening_density, 0.
        change = dp*full.h_p[:,i]
        if (a,b) in r_index:
            change = change+dr*full.h_r[:,r_index[a,b]]
        gain = float(v@change.reshape(candidate.m-1,candidate.m-1)@v)
        moves.append(TopologyMove((a,b),operation,gain,float(dp),float(dr)))
    return sorted(moves,key=lambda move:(-move.predicted_gain,move.edge))


def apply_topology_move(candidate, move):
    a,b = move.edge
    mask,p = candidate.allowed_mask.copy(),candidate.p0.copy()
    if move.operation == "close":
        mask[a,b] = mask[b,a] = False
        p[a,b] = p[b,a] = 0.
    elif move.operation == "open":
        mask[a,b] = mask[b,a] = True
        # The new point is outer input only; P and R will be jointly re-solved.
        p[a,b] = p[b,a] = 0.
    else:
        raise ValueError("Unknown topology operation")
    return replace(candidate,allowed_mask=mask,p0=p)


def continuous_move(candidate, rng, *, mass_radius=.03, offset_radius=.03, min_mass=1e-5):
    logits = np.log(candidate.x)+rng.normal(0,mass_radius,candidate.m)
    masses = np.exp(logits-logits.max())
    masses /= masses.sum()
    if masses.min() < min_mass:
        raise ValueError("Mass-logit move violates the mass floor")
    offsets = normalize_offsets(masses,candidate.s+rng.normal(0,offset_radius,candidate.m))
    return replace(candidate,x=masses,s=offsets)


def high_precision_limit_audit(result, *, digits=60):
    """Independent Decimal polynomial assembly and LDL of the full fixed gauge.

    Saved binary64 x/s/p/r/U/Z are interpreted exactly. No interval or root
    enclosure is performed. An audit never changes a rigorous-proof status.
    """
    if not isinstance(digits,int) or not 30 <= digits <= 80 or result.candidate is None:
        raise ValueError("Require a candidate and 30..80 digits")
    c, r = result.candidate,result.r
    Z = internal_basis(c)
    with localcontext() as context:
        context.prec = digits+10
        D = lambda value: Decimal.from_float(float(value))
        zero = Decimal(0)
        x,s = list(map(D,c.x)),list(map(D,c.s))
        p = [list(map(D,row)) for row in c.p0]
        correction = [list(map(D,row)) for row in r]
        d = c.m-1
        H = [[zero]*d for _ in range(d)]
        F0,F1,K,k = [zero]*c.m,[zero]*c.m,[zero]*4,[zero]*4
        energy = zero
        restoring = [x[a] for a in range(c.m)]
        for a in range(c.m):
            for b in range(a+1,c.m):
                mass,ds = x[a]*x[b],s[b]-s[a]
                if c.groups[a] == c.groups[b]:
                    z = [zero]*3+[D(Z[a,i])-D(Z[b,i]) for i in range(c.m-4)]
                    for i in range(d):
                        for j in range(d):
                            H[i][j] += mass*p[a][b]*z[i]*z[j]
                    energy += mass*p[a][b]*ds*ds
                    F1[a] += x[b]*p[a][b]*ds
                    F1[b] -= x[a]*p[a][b]*ds
                    restoring[a] += x[b]*p[a][b]
                    restoring[b] += x[a]*p[a][b]
                elif (c.groups[b]-c.groups[a])%2 == 1:
                    aa,bb = (a,b) if (c.groups[b]-c.groups[a])%4 == 1 else (b,a)
                    delta = s[bb]-s[aa]
                    jgroup = c.groups[aa]
                    F0[aa] += x[bb]*p[a][b]
                    F0[bb] -= x[aa]*p[a][b]
                    F1[aa] += x[bb]*correction[a][b]
                    F1[bb] -= x[aa]*correction[a][b]
                    K[jgroup] += mass*p[a][b]*delta
                    k[jgroup] -= mass*correction[a][b]*delta
                    u = [D(U4[c.groups[aa],i])-D(U4[c.groups[bb],i]) for i in range(3)]
                    z = [D(Z[aa,i])-D(Z[bb,i]) for i in range(c.m-4)]
                    for i in range(3):
                        for j in range(3):
                            H[i][j] -= mass*correction[a][b]*delta*u[i]*u[j]
                        for j in range(c.m-4):
                            value = -mass*p[a][b]*delta*u[i]*z[j]
                            H[i][j+3] += value
                            H[j+3][i] += value
        lower = [[zero]*d for _ in range(d)]
        pivots = []
        for i in range(d):
            pivot = H[i][i]-sum((lower[i][j]**2*pivots[j] for j in range(i)),zero)
            pivots.append(pivot)
            if pivot == 0:
                break
            for j in range(i+1,d):
                lower[j][i] = (H[j][i]-sum((lower[j][q]*lower[i][q]*pivots[q] for q in range(i)),zero))/pivot
        fmt = lambda value: format(value,f".{digits-5}g")
        return dict(precision_digits=digits,force0_inf=fmt(max(map(abs,F0))),
                    force1_inf=fmt(max(map(abs,F1))),cut_inf=fmt(max(map(abs,K))),
                    budget_identity_error=fmt(abs(sum(k)-energy)),inner_restoring=fmt(min(restoring)),
                    pivots=[fmt(v) for v in pivots],positive_pivots=len(pivots)==d and min(pivots)>0,
                    matrix_error_vs_float=fmt(max(abs(H[i][j]-D(result.hlim[i,j])) for i in range(d) for j in range(d))),
                    proof_available=False)
