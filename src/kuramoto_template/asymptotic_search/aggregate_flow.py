"""Necessary four-parent flow bounds, with explicit scope and one-sided gates."""

from dataclasses import asdict,dataclass
from decimal import Decimal,localcontext

import numpy as np
from scipy.optimize import brentq

from .models import MU_RECORD

CONCRETE_PAPER_EXAMPLE = .6929050246493944
UNIFORM_BLOCK_CAP = (1+np.sqrt(2))/(3+1/np.sqrt(2))


def parent_masses(x,groups):
    return np.array([np.sum(np.asarray(x)[np.asarray(groups)==j]) for j in range(4)])


def parent_flows(candidate):
    c=candidate
    return np.array([sum(c.x[a]*c.x[b]*c.p0[a,b] for a in np.flatnonzero(c.groups==j)
        for b in np.flatnonzero(c.groups==(j+1)%4)) for j in range(4)])


@dataclass(frozen=True)
class AggregateScreen:
    masses: tuple
    cmax: float | None
    cap: float | None
    target: float
    tolerance: float
    excess: float | None
    status: str
    in_scope: bool
    reason: str
    flow_target: float | None = None
    decimal_excess: str | None = None

    def data(self):
        return asdict(self)


def screen_aggregate(x,groups,target=MU_RECORD,*,allowed_mask=None,leading_phases=None,
                     flow_target=None,tolerance=1e-12):
    """Reject only a resolved excess in the mass-cap inequality.

    Scope is checked before any bound. Decimal arithmetic interprets the input
    binary64 masses exactly; a conservative absolute/scale-aware buffer absorbs
    reduction and input uncertainty. This is not an interval proof for unknown
    physical parameters, and equality is never rejected.
    """
    x,groups=np.asarray(x,dtype=float),np.asarray(groups)
    valid=(x.ndim==1 and groups.shape==x.shape and np.all(np.isfinite(x)) and np.all(x>0)
           and abs(x.sum()-1)<=1e-10 and set(groups.tolist())=={0,1,2,3})
    reason="cardinal four-parent adjacent-support scope"
    if valid and allowed_mask is not None:
        mask=np.asarray(allowed_mask,dtype=bool)
        valid=mask.shape==(len(x),len(x)) and np.array_equal(mask,mask.T) and not np.any(mask & ((groups[:,None]-groups[None,:])%4==2))
        if not valid:
            reason="opposite-parent support or invalid mask: bound disabled"
    if valid and leading_phases is not None:
        phases=np.asarray(leading_phases,dtype=float)
        residual=((phases-groups*np.pi/2+np.pi)%(2*np.pi))-np.pi if phases.shape==x.shape else np.array([np.inf])
        # A common phase rotation is harmless; arbitrary leading offsets are not.
        valid=bool(np.all(np.isfinite(residual)) and np.max(np.abs(((residual-residual[0]+np.pi)%(2*np.pi))-np.pi))<1e-12)
        if not valid:
            reason="non-cardinal leading phases: bound disabled"
    if not np.isfinite(target) or not 0<=target<=1 or not np.isfinite(tolerance) or tolerance<=0:
        raise ValueError("Require a target in [0,1] and positive finite tolerance")
    if not valid:
        return AggregateScreen((),None,None,float(target),float(tolerance),None,"out_of_scope",False,reason)
    M=parent_masses(x,groups)
    C=float(np.min(M*np.roll(M,-1)))
    if flow_target is not None and (not np.isfinite(flow_target) or flow_target<0):
        raise ValueError("flow_target must be finite and nonnegative")
    usable=min(C,flow_target) if flow_target is not None else C
    cap=float(np.min(M+2*usable/M))
    excess=float(np.max(M*(target-M))-2*usable)
    tol=max(float(tolerance),256*np.finfo(float).eps*max(1.,abs(2*usable),abs(target)))
    with localcontext() as context:
        context.prec=70
        D=lambda v:Decimal.from_float(float(v))
        dm=[sum((D(x[a]) for a in range(len(x)) if groups[a]==j),Decimal(0)) for j in range(4)]
        dc=min(dm[j]*dm[(j+1)%4] for j in range(4))
        if flow_target is not None:
            dc=min(dc,D(flow_target))
        exact=max(m*(D(target)-m) for m in dm)-2*dc
        rejected=exact>D(tol)
    status="aggregate_rejected" if rejected else "boundary" if abs(excess)<=tol else "not_excluded"
    return AggregateScreen(tuple(float(v) for v in M),C,cap,float(target),tol,excess,status,True,
        "target exceeds necessary common-flow capacity" if rejected else "necessary bound only; solve still required",
        flow_target,str(exact))


def q_mass_cap(q):
    if not np.isfinite(q) or q<=1:
        raise ValueError("q is a mass ratio here and must exceed one")
    return float(min(6/(7+q),((1+q)+8/(1+q))/(7+q)))


def q_mass_critical(target=MU_RECORD):
    return float(brentq(lambda q:((1+q)+8/(1+q))/(7+q)-target,1.,2.,xtol=5e-15))


def uniform_block_rule(candidate,*,metadata=None,tolerance=1e-10):
    """Optional narrowly scoped analytic rule, never inferred from family name.

    Metadata must explicitly assert uniform blocks and complete internals; the
    numerical arrays are then checked as well. It does not apply to S7's crossed
    densities or I=0. The theorem's bound is analytical; numeric scope checks and
    observed candidate diagnostics do not certify a graph witness.
    """
    c=candidate
    if metadata is None or not all(metadata.get(k) is True for k in ("uniform_parent_blocks","complete_internal","singleton_anchor")):
        return dict(status="out_of_scope",in_scope=False,reason="assumptions not guaranteed by metadata")
    M=parent_masses(c.x,c.groups)
    same=c.groups[:,None]==c.groups[None,:]
    if (np.max(np.abs(M[:3]-M[0]))>tolerance or np.sum(c.groups==3)!=1
        or np.max(np.abs(c.p0[same]-1))>tolerance):
        return dict(status="out_of_scope",in_scope=False,reason="mass/internal/anchor scope failed")
    for j in range(4):
        block=c.p0[np.ix_(c.groups==j,c.groups==(j+1)%4)]
        if np.ptp(block)>tolerance:
            return dict(status="out_of_scope",in_scope=False,reason="heterogeneous interparent block")
    from .expansions import coefficients
    e=coefficients(c)
    flows=parent_flows(c)
    if max(np.max(np.abs(e.f0)),np.max(np.abs(e.cuts)),np.ptp(flows))>tolerance:
        return dict(status="out_of_scope",in_scope=False,reason="equilibrium/cut hypotheses unresolved")
    if e.energy<=tolerance:
        return dict(status="trivial_splitting",in_scope=False,reason="I=0 is excluded from the theorem")
    bound=M[0]**2/np.sqrt(2)
    return dict(status="uniform_block_no_go" if flows.mean()>bound+tolerance else "not_excluded",
                in_scope=True,flow=float(flows.mean()),flow_cap=float(bound),mu_cap=float(UNIFORM_BLOCK_CAP),
                energy=e.energy,proof_available=False)


def uniform_block_penalty(candidate,eta):
    """Closed form in the zero parent-mean gauge; audit scope separately."""
    c=candidate
    M=parent_masses(c.x,c.groups)
    means=np.array([np.dot(c.x[c.groups==j],c.s[c.groups==j])/M[j] for j in range(4)])
    if np.ptp(means)>1e-9:
        raise ValueError("Uniform-flow cuts require a common parent mean")
    variances=np.array([M[j]*np.dot(c.x[c.groups==j],(c.s[c.groups==j]-means[j])**2) for j in range(4)])
    delta=np.asarray(eta)-np.roll(np.asarray(eta),-1)
    C=float(parent_flows(c).mean())
    penalty=C*C*np.sum(variances/M**4*(np.roll(delta,1)+delta)**2)
    return float(penalty),variances
