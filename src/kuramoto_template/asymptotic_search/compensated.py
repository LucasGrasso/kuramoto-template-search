"""Linked-edge proposals in ker(F0,K), never an acceptance evaluator."""

from dataclasses import dataclass,field,replace
from itertools import combinations

import cvxpy as cp
import numpy as np
from scipy.linalg import eigh

from .expansions import DIPOLES,coefficients,internal_solve
from .families import parent_mask
from .joint_limit import build_affine_model
from .physical import mass_quotient_metrics,physical_spectrum
from .solvers import solve


def dipole_slacks(candidate):
    e=coefficients(candidate)
    z,_,residual=internal_solve(candidate,e.lin,e.g)
    return np.array([e.energy-v@e.g.T@z@v for v in DIPOLES]),residual


def _cycles(c,model,radius):
    indices={pair:i for i,pair in enumerate(model.p_pairs)}
    values=np.array([c.p0[a,b] for a,b in model.p_pairs])
    candidates=[]
    for j in range(4):
        left=np.flatnonzero(c.groups==j); right=np.flatnonzero(c.groups==(j+1)%4)
        for a,b in combinations(left,2):
            for d,e in combinations(right,2):
                direction=np.zeros(len(values))
                for aa,bb,sign in ((a,d,1),(b,e,1),(a,e,-1),(b,d,-1)):
                    direction[indices[tuple(sorted((int(aa),int(bb))))]]=sign/(c.x[aa]*c.x[bb])
                direction/=np.max(np.abs(direction))
                for sign in (1,-1):
                    v=sign*direction
                    bound=min(radius,np.min((1-values[v>0])/v[v>0]),np.min(-values[v<0]/v[v<0]))
                    if bound>1e-10:
                        candidates.append(v*(.8*bound))
    return candidates


def _derivatives(c,model,r):
    e=coefficients(c,r)
    rows=[]
    for eta in DIPOLES:
        z=internal_solve(c,e.lin,e.g@eta)[0]
        row=[]
        for a,b in model.p_pairs:
            mass=c.x[a]*c.x[b]
            if c.groups[a]==c.groups[b]:
                row.append(mass*((c.s[a]-c.s[b])**2+(z[a]-z[b])**2))
            else:
                aa,bb=(a,b) if (c.groups[b]-c.groups[a])%4==1 else (b,a)
                j=c.groups[aa]
                dg=mass*(c.s[bb]-c.s[aa])*(eta[j]-eta[(j+1)%4])
                row.append(-2*(z[aa]-z[bb])*dg)
        rows.append(row)
    return np.asarray(rows)


def _dual_price(result,model):
    selected=next((r for r in result.solve_records if r.get("selected")),{})
    degree=selected.get("dual_degree")
    hessian=selected.get("dual_full_hessian")
    if degree is None or hessian is None:
        degrees=result.candidate.p0@result.candidate.x
        active=degrees<=degrees.min()+1e-7
        degree=active/active.sum()
        _,vectors=np.linalg.eigh(result.hlim)
        hessian=np.outer(vectors[:,0],vectors[:,0])
        source="active-constraint surrogate, not KKT-certified"
    else:
        degree,hessian=np.asarray(degree),np.asarray(hessian)
        source="saved selected-solve multipliers; local price only"
    price=np.array([degree[a]*result.candidate.x[b]+degree[b]*result.candidate.x[a] for a,b in model.p_pairs])
    price+=np.asarray(hessian).ravel()@model.h_p
    price/=max(1.,np.linalg.norm(price))
    return price,source


@dataclass
class CompensatedMove:
    candidate: object | None
    delta: np.ndarray
    status: str
    method: str
    policy: str
    solver_status: str
    diagnostics: dict=field(default_factory=dict)
    selected_edges: tuple=()


def compensated_move(result,rng,*,policy="balanced",method="projection",radius=.15,ledger=None):
    if policy not in ("balanced","random") or method not in ("projection","qp"):
        raise ValueError("Unknown compensation policy or method")
    c=result.candidate
    if c is None or result.r is None:
        raise ValueError("A saved candidate/correction is required")
    model=build_affine_model(c.x,c.s,c.groups,parent_mask(c.groups),backbone_cycle=c.backbone_cycle)
    p=np.array([c.p0[a,b] for a,b in model.p_pairs])
    A=np.vstack((model.f0_map,model.cut_map))
    cycles=_cycles(c,model,radius)
    zero=np.zeros_like(c.p0)
    if not cycles:
        return CompensatedMove(None,zero,"no_feasible_direction",method,policy,"not_run")
    before,range_residual=dipole_slacks(c)
    e=coefficients(c,result.r)
    scale=max(e.energy,1e-8)
    deriv=_derivatives(c,model,result.r)
    price,price_source=_dual_price(result,model)
    metric,_,_=mass_quotient_metrics(c)
    _,vectors=eigh(result.hlim,metric)
    modes=vectors[:,:min(3,c.m-1)]
    h_rows=np.array([
        [v@model.h_p[:,i].reshape(c.m-1,c.m-1)@v for i in range(len(p))] for v in modes.T])

    def rank(v):
        return float(.05*np.min(deriv@v)/scale+.02*np.min(h_rows@v)/scale+.001*price@v)

    ordered=sorted(cycles,key=rank,reverse=True) if policy=="balanced" else [cycles[i] for i in rng.permutation(len(cycles))]
    selected=np.flatnonzero(np.abs(ordered[0])>0).tolist()
    for v in ordered[1:]:
        union=sorted(set(selected)|set(np.flatnonzero(np.abs(v)>0)))
        if len(union)<=8 and len(union)>len(selected):
            selected=union
            break
    solver_status="analytic"
    if method=="projection":
        def project():
            return ordered[0].copy()
        values=ledger.invoke("compensation_projection",project) if ledger is not None else project()
    else:
        v=cp.Variable(len(selected))
        constraints=[A[:,selected]@v==0,p[selected]+v>=0,p[selected]+v<=1,v>=-radius,v<=radius]
        if policy=="random":
            objective=rng.normal(size=len(selected))@v-.01*cp.sum_squares(v)/radius**2
        else:
            degree_gain,slack_gain,mode_gain=cp.Variable(),cp.Variable(),cp.Variable()
            dm=np.zeros((c.m,len(selected)))
            for i,index in enumerate(selected):
                a,b=model.p_pairs[index]
                dm[a,i],dm[b,i]=c.x[b],c.x[a]
            constraints += [c.p0@c.x+dm@v>=result.mu0+degree_gain,
                            deriv[:,selected]@v>=slack_gain,h_rows[:,selected]@v>=mode_gain]
            objective=degree_gain+.05*slack_gain/scale+.02*mode_gain/scale+.001*price[selected]@v-.01*cp.sum_squares(v)/radius**2
        problem=cp.Problem(cp.Maximize(objective),constraints)
        outcome=solve(problem,"compensation_qp",result.config,ledger)
        solver_status=outcome.status
        if outcome.status!="optimal" or v.value is None:
            return CompensatedMove(None,zero,"uncertain",method,policy,solver_status)
        values=np.zeros_like(p); values[selected]=v.value
    delta=np.zeros_like(c.p0)
    mask=c.allowed_mask.copy()
    for index in selected:
        a,b=model.p_pairs[index]
        delta[a,b]=delta[b,a]=values[index]
        mask[a,b]=mask[b,a]=True
    predicted=c.p0+delta
    violation=float(max(0.,-predicted.min(),predicted.max()-1))
    residual=float(np.max(np.abs(A@values)))
    if max(violation,residual)>result.config.residual_tolerance:
        return CompensatedMove(None,delta,"rejected_residual",method,policy,solver_status,
                               dict(box_violation=violation,equation_residual=residual))
    candidate=replace(c,p0=predicted,allowed_mask=mask,family_id=c.family_id+"/compensated")
    after,after_residual=dipole_slacks(candidate)
    H=result.hlim+(model.h_p@values).reshape(c.m-1,c.m-1)
    diagnostics=dict(force0_delta_inf=float(np.max(np.abs(model.f0_map@values))),
        cut_delta_inf=float(np.max(np.abs(model.cut_map@values))),box_violation=violation,
        range_residual=max(range_residual,after_residual),dipoles_before=before,dipoles_after=after,
        predicted_degree_gain=float((predicted@c.x).min()-(c.p0@c.x).min()),
        predicted_full_margin_gain=float(physical_spectrum(c,H).min()-physical_spectrum(c,result.hlim).min()),
        weakest_three_rayleigh_changes=h_rows@values,approximate_price=float(price@values),price_source=price_source,
        changed_edges=int(np.count_nonzero(np.abs(values)>1e-12)),opened_edges=int(np.count_nonzero(mask & ~c.allowed_mask)//2),
        requires_joint_reoptimization=True,accepted_as_witness=False)
    return CompensatedMove(candidate,delta,"proposal",method,policy,solver_status,diagnostics,
                           tuple(model.p_pairs[i] for i in selected))
