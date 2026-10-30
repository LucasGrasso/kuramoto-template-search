"""Finite-h convex probes whose phases remain C4 perturbations.

This optional research control is independent of the historical finite-lift
schedule and never enables its disabled full-template fallback.
"""

from dataclasses import asdict
from time import perf_counter

import cvxpy as cp
import numpy as np

from .expansions import U4,internal_basis
from .joint_limit import _solver_diagnostics
from .joint_outer import repair_tree
from .solvers import solve,SolverLedger
from .families import winding
from kuramoto_template.sdp_optimizer.template_metrics import template_metrics


class FiniteC4Probe:
    def __init__(self,geometry,config,*,allow_opposite=False,ledger=None):
        self.config=config
        self.ledger=ledger if ledger is not None else SolverLedger()
        self.groups=geometry.groups.copy()
        self.cycle=geometry.backbone_cycle
        self.m=geometry.m
        self.mask=np.ones((self.m,self.m),dtype=bool) if allow_opposite else geometry.allowed_mask.copy()
        self.pairs=tuple((a,b) for a in range(self.m) for b in range(a+1,self.m) if self.mask[a,b])
        n=len(self.pairs);d=self.m-1
        self.p=cp.Variable(n)
        self.mu=cp.Variable()
        self.x=cp.Parameter(self.m)
        self.degree=cp.Parameter((self.m,n))
        self.force=cp.Parameter((self.m,n))
        self.restoring=cp.Parameter((self.m,n))
        self.hessian=cp.Parameter((d*d,n))
        H=cp.reshape(self.hessian@self.p,(d,d),order='C')
        gamma=0. if config.neutral else config.gamma0
        cons=[self.p>=config.box_buffer,self.p<=1-config.box_buffer,
              self.x+self.degree@self.p>=self.mu,self.force@self.p==0,
              self.x+self.restoring@self.p>=config.restoring_floor,H>>gamma*np.eye(d)]
        idx={pair:i for i,pair in enumerate(self.pairs)}
        for a,b in zip(self.cycle,self.cycle[1:]+self.cycle[:1]):
            cons.append(self.p[idx[tuple(sorted((a,b)))]]>=config.backbone_density)
        self.problem=cp.Problem(cp.Maximize(self.mu),cons)
        self.allow_opposite=allow_opposite
        if not self.problem.is_dpp():raise ValueError('Expected finite DPP formulation')

    def run(self,c,h):
        if not np.array_equal(c.groups,self.groups) or c.backbone_cycle!=self.cycle or h<=0:
            raise ValueError('Invalid finite C4 geometry')
        started=perf_counter()
        m,d,n=self.m,self.m-1,len(self.pairs)
        delta=(c.groups[None,:]-c.groups[:,None])%4
        ds=h*(c.s[None,:]-c.s[:,None])
        sine=np.choose(delta,[np.sin(ds),np.cos(ds),-np.sin(ds),-np.cos(ds)])
        cosine=np.choose(delta,[np.cos(ds),-np.sin(ds),-np.cos(ds),np.sin(ds)])
        S=np.eye(4)[c.groups]@U4
        Z=internal_basis(c)
        T=np.column_stack((S,h*Z))
        D,F,C=np.zeros((m,n)),np.zeros((m,n)),np.zeros((m,n))
        H=np.zeros((d*d,n))
        for i,(a,b) in enumerate(self.pairs):
            D[a,i],D[b,i]=c.x[b],c.x[a]
            F[a,i],F[b,i]=c.x[b]*sine[a,b],-c.x[a]*sine[a,b]
            C[a,i],C[b,i]=c.x[b]*cosine[a,b],c.x[a]*cosine[a,b]
            v=(T[a]-T[b])/h
            H[:,i]=(c.x[a]*c.x[b]*cosine[a,b]*np.outer(v,v)).ravel()
        self.x.value=c.x;self.degree.value=D;self.force.value=F
        self.restoring.value=C;self.hessian.value=H
        outcome=solve(self.problem,'c4_finite_sdp',self.config,self.ledger)
        record=_solver_diagnostics(self.problem,outcome,self.problem.value)
        result=dict(candidate=asdict(c),h=h,config=asdict(self.config),allow_opposite=self.allow_opposite,
                    status='uncertain',solver_status=outcome.status,P=None,mu=None,solve_records=[record],
                    proof_available=False,graph_realization_certified=False,elapsed_seconds=perf_counter()-started)
        if outcome.status not in ('optimal','optimal_inaccurate') or self.p.value is None:return result
        P=np.eye(m)
        for (a,b),value in zip(self.pairs,self.p.value):P[a,b]=P[b,a]=value
        phi=c.groups*np.pi/2+h*c.s
        metrics=template_metrics(c.x,phi,P)
        reduced=(H@self.p.value).reshape(d,d)
        scaled_gap=float(np.linalg.eigvalsh((reduced+reduced.T)/2).min())
        gamma=0. if self.config.neutral else self.config.gamma0
        box_violation=max(0.,-P.min(),P.max()-1)
        violation=max(metrics.max_force_error,box_violation,
            gamma-scaled_gap,self.config.restoring_floor-metrics.min_restoring,
            abs(metrics.mu-float(self.mu.value)))
        valid=violation<=self.config.residual_tolerance and winding(c,P,h,min_density=self.config.backbone_density)
        if valid and metrics.min_cluster_eigenvalue>0 and metrics.min_restoring>0 and outcome.status=='optimal':
            result['status']='finite_numeric_positive'
        result.update(P=P,phi=phi,mu=metrics.mu,tree=repair_tree(c),
            diagnostics=dict(force_inf=metrics.max_force_error,restoring=metrics.min_restoring,
                unscaled_gap=metrics.min_cluster_eigenvalue,scaled_gap=scaled_gap,
                box_violation=float(box_violation),independent_primal_check_inf=float(violation),
                winding_valid=winding(c,P,h,min_density=self.config.backbone_density)),
            elapsed_seconds=perf_counter()-started)
        return result
