"""Reusable joint SDP for C4-only geometry experiments.

This is an optional research engine, not a change to the historical solver.
It retains its constraints and calls the same independent primal validator.
Dual envelope derivatives are search directions, not certificates of optimality.
"""

from time import perf_counter

import cvxpy as cp
import numpy as np

from .joint_limit import (JointLimitResult, build_affine_model,
                          validate_joint, _solver_diagnostics)
from .expansions import U4
from .joint_outer import repair_tree
from .models import LimitingCandidate
from .solvers import SolverLedger, solve
from .physical import mass_quotient_metrics


class ParametricJoint:
    def __init__(self, geometry, config, *, ledger=None):
        if config.secondary:
            raise ValueError("Parametric engine requires primary solves")
        self.config = config
        self.ledger = ledger if ledger is not None else SolverLedger()
        self.groups = geometry.groups.copy()
        self.mask = geometry.allowed_mask.copy()
        self.cycle = geometry.backbone_cycle
        self.tree = repair_tree(geometry)
        self.model = build_affine_model(geometry.x, geometry.s, self.groups, self.mask,
                                       backbone_cycle=self.cycle)
        model, m = self.model, geometry.m
        n, nr, d = len(model.p_pairs), len(model.r_pairs), m-1
        self.p, self.r, self.mu = cp.Variable(n), cp.Variable(nr), cp.Variable()
        self.parameters = {name: cp.Parameter(shape) for name, shape in dict(
            x=(m,), degree=(m,n), f0=(m,n), f1p=(m,n), f1r=(m,nr),
            cuts=(4,n), hp=(d*d,n), hr=(d*d,nr), restoring=(m,n), energy=(n,)).items()}
        a = self.parameters
        P = cp.reshape(np.eye(m).ravel()+model.p_map@self.p, (m,m), order="C")
        R = cp.reshape(model.r_map@self.r, (m,m), order="C")
        H = cp.reshape(a["hp"]@self.p+a["hr"]@self.r, (d,d), order="C")
        self.gamma=cp.Variable() if config.mode=='margin' else (0. if config.neutral else config.gamma0)
        gamma=self.gamma
        self.constraints = dict(
            p_lower=self.p>=0, p_upper=self.p<=1,
            r_lower=self.r>=-config.r_max, r_upper=self.r<=config.r_max,
            degree=a["x"]+a["degree"]@self.p>=self.mu,
            f0=a["f0"]@self.p==0, f1=a["f1p"]@self.p+a["f1r"]@self.r==0,
            cuts=a["cuts"]@self.p==0, full=H>>gamma*np.eye(d),
            restoring=a["x"]+a["restoring"]@self.p>=config.restoring_floor,
            energy=a["energy"]@self.p>=config.energy_floor,
            guard_lower=P+config.h_guard*R>=0,
            guard_upper=P+config.h_guard*R<=1,
            buffer_lower=self.p>=config.box_buffer,
            buffer_upper=self.p<=1-config.box_buffer)
        p_index = {pair:i for i,pair in enumerate(model.p_pairs)}
        p_neighbor = self.p[[p_index[pair] for pair in model.r_pairs]]
        self.constraints.update(
            trial_buffer_lower=p_neighbor+config.h_guard*self.r>=config.box_buffer,
            trial_buffer_upper=p_neighbor+config.h_guard*self.r<=1-config.box_buffer)
        if m>4:
            self.constraints["hard"] = H[3:,3:]>>config.hard_floor*np.eye(m-4)
        if config.mode=='margin':
            self.constraints['target']=self.mu>=config.mu_target
        for i,(aa,bb) in enumerate(zip(self.cycle,self.cycle[1:]+self.cycle[:1])):
            self.constraints[f"backbone_{i}"] = P[aa,bb]>=config.backbone_density
        for i,(aa,bb) in enumerate(self.tree):
            self.constraints[f"repair_inward_{i}"] = R[aa,bb]<=-config.inward_slack
            self.constraints[f"repair_base_{i}"] = P[aa,bb]>=config.repair_density
            self.constraints[f"repair_trial_{i}"] = P[aa,bb]+config.h_guard*R[aa,bb]>=config.repair_density
        self.problem = cp.Problem(cp.Maximize(self.mu if config.mode=='density' else self.gamma), list(self.constraints.values()))
        if not self.problem.is_dpp():
            raise ValueError("The reusable formulation must satisfy DPP")
        self.last = None

    @staticmethod
    def arrays(model):
        c = model.geometry
        degree = np.zeros_like(model.f0_map)
        for i,(a,b) in enumerate(model.p_pairs):
            degree[a,i], degree[b,i] = c.x[b], c.x[a]
        return dict(x=c.x, degree=degree, f0=model.f0_map, f1p=model.f1_p,
                    f1r=model.f1_r, cuts=model.cut_map, hp=model.h_p, hr=model.h_r,
                    restoring=model.restoring_map, energy=model.energy_map)

    def solver_arrays(self,model):
        arrays=self.arrays(model)
        d=model.geometry.m-1
        transform=np.eye(d)
        if self.config.margin_metric=='mass':
            metric,_,_=mass_quotient_metrics(model.geometry)
            transform=np.linalg.inv(np.linalg.cholesky(metric).T)
            for name in ('hp','hr'):
                source=arrays[name]
                arrays[name]=np.column_stack([(transform.T@source[:,i].reshape(d,d)@transform).ravel()
                                              for i in range(source.shape[1])])
        return arrays,transform

    @staticmethod
    def raw_duals(duals,transform):
        duals=dict(duals)
        duals['full']=transform@duals['full']@transform.T
        if 'hard' in duals:
            t=transform[3:,3:]
            duals['hard']=t@duals['hard']@t.T
        return duals

    def run(self, geometry):
        c = geometry
        if (not np.array_equal(c.groups,self.groups) or not np.array_equal(c.allowed_mask,self.mask)
                or c.backbone_cycle!=self.cycle):
            raise ValueError("A reusable problem cannot change labels, support or backbone")
        started = perf_counter()
        model = build_affine_model(c.x,c.s,c.groups,c.allowed_mask,backbone_cycle=c.backbone_cycle)
        arrays,transform = self.solver_arrays(model)
        for name,value in arrays.items():
            self.parameters[name].value = value
        outcome = solve(self.problem,"joint_sdp",self.config,self.ledger)
        record = _solver_diagnostics(self.problem,outcome,self.problem.value)
        record["engine"] = "C4 DPP, original independent validator"
        if outcome.status not in ("optimal","optimal_inaccurate") or self.p.value is None or self.r.value is None:
            self.last = None
            return JointLimitResult(None,None,"conditional_infeasible" if outcome.status=="infeasible" else "uncertain",
                outcome.status,self.config,solve_records=[record],repair_edges=self.tree,
                elapsed_seconds=perf_counter()-started,message=outcome.message)
        p, r = np.eye(c.m), np.zeros_like(c.p0)
        for (a,b),value in zip(model.p_pairs,self.p.value):
            p[a,b] = p[b,a] = value
        for (a,b),value in zip(model.r_pairs,self.r.value):
            r[a,b] = r[b,a] = value
        gamma = float(self.gamma.value) if self.config.mode=='margin' else (0. if self.config.neutral else self.config.gamma0)
        try:
            candidate = LimitingCandidate(c.x,c.s,c.groups,p,c.allowed_mask,c.backbone_cycle,c.family_id)
            result = validate_joint(candidate,r,self.config,solver_status=outcome.status,
                                    repair_edges=self.tree,reported_gamma=gamma)
        except ValueError as exc:
            result = JointLimitResult(None,r,"failed",outcome.status,self.config,
                                      repair_edges=self.tree,message=str(exc))
        result.reported_objective = float(self.mu.value if self.config.mode=='density' else self.gamma.value)
        if result.mu0 is not None and self.config.mode=='density':
            gap = abs(result.mu0-result.reported_objective)
            result.diagnostics["density_objective_gap"] = gap
            if gap>self.config.residual_tolerance:
                result.status,result.message = "uncertain","Primary density objective disagrees with measured degree"
        result.solve_records = [record]
        result.elapsed_seconds = perf_counter()-started
        self.last = dict(arrays=arrays,geometry=c,p=self.p.value.copy(),r=self.r.value.copy(),gamma=gamma,
                         duals=self.raw_duals({name:np.array(con.dual_value) for name,con in self.constraints.items()
                                if con.dual_value is not None},transform))
        return result

    def envelope(self, geometry):
        """Lagrangian at fixed last primal/dual values, for cheap differences.

        This does not run a conic solver and does not assert differentiability.
        Equality multipliers have the opposite sign in a maximization envelope.
        """
        if self.last is None:
            raise ValueError("Need a completed primal/dual solve")
        model = build_affine_model(geometry.x,geometry.s,geometry.groups,geometry.allowed_mask,
                                   backbone_cycle=geometry.backbone_cycle)
        a = self.arrays(model)
        p,r,dual = self.last["p"],self.last["r"],self.last["duals"]
        H = (a["hp"]@p+a["hr"]@r).reshape(geometry.m-1,geometry.m-1)
        value = dual["degree"]@(a["x"]+a["degree"]@p)
        value -= dual["f0"]@(a["f0"]@p)
        value -= dual["f1"]@(a["f1p"]@p+a["f1r"]@r)
        value -= dual["cuts"]@(a["cuts"]@p)
        value += np.sum(dual["full"]*H)
        value += dual["restoring"]@(a["x"]+a["restoring"]@p)
        value += float(dual["energy"])*(a["energy"]@p)
        if "hard" in dual:
            value += np.sum(dual["hard"]*H[3:,3:])
        if self.config.margin_metric=='mass':
            metric,hard_metric,_=mass_quotient_metrics(geometry)
            value-=self.last['gamma']*np.sum(dual['full']*metric)
            value-=self.config.hard_floor*np.sum(dual.get('hard',0.)*hard_metric)
        return float(value)

    def gradient(self, geometry_function, parameters, *, step=1e-5):
        dx,ds=self.physical_gradient()
        parameters = np.asarray(parameters,dtype=float)
        gradient = np.empty_like(parameters)
        for i in range(len(parameters)):
            left,right = parameters.copy(),parameters.copy()
            left[i] -= step
            right[i] += step
            a,b=geometry_function(left),geometry_function(right)
            gradient[i]=(dx@(b.x-a.x)+ds@(b.s-a.s))/(2*step)
        return gradient

    def physical_gradient(self):
        """Analytic envelope partials in x and s, before outer normalization."""
        if self.last is None:
            raise ValueError("Need a completed primal/dual solve")
        c=self.last['geometry']
        p=(np.eye(c.m).ravel()+self.model.p_map@self.last['p']).reshape(c.m,c.m)
        r=(self.model.r_map@self.last['r']).reshape(c.m,c.m)
        dual=self.last['duals']
        same=c.groups[:,None]==c.groups[None,:]
        gx=p.T@dual['degree']+(p*same).T@dual['restoring']
        gs=np.zeros(c.m)
        l0,l1,kdual=dual['f0'],dual['f1'],dual['cuts']
        Y=dual['full']
        Yhard=Y[3:,3:]+dual.get('hard',0.)
        Z=self.model.z
        energy_dual=float(dual['energy'])
        for a,b in self.model.p_pairs:
            mass=c.x[a]*c.x[b]
            delta=c.s[b]-c.s[a]
            if c.groups[a]==c.groups[b]:
                z=Z[a]-Z[b]
                hcoef=p[a,b]*(z@Yhard@z)
                ecoef=energy_dual*p[a,b]*delta**2
                gx[a]+=c.x[b]*(hcoef+ecoef)+l1[b]*p[a,b]*delta
                gx[b]+=c.x[a]*(hcoef+ecoef)-l1[a]*p[a,b]*delta
                first=p[a,b]*(l1[a]*c.x[b]-l1[b]*c.x[a])
                force=2*energy_dual*mass*p[a,b]*delta
                gs[a]+=first-force
                gs[b]+=-first+force
            else:
                aa,bb=(a,b) if (c.groups[b]-c.groups[a])%4==1 else (b,a)
                delta=c.s[bb]-c.s[aa]
                z=Z[aa]-Z[bb]
                u=U4[c.groups[aa]]-U4[c.groups[bb]]
                hcoef=-2*p[a,b]*(u@Y[:3,3:]@z)-r[a,b]*(u@Y[:3,:3]@u)
                kcoef=-kdual[c.groups[aa]]*p[a,b]
                coefficient=hcoef+kcoef
                gx[aa]+=l0[bb]*p[a,b]+l1[bb]*r[a,b]+c.x[bb]*delta*coefficient
                gx[bb]+=-l0[aa]*p[a,b]-l1[aa]*r[a,b]+c.x[aa]*delta*coefficient
                gs[aa]-=mass*coefficient
                gs[bb]+=mass*coefficient
        if self.config.margin_metric=='mass':
            parent=np.array([c.x[c.groups==j].sum() for j in range(4)])
            projected=U4.T@parent
            for a in range(c.m):
                j=c.groups[a]
                soft=(np.outer(U4[j],U4[j])-np.outer(U4[j],projected)-np.outer(projected,U4[j]))
                idx=c.groups==j
                vz=c.x[idx]@Z[idx]
                hard=(np.outer(Z[a],Z[a])-(np.outer(Z[a],vz)+np.outer(vz,Z[a]))/parent[j]
                      +np.outer(vz,vz)/parent[j]**2)
                gx[a]-=self.last['gamma']*(np.sum(Y[:3,:3]*soft)+np.sum(Y[3:,3:]*hard))
                gx[a]-=self.config.hard_floor*np.sum(dual.get('hard',0.)*hard)
        return gx,gs
