"""Limiting SDP with free cut stiffness, followed by checked R recovery.

The sum(k)=internal-energy identity is necessary. Recovery and the original
independent validator are mandatory: the relaxation alone is not a witness.
Only after successful exact rational/interval verification is a bound certified.
"""

from dataclasses import asdict,replace
from time import perf_counter

import cvxpy as cp
import numpy as np

from .c4_parametric import ParametricJoint
from .expansions import U4
from .joint_limit import JointLimitResult,build_affine_model,validate_joint,_solver_diagnostics
from .models import LimitingCandidate
from .solvers import solve


class RecoveredJoint(ParametricJoint):
    def __init__(self,geometry,config,*,ledger=None,elastic_penalty=0.):
        super().__init__(geometry,config,ledger=ledger)
        d=geometry.m-1
        self.k=cp.Variable(4)
        self.k_h=np.zeros((d*d,4))
        for j in range(4):
            u=U4[j]-U4[(j+1)%4]
            block=np.zeros((d,d));block[:3,:3]=np.outer(u,u)
            self.k_h[:,j]=block.ravel()
        self.k_parameter=cp.Parameter((d*d,4))
        H=cp.reshape(self.parameters['hp']@self.p+self.k_parameter@self.k,(d,d),order='C')
        keep={'p_lower','p_upper','degree','f0','cuts','restoring','energy','buffer_lower','buffer_upper','target'}
        self.constraints={name:value for name,value in self.constraints.items()
                          if name in keep or name.startswith('backbone_')}
        gamma=self.gamma
        self.constraints['full']=H>>gamma*np.eye(d)
        self.constraints['hard']=H[3:,3:]>>config.hard_floor*np.eye(geometry.m-4)
        self.constraints['budget']=cp.sum(self.k)-self.parameters['energy']@self.p==0
        self.elastic_penalty=float(elastic_penalty)
        self.target_slack=None
        objective=self.mu if config.mode=='density' else self.gamma
        if config.mode=='margin' and self.elastic_penalty>0:
            self.target_slack=cp.Variable(nonneg=True)
            self.constraints['target']=self.mu+self.target_slack>=config.mu_target
            objective=self.gamma-self.elastic_penalty*self.target_slack
        self.problem=cp.Problem(cp.Maximize(objective),list(self.constraints.values()))
        if not self.problem.is_dpp():
            raise ValueError('Expected a DPP relaxation')

    def run(self,geometry):
        c=geometry
        if (not np.array_equal(c.groups,self.groups) or not np.array_equal(c.allowed_mask,self.mask)
                or c.backbone_cycle!=self.cycle):
            raise ValueError('Recovery engine cannot change labels, support or backbone')
        started=perf_counter()
        model=build_affine_model(c.x,c.s,c.groups,c.allowed_mask,backbone_cycle=c.backbone_cycle)
        arrays,transform=self.solver_arrays(model)
        for name,value in arrays.items():self.parameters[name].value=value
        self.k_parameter.value=np.column_stack([(transform.T@self.k_h[:,i].reshape(c.m-1,c.m-1)@transform).ravel()
                                                for i in range(4)])
        outcome=solve(self.problem,'joint_sdp',self.config,self.ledger)
        record=_solver_diagnostics(self.problem,outcome,self.problem.value)
        record['engine']='free cut stiffness, checked R recovery'
        record['optimization_config']=asdict(self.config)
        if outcome.status not in ('optimal','optimal_inaccurate') or self.p.value is None or self.k.value is None:
            self.last=None
            return JointLimitResult(None,None,'conditional_infeasible' if outcome.status=='infeasible' else 'uncertain',
                outcome.status,self.config,solve_records=[record],elapsed_seconds=perf_counter()-started)
        pv=self.p.value.copy()
        kv=self.k.value.copy()
        p=np.eye(c.m)
        for (a,b),value in zip(model.p_pairs,pv):p[a,b]=p[b,a]=value
        kmap=np.zeros((4,len(model.r_pairs)))
        for j,(a,b) in enumerate(model.r_pairs):
            aa,bb=(a,b) if (c.groups[b]-c.groups[a])%4==1 else (b,a)
            kmap[c.groups[aa],j]=-c.x[a]*c.x[b]*(c.s[bb]-c.s[aa])
        A=np.vstack((model.f1_r,kmap))
        target=np.r_[-model.f1_p@pv,kv]

        def recover():
            scale=1/np.maximum(np.max(np.abs(A),axis=1),1e-20)
            solution,residual,rank,singular=np.linalg.lstsq(scale[:,None]*A,scale*target,rcond=1e-13)
            return dict(solution=solution,rank=int(rank),residual=float(np.max(np.abs(A@solution-target))),
                        singular_values=singular)
        recovery=self.ledger.invoke('correction_linear',recover)
        record['recovery']=recovery
        r=np.zeros_like(p)
        for (a,b),value in zip(model.r_pairs,recovery['solution']):r[a,b]=r[b,a]=value
        buffer=self.config.box_buffer/4
        upper=self.config.h_guard
        for a,b in model.r_pairs:
            if r[a,b]>0:upper=min(upper,(1-buffer-p[a,b])/r[a,b])
            elif r[a,b]<0:upper=min(upper,(p[a,b]-buffer)/(-r[a,b]))
        guard=max(1e-15,min(self.config.h_guard,.2*upper,.1/(np.max(np.abs(c.s))+1)))
        cfg=replace(self.config,h_guard=guard,box_buffer=buffer,
                    r_max=max(self.config.r_max,1.01*float(np.max(np.abs(r)))+1))
        shortfall=max(0.,float(self.target_slack.value)) if self.target_slack is not None else 0.
        if self.target_slack is not None:
            cfg=replace(cfg,mu_target=max(1e-9,self.config.mu_target-shortfall))
            record['elastic_target']=dict(requested=self.config.mu_target,shortfall=shortfall,
                                         penalty=self.elastic_penalty,merit=float(self.problem.value))
        try:
            candidate=LimitingCandidate(c.x,c.s,c.groups,p,c.allowed_mask,c.backbone_cycle,c.family_id)
            result=validate_joint(candidate,r,cfg,solver_status=outcome.status,repair_edges=(),
                reported_gamma=float(self.gamma.value) if cfg.mode=='margin' else (0. if cfg.neutral else cfg.gamma0))
        except ValueError as exc:
            result=JointLimitResult(None,r,'failed',outcome.status,cfg,message=str(exc))
        result.reported_objective=float(self.mu.value if cfg.mode=='density' else self.gamma.value)
        if self.target_slack is not None:
            result.diagnostics.update(search_merit=float(self.problem.value),target_shortfall=shortfall,
                                      requested_target=self.config.mu_target)
        if result.mu0 is not None and cfg.mode=='density':
            gap=abs(result.mu0-result.reported_objective)
            result.diagnostics['density_objective_gap']=gap
            if gap>cfg.residual_tolerance:
                result.status,result.message='uncertain','Density objective disagrees with degree'
        result.solve_records=[record]
        result.elapsed_seconds=perf_counter()-started
        duals=self.raw_duals({name:np.array(con.dual_value) for name,con in self.constraints.items() if con.dual_value is not None},transform)
        duals['f1']=np.zeros(c.m)
        duals['energy']=duals['energy']+duals['budget']
        self.last=dict(arrays=arrays,geometry=c,p=pv,r=np.zeros(len(model.r_pairs)),duals=duals,
                       gamma=float(self.gamma.value) if cfg.mode=='margin' else (0. if cfg.neutral else cfg.gamma0))
        return result
