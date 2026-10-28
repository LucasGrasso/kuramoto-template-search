import os,sys,pathlib,json,time,itertools,collections
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=pathlib.Path(os.environ['KURAMOTO_REPO'])
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from dataclasses import replace
from kuramoto_template.asymptotic_search import joint_limit as jl
from kuramoto_template.asymptotic_search.solvers import SolverLedger
from kuramoto_template.asymptotic_search.families import normalize_offsets,parent_mask
from kuramoto_template.asymptotic_search.joint_outer import repair_tree
from kuramoto_template.asymptotic_search.models import LimitingCandidate
from kuramoto_template.asymptotic_search.weak_modes import high_precision_limit_audit
from kuramoto_template.asymptotic_search.persistence import json_safe
RUN=pathlib.Path(os.environ['KURAMOTO_SOURCE_RUN']); OUT=pathlib.Path('outputs'); EV=OUT/'evidencia'; EV.mkdir(exist_ok=True)
analysis=json.loads((OUT/'analisis-datos.json').read_text(encoding='utf-8'))
selected=analysis['selected']; originals={i:jl.result_from_data(json.loads((RUN/f'sweep/joint-results/{i:06d}.json').read_text())) for i in selected}
reference=analysis['reference']; started=time.perf_counter(); limit_seconds=180
ledger=SolverLedger(dict(joint_sdp=120,high_precision_audit=4))
calls=[]; captured=[]; native=jl.solve
def observed(problem,kind,cfg,ledger_):
    out=native(problem,kind,cfg,ledger_)
    captured.append(problem)
    return out
jl.solve=observed
def dump(path,data):path.write_text(json.dumps(json_safe(data),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def dual_summary(problem,result):
    c=result.candidate
    labels=['p_lower','p_upper','r_lower','r_upper','degree','F0','F1','K','full_psd','restoring','energy','guard_lower','guard_upper',
            'buffer_lower','buffer_upper','neighbor_buffer_lower','neighbor_buffer_upper','hard_psd']
    labels+=['backbone_'+str(k) for k in range(4)]
    out={}
    for k,con in enumerate(problem.constraints):
        value=con.dual_value
        if value is None: continue
        value=np.asarray(value)
        out[labels[k] if k<len(labels) else 'restriction_'+str(k)]=dict(value=value,norm=float(np.linalg.norm(value)))
    if c is not None:
        H=result.hlim-result.config.gamma0*np.eye(c.m-1); Y=np.asarray(problem.constraints[8].dual_value)
        if Y.shape==H.shape:
            out['full_psd'].update(eigenvalues=np.linalg.eigvalsh(Y),complementarity_trace=float(np.sum(Y*H)),gamma_derivative=-float(np.trace(Y)))
    return out
def run(name,c,cfg,*,fixed=None,tree=None,meta=None):
    if time.perf_counter()-started>limit_seconds:raise RuntimeError('Wall budget exceeded')
    captured.clear(); cfg=replace(cfg,secondary=False)
    if tree is None:tree=repair_tree(c)
    r=jl.solve_joint_limit(c.x,c.s,c.groups,c.allowed_mask,cfg,backbone_cycle=c.backbone_cycle,repair_edges=tree,
                           fixed_p0=fixed,family_id=c.family_id,ledger=ledger)
    dual=dual_summary(captured[-1],r) if captured and r.candidate is not None else {}
    item=dict(name=name,meta=meta or {},status=r.status,solver_status=r.solver_status,mu=r.mu0,diagnostics=r.diagnostics,
              file='evidencia/'+name+'.json',dual=dual)
    calls.append(item); dump(EV/(name+'.json'),dict(meta=meta,result=r.data(),dual=dual))
    dump(OUT/'experimentos.json',dict(budget=dict(joint_sdp=120,high_precision_audit=4,wall_seconds=limit_seconds),counts=ledger.counts,elapsed=time.perf_counter()-started,calls=calls))
    print(name,r.status,r.mu0,flush=True)
    return r

for i in selected:
    r=originals[i]
    replay=run(f'replay-{i:06d}',r.candidate,r.config,fixed=r.fixed_p0,tree=r.repair_edges,meta=dict(kind='replay',original_id=i,original_mu=r.mu0))
    if i in (600,1,782,912) and replay.candidate is not None:
        audit=ledger.invoke('high_precision_audit',lambda:high_precision_limit_audit(replay))
        dump(EV/f'audit-{i:06d}.json',audit)

best=originals[600]; c=best.candidate
for gamma,guard in itertools.product((1e-7,1e-6,1e-5),(.0003,.003,.01)):
    run(f'floor-{gamma:g}-guard-{guard:g}',c,replace(best.config,gamma0=gamma,h_guard=guard),fixed=best.fixed_p0,tree=best.repair_edges,
        meta=dict(kind='floor_guard',gamma=gamma,guard=guard))

# Central paired perturbations, normalized mass-logit directions and offset tangent directions.
g=c.groups; directions={
 'mass_opposite_02':('mass',np.where(g==0,1.,np.where(g==2,-1.,0.))),
 'mass_opposite_13':('mass',np.where(g==1,1.,np.where(g==3,-1.,0.))),
 'half_mass_02':('mass',np.array([1.,-1.,0.,0.,-1.,1.,0.])),
 'half_mass_1':('mass',np.array([0.,0.,1.,-1.,0.,0.,0.])),
 'offset_w':('offset',np.where(g==1,c.s,0.)),
 'offset_parent_02':('offset',np.where(g==0,1.,np.where(g==2,-1.,0.))),
 'offset_half_02':('offset',np.array([-1.,1.,0.,0.,1.,-1.,0.]))}
for name,(kind,v) in directions.items():
    v=v-c.x@v
    if kind=='offset':v=v-c.s*(c.x@(c.s*v))
    v=v/np.sqrt(c.x@(v*v))
    for epsilon in (-.005,-.001,.001,.005):
        x=c.x*np.exp(epsilon*v) if kind=='mass' else c.x.copy();x=x/x.sum()
        s=normalize_offsets(x,c.s+(epsilon*v if kind=='offset' else 0.))
        child=replace(c,x=x,s=s)
        run(f'local-{name}-{epsilon:+g}',child,best.config,fixed=best.fixed_p0,tree=best.repair_edges,
            meta=dict(kind='local',direction=name,parameterization=kind,epsilon=epsilon,direction_vector=v))

# Shrink extra splits continuously to the nearest S7 offset, then genuinely fuse the cells.
rx=np.array(reference['x']);rs=np.array(reference['s']);rg=np.array([0,0,1,1,2,2,3])
for i in (1,62,782,304):
    original=originals[i]; cc=original.candidate
    # All four selected complete seeds have the original orientation.
    assignment=np.array([min(np.flatnonzero(rg==cc.groups[a]),key=lambda b:abs(cc.s[a]-rs[b])) for a in range(cc.m)])
    target=rs[assignment]
    for alpha in (0.,.25,.5,.75,1.):
        child=replace(cc,s=normalize_offsets(cc.x,(1-alpha)*target+alpha*cc.s))
        run(f'collapse-{i:06d}-{alpha:g}',child,original.config,fixed=original.fixed_p0,tree=original.repair_edges,
            meta=dict(kind='collapse',original_id=i,alpha=alpha,assignment=assignment))
    sets=[np.flatnonzero(assignment==k) for k in range(7)]
    assert all(len(a) for a in sets)
    x=np.array([cc.x[a].sum() for a in sets]);s=np.array([cc.x[a]@cc.s[a]/x[k] for k,a in enumerate(sets)])
    child=LimitingCandidate(x,normalize_offsets(x,s),rg,np.eye(7),parent_mask(rg),(0,2,4,6),'fused:'+str(i))
    fixed=np.where(child.allowed_mask,np.nan,0.);np.fill_diagonal(fixed,1.)
    run(f'fused-{i:06d}',child,original.config,fixed=fixed,meta=dict(kind='fused',original_id=i,assignment=assignment))

# Reopen all non-parent prohibitions at four leading non-S7 representatives.
for i in (994,1039,912,1012):
    original=originals[i];cc=original.candidate;child=replace(cc,allowed_mask=parent_mask(cc.groups))
    fixed=np.where(child.allowed_mask,np.nan,0.);np.fill_diagonal(fixed,1.)
    run(f'reopen-{i:06d}',child,original.config,fixed=fixed,meta=dict(kind='reopen',original_id=i,original_mu=original.mu0))

dump(OUT/'experimentos.json',dict(budget=dict(joint_sdp=120,high_precision_audit=4,wall_seconds=limit_seconds),counts=ledger.counts,
     elapsed=time.perf_counter()-started,calls=calls,events=ledger.events,proof_available=False))
print('COUNTS',ledger.counts,'SECONDS',time.perf_counter()-started)
