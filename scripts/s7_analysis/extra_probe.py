import pathlib,json,os,sys,time
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=pathlib.Path(os.environ['KURAMOTO_REPO']);sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from dataclasses import replace
from kuramoto_template.asymptotic_search import joint_limit as jl
from kuramoto_template.asymptotic_search.families import normalize_offsets
from kuramoto_template.asymptotic_search.solvers import SolverLedger
from kuramoto_template.asymptotic_search.persistence import json_safe
OUT=pathlib.Path('outputs'); exp=json.loads((OUT/'experimentos.json').read_text(encoding='utf-8'))
best=jl.result_from_data(json.loads((pathlib.Path(os.environ['KURAMOTO_SOURCE_RUN'])/'sweep/top-limit.json').read_text()))
c=best.candidate;v=np.array([-1.,1.,0.,0.,1.,-1.,0.]);v-=c.x@v;v-=c.s*(c.x@(c.s*v));v/=np.sqrt(c.x@(v*v))
ledger=SolverLedger(dict(joint_sdp=13));start=time.perf_counter()
for gamma,eps in [(1e-6,e) for e in (-.00025,-.0005,-.00075,-.00125,-.0015,-.002)]+[(1e-7,e) for e in (-.0001,-.0003,-.0006,-.001)]:
    s=normalize_offsets(c.x,c.s+eps*v);cfg=replace(best.config,gamma0=gamma,h_guard=.003,secondary=False)
    r=jl.solve_joint_limit(c.x,s,c.groups,c.allowed_mask,cfg,backbone_cycle=c.backbone_cycle,repair_edges=best.repair_edges,fixed_p0=best.fixed_p0,ledger=ledger)
    name=f'fine-offset-{gamma:g}-{eps:g}'
    meta=dict(kind='fine_offset',gamma=gamma,epsilon=eps,direction_vector=v)
    path=OUT/'evidencia'/(name+'.json');path.write_text(json.dumps(json_safe(dict(meta=meta,result=r.data())),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    exp['calls'].append(dict(name=name,meta=meta,status=r.status,solver_status=r.solver_status,mu=r.mu0,diagnostics=json_safe(r.diagnostics),file='evidencia/'+name+'.json',dual={}))
    print(name,r.status,r.mu0)
exp['counts']['joint_sdp']+=ledger.counts['joint_sdp'];exp['elapsed']+=time.perf_counter()-start;exp['events']+=ledger.events
(OUT/'experimentos.json').write_text(json.dumps(json_safe(exp),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
print('TOTAL',exp['counts'])
