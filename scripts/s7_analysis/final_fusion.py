import os,sys,pathlib,json,time
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=pathlib.Path(os.environ['KURAMOTO_REPO']);sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from dataclasses import replace
from kuramoto_template.asymptotic_search import joint_limit as jl
from kuramoto_template.asymptotic_search.families import normalize_offsets,parent_mask
from kuramoto_template.asymptotic_search.joint_outer import repair_tree
from kuramoto_template.asymptotic_search.models import LimitingCandidate
from kuramoto_template.asymptotic_search.solvers import SolverLedger
from kuramoto_template.asymptotic_search.persistence import json_safe
OUT=pathlib.Path('outputs');data=json.loads((OUT/'analisis-datos.json').read_text(encoding='utf-8'));exp=json.loads((OUT/'experimentos.json').read_text(encoding='utf-8'))
ledger=SolverLedger(dict(joint_sdp=3));rg=np.array([0,0,1,1,2,2,3]);rs=np.array(data['reference']['s'])
for ident,kind in [(991,'fused'),(991,'collapse'),(153,'fused')]:
 original=jl.result_from_data(json.loads((pathlib.Path(os.environ['KURAMOTO_SOURCE_RUN'])/f'sweep/joint-results/{ident:06d}.json').read_text()));cc=original.candidate
 assignment=np.array([min(np.flatnonzero(rg==cc.groups[a]),key=lambda b:abs(cc.s[a]-rs[b])) for a in range(cc.m)])
 sets=[np.flatnonzero(assignment==k) for k in range(7)];assert all(len(a) for a in sets)
 if kind=='fused':
  x=np.array([cc.x[a].sum() for a in sets]);s=np.array([cc.x[a]@cc.s[a]/x[k] for k,a in enumerate(sets)])
  child=LimitingCandidate(x,normalize_offsets(x,s),rg,np.eye(7),parent_mask(rg),(0,2,4,6),'fused:'+str(ident))
  fixed=np.where(child.allowed_mask,np.nan,0.);np.fill_diagonal(fixed,1.);tree=repair_tree(child)
 else:
  child=replace(cc,s=normalize_offsets(cc.x,rs[assignment]));fixed=original.fixed_p0;tree=original.repair_edges
 r=jl.solve_joint_limit(child.x,child.s,child.groups,child.allowed_mask,replace(original.config,secondary=False),backbone_cycle=child.backbone_cycle,
                       repair_edges=tree,fixed_p0=fixed,ledger=ledger)
 name=f'final-{kind}-{ident:06d}';meta=dict(kind=kind,original_id=ident,assignment=assignment,alpha=0 if kind=='collapse' else None)
 artifact=json_safe(dict(meta=meta,result=r.data()));(OUT/'evidencia'/(name+'.json')).write_text(json.dumps(artifact,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
 exp['calls'].append(json_safe(dict(name=name,meta=meta,status=r.status,solver_status=r.solver_status,mu=r.mu0,diagnostics=r.diagnostics,file='evidencia/'+name+'.json',dual={})))
 print(name,r.status,r.mu0)
exp['counts']['joint_sdp']+=ledger.counts['joint_sdp'];exp['events']+=ledger.events
(OUT/'experimentos.json').write_text(json.dumps(exp,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
print('TOTAL',exp['counts'])
