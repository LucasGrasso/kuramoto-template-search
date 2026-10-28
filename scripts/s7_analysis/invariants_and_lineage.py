import pathlib,sys,os,json,collections,itertools
os.environ['PYTHONDONTWRITEBYTECODE']='1'
# Reuse analysis definitions without regenerating its outputs.
script=pathlib.Path('work/analyze_sweep.py').read_text(encoding='utf-8');exec(script.split('trace={t')[0])
analysis=json.loads((OUT/'analisis-datos.json').read_text(encoding='utf-8'));selected=analysis['selected'];rng=np.random.default_rng(20261026)
checks=[]
from dataclasses import replace
from kuramoto_template.asymptotic_search.models import LimitingCandidate
for i in selected:
    c=results[i].candidate;ref=distance(c,reference)[0]
    for _ in range(3):
        p=rng.permutation(c.m);inv=np.argsort(p);shift=int(rng.integers(4));e=int(rng.choice([-1,1]))
        # Backbone order is a schema constraint. Use the first valid node per rotated parent.
        gg=(e*c.groups[p]+shift)%4;mask=c.allowed_mask[np.ix_(p,p)]
        backbone=next(cycle for cycle in itertools.product(*(np.flatnonzero(gg==g) for g in range(4))) if all(mask[a,b] for a,b in zip(cycle,cycle[1:]+cycle[:1])))
        child=LimitingCandidate(c.x[p],e*c.s[p]+rng.normal(),gg,c.p0[np.ix_(p,p)],mask,backbone)
        out=distance(child,reference)[0];err=max(abs(out[k]-ref[k]) for k in ('score','mass_l1','offset_w2','p0_rms','mask_disagreement'))
        checks.append(dict(id=i,error=err))
trace={t['id']:t for t in state['trace']};solve2id={r['trace_id']:r['id'] for r in state['records']}
lineage=[]
for rec in state['records']:
    event=trace[trace[rec['trace_id']]['parent_id']];parent_id=solve2id.get(event['parent_id'])
    if parent_id is None:continue
    pr=state['records'][parent_id];comp=(pr['mode']==rec['mode']=='density' and pr['h_guard']==rec['h_guard'])
    lineage.append(dict(id=rec['id'],parent=parent_id,operation=event['move']['operation'],predicted_gain=event['move'].get('predicted_gain'),
                        comparable=comp,parent_status=pr['status'],child_status=rec['status'],delta_mu=rec['mu0']-pr['mu0'] if comp and rec['mu0'] is not None and pr['mu0'] is not None else None))
summary={}
for op in ('weak_topology','mass_logits_offsets'):
    pp=[l for l in lineage if l['operation']==op];com=[l for l in pp if l['comparable']];ss=[l for l in com if l['child_status']=='joint_asymptotic_strict' and l['parent_status']=='joint_asymptotic_strict']
    summary[op]=dict(paid=len(pp),same_guard_density=len(com),strict_pairs=len(ss),improvements=sum(l['delta_mu']>1e-8 for l in ss),
                     losses=sum(l['delta_mu']<-1e-8 for l in ss),ties=sum(abs(l['delta_mu'])<=1e-8 for l in ss),
                     positive_predictions=sum((l['predicted_gain'] or 0)>0 for l in pp))
dump(OUT/'invariantes-y-linaje.json',dict(checks=checks,max_invariance_error=max(c['error'] for c in checks),lineage=lineage,summary=summary,
     proposal_operations=dict(collections.Counter(t['move']['operation'] for t in state['trace'] if t['kind']=='outer_proposal'))))
print('INVARIANCE',max(c['error'] for c in checks),'LINEAGE',summary)
