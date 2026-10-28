import os, sys, json, pathlib, csv, itertools, collections, hashlib, time
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=pathlib.Path(os.environ['KURAMOTO_REPO'])
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from kuramoto_template.asymptotic_search.models import MU_RECORD, Q_STAR, W_STAR
from kuramoto_template.asymptotic_search.families import s7, normalize_offsets
from kuramoto_template.asymptotic_search.joint_limit import result_from_data, validate_joint
from kuramoto_template.asymptotic_search.expansions import coefficients, DIPOLES, internal_solve, U4, effective_matrix, internal_basis
from kuramoto_template.asymptotic_search.stage1 import screen_candidate
from kuramoto_template.asymptotic_search.persistence import _code_hash
RUN=pathlib.Path(os.environ['KURAMOTO_SOURCE_RUN'])
OUT=pathlib.Path('outputs'); OUT.mkdir(exist_ok=True)
state=json.loads((RUN/'sweep/checkpoint.json').read_text())
raw={r['id']:json.loads((RUN/'sweep'/r['file']).read_text()) for r in state['records']}
results={k:result_from_data(v) for k,v in raw.items() if v['candidate'] is not None}
reference=s7(0.)

def dump(path,data):
    def default(a):
        if isinstance(a,np.ndarray): return a.tolist()
        if isinstance(a,np.generic): return a.item()
        raise TypeError(type(a).__name__)
    path.write_text(json.dumps(data,indent=2,default=default,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def csvwrite(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,keys); w.writeheader(); w.writerows(rows)

def transport(c,d,orientation=1,shift=0):
    # Conditional monotone transport within parents; global mass mismatch is measured separately.
    cg=(orientation*c.groups+shift)%4
    cs=orientation*normalize_offsets(c.x,c.s)
    ds=normalize_offsets(d.x,d.s)
    C=np.zeros((c.m,d.m))
    mass=np.array([c.x[cg==g].sum() for g in range(4)])
    target=np.array([d.x[d.groups==g].sum() for g in range(4)])
    for g in range(4):
        aa=np.flatnonzero(cg==g); aa=aa[np.argsort(cs[aa],kind='stable')]
        bb=np.flatnonzero(d.groups==g); bb=bb[np.argsort(ds[bb],kind='stable')]
        ax=c.x[aa].copy(); bx=d.x[bb]*mass[g]/target[g]
        i=j=0
        while i<len(aa) and j<len(bb):
            weight=min(ax[i],bx[j]); C[aa[i],bb[j]]+=weight
            ax[i]-=weight; bx[j]-=weight
            if ax[i]<1e-15:i+=1
            if bx[j]<1e-15:j+=1
    offset=float(np.sqrt(np.sum(C*(cs[:,None]-ds[None,:])**2)))
    rows=C.sum(axis=1); cols=C.sum(axis=0)
    perr2=rows@(c.p0**2)@rows+cols@(d.p0**2)@cols-2*np.sum(c.p0*(C@d.p0@C.T))
    maskc=c.allowed_mask.astype(float); maskd=d.allowed_mask.astype(float)
    merr=rows@maskc@rows+cols@maskd@cols-2*np.sum(maskc*(C@maskd@C.T))
    p=float(np.sqrt(max(0,perr2))); m=float(np.linalg.norm(mass-target))
    return dict(score=float(np.sqrt(m*m+offset*offset+p*p)),mass_l1=float(abs(mass-target).sum()),mass_l2=m,
                offset_w2=offset,p0_rms=p,mask_disagreement=float(max(0,merr)),orientation=orientation,shift=shift),C

def distance(c,d):
    return min((transport(c,d,e,k) for e in (1,-1) for k in range(4)),key=lambda q:q[0]['score'])

def merge_signature(c, eps=.05):
    # Agglomerate only if all offsets in a cell fit in an eps diameter (no chaining).
    sn=normalize_offsets(c.x,c.s); sets=[]
    for g in range(4):
        ids=sorted(np.flatnonzero(c.groups==g),key=lambda a:sn[a]); local=[]
        for a in ids:
            if not local or sn[a]-sn[local[-1][0]]>eps:local.append([int(a)])
            else:local[-1].append(int(a))
        sets.extend(local)
    masses=np.array([c.x[a].sum() for a in sets]); offsets=np.array([c.x[a]@sn[a]/c.x[a].sum() for a in sets])
    assignment=np.zeros((c.m,len(sets)))
    for j,a in enumerate(sets):assignment[a,j]=c.x[a]/masses[j]
    quotient=assignment.T@c.p0@assignment
    Q=np.zeros((c.m,c.m)); internal_scatter=0.
    for i,a in enumerate(sets):
        for j,b in enumerate(sets):Q[np.ix_(a,b)]=quotient[i,j]
    scatter=float(np.sqrt(np.sum(np.outer(c.x,c.x)*(c.p0-Q)**2)))
    counts=[sum(c.groups[a[0]]==g for a in sets) for g in range(4)]
    return dict(epsilon=eps,cells=sets,count=len(sets),parent_counts=counts,masses=masses,offsets=offsets,p0=quotient,row_scatter=scatter,
                within_offset_rms=float(np.sqrt(sum(np.sum(c.x[a]*(sn[a]-offsets[j])**2) for j,a in enumerate(sets)))))

trace={t['id']:t for t in state['trace']}
solve2id={r['trace_id']:r['id'] for r in state['records']}
rows=[]; details={}
for record in state['records']:
    i=record['id']; d=raw[i]; row=record.copy()
    proposal=trace[trace[record['trace_id']]['parent_id']]
    row.update(proposal=proposal['proposal'],operation=proposal['move']['operation'],parent=solve2id.get(proposal['parent_id']))
    if i in results:
        r=results[i]; c=r.candidate; comparison,C=distance(c,reference)
        row.update({k:v for k,v in comparison.items() if k not in ('orientation','shift')})
        row.update({k:r.diagnostics.get(k) for k in ['effective_gap','hard_gap','inner_restoring','energy','r_norm','r_inf','force0_inf','force1_inf','cut_inf','independent_primal_check_inf']})
        row.update(degree_active=int(np.sum(c.p0@c.x-r.mu0<1e-7)),min_mass=float(c.x.min()),
                   extra_forbidden=int(np.sum(np.triu(((c.groups[:,None]-c.groups)%4!=2)&~c.allowed_mask,1))))
        for eps in (.02,.05,.1,.2):row['merged_'+str(eps)]=merge_signature(c,eps)['count']
        details[i]=dict(comparison=comparison,merge={str(e):merge_signature(c,e) for e in (.02,.05,.1,.2)})
    rows.append(row)

strict=sorted((r for r in rows if r['status']=='joint_asymptotic_strict'),key=lambda a:-a['mu0'])
top=strict[:20]
# Choose the best in every (size,family) stratum plus farthest representatives.
diverse=[]
for m in (8,9,10,12):
    for family in ('complete','anchor_zero','cut_zero','mixed_zero','internal_sparse','nonuniform_anchor'):
        eligible=[r for r in strict if r['size']==m and r['family']==family]
        if eligible: diverse.append(eligible[0])
selected=list(dict.fromkeys([r['id'] for r in top+diverse]))
# Include the strongest mass/offset descendant at every larger size and a topology descendant.
for m in (8,9,10,12):
    r=next(r for r in strict if r['size']==m and r['operation']=='mass_logits_offsets')
    if r['id'] not in selected:selected.append(r['id'])
r=next(r for r in strict if r['operation']=='weak_topology')
if r['id'] not in selected:selected.append(r['id'])
# Distances are computed both ways to make a symmetric dissimilarity, not a certified metric.
n=len(selected); distances=np.zeros((n,n))
for a,b in itertools.combinations(range(n),2):
    ca,cb=(results[selected[k]].candidate for k in (a,b))
    dd=.5*(distance(ca,cb)[0]['score']+distance(cb,ca)[0]['score']); distances[a,b]=distances[b,a]=dd
clusters=fcluster(linkage(squareform(distances),method='average'),.15,criterion='distance')
for row in rows:
    if row['id'] in selected: row['structure_cluster']=int(clusters[selected.index(row['id'])])
csvwrite(OUT/'todos-los-resultados.csv',rows)
csvwrite(OUT/'top20.csv',top)
csvwrite(OUT/'muestra-diversa.csv',[rows[i] for i in selected if i not in [r['id'] for r in top]])
csvwrite(OUT/'distancias-muestra.csv',[dict(id=i,**{str(j):distances[a,b] for b,j in enumerate(selected)}) for a,i in enumerate(selected)])

# Numerical revalidation of every selected witness. No new conic calls.
validation=[]
for i in selected:
    r=results[i]; v=validate_joint(r.candidate,r.r,r.config,solver_status=r.solver_status,fixed_p0=r.fixed_p0,repair_edges=r.repair_edges,
                                reported_gamma=r.reported_objective if r.config.mode=='margin' else r.config.gamma0)
    validation.append(dict(id=i,stored=r.status,recomputed=v.status,mu_difference=v.mu0-r.mu0,
                           max_primal=v.diagnostics['independent_primal_check_inf'],matrix_error=float(np.max(abs(v.hlim-r.hlim)))))
    c=r.candidate; e=coefficients(c,r.r); ev,vec=np.linalg.eigh(effective_matrix(c,r.r));
    # The two-dimensional soft space is stable against arbitrary eigenvector rotation.
    soft=effective_matrix(c,r.r); vals,V=np.linalg.eigh(U4.T@soft@U4); weak=U4@V[:,:2]
    dip=DIPOLES.T/np.sqrt(2)
    overlap=np.linalg.svd(dip.T@weak,compute_uv=False)
    screen=screen_candidate(c)
    eta=U4@V[:,0]  # Unit weakest effective-soft eigenvector for density partials.
    z,_,_=internal_solve(c,e.lin,e.g@eta)
    sens=[]
    for a,b in itertools.combinations(range(c.m),2):
        if c.groups[a]==c.groups[b]:sens.append(dict(edge=[a,b],physical_weight_partial=float((z[a]-z[b])**2),density_partial=float(c.x[a]*c.x[b]*(z[a]-z[b])**2)))
    details[i].update(parent_masses=[c.x[c.groups==g].sum() for g in range(4)],parent_counts=[int(np.sum(c.groups==g)) for g in range(4)],
                      dipole_slacks=screen.dipole_slacks,soft_dipole_subspace_singular_values=overlap,
                      full_spectrum=r.spectrum,internal_sensitivities=sorted(sens,key=lambda d:-d['density_partial']),
                      degrees=c.p0@c.x,offsets=c.s,masses=c.x,mask=c.allowed_mask,p0=c.p0,r=r.r,
                      provenance=dict(proposal=trace[trace[state['records'][i]['trace_id']]['parent_id']],source_file=str(RUN/'sweep'/state['records'][i]['file'])))

counts=collections.Counter(r['status'] for r in rows)
summary=dict(reference=dict(mu=MU_RECORD,q=Q_STAR,w=W_STAR,x=reference.x,s=normalize_offsets(reference.x,reference.s),p0=reference.p0),
             source_hash_saved=state['identity']['source_hash'],source_hash_current=_code_hash(),counts=dict(counts),
             recorded_costs=state['counts'],proposals=state['cursor'],cache_hits=state['cache_hits'],snapshot_count=len(raw),
             selected=selected,top20=[r['id'] for r in top],diverse=[r['id'] for r in diverse],validation=validation,
             size_summary=[dict(size=m,strict=sum(r['size']==m for r in strict),best=next((r for r in strict if r['size']==m),None)) for m in (7,8,9,10,12)],
             top20_range=[top[-1]['mu0'],top[0]['mu0']],all_rows=rows,details={str(i):details[i] for i in selected},
             distance_matrix=distances,structure_clusters={str(i):int(clusters[a]) for a,i in enumerate(selected)},
             shortlist=state['shortlist'],finite=state['finite'],audits=state['audits'])
dump(OUT/'analisis-datos.json',summary)
print('COUNTS',counts,'SOURCE MATCH',summary['source_hash_current']==summary['source_hash_saved'])
print('SELECTED',selected,'CLUSTERS',summary['structure_clusters'])
for r in top: print('TOP', {k:r.get(k) for k in ['id','size','family','mu0','h_guard','score','mass_l1','offset_w2','p0_rms','merged_0.05','operation','parent']})
for r in diverse:print('DIVERSE',{k:r.get(k) for k in ['id','size','family','mu0','score','mass_l1','offset_w2','p0_rms','merged_0.05','extra_forbidden']})
print('REVALIDATION',validation)
