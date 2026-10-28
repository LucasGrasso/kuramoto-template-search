"""Independent flow-balanced starts, legal mass-ratio windows, and winding two."""

from dataclasses import replace

import numpy as np
from scipy.stats import wasserstein_distance

from .aggregate_flow import parent_masses,q_mass_critical
from .families import normalize_offsets,parent_mask,s7
from .models import LimitingCandidate,Q_STAR
from .topology import shortest_nonzero_cycle


def _partition(size,rng,*,singleton_anchor=False,winding_two=False):
    counts=np.array([2,2,2,1] if singleton_anchor else [2,2,2,2] if winding_two else [1,1,1,1])
    if size<counts.sum():
        raise ValueError("Insufficient vertices for the requested family")
    while counts.sum()<size:
        allowed=[j for j in range(4) if counts[j]<4 and (not singleton_anchor or j!=3)]
        counts[int(rng.choice(allowed))]+=1
    return counts


def flow_balanced_seed(size,rng,*,q_mass_ratio=None,flow_fraction=.97,uniform=False):
    if not 0<flow_fraction<1:
        raise ValueError("Use a strictly subcapacity common flow")
    if q_mass_ratio is not None and not q_mass_ratio>1:
        raise ValueError("q_mass_ratio>1 is separate from a crossed-edge density")
    counts=_partition(size,rng,singleton_anchor=q_mass_ratio is not None)
    if q_mass_ratio is None:
        logits=rng.normal(0,.15,4)
        M=np.exp(logits-logits.max()); M/=M.sum()
    else:
        u=2/(7+q_mass_ratio)
        M=np.array([u,u,u,(1+q_mass_ratio)/(7+q_mass_ratio)])
    groups=np.repeat(np.arange(4),counts)
    x=np.empty(size)
    for j in range(4):
        idx=np.flatnonzero(groups==j)
        weights=np.exp(rng.normal(0,.4,len(idx))); weights/=weights.sum()
        x[idx]=M[j]*weights
    C=flow_fraction*float(np.min(M*np.roll(M,-1)))
    f=np.empty(size)
    amplitude=min(.25,.35*(np.sqrt(1/flow_fraction)-1))
    for j in range(4):
        idx=np.flatnonzero(groups==j)
        deviations=rng.normal(size=len(idx)); deviations-=np.dot(x[idx]/M[j],deviations)
        deviations*=amplitude/max(1.,np.max(np.abs(deviations))) if not uniform else 0.
        f[idx]=C*x[idx]/M[j]*(1+deviations)
    p=np.eye(size)
    for j in range(4):
        idx=np.flatnonzero(groups==j)
        p[np.ix_(idx,idx)]=1.
        right=np.flatnonzero(groups==(j+1)%4)
        W=np.outer(f[idx],f[right])/C
        capacities=np.outer(x[idx],x[right])
        if not uniform and min(len(idx),len(right))>=2:
            for _ in range(3):
                a,b=rng.choice(len(idx),2,replace=False)
                c,d=rng.choice(len(right),2,replace=False)
                move=np.zeros_like(W)
                move[a,c]=move[b,d]=1; move[a,d]=move[b,c]=-1
                if rng.random()<.5:
                    move=-move
                upper=min(np.min((capacities-W)[move>0]),np.min(W[move<0]))
                W+=float(rng.uniform(.1,.8))*max(0.,upper)*move
        block=W/capacities
        p[np.ix_(idx,right)]=block
        p[np.ix_(right,idx)]=block.T
    s=rng.normal(size=size)
    for j in range(4):
        idx=np.flatnonzero(groups==j)
        s[idx]-=np.dot(f[idx],s[idx])/f[idx].sum()
    s=normalize_offsets(x,s)
    cycle=tuple(int(np.flatnonzero(groups==j)[0]) for j in range(4))
    c=LimitingCandidate(x,s,groups,p,parent_mask(groups),cycle,
                         "q_mass_window" if q_mass_ratio is not None else "independent_flow")
    return c,dict(q_mass_ratio=q_mass_ratio,q_cross_density=None,flow_fraction=flow_fraction,
        flow_target=C,uniform_parent_blocks=bool(uniform),complete_internal=True,singleton_anchor=bool(counts[3]==1),
        q_mass_critical=q_mass_critical())


def winding_two_seed(size,rng,*,relaxed_chords=False):
    if size<8:
        raise ValueError("A simple winding-two cycle needs at least eight interparent edges")
    counts=_partition(size,rng,winding_two=True)
    groups=np.array(list(range(4))*2+[j for j in range(4) for _ in range(counts[j]-2)])
    M=np.exp(rng.normal(0,.1,4)); M/=M.sum()
    x=np.empty(size)
    for j in range(4):
        idx=np.flatnonzero(groups==j)
        weights=np.exp(rng.normal(0,.25,len(idx))); weights/=weights.sum()
        x[idx]=M[j]*weights
    mask=groups[:,None]==groups[None,:]
    cycle=tuple(range(8))
    for a,b in zip(cycle,cycle[1:]+cycle[:1]):
        mask[a,b]=mask[b,a]=True
    # Extra vertices join their same-parent block. Additional neighbor links are
    # admitted only when they retain a shortest nonzero cycle of length >=5,
    # unless this explicitly tagged relaxed control permits four-cycles.
    candidates=[(a,b) for a in range(size) for b in range(a+1,size)
                if (groups[a]-groups[b])%2==1 and not mask[a,b]]
    rng.shuffle(candidates)
    for a,b in candidates[:max(4,size-8)]:
        trial=mask.copy(); trial[a,b]=trial[b,a]=True
        if relaxed_chords or shortest_nonzero_cycle(groups,trial)["length"]>=5:
            mask=trial
    p=(groups[:,None]==groups[None,:]).astype(float)
    edges=list(zip(cycle,cycle[1:]+cycle[:1]))
    W=.7*min(x[a]*x[b] for a,b in edges)
    for a,b in edges:
        p[a,b]=p[b,a]=W/(x[a]*x[b])
    s=np.zeros(size)
    for j in range(4):
        width=float(rng.uniform(.6,1.4))
        s[j],s[j+4]=-width,width
    s=normalize_offsets(x,s)
    c=LimitingCandidate(x,s,groups,p,mask,cycle,"winding_two_relaxed" if relaxed_chords else "winding_two_long")
    return c,dict(q_mass_ratio=None,q_cross_density=None,flow_target=None,
                  skeleton_winding=2,skeleton_length=8,
                  minimum_nonzero_cycle=shortest_nonzero_cycle(groups,mask),relaxed_chords=relaxed_chords)


def s7_distance(candidate):
    """Rotation/reflection/permutation-invariant distribution/edge-feature proxy.

    It is not an isomorphism test or a claim of exact twins. Different cluster
    counts are compared as mass distributions, and fresh merge solves remain
    necessary for degree/stability comparisons.
    """
    c=candidate; ref=s7(0.)
    sr=normalize_offsets(ref.x,ref.s); sc=normalize_offsets(c.x,c.s)
    Mr=parent_masses(ref.x,ref.groups)

    def features(x,g,p):
        M=parent_masses(x,g); values=[]
        for j in range(4):
            a=np.flatnonzero(g==j); b=np.flatnonzero(g==(j+1)%4)
            weights=np.outer(x[a],x[b])/(M[j]*M[(j+1)%4])
            block=p[np.ix_(a,b)]
            values.extend([float(np.sum(weights*block)),float(np.sum(weights*block**2))])
        return np.array(values)

    rf=features(ref.x,ref.groups,ref.p0)
    options=[]
    for reflection in (1,-1):
        for rotation in range(4):
            g=(reflection*c.groups+rotation)%4
            M=parent_masses(c.x,g)
            mass=float(np.linalg.norm(M-Mr))
            offsets=sum(float(Mr[j])*wasserstein_distance(reflection*sc[g==j],sr[ref.groups==j],
                c.x[g==j],ref.x[ref.groups==j]) for j in range(4))
            edge=float(np.linalg.norm(features(c.x,g,c.p0)-rf))
            options.append(dict(total=mass+offsets+.25*edge,mass=mass,offsets=float(offsets),
                                edge_features=edge,rotation=rotation,reflection=reflection))
    best=min(options,key=lambda d:d["total"])
    return dict(kind="invariant distribution/edge-feature proxy; not twin equivalence",**best)


def merge_pair(candidate,a,b):
    c=candidate
    if a==b or c.groups[a]!=c.groups[b]:
        raise ValueError("Merge within one parent only")
    if a>b:
        a,b=b,a
    keep=[k for k in range(c.m) if k!=b]
    x=c.x.copy(); x[a]+=x[b]
    s=c.s.copy(); s[a]=(c.x[a]*c.s[a]+c.x[b]*c.s[b])/x[a]
    p=c.p0.copy(); mask=c.allowed_mask.copy()
    for k in range(c.m):
        if k not in (a,b):
            p[a,k]=p[k,a]=(c.x[a]*c.p0[a,k]+c.x[b]*c.p0[b,k])/x[a]
            mask[a,k]=mask[k,a]=bool(c.allowed_mask[a,k] or c.allowed_mask[b,k])
    p[a,a]=1.
    x,s,g=x[keep],s[keep],c.groups[keep]
    p,mask=p[np.ix_(keep,keep)],mask[np.ix_(keep,keep)]
    shortest=shortest_nonzero_cycle(g,mask)
    if shortest["cycle"] is None:
        raise ValueError("No nonzero-winding backbone survives the merge")
    cycle=shortest["cycle"] if shortest["winding"]>0 else tuple(reversed(shortest["cycle"]))
    return LimitingCandidate(x,s,g,p,mask,cycle,c.family_id+"/merged")


def merge_to_seven(candidate):
    c=candidate; trace=[]
    while c.m>7:
        pairs=[(a,b) for a in range(c.m) for b in range(a+1,c.m) if c.groups[a]==c.groups[b]]
        a,b=min(pairs,key=lambda pair:abs(c.s[pair[0]]-c.s[pair[1]]))
        trace.append(dict(a=a,b=b,separation=float(abs(c.s[a]-c.s[b]))))
        c=merge_pair(c,a,b)
    return c,trace


def split_physical(candidate,index,ratio,separation):
    """Preserve inherited offset/h units for a mass-metric split/merge pair."""
    c=candidate
    if not 0<ratio<1:
        raise ValueError("Invalid split ratio")
    x=np.r_[c.x,c.x[index]*(1-ratio)]; x[index]*=ratio
    s=np.r_[c.s,c.s[index]+ratio*separation]; s[index]-=(1-ratio)*separation
    order=np.r_[np.arange(c.m),index]
    return LimitingCandidate(x,s,c.groups[order],c.p0[np.ix_(order,order)],
        c.allowed_mask[np.ix_(order,order)],c.backbone_cycle,c.family_id+"/split")
