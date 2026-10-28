"""Simple support cycles; winding is an integer invariant, not a reward."""

import numpy as np


def parent_step(a,b):
    delta=int((b-a)%4)
    if delta==2:
        raise ValueError("Opposite-parent edge is outside the nearest-parent model")
    return -1 if delta==3 else delta


def cycle_audit(groups,cycle,*,phi=None,P=None,tolerance=1e-10):
    groups=np.asarray(groups)
    cycle=tuple(cycle)
    simple=len(cycle)>=3 and len(set(cycle))==len(cycle)
    if any(not 0<=a<len(groups) for a in cycle) or len(cycle)<3:
        return dict(simple=False,valid=False,winding=None,reason="invalid cycle indices")
    try:
        steps=[parent_step(groups[a],groups[b]) for a,b in zip(cycle,cycle[1:]+cycle[:1])]
    except ValueError:
        return dict(simple=simple,valid=False,winding=None,reason="opposite-parent edge")
    total=sum(steps)
    if total%4:
        return dict(simple=simple,valid=False,winding=None,reason="nonclosed parent walk")
    winding=total//4
    supported=P is None or all(P[a,b]>0 for a,b in zip(cycle,cycle[1:]+cycle[:1]))
    result=dict(simple=simple,valid=bool(simple and supported),winding=winding,
                interparent_edges=sum(step!=0 for step in steps),length=len(cycle))
    if phi is not None:
        phases=np.asarray(phi)
        wrapped=[float((phases[b]-phases[a]+np.pi)%(2*np.pi)-np.pi) for a,b in zip(cycle,cycle[1:]+cycle[:1])]
        branch=any(abs(abs(v)-np.pi)<=tolerance for v in wrapped)
        result.update(wrapped_steps=wrapped,wrapped_total=sum(wrapped),antipodal=branch)
        result["valid"]=bool(result["valid"] and not branch and abs(sum(wrapped)-2*np.pi*winding)<=tolerance)
    return result
