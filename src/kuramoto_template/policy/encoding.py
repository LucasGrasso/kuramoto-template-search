"""Padded variable-size batches; no canonical sorting or absolute phase input."""
import numpy as np
import torch


def batch_observations(observations, device="cpu"):
    if not observations:
        raise ValueError("Cannot batch zero observations")
    # Preserve index identities, including adversarial non-prefix padding.
    n = max(len(o["node_mask"]) for o in observations)
    result = {}
    for key in observations[0]:
        arrays=[]
        for obs in observations:
            a=np.asarray(obs[key]); m=len(obs["node_mask"])
            if key in ("nodes","node_mask"):
                a=np.pad(a,[(0,n-m)]+[(0,0)]*(a.ndim-1))
            elif key=="edges":
                a=np.pad(a,[(0,n-m),(0,n-m),(0,0)])
            elif key=="pair_mask":
                a=np.pad(a,[(0,0),(0,n-m),(0,n-m)])
            arrays.append(a)
        result[key]=torch.as_tensor(np.stack(arrays),device=device,dtype=torch.float32)
    return result


def batch_actions(actions, device="cpu"):
    canonical=[]
    for a in actions:
        a={**a,'parameters':np.array(a['parameters'],dtype=np.float32,copy=True)}
        if a['operation'] in (0,2):a['j']=0
        if a['operation']==1:a['parameters'][:]=0
        elif a['operation']!=0:a['parameters'][1]=0
        canonical.append(a)
    return {k:torch.as_tensor(np.stack([a[k] for a in canonical]),device=device,
            dtype=torch.float32 if k=="parameters" else torch.long)
            for k in ("operation","i","j","parameters")}


def numpy_action(action, index=0):
    return {k:(v[index].detach().cpu().numpy().astype(np.float32) if k=="parameters"
               else int(v[index])) for k,v in action.items()}


def reflect(observation, action=None):
    """Reflect phi -> -phi coherently with the existing decoder child order."""
    obs={k:np.array(v,copy=True) for k,v in observation.items()}
    obs["edges"][...,0]*=-1
    if action is None:return obs
    a={k:(np.array(v,copy=True) if k=="parameters" else int(v)) for k,v in action.items()}
    if a["operation"]==0 and a["parameters"][1]==0:
        raise ValueError("Split q=0 has a decoder sign discontinuity; do not reflect this action")
    if a["operation"]==0:a["parameters"][1]*=-1
    if a["operation"]==2:a["parameters"][0]*=-1
    return obs,a
