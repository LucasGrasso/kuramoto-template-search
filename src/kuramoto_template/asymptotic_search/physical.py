"""Explicit mass quotient norm for comparisons across within-parent refinements."""

import numpy as np
from scipy.linalg import block_diag,eigvalsh

from .expansions import U4,internal_basis


def mass_quotient_metrics(candidate):
    c=candidate
    masses=np.array([c.x[c.groups==j].sum() for j in range(4)])
    soft=U4.T@(np.diag(masses)-np.outer(masses,masses))@U4
    internal=np.diag(c.x)
    for j in range(4):
        v=np.where(c.groups==j,c.x,0.)
        internal-=np.outer(v,v)/masses[j]
    Z=internal_basis(c)
    hard=Z.T@internal@Z
    return block_diag(soft,hard),hard,soft


def physical_spectrum(candidate,hlim):
    metric,_,_=mass_quotient_metrics(candidate)
    return eigvalsh((hlim+hlim.T)/2,metric)
