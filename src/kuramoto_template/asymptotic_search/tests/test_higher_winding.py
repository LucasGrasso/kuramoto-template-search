from dataclasses import replace

import numpy as np
import pytest

from kuramoto_template.asymptotic_search.expansions import coefficients
from kuramoto_template.asymptotic_search.families import winding
from kuramoto_template.asymptotic_search.flow_families import winding_two_seed
from kuramoto_template.asymptotic_search.topology import cycle_audit,shortest_nonzero_cycle


def test_canonical_simple_eight_cycle_has_winding_two():
    groups=np.tile(np.arange(4),2)
    cycle=tuple(range(8))
    support=np.eye(8,dtype=bool)
    for a,b in zip(cycle,cycle[1:]+cycle[:1]):
        support[a,b]=support[b,a]=True
    phi=groups*np.pi/2+1e-3*np.arange(8)
    audit=cycle_audit(groups,cycle,phi=phi,P=support)
    assert audit["simple"] and audit["valid"] and audit["winding"]==2
    assert abs(audit["wrapped_total"]-4*np.pi)<1e-12
    assert shortest_nonzero_cycle(groups,support)["length"]==8
    order=np.array([5,1,7,0,3,2,6,4]); inverse=np.argsort(order)
    perm=cycle_audit(groups[order],tuple(inverse[list(cycle)]),phi=phi[order]+2.8,P=support[np.ix_(order,order)])
    assert perm["valid"] and perm["winding"]==2


def test_concatenation_is_not_a_simple_winding_two_cycle():
    groups=np.arange(4)
    audit=cycle_audit(groups,(0,1,2,3,0,1,2,3))
    assert audit["winding"]==2 and not audit["simple"] and not audit["valid"]
    with pytest.raises(ValueError):
        winding_two_seed(7,np.random.default_rng(1))


def test_supported_eight_backbone_preserves_hard_connectivity_and_signed_branches():
    c,meta=winding_two_seed(8,np.random.default_rng(3))
    assert c.backbone_winding==2
    assert winding(c,c.p0,.001)
    assert meta["minimum_nonzero_cycle"]["length"]==5
    e=coefficients(c)
    assert np.max(np.abs(e.f0))<1e-14
    assert np.max(np.abs(e.cuts))<1e-14
    assert np.linalg.matrix_rank(e.lin,tol=1e-12)==c.m-4
    wrong=c.p0.copy(); wrong[0,1]=wrong[1,0]=0
    assert not winding(c,wrong,.001)
    phi=c.template(.001).phi.copy(); phi[1]=phi[0]+np.pi
    assert not cycle_audit(c.groups,c.backbone_cycle,phi=phi)["valid"]
