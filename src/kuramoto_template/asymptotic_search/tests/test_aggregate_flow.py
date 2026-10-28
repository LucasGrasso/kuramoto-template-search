import numpy as np
import pytest

from kuramoto_template.asymptotic_search.aggregate_flow import (
    UNIFORM_BLOCK_CAP,parent_flows,parent_masses,q_mass_cap,q_mass_critical,
    screen_aggregate,uniform_block_penalty,uniform_block_rule,
)
from kuramoto_template.asymptotic_search.expansions import DIPOLES,coefficients,internal_solve
from kuramoto_template.asymptotic_search.families import parent_mask,s7
from kuramoto_template.asymptotic_search.flow_families import flow_balanced_seed
from kuramoto_template.asymptotic_search.models import LimitingCandidate,MU_RECORD


def test_mass_bound_on_random_equal_flow_templates():
    rng=np.random.default_rng(23)
    for i in range(40):
        c,meta=flow_balanced_seed((7,8,9,10,12)[i%5],rng,flow_fraction=.8)
        e=coefficients(c)
        assert np.max(np.abs(e.f0))<1e-14
        assert np.max(np.abs(e.cuts))<1e-14
        assert c.p0.min()>=0 and c.p0.max()<=1
        flows=parent_flows(c)
        np.testing.assert_allclose(flows,meta["flow_target"],atol=1e-15)
        bound=screen_aggregate(c.x,c.groups,.69,allowed_mask=c.allowed_mask)
        assert (c.p0@c.x).min()<=bound.cap+1e-14
        M=parent_masses(c.x,c.groups)
        excess=max(M*(.69-M))-2*min(M*np.roll(M,-1))
        assert bound.excess==pytest.approx(excess,abs=1e-15)
        assert (bound.status=="aggregate_rejected")==bool(excess>bound.tolerance)


@pytest.mark.parametrize("t",[.03,.01,.003,0.])
def test_s7_exact_mass_ceiling_and_conservative_boundary(t):
    c=s7(t)
    q=c.p0[0,3]
    mu=(4+2*q)/(7+q)
    bound=screen_aggregate(c.x,c.groups,mu)
    assert bound.cap==pytest.approx(mu,abs=1e-15)
    assert bound.status=="boundary"
    assert screen_aggregate(c.x,c.groups,mu+1e-14).status!="aggregate_rejected"
    assert screen_aggregate(c.x,c.groups,mu+1e-7).status=="aggregate_rejected"


def test_out_of_scope_is_not_pruned():
    c=s7()
    assert screen_aggregate(c.x,c.groups,allowed_mask=np.ones((7,7))).status=="out_of_scope"
    phi=c.groups*np.pi/2+.1*np.arange(7)
    assert screen_aggregate(c.x,c.groups,leading_phases=phi).status=="out_of_scope"
    assert screen_aggregate(c.x,c.groups,leading_phases=c.groups*np.pi/2+1.23).in_scope


def test_uniform_block_identity_and_narrow_scope():
    x=np.array([.1,.15]*3+[.25])
    groups=np.array([0,0,1,1,2,2,3])
    s=np.array([1,-2/3,.7,-.7*2/3,-1,2/3,0])
    mask=parent_mask(groups)
    p=np.where(groups[:,None]==groups[None,:],1.,.9)
    p[~mask]=0
    c=LimitingCandidate(x,s,groups,p,mask,(0,2,4,6))
    metadata=dict(uniform_parent_blocks=True,complete_internal=True,singleton_anchor=True)
    assert uniform_block_rule(c,metadata=metadata)["status"]=="uniform_block_no_go"
    assert UNIFORM_BLOCK_CAP==pytest.approx(.6512392830509103,abs=1e-15)
    e=coefficients(c)
    for eta in DIPOLES:
        penalty,energies=uniform_block_penalty(c,eta)
        g=e.g@eta
        reference=g@internal_solve(c,e.lin,g)[0]
        assert penalty==pytest.approx(reference,abs=1e-15)
        assert energies.sum()==pytest.approx(e.energy,abs=1e-15)
        assert g@np.linalg.pinv(e.lin)@g==pytest.approx(penalty,abs=1e-15)
    assert uniform_block_rule(s7(),metadata=metadata)["status"]=="out_of_scope"
    assert uniform_block_rule(c)["status"]=="out_of_scope"
    from dataclasses import replace
    assert uniform_block_rule(replace(c,s=np.zeros(7)),metadata=metadata)["status"]=="trivial_splitting"


@pytest.mark.parametrize("q",[1.1,1.3,1.5])
def test_mass_ratio_window_is_not_an_illegal_crossed_density(q):
    c,meta=flow_balanced_seed(9,np.random.default_rng(41),q_mass_ratio=q,flow_fraction=.97)
    assert meta["q_mass_ratio"]==q and meta["q_cross_density"] is None
    assert c.p0.max()<=1
    cap=screen_aggregate(c.x,c.groups).cap
    assert cap==pytest.approx(q_mass_cap(q),abs=1e-15)
    assert (cap>MU_RECORD)==(q<q_mass_critical())
    root=q_mass_critical()
    assert abs(((1+root)+8/(1+root))/(7+root)-MU_RECORD)<1e-15
