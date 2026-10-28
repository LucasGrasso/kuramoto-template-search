from dataclasses import replace

import numpy as np
import pytest

from kuramoto_template.asymptotic_search.compensated import compensated_move
from kuramoto_template.asymptotic_search.expansions import coefficients
from kuramoto_template.asymptotic_search.families import s7,appendix_a_correction
from kuramoto_template.asymptotic_search.flow_families import merge_to_seven,s7_distance,split_physical
from kuramoto_template.asymptotic_search.joint_limit import JointLimitConfig,solve_joint_limit,validate_joint
from kuramoto_template.asymptotic_search.physical import mass_quotient_metrics
from kuramoto_template.asymptotic_search.solvers import SolverLedger


@pytest.mark.parametrize("policy",["balanced","random"])
@pytest.mark.parametrize("method",["projection","qp"])
def test_compensated_changes_preserve_f0_k_support_and_tangents(policy,method):
    c=s7(.03)
    result=validate_joint(c,appendix_a_correction(c),fixed_p0=c.p0)
    ledger=SolverLedger(dict(compensation_qp=1,compensation_projection=1))
    move=compensated_move(result,np.random.default_rng(12),policy=policy,method=method,ledger=ledger)
    assert move.status=="proposal"
    assert len(move.selected_edges)<=8
    assert np.max(np.abs(move.delta-move.delta.T))==0
    e=coefficients(c); changed=coefficients(move.candidate)
    np.testing.assert_allclose(changed.f0,e.f0,atol=1e-9)
    np.testing.assert_allclose(changed.cuts,e.cuts,atol=1e-9)
    assert move.diagnostics["box_violation"]<1e-9
    assert np.min(move.delta[c.p0==0],initial=0)>=-1e-9
    assert np.max(move.delta[c.p0==1],initial=0)<=1e-9
    assert move.diagnostics["requires_joint_reoptimization"]
    assert not move.diagnostics["accepted_as_witness"]
    assert ledger.counts["compensation_qp" if method=="qp" else "compensation_projection"]==1


def test_s7_distance_is_invariant_and_merge_is_not_accepted_without_solve():
    c=s7(0.)
    assert s7_distance(c)["total"]<1e-14
    order=np.array([1,0,3,2,5,4,6]); inverse=np.argsort(order)
    p=replace(c,x=c.x[order],s=c.s[order],groups=c.groups[order],p0=c.p0[np.ix_(order,order)],
        allowed_mask=c.allowed_mask[np.ix_(order,order)],backbone_cycle=tuple(inverse[list(c.backbone_cycle)]))
    assert s7_distance(p)["total"]<1e-14
    reflection=replace(c,groups=(-c.groups)%4,s=-c.s,backbone_cycle=tuple(reversed(c.backbone_cycle)))
    assert s7_distance(reflection)["total"]<1e-14
    split=split_physical(c,0,.3,.001)
    merged,trace=merge_to_seven(split)
    assert merged.m==7 and trace
    np.testing.assert_allclose([merged.x[merged.groups==j].sum() for j in range(4)],
                                [c.x[c.groups==j].sum() for j in range(4)],atol=1e-15)
    assert merged.x@merged.s==pytest.approx(c.x@c.s,abs=1e-15)
    _,_,soft_before=mass_quotient_metrics(split)
    _,_,soft_after=mass_quotient_metrics(merged)
    np.testing.assert_allclose(soft_before,soft_after,atol=1e-15)


@pytest.mark.integration
def test_fixed_flow_and_mass_metric_are_independently_validated():
    c=s7(.03)
    e=coefficients(c)
    from kuramoto_template.asymptotic_search.aggregate_flow import parent_flows
    flow=.999*parent_flows(c).mean()  # Free-density buffers explicitly exclude saturation.
    cfg=JointLimitConfig(margin_metric="mass",gamma0=1e-5,hard_floor=1e-4,secondary=False)
    r=solve_joint_limit(c.x,c.s,c.groups,c.allowed_mask,cfg,backbone_cycle=c.backbone_cycle,flow_target=flow)
    assert r.status=="joint_asymptotic_strict"
    assert r.diagnostics["flow_target_error"]<1e-9
    assert r.diagnostics["physical_gap"]>=.95*cfg.gamma0
    assert abs(np.sum(r.solve_records[0]["dual_degree"])-1)<1e-6


def test_both_dipole_derivatives_match_independent_perturbations():
    from kuramoto_template.asymptotic_search.compensated import _derivatives,dipole_slacks
    from kuramoto_template.asymptotic_search.joint_limit import build_affine_model
    c=s7(.03)
    model=build_affine_model(c.x,c.s,c.groups,c.allowed_mask,backbone_cycle=c.backbone_cycle)
    rows=_derivatives(c,model,appendix_a_correction(c))
    before,_=dipole_slacks(c)
    for i,(a,b) in enumerate(model.p_pairs):
        step=-1e-7 if c.p0[a,b]>.5 else 1e-7
        p=c.p0.copy(); p[a,b]+=step; p[b,a]+=step
        after,_=dipole_slacks(replace(c,p0=p))
        np.testing.assert_allclose((after-before)/step,rows[:,i],rtol=2e-5,atol=2e-7)
