from dataclasses import replace

import numpy as np
import pytest

from kuramoto_template.asymptotic_search.families import appendix_a_correction,s7
from kuramoto_template.asymptotic_search.finite_lift import lift_candidate
from kuramoto_template.asymptotic_search.joint_limit import JointLimitConfig,solve_joint_limit,validate_joint
from kuramoto_template.asymptotic_search.solvers import CallLimit,SolverLedger
from kuramoto_template.asymptotic_search.stage2 import validate_correction

TREE = ((0,2),(2,4),(4,6),(6,5),(5,3),(3,1))


@pytest.mark.integration
def test_free_joint_model_guard_interpolation_and_actual_finite_lift():
    c = s7(.01)
    cfg = JointLimitConfig(secondary=False)
    ledger = SolverLedger(dict(joint_sdp=1))
    result = solve_joint_limit(c.x,c.s,c.groups,c.allowed_mask,cfg,
                               backbone_cycle=c.backbone_cycle,repair_edges=TREE,ledger=ledger)
    assert result.status=="joint_asymptotic_strict"
    assert ledger.counts==dict(joint_sdp=1)
    assert result.diagnostics["full_gap"]>0
    assert result.diagnostics["hard_gap"]>0
    for h in np.linspace(0,cfg.h_guard,11):
        P = result.candidate.p0+h*result.r
        assert P.min()>=-1e-9 and P.max()<=1+1e-9
    assert all(result.r[a,b]<0 for a,b in TREE)
    correction = validate_correction(result.candidate,result.r,numerics=cfg.lift_numerics(),solver_status=result.solver_status)
    lifted = lift_candidate(correction,.0003,numerics=cfg.lift_numerics())
    assert lifted.status=="finite_h_numeric_strict"
    assert not result.proof_available
    assert result.solve_records[0]["primal_residual"] is not None
    with pytest.raises(CallLimit):
        solve_joint_limit(c.x,c.s,c.groups,c.allowed_mask,cfg,ledger=ledger)


@pytest.mark.integration
def test_target_margin_feasible_and_conditionally_infeasible():
    c = s7(.03)
    feasible = solve_joint_limit(c.x,c.s,c.groups,c.allowed_mask,
        JointLimitConfig(mode="margin",mu_target=.68,secondary=False),backbone_cycle=c.backbone_cycle)
    assert feasible.status=="joint_asymptotic_strict"
    assert feasible.mu0>=.68-1e-9
    infeasible = solve_joint_limit(c.x,c.s,c.groups,c.allowed_mask,
        JointLimitConfig(mode="margin",mu_target=.695,secondary=False),backbone_cycle=c.backbone_cycle)
    assert infeasible.status=="conditional_infeasible"
    assert infeasible.candidate is None


def test_false_solver_flags_and_disconnected_hard_block_never_strict():
    c = s7()
    r = appendix_a_correction(c)
    assert validate_joint(c,r,fixed_p0=c.p0,solver_status="optimal_inaccurate").status=="uncertain"
    assert validate_joint(c,r,fixed_p0=c.p0,reported_gamma=1).status=="uncertain"
    assert validate_joint(c,np.zeros_like(r),fixed_p0=c.p0).status=="failed"
    p = c.p0.copy()
    p[0,1] = p[1,0] = 0
    disconnected = replace(c,p0=p)
    assert validate_joint(disconnected,r,fixed_p0=p).status!="joint_asymptotic_strict"


@pytest.mark.integration
def test_decreasing_guards_are_recorded_and_s7_stays_feasible():
    c = s7(.03)
    for guard in (.01,.003,.001):
        cfg = JointLimitConfig(h_guard=guard,secondary=False)
        result = solve_joint_limit(c.x,c.s,c.groups,c.allowed_mask,cfg,fixed_p0=c.p0,
                                   backbone_cycle=c.backbone_cycle,repair_edges=TREE)
        assert result.status=="joint_asymptotic_strict"
        assert result.diagnostics["h_guard"]==guard
