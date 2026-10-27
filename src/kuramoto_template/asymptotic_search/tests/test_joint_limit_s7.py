from dataclasses import replace

import numpy as np
import pytest

from kuramoto_template.asymptotic_search.families import appendix_a_correction,s7
from kuramoto_template.asymptotic_search.joint_limit import JointLimitConfig,s7_sym_dipoles,solve_joint_limit,validate_joint
from kuramoto_template.asymptotic_search.models import Q_STAR,W_STAR
from kuramoto_template.asymptotic_search.stage1 import screen_candidate


@pytest.mark.parametrize("t",[.03,.01,.003,0.,-.01])
def test_narrow_s7_polynomials_equal_direct_dipole_slacks(t):
    c = s7(t)
    np.testing.assert_allclose(s7_sym_dipoles(Q_STAR-t,W_STAR),
        np.asarray(screen_candidate(c).dipole_slacks)/c.x[0]**2,atol=3e-14)
    if t<0:
        assert screen_candidate(c).status=="screen_rejected"
    elif t==0:
        result = validate_joint(c,appendix_a_correction(c),JointLimitConfig(neutral=True),fixed_p0=c.p0)
        assert result.status=="joint_neutral"
    else:
        assert min(s7_sym_dipoles(Q_STAR-t,W_STAR))>0


def test_s7_sym_no_go_is_independent_of_first_order_correction():
    # Only the ORIGINAL equal-mass/equal-crossed-density family. These are
    # R-independent necessary polynomials, not an outer-family no-go claim.
    rng = np.random.default_rng(912)
    for q,w in zip(rng.uniform(Q_STAR+1e-5,1,1000),rng.uniform(0,8,1000)):
        assert min(s7_sym_dipoles(q,w))<0
    np.testing.assert_allclose(s7_sym_dipoles(Q_STAR,W_STAR),0,atol=1e-14)
    for q in (.66,.8,1):
        # If Psi2 can be nonnegative, w must exceed w*, where Psi1 decreases.
        assert s7_sym_dipoles(q,W_STAR)[1]<0
        assert s7_sym_dipoles(q,W_STAR+.01)[0]<0


@pytest.mark.integration
@pytest.mark.parametrize("t",[.03,.01,.003])
def test_fixed_s7_joint_sdp_retains_the_known_positive_controls(t):
    c = s7(t)
    result = solve_joint_limit(c.x,c.s,c.groups,c.allowed_mask,fixed_p0=c.p0,
                               backbone_cycle=c.backbone_cycle)
    assert result.status=="joint_asymptotic_strict"
    q = Q_STAR-t
    assert result.mu0==pytest.approx((4+2*q)/(7+q),abs=1e-14)
    assert result.diagnostics["force0_inf"]<1e-14
    assert result.diagnostics["force1_inf"]<1e-9
    assert result.diagnostics["budget_identity_error"]<1e-9
    assert result.diagnostics["affine_error"]<1e-14


@pytest.mark.integration
def test_negative_control_cannot_be_saved_by_joint_r_optimization():
    c = s7(-.01)
    cfg = JointLimitConfig(mode="margin",mu_target=.69,secondary=False)
    result = solve_joint_limit(c.x,c.s,c.groups,c.allowed_mask,cfg,fixed_p0=c.p0,
                               backbone_cycle=c.backbone_cycle)
    assert result.status in ("joint_unstable","uncertain")
    assert result.diagnostics["full_gap"]<0
