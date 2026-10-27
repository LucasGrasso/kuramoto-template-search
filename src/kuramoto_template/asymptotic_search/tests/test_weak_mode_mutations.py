from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from kuramoto_template.asymptotic_search.expansions import coefficients,full_rescaled_matrix,U4,internal_solve
from kuramoto_template.asymptotic_search.families import appendix_a_correction,s7
from kuramoto_template.asymptotic_search.joint_limit import validate_joint
from kuramoto_template.asymptotic_search.weak_modes import (
    apply_topology_move,continuous_move,high_precision_limit_audit,
    internal_partial_sensitivities,rank_topology_moves,weakest_mode,
)


def test_weak_vector_rankings_are_exact_partial_changes_not_acceptance():
    c = s7(.03)
    r = appendix_a_correction(c)
    H = full_rescaled_matrix(c,r)
    gap,v = weakest_mode(H)
    assert np.linalg.norm(v)==pytest.approx(1)
    assert np.linalg.norm(H@v-gap*v)<1e-14
    moves = rank_topology_moves(c,r,H)
    assert [m.predicted_gain for m in moves]==sorted([m.predicted_gain for m in moves],reverse=True)
    move = moves[0]
    changed = apply_topology_move(c,move)
    rr = r.copy()
    a,b = move.edge
    rr[a,b] += move.delta_r
    rr[b,a] += move.delta_r
    assert v@(full_rescaled_matrix(changed,rr)-H)@v==pytest.approx(move.predicted_gain,abs=1e-14)
    assert validate_joint(changed,rr,fixed_p0=changed.p0).status!="joint_asymptotic_strict"


def test_internal_partial_derivative_holds_g_and_cuts_fixed():
    c = s7()
    eta = U4[:,0]
    e = coefficients(c)
    sensitivities = internal_partial_sensitivities(c,eta)
    assert all(v>=0 for v in sensitivities.values())
    a,b = next(iter(sensitivities))
    eye = np.eye(c.m)
    edge = eye[a]-eye[b]
    g = e.g@eta
    base = g@internal_solve(c,e.lin,g)[0]
    epsilon = 1e-7
    perturbed = g@internal_solve(c,e.lin+epsilon*np.outer(edge,edge),g)[0]
    assert -(perturbed-base)/epsilon==pytest.approx(sensitivities[a,b],rel=1e-5)


def test_continuous_moves_replay_and_decimal_limit_audit():
    c = s7()
    a = continuous_move(c,np.random.default_rng(3))
    b = continuous_move(c,np.random.default_rng(3))
    np.testing.assert_array_equal(a.x,b.x)
    np.testing.assert_array_equal(a.s,b.s)
    assert a.x@a.s==pytest.approx(0,abs=1e-15)
    assert a.x@(a.s*a.s)==pytest.approx(1)
    result = validate_joint(c,appendix_a_correction(c),fixed_p0=c.p0)
    audit = high_precision_limit_audit(result)
    assert audit["positive_pivots"]
    assert Decimal(audit["force1_inf"])<Decimal("1e-14")
    assert Decimal(audit["matrix_error_vs_float"])<Decimal("1e-14")
    assert not audit["proof_available"]
