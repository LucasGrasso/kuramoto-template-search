import numpy as np
import pytest

from kuramoto_template.asymptotic_search.expansions import U4, effective_matrix
from kuramoto_template.asymptotic_search.families import appendix_a_correction, s7
from kuramoto_template.asymptotic_search.solvers import SolverLedger
from kuramoto_template.asymptotic_search.stage2 import solve_correction, validate_correction


@pytest.mark.integration
@pytest.mark.parametrize("t",[.03,.01,.003])
def test_s7_conic_correction_independent_validation(t):
    c = s7(t)
    ledger = SolverLedger(dict(stage2_correction_sdp=2))
    result = solve_correction(c,ledger=ledger)
    assert result.status=="asymptotic_full_second_order_strict"
    assert result.force1_inf<1e-9
    assert result.budget_identity_error<1e-9
    assert result.cone_violation<1e-9
    assert result.schur_error<1e-14
    assert result.full_rescaled_gap>0 and result.hard_gap>0
    assert result.effective_gap == pytest.approx(np.linalg.eigvalsh(U4.T@effective_matrix(c,result.r)@U4).min(),abs=1e-14)
    assert ledger.counts["stage2_correction_sdp"]==2


def test_invalid_and_inaccurate_corrections_never_strict():
    c = s7()
    r = appendix_a_correction(c)
    assert validate_correction(c,r).status=="asymptotic_full_second_order_strict"
    assert validate_correction(c,r,solver_status="optimal_inaccurate").status=="uncertain"
    assert validate_correction(c,r,reported_gamma=1).status=="uncertain"
    assert validate_correction(c,np.zeros_like(r)).status=="failed"
    wrong = r.copy()
    wrong[0,2] = wrong[2,0] = 1
    assert validate_correction(c,wrong).status=="failed"
    wrong = r.copy()
    wrong[0,4] = wrong[4,0] = 1
    assert validate_correction(c,wrong).status=="failed"
    assert validate_correction(s7(0),appendix_a_correction(s7(0))).status=="asymptotic_degenerate"


def test_zero_density_allowed_edge_tangent_direction():
    c = s7(.6505691669447988)  # q=0: crossed neighbor edges are allowed, but absent.
    r = appendix_a_correction(c)
    wrong = r.copy()
    wrong[0,3] = wrong[3,0] = -1
    assert validate_correction(c,wrong).status=="failed"


@pytest.mark.integration
def test_allowed_zero_edge_cone_is_imposed_in_optimization():
    c = s7(.6505691669447988)
    result = solve_correction(c)
    assert result.r is not None
    assert result.r[0,3]>=0
    assert result.cone_violation<1e-9
