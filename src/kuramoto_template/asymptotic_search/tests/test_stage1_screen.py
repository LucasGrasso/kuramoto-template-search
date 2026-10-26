from dataclasses import replace

import numpy as np
import pytest

from kuramoto_template.asymptotic_search.expansions import DIPOLES, coefficients
from kuramoto_template.asymptotic_search.families import s7
from kuramoto_template.asymptotic_search.solvers import CallLimit, SolverLedger
from kuramoto_template.asymptotic_search.stage1 import optimize_limiting, screen_candidate


@pytest.mark.parametrize("t",[.03,.01,.003])
def test_direct_screen_and_schur_lmis_agree(t):
    c = s7(t)
    metrics = screen_candidate(c)
    assert metrics.status=="stage1_screened"
    e = coefficients(c)
    for eta,slack in zip(DIPOLES,metrics.dipole_slacks):
        g = e.g@eta
        LMI = np.block([[e.lin,g[:,None]],[g[None,:],np.array([[e.energy]])]])
        assert np.linalg.eigvalsh(LMI).min()>-1e-14
        assert slack>0


@pytest.mark.integration
def test_lp_fail_does_not_exclude_necessary_lmi_solution():
    c = s7()
    ledger = SolverLedger(dict(stage1_lp=1,stage1_screen_sdp=1))
    lp = optimize_limiting(c,ledger=ledger)
    assert lp.status=="lp_screen_fail"
    sdp = optimize_limiting(c,method="sdp",ledger=ledger)
    assert sdp.status=="stage1_screened"
    assert sdp.metrics.mu0 == pytest.approx(screen_candidate(c).mu0,abs=2e-8)
    assert ledger.counts==dict(stage1_lp=1,stage1_screen_sdp=1)
    with pytest.raises(CallLimit):
        optimize_limiting(c,ledger=ledger)


@pytest.mark.integration
def test_fixed_outer_infeasibility_has_distinct_status():
    c = replace(s7(),s=np.zeros(7))
    assert optimize_limiting(c,method="sdp").status=="necessary_screen_infeasible"
    assert optimize_limiting(c,method="lp").status=="lp_screen_fail"
