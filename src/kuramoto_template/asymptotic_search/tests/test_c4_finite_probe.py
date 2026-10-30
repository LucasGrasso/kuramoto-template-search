import pytest

from kuramoto_template.asymptotic_search.c4_multisplit import MirrorFamily
from kuramoto_template.asymptotic_search.c4_finite_probe import FiniteC4Probe
from kuramoto_template.asymptotic_search.joint_limit import JointLimitConfig
from kuramoto_template.asymptotic_search.solvers import SolverLedger


@pytest.mark.integration
@pytest.mark.parametrize('opposite',[False,True])
def test_finite_c4_probe_checks_actual_equilibrium_and_stability(opposite):
    family=MirrorFamily((1,2,2,2))
    c=family.geometry([-.1,.1,-.3])
    ledger=SolverLedger()
    engine=FiniteC4Probe(c,JointLimitConfig(gamma0=1e-6,secondary=False),allow_opposite=opposite,ledger=ledger)
    result=engine.run(c,.03)
    assert result['mu'] is not None and result['mu']>.5
    assert result['diagnostics']['force_inf']<1e-9
    assert result['diagnostics']['restoring']>0
    assert result['diagnostics']['unscaled_gap']>0
    assert result['diagnostics']['scaled_gap']>1e-7
    assert result['diagnostics']['winding_valid']
    assert ledger.counts=={'c4_finite_sdp':1}
