from decimal import Decimal
import pytest

from kuramoto_template.asymptotic_search.audit import high_precision_audit
from kuramoto_template.asymptotic_search.families import appendix_a_correction,s7
from kuramoto_template.asymptotic_search.finite_lift import lift_candidate
from kuramoto_template.asymptotic_search.stage2 import validate_correction


def test_increased_precision_audits_stored_floats_without_proof_claim():
    c = s7(.03)
    result = lift_candidate(validate_correction(c,appendix_a_correction(c)),.001)
    audit = high_precision_audit(c.x,result.phi,result.P)
    assert audit["precision_digits"]==60
    assert audit["positive_grounded_pivots"]
    assert audit["box_valid"] and audit["symmetry_valid"]
    assert Decimal(audit["max_force_error"])<Decimal("1e-14")
    assert float(audit["min_restoring"])==pytest.approx(result.diagnostics["c_min"],abs=1e-15)
    assert not audit["proof_available"]
    bad = high_precision_audit(c.x,c.template(.001).phi,c.p0)
    assert not bad["positive_grounded_pivots"]
