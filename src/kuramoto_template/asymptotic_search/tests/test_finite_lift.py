from dataclasses import replace

import numpy as np
import pytest

from kuramoto_template.asymptotic_search.families import appendix_a_correction, s7
from kuramoto_template.asymptotic_search.finite_lift import lift_candidate, scaling_slopes, validate_finite
from kuramoto_template.asymptotic_search.stage2 import validate_correction
from kuramoto_template.sdp_optimizer.template_metrics import template_metrics


def test_targeted_tree_repair_and_project_metrics():
    c = s7(.03)
    stage2 = validate_correction(c,appendix_a_correction(c))
    lifts = [lift_candidate(stage2,h) for h in [.003,.001,.0003,.0001]]
    for result in lifts:
        assert result.status=="finite_h_numeric_strict"
        assert len(result.tree)==c.m-1
        assert result.diagnostics["force_inf"]<1e-14
        assert result.diagnostics["c_min"]>0
        assert result.diagnostics["cluster_gap"]>result.diagnostics["roundoff_floor"]
        assert result.repair_inf<10*result.h**2
        metrics = template_metrics(c.x,c.template(result.h).phi,result.P)
        assert metrics.mu==result.mu
        assert metrics.max_force_error==result.diagnostics["force_inf"]
    slopes = scaling_slopes(lifts)
    assert 1.95<slopes["base_force_inf"]<2.05
    assert 1.95<slopes["repair_inf"]<2.05
    assert 1.9<slopes["cluster_gap"]<2.1
    mu0 = (c.p0@c.x).min()
    assert abs(lifts[-1].mu-mu0)<abs(lifts[0].mu-mu0)/20


def test_box_bad_correction_and_negative_hessian_not_clamped():
    c = s7()
    stage2 = validate_correction(c,appendix_a_correction(c))
    assert lift_candidate(replace(stage2,r=np.zeros((7,7))),.001).status=="correction_rejected"
    assert lift_candidate(stage2,1.).status=="box_rejected"
    good = lift_candidate(stage2,.001)
    bad = good.P.copy()
    bad[0,0] = 1.00001
    assert validate_finite(c,bad,.001)[0]=="finite_invalid"
    assert bad[0,0]==1.00001
    # Exact four-parent limiting densities at finite h are not a stable witness.
    assert validate_finite(c,c.p0,.001)[0]!="finite_h_numeric_strict"
    bad_r = -stage2.r
    assert validate_correction(c,bad_r).status!="asymptotic_full_second_order_strict"
