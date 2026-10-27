from dataclasses import replace

import numpy as np
import pytest

from kuramoto_template.asymptotic_search.expansions import (
    U4, coefficients, finite_rescaled_matrix, full_rescaled_matrix,
    group_indicator, internal_basis, reference_rescaled_matrix,
)
from kuramoto_template.asymptotic_search.families import appendix_a_correction,parent_mask,s7
from kuramoto_template.asymptotic_search.joint_limit import build_affine_model
from kuramoto_template.asymptotic_search.models import LimitingCandidate


def test_affine_coefficients_match_independent_edge_and_dense_routes():
    c = s7()
    model = build_affine_model(c.x,c.s,c.groups,c.allowed_mask,backbone_cycle=c.backbone_cycle)
    rng = np.random.default_rng(11)
    p,r = np.eye(c.m),np.zeros((c.m,c.m))
    for a,b in model.p_pairs:
        p[a,b] = p[b,a] = rng.uniform(.1,.9)
    for a,b in model.r_pairs:
        r[a,b] = r[b,a] = rng.normal()
    raw = replace(c,p0=p)
    values = model.evaluate(p,r)
    e = coefficients(raw,r)
    np.testing.assert_allclose(values["f0"],e.f0,atol=1e-15)
    np.testing.assert_allclose(values["f1"],e.f1,atol=1e-15)
    np.testing.assert_allclose(values["cuts"],e.cuts,atol=1e-15)
    np.testing.assert_allclose(values["hlim"],full_rescaled_matrix(raw,r),atol=1e-15)
    np.testing.assert_allclose(values["hlim"],reference_rescaled_matrix(raw,r),atol=1e-15)
    # Perturb EACH coefficient independently, not just a whole random vector.
    baseline = values["hlim"]
    for pairs,tensor,which in ((model.p_pairs,model.h_p,"p"),(model.r_pairs,model.h_r,"r")):
        for i,(a,b) in enumerate(pairs):
            pp,rr = p.copy(),r.copy()
            matrix = pp if which=="p" else rr
            matrix[a,b] += .01
            matrix[b,a] += .01
            np.testing.assert_allclose(model.evaluate(pp,rr)["hlim"]-baseline,
                                       .01*tensor[:,i].reshape(c.m-1,c.m-1),atol=1e-15)


def test_random_gauge_vectors_converge_from_exact_trigonometry():
    c = s7(.03)
    r = appendix_a_correction(c)
    model = build_affine_model(c.x,c.s,c.groups,c.allowed_mask,backbone_cycle=c.backbone_cycle)
    H = model.evaluate(c.p0,r)["hlim"]
    rng = np.random.default_rng(41)
    for _ in range(6):
        v = rng.normal(size=c.m-1)
        v /= np.linalg.norm(v)
        errors = [abs(v@finite_rescaled_matrix(c,c.p0+h*r,h)@v-v@H@v) for h in (.01,.003,.001,.0003)]
        assert errors[-1]<errors[0]/20
    Z = internal_basis(c)
    np.testing.assert_allclose(U4.T@U4,np.eye(3),atol=1e-15)
    np.testing.assert_allclose(Z.T@Z,np.eye(c.m-4),atol=1e-15)
    np.testing.assert_allclose(group_indicator(c).T@Z,0,atol=1e-15)


def test_gauges_scaling_and_singleton_empty_hard_block():
    c = s7()
    r = appendix_a_correction(c)
    base = full_rescaled_matrix(c,r)
    np.testing.assert_allclose(full_rescaled_matrix(replace(c,s=c.s+3),r),base,atol=1e-14)
    D = np.diag([2.]*3+[1.]*(c.m-4))
    np.testing.assert_allclose(full_rescaled_matrix(replace(c,s=2*c.s),2*r),D@base@D,atol=1e-14)
    groups = np.arange(4)
    mask = parent_mask(groups)
    model = build_affine_model(np.ones(4)/4,[-1,1,-1,1],groups,mask)
    values = model.evaluate(mask.astype(float),np.zeros((4,4)))
    assert values["hlim"].shape==(3,3)
    assert model.z.shape==(4,0)
    assert values["energy"]==0
