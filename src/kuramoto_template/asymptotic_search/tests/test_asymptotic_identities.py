from dataclasses import replace

import numpy as np
import pytest

from kuramoto_template.asymptotic_search.expansions import (
    DIPOLES, U4, coefficients, effective_matrix, finite_rescaled_matrix,
    full_rescaled_matrix, group_indicator, internal_basis, internal_solve,
    reference_rescaled_matrix,
)
from kuramoto_template.asymptotic_search.families import appendix_a_correction, normalize_offsets, s7
from kuramoto_template.asymptotic_search.finite_lift import exact_forces


@pytest.mark.parametrize("t",[.03,.01,.003,0.])
def test_s7_analytic_identities(t):
    c = s7(t)
    r = appendix_a_correction(c)
    e = coefficients(c,r)
    assert c.x.sum() == pytest.approx(1,abs=1e-15)
    q = c.p0[0,3]
    assert (c.p0@c.x).min() == pytest.approx((4+2*q)/(7+q),abs=1e-15)
    assert np.max(np.abs(e.f0)) < 1e-15
    assert np.max(np.abs(e.f1)) < 1e-15
    assert abs(c.x@e.f0) < 1e-15
    assert abs(c.x@e.f1) < 1e-15
    assert np.max(np.abs(e.cuts)) < 1e-15
    assert abs(e.cuts.sum()) < 1e-15
    assert abs(e.k.sum()-e.energy) < 1e-15
    # Deliberately reverse a correction on one oriented edge: budget/F1 must fail.
    wrong = r.copy()
    wrong[0,2] = wrong[2,0] = -wrong[0,2]
    bad = coefficients(c,wrong)
    assert np.max(np.abs(bad.f1)) > 1e-4
    assert abs(bad.k.sum()-bad.energy) > 1e-4


def test_coefficients_against_independent_exact_trig():
    c = s7()
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(c.m,c.m))
    r = (raw+raw.T)/2
    neighbor = (c.groups[:,None]-c.groups[None,:])%2 == 1
    r[~neighbor] = 0
    e = coefficients(c,r)
    errors = []
    for h in [.003,.001,.0003]:
        error = np.max(np.abs(exact_forces(c,c.p0+h*r,h)-e.f0-h*e.f1))
        errors.append(error)
        assert error < 5*h*h
    assert errors[-1] < errors[0]/50


def test_internal_energy_range_and_constrained_minimization():
    c = s7()
    e = coefficients(c)
    same = c.groups[:,None] == c.groups[None,:]
    W = c.p0*np.outer(c.x,c.x)*same
    np.fill_diagonal(W,0)
    reference = np.diag(W.sum(axis=1))-W
    np.testing.assert_allclose(e.lin,reference,atol=1e-16)
    assert np.linalg.matrix_rank(e.lin,tol=1e-10)==c.m-4
    H = internal_basis(c)
    z,gap,res = internal_solve(c,e.lin,e.g)
    assert gap>0 and res<1e-15
    for j in range(4):
        np.testing.assert_allclose(e.g[c.groups==j].sum(axis=0),0,atol=1e-15)
    for eta in DIPOLES:
        g = e.g@eta
        minimizer = H@np.linalg.solve(H.T@e.lin@H,H.T@g)
        assert g@minimizer == pytest.approx(g@(z@eta),abs=1e-15)
        assert minimizer@e.lin@minimizer-2*minimizer@g == pytest.approx(-g@(z@eta),abs=1e-15)


def test_full_schur_and_finite_rescaling_converge():
    c = s7(.03)
    r = appendix_a_correction(c)
    full = full_rescaled_matrix(c,r)
    np.testing.assert_allclose(full,reference_rescaled_matrix(c,r),atol=1e-15)
    schur = full[:3,:3]-full[:3,3:]@np.linalg.solve(full[3:,3:],full[3:,:3])
    np.testing.assert_allclose(schur,U4.T@effective_matrix(c,r)@U4,atol=1e-15)
    errors = [np.linalg.norm(finite_rescaled_matrix(c,c.p0+h*r,h)-full) for h in [.01,.003,.001]]
    assert errors[-1] < errors[0]/8
    # Compare finite edge-based form to the full exact dense quadratic form.
    h = .003
    phi = c.template(h).phi
    W = np.outer(c.x,c.x)*(c.p0+h*r)*np.cos(phi[:,None]-phi[None,:])
    np.fill_diagonal(W,0)
    L = np.diag(W.sum(axis=1))-W
    T = np.column_stack((group_indicator(c)@U4,h*internal_basis(c)))
    np.testing.assert_allclose(finite_rescaled_matrix(c,c.p0+h*r,h),T.T@L@T/h**2,atol=2e-11)


def test_gauge_scale_and_permutation_covariance():
    c = s7()
    r = appendix_a_correction(c)
    Q = effective_matrix(c,r)
    shifted = replace(c,s=c.s+17)
    np.testing.assert_allclose(effective_matrix(shifted,r),Q,atol=1e-14)
    scaled = replace(c,s=2*c.s)
    np.testing.assert_allclose(effective_matrix(scaled,2*r),4*Q,atol=1e-14)
    order = np.array([1,0,3,2,5,4,6])
    inv = np.argsort(order)
    permuted = replace(c,x=c.x[order],s=c.s[order],groups=c.groups[order],p0=c.p0[np.ix_(order,order)],
                       allowed_mask=c.allowed_mask[np.ix_(order,order)],backbone_cycle=tuple(inv[list(c.backbone_cycle)]))
    np.testing.assert_allclose(effective_matrix(permuted,r[np.ix_(order,order)]),Q,atol=1e-14)
    normalized = normalize_offsets(c.x,c.s)
    assert abs(c.x@normalized)<1e-15
    assert c.x@(normalized**2)==pytest.approx(1)
