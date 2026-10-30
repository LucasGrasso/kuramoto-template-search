import copy
import json
from pathlib import Path

import numpy as np
import pytest

flint = pytest.importorskip("flint", reason="requires the certification extra")

from kuramoto_template.asymptotic_search.interval_certification import (
    CertificationFailure, decimal_bounds, interval_cholesky, limiting_hessian, rational,
    rational_finite_repair, rational_limiting_repair, verify_finite, verify_limiting,
)

ROOT = Path(__file__).resolve().parents[4]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_interval_cholesky_rejects_indefinite_and_unresolved_pivots():
    with flint.ctx.workprec(256):
        assert interval_cholesky([[2, 1], [1, 2]])["positive_definite"]
        for matrix in ([[1, 2], [2, 1]], [[flint.arb(0, 1)]], [[0]]):
            with pytest.raises(CertificationFailure, match="pivot"):
                interval_cholesky(matrix)


@pytest.mark.parametrize('sign',[1,-1])
def test_decimal_bounds_handle_native_rationals_beyond_python_string_limit(sign):
    value=flint.fmpq(sign*(10**5000+1),3*10**5000)
    bounds=decimal_bounds(value)
    if sign>0:
        assert bounds['lower']=='0.33333333333333333333'
        assert bounds['upper']=='0.33333333333333333334'
    else:
        assert bounds['lower']=='-0.33333333333333333334'
        assert bounds['upper']=='-0.33333333333333333333'
    assert rational(bounds['exact'])==value


def test_finite_exact_repair_krawczyk_and_interval_stability():
    saved = read("research/joint-research-final/finalist-refinement.json")["best_finite"]
    data, changes = rational_finite_repair(saved)
    proof = verify_finite(data)
    assert proof["certified"] and proof["exact_equilibrium"]
    assert not proof["original_snapshot_certified"]
    assert rational(changes["max_density_change"]) < rational("1/100000000000000")
    assert proof["proof"]["strict_inclusion"]
    assert rational(proof["proof"]["derivative_infinity_norm_upper"]) < rational("1/10000000000000000")
    assert abs(float(rational(proof["density"]["exact"])) - saved["mu"]) < 1e-15
    # A fresh verification uses the recorded exact box, not trusted status flags.
    assert verify_finite(data, box=proof["proof"])["certified"]
    damaged = copy.deepcopy(data)
    damaged["P"][0][2] = damaged["P"][2][0] = str(rational(damaged["P"][0][2]) + rational("1/10000000000"))
    with pytest.raises(CertificationFailure, match="equilibrium"):
        verify_finite(damaged)


@pytest.mark.parametrize("relative", [
    "research/joint-research-final/sweep/top-limit.json",
    "research/s7-analysis/evidencia/fine-offset-1e-07--0.0001.json",
])
def test_exact_limiting_repair_and_independent_dense_hessian(relative):
    saved = read(relative)
    saved = saved.get("result", saved)
    data, changes = rational_limiting_repair(saved)
    proof = verify_limiting(data)
    assert proof["certified"] and proof["exact_F0_F1_K_and_budget"]
    assert proof["finite_family_for_sufficiently_small_h"]
    assert not proof["explicit_finite_h_certified"]
    assert rational(changes["max_density_change"]) < rational("1/100000000000000")
    x = np.array([float(rational(v)) for v in data["x"]])
    s = np.array([float(rational(v)) for v in data["s"]])
    p = np.array([[float(rational(v)) for v in row] for row in data["P0"]])
    r = np.array([[float(rational(v)) for v in row] for row in data["R"]])
    g = np.array(data["groups"])
    m = len(x)
    soft = np.eye(4)[g, :3]
    cols = []
    for j in range(4):
        ids = np.flatnonzero(g == j)
        cols.extend(np.eye(m)[a] - np.eye(m)[ids[-1]] for a in ids[:-1])
    hard = np.array(cols).T
    delta = s[None, :] - s[:, None]
    oriented = (g[None, :] - g[:, None]) % 4
    sin0 = np.where(oriented == 1, 1., np.where(oriented == 3, -1., 0.))
    same = g[:, None] == g[None, :]

    def lap(weights):
        weights = weights.copy()
        np.fill_diagonal(weights, 0)
        return np.diag(weights.sum(axis=1)) - weights

    mass = np.outer(x, x)
    l0 = lap(mass * p * same)
    l1 = lap(-mass * p * sin0 * delta)
    l2 = lap(mass * (-r * sin0 * delta - .5 * p * same * delta ** 2))
    expected = np.block([[soft.T @ l2 @ soft, soft.T @ l1 @ hard],
                         [hard.T @ l1 @ soft, hard.T @ l0 @ hard]])
    exact = limiting_hessian([rational(v) for v in data["x"]], [rational(v) for v in data["s"]],
                             data["groups"], [[rational(v) for v in row] for row in data["P0"]],
                             [[rational(v) for v in row] for row in data["R"]])
    np.testing.assert_allclose(np.array([[float(v) for v in row] for row in exact]), expected, atol=1e-15, rtol=0)
    broken = copy.deepcopy(data)
    broken["R"][0][2] = broken["R"][2][0] = str(rational(broken["R"][0][2]) + 1)
    with pytest.raises(CertificationFailure, match="identities"):
        verify_limiting(broken)
