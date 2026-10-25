import cvxpy as cp
import numpy as np
import pytest

from kuramoto_template.sdp_optimizer.stability_sdp import maximize_stability_margin


@pytest.fixture
def ring():
    m = 5
    allowed = np.eye(m, dtype=bool)
    for i in range(m):
        allowed[i, (i + 1) % m] = allowed[(i + 1) % m, i] = True
    return np.full(m, 0.2), np.arange(m) * (2 * np.pi / m), allowed


@pytest.mark.parametrize("target", [0.5, 0.6])
def test_ring_analytic_optimum_and_diagnostics(ring, target):
    x, phi, allowed = ring
    result = maximize_stability_margin(x, phi, target, allowed=allowed)
    # For full ring wiring: lambda_1 = 2*x^2*cos(2*pi/5)*(1-cos(2*pi/5)).
    expected = 2 * 0.2**2 * np.cos(2 * np.pi / 5) * (1 - np.cos(2 * np.pi / 5))
    assert result.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    assert result.numerically_valid and result.numerically_feasible
    assert result.gamma == pytest.approx(expected, abs=1e-7)
    assert result.mu == pytest.approx(0.6, abs=1e-6)
    assert result.target_mu == target
    assert result.P is not None
    assert result.degrees is not None
    np.testing.assert_allclose(result.P, allowed, atol=1e-5)
    np.testing.assert_allclose(result.degrees, result.P @ x, atol=1e-12)
    assert max(result.residuals.values()) <= 1e-7
    assert set(result.residuals) == {
        "symmetry", "lower_bound", "upper_bound", "forbidden", "cliques",
        "degree", "equilibrium", "restoring", "cluster_eigenvalue",
    }


def test_negative_margin_is_valid_but_unstable():
    result = maximize_stability_margin([0.5, 0.5], [0, np.pi], 1.0)
    assert result.numerically_valid
    assert not result.numerically_feasible
    assert result.gamma == pytest.approx(-0.5, abs=1e-7)
    assert result.min_cluster_eigenvalue == pytest.approx(-0.5, abs=1e-7)


def test_quarter_turn_degeneracy_is_not_a_stable_witness():
    result = maximize_stability_margin(np.full(4, 0.25), np.arange(4) * np.pi / 2, 0.75)
    assert result.numerically_valid
    assert result.gamma is not None
    assert abs(result.gamma) < 1e-7
    assert not result.numerically_feasible


@pytest.mark.parametrize("pattern,target", [("ring", 0.61), ("diagonal", 0.3)])
def test_infeasible_constraints(ring, pattern, target):
    x, phi, allowed = ring
    if pattern == "diagonal":
        allowed = np.eye(5, dtype=bool)
    result = maximize_stability_margin(x, phi, target, allowed=allowed)
    assert result.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE)
    assert result.P is None and result.gamma is None
    assert not result.numerically_valid and not result.numerically_feasible


def test_disconnected_zero_margin_is_allowed(ring):
    x, phi, _ = ring
    result = maximize_stability_margin(x, phi, 0.2, allowed=np.eye(5, dtype=bool))
    assert result.numerically_valid
    assert result.gamma == pytest.approx(0, abs=1e-8)
    assert not result.numerically_feasible


def test_rotation_and_label_invariance(ring):
    x, phi, allowed = ring
    original = maximize_stability_margin(x, phi, 0.6, allowed=allowed)
    perm = np.array([3, 0, 4, 1, 2])
    result = maximize_stability_margin(
        x[perm], phi[perm] + 1.234, 0.6, allowed=allowed[np.ix_(perm, perm)],
    )
    assert result.numerically_feasible
    assert result.gamma == pytest.approx(original.gamma, abs=1e-8)
    assert result.P is not None
    assert original.P is not None
    np.testing.assert_allclose(result.P, original.P[np.ix_(perm, perm)], atol=1e-6)


def test_optional_cliques_and_solver_options(ring):
    x, phi, allowed = ring
    np.fill_diagonal(allowed, False)
    result = maximize_stability_margin(
        x, phi, 0.4, allowed=allowed, force_cliques=False,
        solver_options={"tol_gap_abs": 1e-10, "tol_feas": 1e-10},
    )
    assert result.numerically_valid and result.numerically_feasible
    assert result.P is not None
    np.testing.assert_allclose(np.diag(result.P), 0, atol=1e-8)


def test_solver_error(ring):
    x, phi, allowed = ring
    result = maximize_stability_margin(x, phi, 0.6, allowed=allowed, solver="MISSING_SOLVER")
    assert result.status == "solver_error" and result.message
    assert result.gamma is None and result.P is None
    assert not result.numerically_valid


@pytest.mark.parametrize("target", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_target(ring, target):
    x, phi, _ = ring
    with pytest.raises(ValueError, match="target_mu"):
        maximize_stability_margin(x, phi, target)


@pytest.mark.parametrize("name", ["residual_tolerance", "stability_tolerance"])
@pytest.mark.parametrize("value", [-1, np.nan, np.inf])
def test_invalid_tolerances(ring, name, value):
    x, phi, _ = ring
    with pytest.raises(ValueError, match=name):
        maximize_stability_margin(x, phi, 0.6, **{name: value})


@pytest.mark.parametrize("x,phi,allowed,match", [
    ([0.5, 0.5], [0], None, "shape"),
    ([0, 1], [0, 1], None, "positive"),
    ([0.5, 0.500001], [0, 1], None, "sum to one"),
    ([0.5, 0.5], [0, np.inf], None, "finite"),
    ([0.5, 0.5], [0, 2 * np.pi], None, "synchronized"),
    ([0.5, 0.5], [0, 1], [[True]], "shape"),
    ([0.5, 0.5], [0, 1], [[True, False], [True, True]], "symmetric"),
    ([0.5, 0.5], [0, 1], [[False, True], [True, True]], "conflicts"),
])
def test_input_validation(x, phi, allowed, match):
    with pytest.raises(ValueError, match=match):
        maximize_stability_margin(x, phi, 0.5, allowed=allowed)


def fake_solve(monkeypatch, density, margin, status=cp.OPTIMAL_INACCURATE):
    def solve(problem, **kwargs):
        problem._status = status
        for variable in problem.variables():
            variable.save_value(density if variable.name() == "P" else margin)
    monkeypatch.setattr(cp.Problem, "solve", solve)


def test_inaccurate_status_is_checked_independently(ring, monkeypatch):
    x, phi, allowed = ring
    fake_solve(monkeypatch, allowed.astype(float), 0.01)
    valid = maximize_stability_margin(x, phi, 0.6, allowed=allowed)
    assert valid.numerically_valid and valid.numerically_feasible
    fake_solve(monkeypatch, allowed.astype(float), 0.1)
    invalid = maximize_stability_margin(x, phi, 0.6, allowed=allowed)
    assert not invalid.numerically_valid and not invalid.numerically_feasible
    assert invalid.residuals["cluster_eigenvalue"] > 0.08


@pytest.mark.parametrize("density,margin,match", [
    (None, 0.01, "no P or gamma"),
    (np.eye(5), None, "no P or gamma"),
    (np.full((5, 5), np.nan), 0.01, "nonfinite"),
    (np.eye(5), np.inf, "nonfinite"),
])
def test_missing_or_nonfinite_solver_values(ring, monkeypatch, density, margin, match):
    x, phi, _ = ring
    fake_solve(monkeypatch, density, margin)
    result = maximize_stability_margin(x, phi, 0.6)
    assert not result.numerically_valid
    assert match in result.message


def test_all_constraint_residuals_use_unclipped_output(ring, monkeypatch):
    x, phi, allowed = ring
    density = allowed.astype(float)
    density[0, 0] = 1.1
    density[0, 2] = -0.1
    fake_solve(monkeypatch, density, 0.5)
    result = maximize_stability_margin(x, phi, 0.9, allowed=allowed)
    assert not result.numerically_valid
    np.testing.assert_array_equal(result.P, density)
    for name in result.residuals:
        assert result.residuals[name] > 0
    assert result.residuals["lower_bound"] == pytest.approx(0.1)
    assert result.residuals["upper_bound"] == pytest.approx(0.1)


def test_stability_screen_is_configurable(ring, monkeypatch):
    x, phi, allowed = ring
    fake_solve(monkeypatch, allowed.astype(float), 1e-12)
    assert not maximize_stability_margin(x, phi, 0.6).numerically_feasible
    assert maximize_stability_margin(x, phi, 0.6, stability_tolerance=0).numerically_feasible
