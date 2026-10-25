import numpy as np
import pytest

from ..sdp import (
    optimize_template,
    template_metrics,
)


@pytest.fixture
def five_cluster_template():
    """Five-cluster twisted state with cyclic connectivity."""
    m = 5

    x = np.full(m, 1.0 / m)
    phi = np.arange(m) * (2 * np.pi / m)

    allowed = np.eye(m, dtype=bool)

    for i in range(m):
        j = (i + 1) % m
        allowed[i, j] = True
        allowed[j, i] = True

    return x, phi, allowed


def test_five_cluster_optimal_density(five_cluster_template):
    x, phi, allowed = five_cluster_template

    result = optimize_template(x, phi, allowed=allowed)

    assert result.status == "optimal"
    assert result.numerically_feasible
    assert result.mu == pytest.approx(0.6, abs=1e-5)


def test_five_cluster_adjacency(five_cluster_template):
    x, phi, allowed = five_cluster_template

    result = optimize_template(x, phi, allowed=allowed)

    expected_P = allowed.astype(float)

    assert result.P is not None

    np.testing.assert_allclose(
        result.P,
        expected_P,
        atol=1e-4,
    )


def test_five_cluster_equilibrium(five_cluster_template):
    x, phi, allowed = five_cluster_template

    result = optimize_template(x, phi, allowed=allowed)

    metrics = template_metrics(x, phi, result.P)

    assert metrics.max_force_error < 1e-6


def test_five_cluster_stability(five_cluster_template):
    x, phi, allowed = five_cluster_template

    result = optimize_template(x, phi, allowed=allowed)

    assert result.min_restoring is not None
    assert result.min_cluster_eigenvalue is not None
    assert result.min_restoring > 0
    assert result.min_cluster_eigenvalue > 0


def test_synchronized_phases_rejected():
    x = np.full(5, 0.2)
    phi = np.zeros(5)

    with pytest.raises(ValueError, match="synchronized"):
        optimize_template(x, phi)
