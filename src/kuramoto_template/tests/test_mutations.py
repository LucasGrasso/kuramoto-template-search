import numpy as np
import pytest

from kuramoto_template.template import PhaseTemplate
from kuramoto_template.mutations import (
    merge_clusters,
    perturb_phase,
    resize_clusters,
    split_cluster,
)


@pytest.fixture
def template():
    return PhaseTemplate(np.array([0.2, 0.3, 0.5]), np.array([0.01, 2.0, 5.0]))


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("separation", [-0.1, 0, 0.1])
def test_split_mean_and_order(template, index, separation):
    ratio = 0.3
    candidate = split_cluster(template, index, ratio, separation)
    assert candidate.n_clusters == 4
    children = [
        template.phi[index] - (1 - ratio) * separation,
        template.phi[index] + ratio * separation,
    ]
    np.testing.assert_allclose(
        candidate.phi[index : index + 2], np.mod(children, 2 * np.pi)
    )
    assert np.dot([ratio, 1 - ratio], children) == pytest.approx(template.phi[index])
    np.testing.assert_allclose(
        candidate.x[index : index + 2], template.x[index] * np.array([ratio, 1 - ratio])
    )
    np.testing.assert_array_equal(candidate.phi[:index], template.phi[:index])
    np.testing.assert_array_equal(candidate.phi[index + 2 :], template.phi[index + 1 :])
    assert candidate.x.sum() == pytest.approx(1)
    assert np.all(candidate.x > 0)


def test_merge_boundary_order_and_argument_symmetry():
    """Test that merging preserves the order of clusters and is symmetric in its arguments."""
    template = PhaseTemplate(
        np.array([0.2, 0.5, 0.3]), np.array([0.1, 2, 2 * np.pi - 0.1])
    )
    candidate = merge_clusters(template, 2, 0)
    expected = np.angle(0.2 * np.exp(0.1j) + 0.3 * np.exp(-0.1j)) % (2 * np.pi)
    np.testing.assert_allclose(candidate.x, [0.5, 0.5])
    np.testing.assert_allclose(candidate.phi, [expected, 2])
    np.testing.assert_array_equal(candidate.phi, merge_clusters(template, 0, 2).phi)


def test_merge_rejects_antipodal_cancellation():
    template = PhaseTemplate(np.array([0.25, 0.25, 0.5]), np.array([0, np.pi, 1]))
    with pytest.raises(ValueError, match="antipodal"):
        merge_clusters(template, 0, 1)


def test_merge_two_clusters_rejected():
    with pytest.raises(ValueError, match="at least two"):
        merge_clusters(PhaseTemplate(np.array([0.5, 0.5]), np.array([0, 1])), 0, 1)


def test_perturb_wraps_and_resize_preserves_phases(template):
    candidate = perturb_phase(template, 0, -0.1)
    assert candidate.phi[0] == pytest.approx(2 * np.pi - 0.09)
    np.testing.assert_array_equal(candidate.x, template.x)
    np.testing.assert_array_equal(candidate.phi[1:], template.phi[1:])
    candidate = resize_clusters(template, 2, 0, 0.1)
    np.testing.assert_allclose(candidate.x, [0.3, 0.3, 0.4])
    np.testing.assert_array_equal(candidate.phi, template.phi)


@pytest.mark.parametrize(
    "operation",
    [
        lambda t: split_cluster(t, 1, 0.2, 0.1),
        lambda t: merge_clusters(t, 0, 1),
        lambda t: perturb_phase(t, 1, 0.1),
        lambda t: resize_clusters(t, 0, 1, 0.01),
    ],
)
def test_operations_are_pure(template, operation):
    x, phi = template.x.copy(), template.phi.copy()
    candidate = operation(template)
    np.testing.assert_array_equal(template.x, x)
    np.testing.assert_array_equal(template.phi, phi)
    assert candidate is not template
    for name in ("x", "phi"):
        assert not getattr(candidate, name).flags.writeable
        assert not np.shares_memory(getattr(candidate, name), getattr(template, name))


@pytest.mark.parametrize(
    "operation,error",
    [
        (lambda t: split_cluster(t, -1, 0.5, 0.1), IndexError),
        (lambda t: split_cluster(t, 3, 0.5, 0.1), IndexError),
        (lambda t: split_cluster(t, 0.5, 0.5, 0.1), TypeError),  # type: ignore
        (lambda t: split_cluster(t, True, 0.5, 0.1), TypeError),
        (lambda t: split_cluster(t, 0, 0, 0.1), ValueError),
        (lambda t: split_cluster(t, 0, 1, 0.1), ValueError),
        (lambda t: split_cluster(t, 0, np.nan, 0.1), ValueError),
        (lambda t: split_cluster(t, 0, 0.5, np.inf), ValueError),
        (lambda t: merge_clusters(t, 0, 0), ValueError),
        (lambda t: merge_clusters(t, 0, -1), IndexError),
        (lambda t: perturb_phase(t, 3, 0.1), IndexError),
        (lambda t: perturb_phase(t, 0, np.nan), ValueError),
        (lambda t: resize_clusters(t, 0, 0, 0.1), ValueError),
        (lambda t: resize_clusters(t, -1, 0, 0.1), IndexError),
        (lambda t: resize_clusters(t, 0, 1, 0), ValueError),
        (lambda t: resize_clusters(t, 0, 1, -0.1), ValueError),
        (lambda t: resize_clusters(t, 0, 1, 0.2), ValueError),
        (lambda t: resize_clusters(t, 0, 1, np.inf), ValueError),
        (lambda t: resize_clusters(t, 0, 1, 0.1, min_mass=0.1), ValueError),
        (lambda t: resize_clusters(t, 0, 1, 0.1, min_mass=-1), ValueError),
        (lambda t: resize_clusters(t, 0, 1, 0.1, min_mass=np.nan), ValueError),
        (lambda t: resize_clusters(t, 0, 1, 1e-300), ValueError),
    ],
)
def test_invalid_parameters(template, operation, error):
    with pytest.raises(error):
        operation(template)


def test_seeded_repeated_sequences(template):
    def sequence(seed):
        rng = np.random.default_rng(seed)
        state = template
        for _ in range(100):
            i = int(rng.integers(state.n_clusters))
            ratio = rng.uniform(0.2, 0.8)
            state = split_cluster(state, i, ratio, 10 ** rng.uniform(-4, -1))
            state = perturb_phase(state, i, rng.normal(scale=0.01))
            state = resize_clusters(state, i, i + 1, state.x[i] * 0.1)
            state = merge_clusters(state, i, i + 1)
            assert state.n_clusters == template.n_clusters
            assert state.x.sum() == pytest.approx(1, abs=1e-12)
            assert np.all(state.x > 0)
            assert np.all((state.phi >= 0) & (state.phi < 2 * np.pi))
        return state

    first, second = sequence(42), sequence(42)
    np.testing.assert_array_equal(first.x, second.x)
    np.testing.assert_array_equal(first.phi, second.phi)
