from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from kuramoto_template.template import PhaseTemplate


def test_defensive_copies_and_immutability():
    x = np.array([0.4, 0.6])
    phi = np.array([0.0, 1.0])
    template = PhaseTemplate(x, phi)
    x[0] = 0.1
    phi[0] = 3.0
    np.testing.assert_array_equal(template.x, [0.4, 0.6])
    np.testing.assert_array_equal(template.phi, [0.0, 1.0])
    for array in (template.x, template.phi):
        assert array.dtype == np.float64
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0
    with pytest.raises(FrozenInstanceError):
        template.x = x  # type: ignore


def test_normalization_and_identity_equality():
    template = PhaseTemplate(np.array([0.5, 0.5]), np.array([-0.1, 4 * np.pi]))
    assert template.n_clusters == 2
    np.testing.assert_allclose(template.phi, [2 * np.pi - 0.1, 0])
    assert template == template
    assert template != PhaseTemplate(template.x, template.phi)
    assert len({template, PhaseTemplate(template.x, template.phi)}) == 2


@pytest.mark.parametrize(
    "x,phi,match",
    [
        (np.array([[0.5, 0.5]]), np.array([0, 1]), "1D"),
        (np.array([0.5, 0.5]), np.array([[0, 1]]), "1D"),
        (np.array([0.5, 0.5]), np.array([0]), "1D"),
        (np.array([]), np.array([]), "two clusters"),
        (np.array([1]), np.array([0]), "two clusters"),
        (np.array([0, 1]), np.array([0, 1]), "positive"),
        (np.array([-0.1, 1.1]), np.array([0, 1]), "positive"),
        (np.array([0.5, 0.4]), np.array([0, 1]), "sum to one"),
        (np.array([0.5, 0.500001]), np.array([0, 1]), "sum to one"),
        (np.array([np.nan, 0.5]), np.array([0, 1]), "finite"),
        (np.array([np.inf, 0.5]), np.array([0, 1]), "finite"),
        (np.array([0.5, 0.5]), np.array([np.nan, 1]), "finite"),
        (np.array([0.5, 0.5]), np.array([0, np.inf]), "finite"),
    ],
)
def test_invalid_inputs(x, phi, match):
    with pytest.raises(ValueError, match=match):
        PhaseTemplate(x, phi)


def test_mass_tolerance_does_not_normalize():
    x = [0.5, 0.5 + 5e-11]
    np.testing.assert_array_equal(PhaseTemplate(np.array(x), np.array([0, 1])).x, x)


def test_synchrony_allowed_in_representation():
    template = PhaseTemplate(np.array([0.5, 0.5]), np.array([0, 2 * np.pi]))
    np.testing.assert_array_equal(template.phi, [0, 0])


def test_tiny_negative_phase_remains_in_half_open_interval():
    template = PhaseTemplate(np.array([0.5, 0.5]), np.array([-1e-300, 1]))
    assert np.all((0 <= template.phi) & (template.phi < 2 * np.pi))
