"""Initial-template families are deterministic, valid, and distinct."""

import numpy as np
import pytest

from kuramoto_template.search.random_search import twisted_seed
from kuramoto_template.template import PhaseTemplate

from ..seeds import INITIAL_FAMILIES, build_initial, register_initial


def test_default_families_are_registered() -> None:
    assert {"five_twisted", "eight_twisted", "four_perturbed", "random_phases"} <= set(
        INITIAL_FAMILIES
    )


def test_build_initial_is_reproducible() -> None:
    for name in INITIAL_FAMILIES:
        first = build_initial(name, 123)
        second = build_initial(name, 123)
        assert np.array_equal(first.x, second.x)
        assert np.array_equal(first.phi, second.phi)


def test_fixed_families() -> None:
    five = build_initial("five_twisted", 0)
    eight = build_initial("eight_twisted", 0)
    assert five.n_clusters == 5
    assert eight.n_clusters == 8
    assert np.allclose(five.x, twisted_seed(5).x)
    assert np.allclose(five.phi, twisted_seed(5).phi)

    four = build_initial("four_perturbed", 7)
    assert four.n_clusters == 4
    assert np.isclose(four.x.sum(), 1.0)
    assert np.all(four.x > 0)
    assert not np.allclose(four.phi, twisted_seed(4).phi)


def test_random_phase_family_varies_with_seed() -> None:
    first = build_initial("random_phases", 1)
    second = build_initial("random_phases", 2)
    assert 3 <= first.n_clusters <= 8
    assert np.isclose(first.x.sum(), 1.0)
    assert np.all(first.x > 0)
    assert first.n_clusters != second.n_clusters or not np.allclose(first.phi, second.phi)


def test_unknown_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown initial family"):
        build_initial("does_not_exist", 0)


def test_register_initial_adds_family() -> None:
    family = INITIAL_FAMILIES["five_twisted"]
    registered = type(family)(
        "custom_five",
        "test-only alias",
        lambda rng: twisted_seed(5),
    )
    register_initial(registered)
    try:
        assert isinstance(build_initial("custom_five", 0), PhaseTemplate)
    finally:
        del INITIAL_FAMILIES["custom_five"]
