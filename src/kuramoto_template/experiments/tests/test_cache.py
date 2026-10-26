"""Canonical cache behavior: symmetry, accounting, and truncation."""

import numpy as np
import pytest

from kuramoto_template.mutations import perturb_phase
from kuramoto_template.search.random_search import twisted_seed
from kuramoto_template.template import PhaseTemplate

from ..cache import CachedEvaluator, canonical_key
from .support import FakeEvaluator


def rotated(template: PhaseTemplate, angle: float) -> PhaseTemplate:
    return PhaseTemplate(template.x, template.phi + angle)


def permuted(template: PhaseTemplate, order: list[int]) -> PhaseTemplate:
    return PhaseTemplate(template.x[order], template.phi[order])


def test_canonical_key_ignores_global_rotation_and_cluster_order() -> None:
    template = twisted_seed(5)
    order = [3, 0, 4, 1, 2]
    variants = [
        template,
        rotated(template, 0.37),
        rotated(template, -2.1),
        permuted(template, order),
        rotated(permuted(template, order), 1.25),
    ]
    keys = {canonical_key(variant) for variant in variants}
    assert len(keys) == 1


def test_canonical_key_distinguishes_geometry() -> None:
    template = twisted_seed(5)
    assert canonical_key(template) != canonical_key(perturb_phase(template, 0, 0.1))
    masses = template.x.copy()
    masses[0] += 0.01
    masses[1] -= 0.01
    assert canonical_key(template) != canonical_key(PhaseTemplate(masses, template.phi))
    assert canonical_key(template) != canonical_key(twisted_seed(6))


def test_cache_hit_marks_and_zeroes_solver_calls() -> None:
    inner = FakeEvaluator()
    cache = CachedEvaluator(inner)
    template = twisted_seed(5)
    variant = rotated(permuted(template, [2, 0, 3, 1, 4]), 0.9)

    first = cache.evaluate(template)
    second = cache.evaluate(variant)

    assert inner.calls == 1
    assert first.cached is False
    assert second.cached is True
    assert second.template is variant
    assert second.sdp_calls == 0
    assert second.stable_mu == first.stable_mu
    assert cache.stats.hits == 1 and cache.stats.misses == 1
    assert cache.stats.stored == 1
    assert cache.stats.hit_rate == 0.5


def test_cache_distinguishes_different_templates() -> None:
    inner = FakeEvaluator()
    cache = CachedEvaluator(inner)
    template = twisted_seed(5)
    other = perturb_phase(template, 1, 0.05)
    cache.evaluate(template)
    cache.evaluate(other)
    assert inner.calls == 2
    assert cache.stats.misses == 2


def test_allowed_patterns_bypass_cache() -> None:
    inner = FakeEvaluator()
    cache = CachedEvaluator(inner)
    template = twisted_seed(5)
    allowed = np.ones((5, 5), dtype=bool)
    cache.evaluate(template, allowed=allowed)
    cache.evaluate(template, allowed=allowed)
    assert inner.calls == 2
    assert cache.stats.hits == 0 and cache.stats.misses == 0
    assert cache.stats.stored == 0


def test_truncated_evaluation_is_not_stored() -> None:
    inner = FakeEvaluator()
    cache = CachedEvaluator(inner)
    template = twisted_seed(4)  # near-feasible, normally two calls

    truncated = cache.evaluate(template, max_solver_calls=1)
    assert truncated.status == "infeasible"
    assert truncated.sdp_calls == 1
    assert cache.stats.misses == 1 and cache.stats.stored == 0

    complete = cache.evaluate(template, max_solver_calls=2)
    assert complete.status == "near_feasible"
    assert complete.sdp_calls == 2
    assert cache.stats.misses == 2 and cache.stats.stored == 1

    hit = cache.evaluate(template, max_solver_calls=2)
    assert hit.cached is True and hit.sdp_calls == 0
    assert cache.stats.hits == 1


@pytest.mark.parametrize("decimals", [-1, 1.5, True])
def test_invalid_cache_precision(decimals) -> None:
    with pytest.raises(ValueError):
        CachedEvaluator(FakeEvaluator(), decimals=decimals)
    with pytest.raises(ValueError):
        canonical_key(twisted_seed(4), decimals=decimals)


def test_clear_resets_cache_and_stats() -> None:
    inner = FakeEvaluator()
    cache = CachedEvaluator(inner)
    template = twisted_seed(5)
    cache.evaluate(template)
    cache.evaluate(template)
    assert cache.stats.hits == 1
    cache.clear()
    cache.evaluate(template)
    assert inner.calls == 2
    assert cache.stats.hits == 0 and cache.stats.misses == 1
