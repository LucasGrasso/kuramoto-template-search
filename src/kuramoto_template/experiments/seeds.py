"""Initial-template families used by benchmarks.

Families are deliberately small and explicit. Each family is a deterministic
function of a NumPy generator, so a ``(master seed, family, repetition)`` triple
reproduces exactly one starting template. Add literature-specific constructions
through :func:`register_initial` rather than editing the benchmark runner.

The eight-cluster family is the equally spaced symmetric construction. It is a
useful reference point, not a claim that it is optimal or literature-derived.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from kuramoto_template.search.random_search import twisted_seed
from kuramoto_template.template import PhaseTemplate


@dataclass(frozen=True)
class InitialFamily:
    """One named way to draw a starting template."""

    name: str
    description: str
    build: Callable[[np.random.Generator], PhaseTemplate]


_NUMBER_WORDS = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def _name(prefix: str, n_clusters: int) -> str:
    word = _NUMBER_WORDS.get(n_clusters, str(n_clusters))
    return f"{word}_{prefix}"


def twisted(n_clusters: int) -> InitialFamily:
    """Equally spaced equal-mass phases (the classic twisted state)."""

    def build(_rng: np.random.Generator) -> PhaseTemplate:
        return twisted_seed(n_clusters)

    return InitialFamily(
        _name("twisted", n_clusters),
        f"{n_clusters} equally spaced equal-mass clusters",
        build,
    )


def perturbed_twisted(
    n_clusters: int,
    *,
    phase_scale: float = 0.08,
    mass_log_scale: float = 0.15,
) -> InitialFamily:
    """Twisted state with log-normal masses and Gaussian phase noise."""

    def build(rng: np.random.Generator) -> PhaseTemplate:
        base = twisted_seed(n_clusters)
        masses = base.x * np.exp(rng.normal(0.0, mass_log_scale, n_clusters))
        masses /= masses.sum()
        phases = base.phi + rng.normal(0.0, phase_scale, n_clusters)
        return PhaseTemplate(masses, phases)

    return InitialFamily(
        _name("perturbed", n_clusters),
        f"perturbed {n_clusters}-cluster twisted state",
        build,
    )


def random_phases(
    *, min_clusters: int = 3, max_clusters: int = 8
) -> InitialFamily:
    """Dirichlet masses and uniform phases with a sampled cluster count."""

    if min_clusters < 2 or max_clusters < min_clusters:
        raise ValueError("Require 2 <= min_clusters <= max_clusters")

    def build(rng: np.random.Generator) -> PhaseTemplate:
        n_clusters = int(rng.integers(min_clusters, max_clusters + 1))
        masses = rng.dirichlet(np.ones(n_clusters))
        phases = rng.uniform(0.0, 2.0 * np.pi, n_clusters)
        return PhaseTemplate(masses, phases)

    return InitialFamily(
        "random_phases",
        f"random masses and phases with {min_clusters}-{max_clusters} clusters",
        build,
    )


INITIAL_FAMILIES: dict[str, InitialFamily] = {
    family.name: family
    for family in (
        twisted(5),
        twisted(8),
        perturbed_twisted(4),
        random_phases(),
    )
}


def register_initial(family: InitialFamily) -> None:
    """Add or replace a family, mainly for notebooks and local experiments."""
    if not family.name:
        raise ValueError("Initial family name must be nonempty")
    INITIAL_FAMILIES[family.name] = family


def build_initial(name: str, seed: int) -> PhaseTemplate:
    """Build one initial template reproducibly."""
    try:
        family = INITIAL_FAMILIES[name]
    except KeyError as exc:
        known = ", ".join(sorted(INITIAL_FAMILIES))
        raise ValueError(f"Unknown initial family {name!r}; known: {known}") from exc
    return family.build(np.random.default_rng(seed))
