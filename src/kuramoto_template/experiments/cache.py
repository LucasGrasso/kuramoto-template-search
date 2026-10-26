"""Canonicalized evaluation cache used by benchmarks.

Why this exists
---------------
Both SDPs depend on phases only through differences ``phi_i - phi_j``. Rotating
every phase by the same constant and permuting clusters therefore describe the
same optimization problem. The search algorithms still treat those templates as
distinct (exact float64 deduplication), so benchmarks can spend the same solver
work on equivalent templates. This cache lets a benchmark reuse the previous
answer and spend its SDP budget elsewhere.

Scope and caveats
-----------------
* This is a *benchmark* device, not part of a search policy. Search algorithms
  do not import it.
* Keys are quantized, so templates closer than the quantum can collide. The
  default quantum is tiny (12 decimal places), but this is screening-level
  equivalence, never certification.
* A cache hit maps ``P`` from the representative back to the requested cluster
  order. Quantized reuse remains screening only; refinement must re-solve.
* Evaluations truncated by ``max_solver_calls`` are returned but not stored, so
  a later call with a larger cap can still run the margin fallback.
"""

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from kuramoto_template.search.evaluator import EvaluationConfig, TemplateEvaluation
from kuramoto_template.template import PhaseTemplate
from kuramoto_template.sdp_optimizer.validate_input import in_open_semicircle, validate_input

_TWO_PI = 2.0 * np.pi


class CacheableEvaluator(Protocol):
    """Evaluator signature accepted by :class:`CachedEvaluator`."""

    config: EvaluationConfig

    def evaluate(
        self,
        template: PhaseTemplate,
        *,
        allowed: NDArray[np.bool_] | None = None,
        max_solver_calls: int | None = None,
    ) -> TemplateEvaluation: ...


@dataclass
class CacheStats:
    """Operational counters for one benchmark run."""

    hits: int = 0
    misses: int = 0
    stored: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stored": self.stored,
            "lookups": self.lookups,
            "hit_rate": self.hit_rate,
        }


def _quantize(value: float, decimals: int) -> float:
    rounded = round(float(value), decimals)
    return 0.0 if rounded == 0.0 else rounded


def canonical_key(
    template: PhaseTemplate, *, decimals: int = 12
) -> tuple[int, tuple[tuple[float, float], ...]]:
    """Return an approximately rotation- and permutation-invariant key.

    For every cluster we rotate the template so that cluster sits at phase 0,
    sort the resulting ``(mass, phase)`` pairs, and quantize them. The smallest
    serialized candidate over all anchors is invariant to both symmetries.
    """
    return _canonical_form(template, decimals=decimals)[0]


def _canonical_form(template: PhaseTemplate, *, decimals: int):
    """Key and canonical-to-input permutation, retaining cluster identities."""
    if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0:
        raise ValueError("decimals must be a nonnegative integer")
    x = np.asarray(template.x, dtype=np.float64)
    phi = np.asarray(template.phi, dtype=np.float64)
    best: tuple[int, tuple[tuple[float, float], ...]] | None = None
    best_order = None
    for anchor in range(x.size):
        shifted = np.remainder(phi - phi[anchor], _TWO_PI)
        pairs: list[tuple[float, float]] = []
        for mass, angle in zip(x, shifted):
            quantized_angle = _quantize(float(angle), decimals)
            if quantized_angle >= _TWO_PI:
                quantized_angle = 0.0
            pairs.append((_quantize(float(mass), decimals), quantized_angle))
        order = np.array(sorted(range(x.size), key=pairs.__getitem__), dtype=int)
        pairs = [pairs[i] for i in order]
        key = (int(x.size), tuple(pairs))
        if best is None or key < best:
            best = key
            best_order = order
    if best is None:  # pragma: no cover - PhaseTemplate always has clusters
        raise ValueError("Cannot canonicalize an empty template")
    return best, best_order


def _is_complete(evaluation: TemplateEvaluation, max_solver_calls: int | None) -> bool:
    """Whether an evaluation used every solver call its completeness needs."""
    if max_solver_calls is None:
        return True
    if evaluation.sdp_calls >= 2:
        return True
    if evaluation.status in ("stable", "invalid"):
        return True
    return evaluation.sdp_calls < max_solver_calls


def _roundoff_bucket(template: PhaseTemplate):
    """Cheap invariant index; detailed matching still checks full geometry."""
    diff = template.phi[:, None] - template.phi[None, :]
    pairs = np.cos(diff)[np.triu_indices(template.n_clusters, 1)]
    return (template.n_clusters, tuple(np.sort(np.round(template.x, 8))),
            tuple(np.sort(np.round(pairs, 8))))


class CachedEvaluator:
    """Memoize evaluations by canonical template key.

    ``stats.misses`` counts direct evaluator calls. ``stats.stored`` can be
    lower because budget-truncated evaluations are deliberately not cached.
    """

    def __init__(self, evaluator: CacheableEvaluator, *, decimals: int = 12):
        if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0:
            raise ValueError("decimals must be a nonnegative integer")
        self.evaluator = evaluator
        self.config = evaluator.config
        self.decimals = decimals
        self.stats = CacheStats()
        self._entries: dict[
            tuple[int, tuple[tuple[float, float], ...]], tuple[TemplateEvaluation, NDArray]
        ] = {}
        self._roundoff_buckets: dict[tuple, list[tuple[TemplateEvaluation, NDArray]]] = {}

    def evaluate(
        self,
        template: PhaseTemplate,
        *,
        allowed: NDArray[np.bool_] | None = None,
        max_solver_calls: int | None = None,
    ) -> TemplateEvaluation:
        if max_solver_calls is not None and max_solver_calls < 1:
            raise ValueError("max_solver_calls must be positive")
        # Restricting edges changes the optimization problem, so those calls
        # bypass the symmetry cache and are never stored.
        if allowed is not None:
            return self.evaluator.evaluate(
                template, allowed=allowed, max_solver_calls=max_solver_calls
            )

        # Validity is checked on the actual input before a quantized cache
        # lookup. A collision must never substitute a result for invalid input.
        try:
            validate_input(template.x, template.phi)
        except ValueError:
            self.stats.misses += 1
            return self.evaluator.evaluate(template, max_solver_calls=max_solver_calls)

        key, order = _canonical_form(template, decimals=self.decimals)
        entry = self._entries.get(key)
        obstruction = in_open_semicircle(template.phi)
        if entry is not None and in_open_semicircle(entry[0].template.phi) != obstruction:
            entry = None
        if entry is not None:
            cached, cached_order = entry
        else:
            cached = None
            # Floating subtraction near decimal rounding boundaries can change
            # a key under an exact rotation. Recover those hits only after
            # checking unquantized geometry to roundoff, never by coarsening
            # the requested key or merging merely close phase templates.
            for candidate, candidate_order in self._roundoff_buckets.get(_roundoff_bucket(template), []):
                if in_open_semicircle(candidate.template.phi) != obstruction:
                    continue
                mapping = np.empty(template.n_clusters, dtype=int)
                mapping[order] = candidate_order
                if not np.allclose(template.x, candidate.template.x[mapping], atol=64*np.finfo(float).eps, rtol=0):
                    continue
                delta = template.phi - candidate.template.phi[mapping]
                relative = np.angle(np.exp(1j * (delta - delta[0])))
                if np.max(np.abs(relative)) <= 64*np.finfo(float).eps:
                    cached, cached_order = candidate, candidate_order
                    break
        if cached is not None:
            self.stats.hits += 1
            mapping = np.empty(template.n_clusters, dtype=int)
            mapping[order] = cached_order

            def remap(diagnostic):
                if diagnostic is None:
                    return None
                return replace(diagnostic, residuals=dict(diagnostic.residuals),
                               P=(None if diagnostic.P is None else diagnostic.P[np.ix_(mapping, mapping)].copy()))

            message = cached.message + " | cached" if cached.message else "cached"
            return replace(
                cached,
                template=template,
                sdp_calls=0,
                elapsed_seconds=0.0,
                message=message,
                cached=True,
                density=remap(cached.density),
                margin=remap(cached.margin),
            )

        result = self.evaluator.evaluate(
            template, allowed=None, max_solver_calls=max_solver_calls
        )
        self.stats.misses += 1
        if result.status != "invalid" and _is_complete(result, max_solver_calls):
            self._entries[key] = (result, order)
            self._roundoff_buckets.setdefault(_roundoff_bucket(template), []).append((result, order))
            self.stats.stored += 1
        return result

    def clear(self) -> None:
        self._entries.clear()
        self._roundoff_buckets.clear()
        self.stats = CacheStats()
