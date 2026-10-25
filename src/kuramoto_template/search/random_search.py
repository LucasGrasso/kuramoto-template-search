"""Seeded archive search. CLI: uv run python -m kuramoto_template.search.random_search.

Iterations count proposal attempts, including rejected/duplicate proposals.
Seeds are always evaluated first (even duplicates), outside that proposal budget.
The SDP budget includes every density/fallback call, including exceptions.
Wall-time limits are checked between evaluations, after mandatory seed evaluation;
they cannot interrupt an in-flight conic solve. Output directories must be new.

Archive, accounting, and persistence live in infrastructure.SearchSession.
Future nested rollouts can share one session while replacing this proposal loop.
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np

from kuramoto_template.mutations import merge_clusters, perturb_phase, resize_clusters, split_cluster
from kuramoto_template.template import PhaseTemplate
from .evaluator import EvaluationConfig, TemplateEvaluator
from .infrastructure import (
    ArchiveEntry as ArchiveEntry,
    DiversityArchive as DiversityArchive,
    Evaluator as Evaluator,
    MutationProposal as MutationProposal,
    SearchRecord as SearchRecord,
    SearchResult as SearchResult,
    SearchSession,
    evaluation_data as evaluation_data,
)


class MutationSamplingConfig(Protocol):
    """Structural settings required by the shared mutation sampler.

    RandomSearchConfig and NestedSearchConfig both provide these fields, so the
    sampler can be reused without constructing a fake search configuration.
    Read-only properties also accept frozen configuration dataclasses.
    """

    @property
    def min_clusters(self) -> int: ...

    @property
    def max_clusters(self) -> int: ...

    @property
    def mutation_weights(self) -> tuple[float, float, float, float]: ...

    @property
    def separation_min(self) -> float: ...

    @property
    def separation_max(self) -> float: ...

    @property
    def phase_scale(self) -> float: ...

    @property
    def resize_fraction(self) -> float: ...


@dataclass(frozen=True)
class RandomSearchConfig:
    iterations: int = 1000
    seed: int = 42
    min_clusters: int = 3
    max_clusters: int = 12
    archive_size: int = 32
    mutation_weights: tuple[float, float, float, float] = (1, 1, 1, 1)
    restart_probability: float = 0.1
    separation_min: float = 1e-4
    separation_max: float = 1e-1
    phase_scale: float = 0.05
    resize_fraction: float = 0.1
    max_sdp_calls: int | None = None
    max_wall_seconds: float | None = None
    parent_selection: str = "uniform_archive"

    def __post_init__(self) -> None:
        for name, lower in (("iterations", 0), ("seed", 0), ("min_clusters", 2),
                            ("max_clusters", 2), ("archive_size", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        if self.max_clusters < self.min_clusters:
            raise ValueError("max_clusters must be >= min_clusters")
        if self.parent_selection != "uniform_archive":
            raise ValueError("Only uniform_archive parent selection is supported")
        if self.max_sdp_calls is not None and (
            isinstance(self.max_sdp_calls, bool) or not isinstance(self.max_sdp_calls, int) or self.max_sdp_calls < 1
        ):
            raise ValueError("max_sdp_calls must be a positive integer")
        if self.max_wall_seconds is not None and (not np.isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0):
            raise ValueError("max_wall_seconds must be finite and positive")
        for name in ("restart_probability", "separation_min", "separation_max", "phase_scale", "resize_fraction"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.restart_probability <= 1:
            raise ValueError("restart_probability must be in [0, 1]")
        if not 0 < self.separation_min <= self.separation_max or self.phase_scale <= 0:
            raise ValueError("Separation bounds and phase_scale must be positive")
        if not 0 < self.resize_fraction < 1:
            raise ValueError("resize_fraction must be in (0, 1)")
        weights = np.asarray(self.mutation_weights)
        if weights.shape != (4,) or not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("mutation_weights must contain four finite nonnegative weights with positive sum")
        if self.min_clusters == self.max_clusters and weights[2:].sum() == 0 and self.restart_probability < 1:
            raise ValueError("No mutations available at fixed cluster count")



def sample_mutation(
    template: PhaseTemplate, rng: np.random.Generator, config: MutationSamplingConfig,
) -> MutationProposal:
    """Thin sampling adapter over existing pure mutation operations."""
    m = template.n_clusters
    names = ("split", "merge", "perturb", "resize")
    weights = np.array(config.mutation_weights, dtype=float)
    if m >= config.max_clusters:
        weights[0] = 0
    if m <= config.min_clusters:
        weights[1] = 0
    if weights.sum() == 0:
        return MutationProposal("unavailable", {}, message="No permitted mutation has positive weight")
    operation = str(rng.choice(names, p=weights / weights.sum()))
    i, j = (int(v) for v in rng.choice(m, size=2, replace=False))
    params: dict[str, float | int]
    if operation == "split":
        params = dict(index=i, ratio=float(rng.uniform(0.2, 0.8)),
                      separation=float(rng.choice([-1, 1]) * np.exp(rng.uniform(np.log(config.separation_min), np.log(config.separation_max)))))
    elif operation == "merge":
        params = dict(i=i, j=j)
    elif operation == "perturb":
        params = dict(index=i, delta=float(rng.normal(scale=config.phase_scale)))
    else:
        params = dict(source=i, target=j, amount=float(template.x[i] * rng.uniform(0.01 * config.resize_fraction, config.resize_fraction)))
    proposal = MutationProposal(operation, params)
    try:
        # Explicit dispatch keeps integer indices typed as integers.
        if operation == "split":
            proposal.template = split_cluster(template, i, float(params["ratio"]), float(params["separation"]))
        elif operation == "merge":
            proposal.template = merge_clusters(template, i, j)
        elif operation == "perturb":
            proposal.template = perturb_phase(template, i, float(params["delta"]))
        else:
            proposal.template = resize_clusters(template, i, j, float(params["amount"]))
    except (ValueError, FloatingPointError) as exc:
        proposal.message = f"{type(exc).__name__}: {exc}"
    return proposal



def twisted_seed(n_clusters: int = 5) -> PhaseTemplate:
    """Equally spaced phases and masses; no edge-pattern restrictions implied."""
    return PhaseTemplate(np.full(n_clusters, 1 / n_clusters), np.arange(n_clusters) * (2 * np.pi / n_clusters))



def restart_proposal(
    rng: np.random.Generator, config: MutationSamplingConfig,
) -> MutationProposal:
    """Sample a fresh template, alternating twisted and unstructured seeds.

    This is an exploration policy, not a mutation. Nested search reuses it so
    both baselines draw restarts from the same distribution.
    """
    m = int(rng.integers(config.min_clusters, config.max_clusters + 1))
    kind = int(rng.integers(2))
    template = twisted_seed(m) if kind == 0 else PhaseTemplate(rng.dirichlet(np.ones(m)), rng.uniform(0, 2 * np.pi, m))
    return MutationProposal("restart", {"n_clusters": m, "twisted": kind == 0}, template)



def run_random_search(
    seeds: list[PhaseTemplate], evaluator: Evaluator, config: RandomSearchConfig = RandomSearchConfig(),
    *, output: str | Path | None = None,
) -> SearchResult:
    """Run uniform archive search using shared recording and budget infrastructure.

    Seed evaluations are mandatory and outside the proposal-attempt budget.
    Persistence and exact-data deduplication are handled by SearchSession.
    """
    with SearchSession(
        seeds, evaluator, seed=config.seed,
        min_clusters=config.min_clusters, max_clusters=config.max_clusters,
        archive_size=config.archive_size, max_sdp_calls=config.max_sdp_calls,
        max_wall_seconds=config.max_wall_seconds, output=output,
        search_config=asdict(config), clock=perf_counter,
    ) as session:
        session.evaluate_seeds()
        stop_reason = "iterations"
        for _ in range(config.iterations):
            reason = session.budget.stop_reason()
            if reason is not None:
                stop_reason = reason
                break
            if session.rng.random() < config.restart_probability:
                session.record(restart_proposal(session.rng, config), None, "restart")
            else:
                parent = session.archive.entries[int(session.rng.integers(len(session.archive.entries)))]
                session.record(sample_mutation(parent.evaluation.template, session.rng, config),
                               parent.id, config.parent_selection)
        return session.finish(stop_reason)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True, help="New output directory (never overwritten)")
    parser.add_argument("--target-mu", type=float, default=0.693)
    parser.add_argument("--min-clusters", type=int, default=3)
    parser.add_argument("--max-clusters", type=int, default=12)
    parser.add_argument("--archive-size", type=int, default=32)
    parser.add_argument("--max-sdp-calls", type=int)
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--internal-margin", type=float, default=1e-7)
    parser.add_argument("--cluster-margin", type=float, default=1e-8)
    parser.add_argument("--stability-tolerance", type=float, default=1e-10)
    parser.add_argument("--residual-tolerance", type=float, default=1e-7)
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--solver-options", type=json.loads, default={}, help="JSON object of conic solver settings")
    args = parser.parse_args()
    try:
        config = RandomSearchConfig(iterations=args.iterations, seed=args.seed, min_clusters=args.min_clusters,
                                    max_clusters=args.max_clusters, archive_size=args.archive_size,
                                    max_sdp_calls=args.max_sdp_calls, max_wall_seconds=args.max_wall_seconds)
        evaluator = TemplateEvaluator(EvaluationConfig(mu_target=args.target_mu, max_clusters=args.max_clusters,
                                       internal_margin=args.internal_margin, cluster_margin=args.cluster_margin,
                                       stability_tolerance=args.stability_tolerance, residual_tolerance=args.residual_tolerance,
                                       solver=args.solver, solver_options=args.solver_options))
        counts = [m for m in (5, 4) if config.min_clusters <= m <= config.max_clusters]
        seeds = [twisted_seed(m) for m in counts or [config.min_clusters]]
        result = run_random_search(seeds, evaluator, config, output=args.output)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    statuses = Counter(r.evaluation.status for r in result.trace if r.evaluation is not None)
    print(json.dumps(dict(best_mu=result.best.evaluation.stable_mu if result.best else None,
                          evaluations=result.evaluations, sdp_calls=result.sdp_calls, proposals=result.proposals,
                          elapsed_seconds=result.elapsed_seconds, stop_reason=result.stop_reason, statuses=dict(statuses))))
    print("Numerical candidates only; no rigorous certificate. Results:", args.output)



if __name__ == "__main__":
    main()
