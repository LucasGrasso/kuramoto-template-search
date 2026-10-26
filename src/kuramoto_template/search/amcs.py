"""Fixed-depth nested rollouts on top of the shared search infrastructure.

Level 0 plays a random mutation sequence. Level L samples several children and
plays a level L-1 rollout from each, then selects a child by its rollout
endpoint. Every visited template passes through ``SearchSession``, so exact
deduplication, global SDP-call accounting, archive updates, JSONL tracing and
persistence are shared with random search.

The two ranking signals stay separate. Stable candidates are compared by
``stable_mu``. Exploratory candidates compare ``gamma`` only inside one
cluster-count bucket; across buckets we use archive representation and cluster
count, never raw gamma. This module is deliberately fixed-depth. Adaptive depth
and nesting levels are deferred to a later driver that can reuse the same
``_rollout`` primitive and session: keep one budget/archive, vary ``level`` and
``depth`` per episode after stalls, reset on a stable density improvement, and
fall back to ``_choose_root`` for archive/fresh restarts. Evaluation details
stay in ``TemplateEvaluator`` and never move into the search policy.
"""

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np

from kuramoto_template.template import PhaseTemplate

from .evaluator import EvaluationConfig, TemplateEvaluation, TemplateEvaluator
from .infrastructure import (
    ArchiveEntry,
    Evaluator,
    MutationProposal,
    SearchBudgetExhausted,
    SearchRecord,
    SearchResult,
    SearchSession,
)
from .random_search import (
    MutationSamplingConfig,
    restart_proposal,
    sample_mutation,
    twisted_seed,
)


@dataclass(frozen=True)
class NestedSearchConfig:
    """Controls for the fixed nested-rollout baseline.

    ``iterations`` counts top-level episodes, not proposals or SDP calls.
    Proposals and calls are counted globally by ``SearchSession``. ``level=0``
    is a random rollout; ``level=1`` evaluates each child through one lower
    rollout, and so on. Branching grows exponentially with level, so the SDP
    budget remains the primary safety control.
    """

    seed: int = 42
    min_clusters: int = 3
    max_clusters: int = 12
    archive_size: int = 32
    iterations: int = 8
    level: int = 1
    depth: int = 2
    branching: int = 8
    restart_probability: float = 0.1
    exploration_probability: float = 0.2
    parent_elite_probability: float = 0.5
    parent_promising_probability: float = 0.3
    parent_random_probability: float = 0.2
    max_sdp_calls: int | None = 256
    max_wall_seconds: float | None = None
    mutation_weights: tuple[float, float, float, float] = (1, 1, 1, 1)
    separation_min: float = 1e-4
    separation_max: float = 1e-1
    phase_scale: float = 0.05
    resize_fraction: float = 0.1
    max_invalid_attempts: int = 4

    def __post_init__(self) -> None:
        for name, lower in (
            ("seed", 0),
            ("min_clusters", 2),
            ("max_clusters", 2),
            ("archive_size", 1),
            ("iterations", 0),
            ("level", 0),
            ("depth", 0),
            ("branching", 1),
            ("max_invalid_attempts", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        if self.max_clusters < self.min_clusters:
            raise ValueError("max_clusters must be >= min_clusters")
        if self.max_sdp_calls is not None and (
            isinstance(self.max_sdp_calls, bool)
            or not isinstance(self.max_sdp_calls, int)
            or self.max_sdp_calls < 1
        ):
            raise ValueError("max_sdp_calls must be a positive integer")
        if self.max_wall_seconds is not None and (
            not np.isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0
        ):
            raise ValueError("max_wall_seconds must be finite and positive")
        for name in (
            "restart_probability",
            "exploration_probability",
            "parent_elite_probability",
            "parent_promising_probability",
            "parent_random_probability",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if (
            self.parent_elite_probability
            + self.parent_promising_probability
            + self.parent_random_probability
            <= 0
        ):
            raise ValueError(
                "At least one parent-selection probability must be positive"
            )
        for name in (
            "separation_min",
            "separation_max",
            "phase_scale",
            "resize_fraction",
        ):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.separation_min <= self.separation_max or self.phase_scale <= 0:
            raise ValueError("Separation bounds and phase_scale must be positive")
        if not 0 < self.resize_fraction < 1:
            raise ValueError("resize_fraction must be in (0, 1)")
        weights = np.asarray(self.mutation_weights)
        if (
            weights.shape != (4,)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or weights.sum() <= 0
        ):
            raise ValueError(
                "mutation_weights must contain four finite nonnegative weights "
                "with positive sum"
            )
        if (
            self.min_clusters == self.max_clusters
            and weights[2:].sum() == 0
            and self.depth > 0
        ):
            raise ValueError("No mutations available at fixed cluster count")


@dataclass(eq=False)
class EpisodeResult:
    """One top-level rollout and the path it selected, by trace record id.

    ``endpoint_id`` is the selected rollout endpoint used for nested decisions.
    ``best_id`` is the best record on this path under the separated ranking; it
    can differ from the global incumbent in ``search.best``.
    """

    index: int
    level: int
    depth: int
    branching: int
    root_id: int
    root_selection: str
    record_ids: list[int]
    endpoint_id: int
    best_id: int
    endpoint_status: str
    endpoint_stable_mu: float | None
    endpoint_margin_gamma: float | None
    endpoint_n_clusters: int
    best_stable_mu: float | None
    evaluations: int
    sdp_calls: int
    complete: bool
    message: str = ""


@dataclass(eq=False)
class NestedSearchResult:
    """Shared search result plus reviewable episode paths."""

    search: SearchResult
    episodes: list[EpisodeResult]
    config: NestedSearchConfig

    @property
    def best(self) -> ArchiveEntry | None:
        return self.search.best

    @property
    def archive(self) -> list[ArchiveEntry]:
        return self.search.archive

    @property
    def trace(self) -> list[SearchRecord]:
        return self.search.trace

    @property
    def evaluations(self) -> int:
        return self.search.evaluations

    @property
    def sdp_calls(self) -> int:
        return self.search.sdp_calls

    @property
    def proposals(self) -> int:
        return self.search.proposals

    @property
    def elapsed_seconds(self) -> float:
        return self.search.elapsed_seconds

    @property
    def stop_reason(self) -> str:
        return self.search.stop_reason

    @property
    def run_id(self) -> str:
        return self.search.run_id

    def episode_records(self, episode: EpisodeResult) -> list[SearchRecord]:
        """Resolve an episode's record ids against the shared trace."""
        return [self.search.trace[record_id] for record_id in episode.record_ids]


@dataclass
class _Rollout:
    records: list[SearchRecord]
    endpoint: SearchRecord
    best: SearchRecord
    complete: bool


class RolloutConfig(MutationSamplingConfig, Protocol):
    """Settings shared by fixed and adaptive nested rollouts.

    ``NestedSearchConfig`` satisfies this structurally. The adaptive driver
    passes its own subclass, so ``_rollout`` can be reused without copying the
    search machinery.
    """

    @property
    def level(self) -> int: ...

    @property
    def depth(self) -> int: ...

    @property
    def branching(self) -> int: ...

    @property
    def exploration_probability(self) -> float: ...

    @property
    def restart_probability(self) -> float: ...

    @property
    def parent_elite_probability(self) -> float: ...

    @property
    def parent_promising_probability(self) -> float: ...

    @property
    def parent_random_probability(self) -> float: ...

    @property
    def max_invalid_attempts(self) -> int: ...


@dataclass
class _Context:
    session: SearchSession
    config: RolloutConfig
    stop: bool = False
    stop_reason: str | None = None
    message: str = ""


def _record_for_entry(session: SearchSession, entry: ArchiveEntry) -> SearchRecord:
    if entry.id < 0 or entry.id >= len(session.trace):
        raise RuntimeError("Archive entry is not present in the current trace")
    record = session.trace[entry.id]
    if record.id != entry.id:
        raise RuntimeError("Archive entry id does not match its trace index")
    return record


def _try_record(
    ctx: _Context,
    proposal: MutationProposal,
    parent_id: int | None,
    selection: str,
) -> SearchRecord | None:
    if ctx.stop:
        return None
    reason = ctx.session.budget.stop_reason()
    if reason is not None:
        ctx.stop = True
        ctx.stop_reason = reason
        return None
    try:
        return ctx.session.record(proposal, parent_id, selection)
    except SearchBudgetExhausted:
        ctx.stop = True
        ctx.stop_reason = "sdp_budget"
        return None
    except Exception as exc:  # noqa: BLE001 - evaluator boundary must not corrupt the run
        ctx.stop = True
        ctx.stop_reason = "evaluator_error"
        ctx.message = f"{type(exc).__name__}: {exc}"
        return None


def _propose(
    ctx: _Context, parent: SearchRecord, selection: str
) -> SearchRecord | None:
    template = parent.template
    if template is None:
        raise RuntimeError("Cannot mutate a record without a template")
    for _ in range(ctx.config.max_invalid_attempts):
        if ctx.stop:
            return None
        proposal = sample_mutation(template, ctx.session.rng, ctx.config)
        record = _try_record(ctx, proposal, parent.id, selection)
        if record is None:
            return None
        if record.evaluation is not None and record.evaluation.status != "invalid":
            return record
    return None


def _select_best_record(records: Sequence[SearchRecord], ctx: _Context) -> SearchRecord:
    """Rank records without ever comparing gamma and mu.

    Stable candidates dominate and use stable_mu globally. Near-feasible
    candidates are compared only within one cluster count; bucket winners are
    then ordered by archive representation and cluster count. If neither signal
    exists, the most recent evaluated record is returned.
    """
    evaluated: list[tuple[SearchRecord, TemplateEvaluation]] = []
    for record in records:
        if record.evaluation is not None:
            evaluated.append((record, record.evaluation))
    if not evaluated:
        raise ValueError("Cannot rank records without evaluations")

    stable: list[tuple[SearchRecord, float]] = []
    for record, evaluation in evaluated:
        mu = evaluation.stable_mu
        if evaluation.status == "stable" and mu is not None and np.isfinite(mu):
            stable.append((record, float(mu)))
    if stable:
        return max(stable, key=lambda item: (item[1], -item[0].id))[0]

    buckets: dict[int, list[tuple[SearchRecord, float]]] = {}
    for record, evaluation in evaluated:
        gamma = evaluation.margin_gamma
        if (
            evaluation.status == "near_feasible"
            and gamma is not None
            and np.isfinite(gamma)
        ):
            m = evaluation.template.n_clusters
            buckets.setdefault(m, []).append((record, float(gamma)))
    if buckets:
        winners: list[tuple[SearchRecord, int]] = []
        for m, group in buckets.items():
            record = max(group, key=lambda item: (item[1], -item[0].id))[0]
            winners.append((record, m))
        counts = Counter(
            entry.evaluation.template.n_clusters
            for entry in ctx.session.archive.entries
        )
        return min(
            winners,
            key=lambda item: (counts[item[1]], item[1], item[0].id),
        )[0]

    return max(evaluated, key=lambda item: item[0].id)[0]


def _select_best_rollout(rollouts: Sequence[_Rollout], ctx: _Context) -> _Rollout:
    if not rollouts:
        raise ValueError("Cannot select from an empty rollout list")
    best_record = _select_best_record([rollout.endpoint for rollout in rollouts], ctx)
    for rollout in rollouts:
        if rollout.endpoint.id == best_record.id:
            return rollout
    raise RuntimeError("Selected endpoint is not present in any rollout")


def _rollout(ctx: _Context, start: SearchRecord, level: int, depth: int) -> _Rollout:
    """Play a rollout from an already evaluated record."""
    if ctx.stop:
        return _Rollout([start], start, start, complete=False)
    if depth <= 0 or level < 0:
        return _Rollout([start], start, start, complete=not ctx.stop)

    if level == 0:
        records = [start]
        current = start
        complete = True
        for _ in range(depth):
            if ctx.stop:
                complete = False
                break
            child = _propose(ctx, current, "nested_L0")
            if child is None:
                complete = False
                break
            records.append(child)
            current = child
        return _Rollout(records, current, _select_best_record(records, ctx), complete)

    rollouts: list[_Rollout] = []
    attempted = 0
    while attempted < ctx.config.branching:
        attempted += 1
        if ctx.stop:
            break
        child = _propose(ctx, start, f"nested_L{level}")
        if child is None:
            if ctx.stop:
                break
            continue
        rollouts.append(_rollout(ctx, child, level - 1, depth))
        if ctx.stop:
            break
    if not rollouts:
        return _Rollout([start], start, start, complete=False)

    best_rollout = _select_best_rollout(rollouts, ctx)
    completed = [rollout for rollout in rollouts if rollout.complete] or rollouts
    if (
        len(completed) > 1
        and ctx.session.rng.random() < ctx.config.exploration_probability
    ):
        chosen = completed[int(ctx.session.rng.integers(len(completed)))]
    else:
        chosen = best_rollout
    records = [start] + chosen.records
    return _Rollout(
        records,
        chosen.endpoint,
        _select_best_record(records, ctx),
        complete=chosen.complete and not ctx.stop and attempted >= ctx.config.branching,
    )


def _choose_root(ctx: _Context) -> tuple[SearchRecord, str] | None:
    """Sample a restart or an archive parent using separate signal groups."""
    rng = ctx.session.rng
    if rng.random() < ctx.config.restart_probability:
        record = _try_record(ctx, restart_proposal(rng, ctx.config), None, "restart")
        if record is not None and record.evaluation is not None:
            return record, "restart"
        if ctx.stop:
            return None

    entries = [
        entry
        for entry in ctx.session.archive.entries
        if entry.evaluation.status != "invalid"
    ]
    if not entries:
        return None

    groups: dict[str, list[ArchiveEntry]] = {
        "elite": [
            entry
            for entry in entries
            if entry.evaluation.status == "stable"
            and entry.evaluation.stable_mu is not None
        ],
        "promising": [
            entry
            for entry in entries
            if entry.evaluation.status == "near_feasible"
            and entry.evaluation.margin_gamma is not None
        ],
        "random": list(entries),
    }
    labels: list[str] = []
    weights: list[float] = []
    for label, weight in (
        ("elite", ctx.config.parent_elite_probability),
        ("promising", ctx.config.parent_promising_probability),
        ("random", ctx.config.parent_random_probability),
    ):
        if groups[label] and weight > 0:
            labels.append(label)
            weights.append(weight)
    if not labels:
        labels, weights = ["random"], [1.0]
    probabilities = np.asarray(weights, dtype=float)
    probabilities /= probabilities.sum()
    label = str(rng.choice(labels, p=probabilities))
    group = groups[label]
    entry = group[int(rng.integers(len(group)))]
    return _record_for_entry(ctx.session, entry), label


def _episode_result(
    index: int,
    root: SearchRecord,
    root_selection: str,
    rollout: _Rollout,
    ctx: _Context,
    evaluations_before: int,
    sdp_calls_before: int,
    *,
    level: int | None = None,
    depth: int | None = None,
    branching: int | None = None,
) -> EpisodeResult:
    endpoint_evaluation = rollout.endpoint.evaluation
    if endpoint_evaluation is None:
        raise RuntimeError("Rollout endpoint has no evaluation")
    stable_mus: list[float] = []
    for record in rollout.records:
        evaluation = record.evaluation
        if evaluation is None or evaluation.status != "stable":
            continue
        if evaluation.stable_mu is not None and np.isfinite(evaluation.stable_mu):
            stable_mus.append(float(evaluation.stable_mu))
    return EpisodeResult(
        index=index,
        level=ctx.config.level if level is None else level,
        depth=ctx.config.depth if depth is None else depth,
        branching=ctx.config.branching if branching is None else branching,
        root_id=root.id,
        root_selection=root_selection,
        record_ids=[record.id for record in rollout.records],
        endpoint_id=rollout.endpoint.id,
        best_id=rollout.best.id,
        endpoint_status=endpoint_evaluation.status,
        endpoint_stable_mu=(
            endpoint_evaluation.stable_mu
            if endpoint_evaluation.status == "stable"
            else None
        ),
        endpoint_margin_gamma=(
            endpoint_evaluation.margin_gamma
            if endpoint_evaluation.status == "near_feasible"
            else None
        ),
        endpoint_n_clusters=endpoint_evaluation.template.n_clusters,
        best_stable_mu=max(stable_mus) if stable_mus else None,
        evaluations=ctx.session.evaluations - evaluations_before,
        sdp_calls=ctx.session.budget.sdp_calls - sdp_calls_before,
        complete=rollout.complete,
        message=ctx.message if not rollout.complete else "",
    )


def _write_paths(
    session: SearchSession,
    result: SearchResult,
    episodes: list[EpisodeResult],
    config: NestedSearchConfig,
) -> None:
    if session.destination is None:
        return
    session.write_artifact(
        "paths.json",
        {
            "schema_version": 1,
            "run_id": result.run_id,
            "algorithm": "nested_rollouts_fixed",
            "level": config.level,
            "depth": config.depth,
            "branching": config.branching,
            "episodes": [asdict(episode) for episode in episodes],
        },
    )


def run_nested_search(
    seeds: list[PhaseTemplate],
    evaluator: Evaluator,
    config: NestedSearchConfig | None = None,
    *,
    output: str | Path | None = None,
) -> NestedSearchResult:
    """Run fixed-depth nested rollouts using one shared session and budget."""
    if config is None:
        config = NestedSearchConfig()
    with SearchSession(
        seeds,
        evaluator,
        seed=config.seed,
        min_clusters=config.min_clusters,
        max_clusters=config.max_clusters,
        archive_size=config.archive_size,
        max_sdp_calls=config.max_sdp_calls,
        max_wall_seconds=config.max_wall_seconds,
        output=output,
        search_config=dict(algorithm="nested_rollouts_fixed", **asdict(config)),
        clock=perf_counter,
    ) as session:
        ctx = _Context(session, config)
        try:
            session.evaluate_seeds()
        except SearchBudgetExhausted:
            result = session.finish("sdp_budget")
            _write_paths(session, result, [], config)
            return NestedSearchResult(result, [], config)

        episodes: list[EpisodeResult] = []
        stop_reason = "iterations"
        for index in range(config.iterations):
            if ctx.stop:
                stop_reason = ctx.stop_reason or "stopped"
                break
            reason = session.budget.stop_reason()
            if reason is not None:
                stop_reason = reason
                break
            chosen = _choose_root(ctx)
            if chosen is None:
                stop_reason = ctx.stop_reason or "no_root"
                break
            root, root_selection = chosen
            evaluations_before = session.evaluations
            sdp_calls_before = session.budget.sdp_calls
            rollout = _rollout(ctx, root, config.level, config.depth)
            episodes.append(
                _episode_result(
                    index,
                    root,
                    root_selection,
                    rollout,
                    ctx,
                    evaluations_before,
                    sdp_calls_before,
                )
            )
            if ctx.stop:
                stop_reason = ctx.stop_reason or "stopped"
                break

        result = session.finish(stop_reason)
        _write_paths(session, result, episodes, config)
        return NestedSearchResult(result, episodes, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--branching", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory (never overwritten)",
    )
    parser.add_argument("--target-mu", type=float, default=0.693)
    parser.add_argument("--min-clusters", type=int, default=3)
    parser.add_argument("--max-clusters", type=int, default=12)
    parser.add_argument("--archive-size", type=int, default=32)
    parser.add_argument("--restart-probability", type=float, default=0.1)
    parser.add_argument("--exploration-probability", type=float, default=0.2)
    parser.add_argument("--max-sdp-calls", type=int, default=256)
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--max-invalid-attempts", type=int, default=4)
    parser.add_argument("--internal-margin", type=float, default=1e-7)
    parser.add_argument("--cluster-margin", type=float, default=1e-8)
    parser.add_argument("--stability-tolerance", type=float, default=1e-10)
    parser.add_argument("--residual-tolerance", type=float, default=1e-7)
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument(
        "--solver-options",
        type=json.loads,
        default={},
        help="JSON object of conic solver settings",
    )
    args = parser.parse_args()
    try:
        config = NestedSearchConfig(
            seed=args.seed,
            iterations=args.iterations,
            level=args.level,
            depth=args.depth,
            branching=args.branching,
            min_clusters=args.min_clusters,
            max_clusters=args.max_clusters,
            archive_size=args.archive_size,
            restart_probability=args.restart_probability,
            exploration_probability=args.exploration_probability,
            max_sdp_calls=args.max_sdp_calls,
            max_wall_seconds=args.max_wall_seconds,
            max_invalid_attempts=args.max_invalid_attempts,
        )
        evaluator = TemplateEvaluator(
            EvaluationConfig(
                mu_target=args.target_mu,
                max_clusters=args.max_clusters,
                internal_margin=args.internal_margin,
                cluster_margin=args.cluster_margin,
                stability_tolerance=args.stability_tolerance,
                residual_tolerance=args.residual_tolerance,
                solver=args.solver,
                solver_options=args.solver_options,
            )
        )
        counts = [m for m in (5, 4) if config.min_clusters <= m <= config.max_clusters]
        seeds = [twisted_seed(m) for m in counts or [config.min_clusters]]
        result = run_nested_search(seeds, evaluator, config, output=args.output)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    statuses = Counter(
        record.evaluation.status
        for record in result.trace
        if record.evaluation is not None
    )
    print(
        json.dumps(
            {
                "best_mu": result.best.evaluation.stable_mu if result.best else None,
                "evaluations": result.evaluations,
                "sdp_calls": result.sdp_calls,
                "proposals": result.proposals,
                "episodes": len(result.episodes),
                "level": result.config.level,
                "depth": result.config.depth,
                "branching": result.config.branching,
                "elapsed_seconds": result.elapsed_seconds,
                "stop_reason": result.stop_reason,
                "statuses": dict(statuses),
            }
        )
    )
    print("Numerical candidates only; no rigorous certificate. Results:", args.output)


if __name__ == "__main__":
    main()
