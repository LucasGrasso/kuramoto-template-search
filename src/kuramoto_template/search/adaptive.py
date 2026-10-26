"""Adaptive nested rollouts with stagnation-driven depth, nesting and pruning.

This driver reuses the fixed-depth rollout primitive in :mod:`search.amcs` and
the shared :class:`SearchSession`. It adds three things the fixed baseline
lacks:

* a controller that raises search depth and nesting level when the best stable
  density stalls, and resets both as soon as a stable improvement appears;
* structural pruning proposals (merge the closest clusters, dissolve the
  smallest cluster, or undo a recent split) that need not preserve stability;
* two explicit signal archives, one ranked by strict stable ``mu`` and one
  ranked by target-margin ``gamma`` within a cluster-count bucket.

Pruning is only a proposal mechanism. Every pruned template is re-evaluated by
the normal evaluator and can never be treated as a valid witness without that
evaluation. ``gamma`` is never compared with ``mu``.

All outcomes remain numerical evidence, not mathematical certificates.
"""

import argparse
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np

from kuramoto_template.mutations import (
    dissolve_cluster,
    merge_clusters,
    perturb_phase,
)
from kuramoto_template.template import PhaseTemplate

from .amcs import (
    NestedSearchConfig,
    _Context,
    _episode_result,
    _record_for_entry,
    _rollout,
    _try_record,
)
from .evaluator import EvaluationConfig, TemplateEvaluator
from .infrastructure import (
    ArchiveEntry,
    Evaluator,
    MutationProposal,
    SearchBudgetExhausted,
    SearchRecord,
    SearchResult,
    SearchSession,
)
from .random_search import restart_proposal, twisted_seed

_MU_TOLERANCE = 1e-12
_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class AdaptiveSearchConfig(NestedSearchConfig):
    """Nested-rollout settings plus adaptation and pruning controls.

    ``level`` and ``depth`` are the baseline (and reset) values. On a stall the
    controller alternates between raising ``depth`` and raising ``level`` up to
    ``max_depth``/``max_level``; at the caps it returns to the baseline while
    keeping a high pruning probability. Any new best stable ``mu`` resets the
    schedule. ``prune_probability`` starts the pruning rate and
    ``prune_increment`` raises it after each escalation.
    """

    max_level: int = 3
    max_depth: int = 5
    stall_threshold: int = 2
    prune_probability: float = 0.2
    prune_increment: float = 0.2
    prune_phase_scale: float = 0.05
    stable_archive_size: int = 16
    promising_archive_size: int = 16
    promising_per_cluster: int = 3
    undo_split_window: int = 64
    prune_sources: int = 3

    def __post_init__(self) -> None:
        super().__post_init__()
        for name, lower in (
            ("max_level", 0),
            ("max_depth", 0),
            ("stall_threshold", 1),
            ("stable_archive_size", 1),
            ("promising_archive_size", 1),
            ("promising_per_cluster", 1),
            ("undo_split_window", 1),
            ("prune_sources", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        if self.max_level < self.level:
            raise ValueError("max_level must be >= the baseline level")
        if self.max_depth < self.depth:
            raise ValueError("max_depth must be >= the baseline depth")
        for name in (
            "prune_probability",
            "prune_increment",
            "prune_phase_scale",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.prune_probability > 1:
            raise ValueError("prune_probability must be <= 1")
        if self.prune_phase_scale <= 0:
            raise ValueError("prune_phase_scale must be positive")


@dataclass
class AdaptiveArchives:
    """Two separate bounded views over evaluated trace records.

    ``stable`` keeps the best strictly stable templates by ``mu``. ``promising``
    keeps the best near-feasible templates *per cluster count*, ranked only by
    ``gamma`` inside that bucket. Neither list is used as a witness; they only
    guide root sampling and pruning.
    """

    stable_capacity: int = 16
    promising_capacity: int = 16
    promising_per_cluster: int = 3
    stable: list[SearchRecord] = field(default_factory=list)
    promising: dict[int, list[SearchRecord]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stable_capacity < 1 or self.promising_capacity < 1:
            raise ValueError("Archive capacities must be positive")
        if self.promising_per_cluster < 1:
            raise ValueError("promising_per_cluster must be positive")

    def observe(self, records: Sequence[SearchRecord]) -> None:
        for record in records:
            self._consider(record)

    def _consider(self, record: SearchRecord) -> None:
        evaluation = record.evaluation
        if evaluation is None:
            return
        if (
            evaluation.status == "stable"
            and evaluation.stable_mu is not None
            and np.isfinite(evaluation.stable_mu)
        ):
            if all(item.id != record.id for item in self.stable):
                self.stable.append(record)
            self.stable.sort(
                key=lambda item: (
                    item.evaluation.stable_mu
                    if item.evaluation is not None
                    and item.evaluation.stable_mu is not None
                    else -np.inf
                ),
                reverse=True,
            )
            del self.stable[self.stable_capacity :]
        if (
            evaluation.status == "near_feasible"
            and evaluation.margin_gamma is not None
            and np.isfinite(evaluation.margin_gamma)
        ):
            m = evaluation.template.n_clusters
            bucket = self.promising.setdefault(m, [])
            if all(item.id != record.id for item in bucket):
                bucket.append(record)
            bucket.sort(
                key=lambda item: (
                    item.evaluation.margin_gamma
                    if item.evaluation is not None
                    and item.evaluation.margin_gamma is not None
                    else -np.inf
                ),
                reverse=True,
            )
            del bucket[self.promising_per_cluster :]
        self._trim_promising()

    def _trim_promising(self) -> None:
        total = sum(len(bucket) for bucket in self.promising.values())
        if total <= self.promising_capacity:
            return
        # Capacity pressure removes the weakest entry (already sorted last)
        # from the most populated bucket. Buckets are never ranked by raw gamma
        # against each other; only within-bucket order is a gamma comparison.
        while sum(len(bucket) for bucket in self.promising.values()) > (
            self.promising_capacity
        ):
            m = max(self.promising, key=lambda key: (len(self.promising[key]), key))
            bucket = self.promising[m]
            bucket.pop()
            if not bucket:
                del self.promising[m]

    def promising_records(self) -> list[SearchRecord]:
        return [record for m in sorted(self.promising) for record in self.promising[m]]


@dataclass(eq=False)
class AdaptiveEpisode:
    """One adaptive episode and the controller decision that followed it."""

    index: int
    level: int
    depth: int
    branching: int
    root_id: int
    source_root_id: int
    root_selection: str
    prune_probability: float
    prune_operation: str | None
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
    incumbent_mu_before: float | None
    incumbent_mu_after: float | None
    improved: bool
    stall_count: int
    decision: str
    message: str = ""


@dataclass(eq=False)
class AdaptiveSearchResult:
    """Shared search result plus adaptation history and signal archives."""

    search: SearchResult
    episodes: list[AdaptiveEpisode]
    config: AdaptiveSearchConfig
    archives: AdaptiveArchives

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

    def episode_records(self, episode: AdaptiveEpisode) -> list[SearchRecord]:
        return [self.search.trace[record_id] for record_id in episode.record_ids]


class AdaptiveController:
    """Stagnation-driven search-depth and nesting-level schedule."""

    def __init__(
        self,
        config: AdaptiveSearchConfig,
        *,
        incumbent_mu: float | None = None,
    ) -> None:
        self.config = config
        self.level = config.level
        self.depth = config.depth
        self.stall = 0
        self.escalations = 0
        self.incumbent_mu = incumbent_mu
        self.prune_probability = config.prune_probability

    def after_episode(self, current_best_mu: float | None) -> str:
        """Update the schedule and return a short decision label.

        A strict stable improvement resets depth, level and pruning. Otherwise
        the episode counts toward a stall; at ``stall_threshold`` the controller
        alternates deepening and nesting, then cycles back to the baseline at
        the caps while keeping the raised pruning rate.
        """
        improved = current_best_mu is not None and (
            self.incumbent_mu is None
            or current_best_mu > self.incumbent_mu + _MU_TOLERANCE
        )
        if improved:
            self.incumbent_mu = current_best_mu
            self.stall = 0
            self.escalations = 0
            self.level = self.config.level
            self.depth = self.config.depth
            self.prune_probability = self.config.prune_probability
            return "reset"

        self.stall += 1
        if self.stall < self.config.stall_threshold:
            return "hold"

        self.stall = 0
        self.escalations += 1
        if self.depth < self.config.max_depth and (
            self.escalations % 2 == 1 or self.level >= self.config.max_level
        ):
            self.depth = min(self.config.max_depth, self.depth + 1)
            decision = "deepen"
        elif self.level < self.config.max_level:
            self.level = min(self.config.max_level, self.level + 1)
            decision = "nest"
        else:
            self.level = self.config.level
            self.depth = self.config.depth
            decision = "restart_schedule"
        self.prune_probability = min(
            1.0,
            self.config.prune_probability
            + self.config.prune_increment * self.escalations,
        )
        return decision


def _phase_gap(a: float, b: float) -> float:
    gap = float(abs((a - b) % _TWO_PI))
    return min(gap, _TWO_PI - gap)


def _closest_pair(template: PhaseTemplate) -> tuple[int, int]:
    best: tuple[int, int] | None = None
    best_gap = np.inf
    for i in range(template.n_clusters):
        for j in range(i + 1, template.n_clusters):
            gap = _phase_gap(float(template.phi[i]), float(template.phi[j]))
            if gap < best_gap:
                best_gap = gap
                best = (i, j)
    if best is None:  # pragma: no cover - templates always have >= 2 clusters
        raise ValueError("No cluster pair available")
    return best


def _latest_split(ctx: _Context, window: int) -> SearchRecord | None:
    for record in reversed(ctx.session.trace[-window:]):
        if record.operation == "split" and record.template is not None:
            return record
    return None


def _valid_session_records(ctx: _Context) -> list[SearchRecord]:
    return [
        _record_for_entry(ctx.session, entry)
        for entry in ctx.session.archive.entries
        if entry.evaluation.status != "invalid"
    ]


def _prune_sources(
    ctx: _Context, archives: AdaptiveArchives, limit: int
) -> list[SearchRecord]:
    """Sample a few distinct pruning sources, favouring promising templates."""
    rng = ctx.session.rng
    groups: list[tuple[str, list[SearchRecord], float]] = [
        ("promising", archives.promising_records(), 0.6),
        ("stable", list(archives.stable), 0.2),
        ("diverse", _valid_session_records(ctx), 0.2),
    ]
    sources: list[SearchRecord] = []
    seen: set[int] = set()
    for _ in range(limit):
        available = [(label, pool, weight) for label, pool, weight in groups if pool]
        available = [
            (label, pool, weight)
            for label, pool, weight in available
            if any(record.id not in seen for record in pool)
        ]
        if not available:
            break
        weights = np.asarray([weight for _, _, weight in available], dtype=float)
        weights /= weights.sum()
        _, pool, _ = available[int(rng.choice(len(available), p=weights))]
        candidates = [record for record in pool if record.id not in seen]
        record = candidates[int(rng.integers(len(candidates)))]
        seen.add(record.id)
        sources.append(record)
    return sources


def _pruning_proposal(
    ctx: _Context, config: AdaptiveSearchConfig, archives: AdaptiveArchives
) -> MutationProposal:
    """Build one structural pruning proposal; never evaluate it here.

    Candidate templates come from recent archive entries. Every operation can
    fail or leave the cluster bounds; failures are collected and the next plan
    is tried. The returned proposal is always re-evaluated by the caller.
    """
    rng = ctx.session.rng
    plans: list[tuple[str, dict[str, float | int], Callable[[], PhaseTemplate]]] = []
    minimum = max(2, config.min_clusters)
    for source in _prune_sources(ctx, archives, config.prune_sources):
        template = source.template
        if template is None or template.n_clusters <= minimum:
            continue
        i, j = _closest_pair(template)
        plans.append(
            (
                "prune_merge",
                {"i": i, "j": j, "source_id": source.id},
                lambda template=template, i=i, j=j: merge_clusters(template, i, j),
            )
        )
        smallest = int(np.argmin(template.x))
        plans.append(
            (
                "prune_dissolve",
                {"index": smallest, "source_id": source.id},
                lambda template=template, index=smallest: dissolve_cluster(
                    template, index
                ),
            )
        )

    split = _latest_split(ctx, config.undo_split_window)
    if (
        split is not None
        and split.template is not None
        and split.template.n_clusters - 1 >= config.min_clusters
    ):
        split_template = split.template
        split_index = int(split.parameters.get("index", 0))
        if 0 <= split_index < split_template.n_clusters - 1:
            undo_parameters: dict[str, float | int] = {
                "split_id": split.id, "index": split_index,
            }

            def undo_split(
                template: PhaseTemplate = split_template, index: int = split_index,
                parameters: dict[str, float | int] = undo_parameters,
            ) -> PhaseTemplate:
                merged = merge_clusters(template, index, index + 1)
                perturb_index = int(rng.integers(merged.n_clusters))
                delta = float(rng.normal(scale=config.prune_phase_scale))
                parameters.update(perturb_index=perturb_index, delta=delta)
                return perturb_phase(merged, perturb_index, delta)

            plans.append(
                (
                    "prune_undo_split",
                    undo_parameters,
                    undo_split,
                )
            )

    rng.shuffle(plans)
    failures: list[str] = []
    for operation, parameters, build in plans:
        try:
            template = build()
        except (ValueError, FloatingPointError) as exc:
            failures.append(f"{operation}: {type(exc).__name__}: {exc}")
            continue
        if not config.min_clusters <= template.n_clusters <= config.max_clusters:
            failures.append(f"{operation}: cluster count outside search bounds")
            continue
        return MutationProposal(operation, parameters, template)
    message = "; ".join(failures[-3:]) if failures else "no pruning plan available"
    return MutationProposal("prune_unavailable", {}, None, message)


def _choose_adaptive_root(
    ctx: _Context, config: AdaptiveSearchConfig, archives: AdaptiveArchives
) -> tuple[SearchRecord, str] | None:
    """Sample a restart or a parent from the two signal archives."""
    rng = ctx.session.rng
    if rng.random() < config.restart_probability:
        record = _try_record(ctx, restart_proposal(rng, config), None, "restart")
        if record is not None:
            archives.observe([record])
        if record is not None and record.evaluation is not None:
            return record, "restart"
        if ctx.stop:
            return None

    groups: dict[str, list[SearchRecord]] = {
        "elite": list(archives.stable),
        "promising": archives.promising_records(),
        "diverse": _valid_session_records(ctx),
    }
    labels: list[str] = []
    weights: list[float] = []
    for label, weight in (
        ("elite", config.parent_elite_probability),
        ("promising", config.parent_promising_probability),
        ("diverse", config.parent_random_probability),
    ):
        if groups[label] and weight > 0:
            labels.append(label)
            weights.append(weight)
    if not labels:
        if not groups["diverse"]:
            return None
        labels, weights = ["diverse"], [1.0]
    probabilities = np.asarray(weights, dtype=float)
    probabilities /= probabilities.sum()
    label = str(rng.choice(labels, p=probabilities))
    pool = groups[label]
    return pool[int(rng.integers(len(pool)))], label


def _archive_snapshot(records: Sequence[SearchRecord]) -> list[dict[str, object]]:
    snapshot: list[dict[str, object]] = []
    for record in records:
        evaluation = record.evaluation
        if evaluation is None:
            continue
        snapshot.append(
            {
                "record_id": record.id,
                "n_clusters": evaluation.template.n_clusters,
                "status": evaluation.status,
                "stable_mu": evaluation.stable_mu,
                "margin_gamma": evaluation.margin_gamma,
            }
        )
    return snapshot


def _write_adaptive(
    session: SearchSession,
    result: SearchResult,
    episodes: Sequence[AdaptiveEpisode],
    config: AdaptiveSearchConfig,
    archives: AdaptiveArchives,
) -> None:
    if session.destination is None:
        return
    session.write_artifact(
        "adaptive.json",
        {
            "schema_version": 1,
            "run_id": result.run_id,
            "algorithm": "adaptive_amcs",
            "baseline_level": config.level,
            "baseline_depth": config.depth,
            "max_level": config.max_level,
            "max_depth": config.max_depth,
            "episodes": [asdict(episode) for episode in episodes],
            "stable_archive": _archive_snapshot(archives.stable),
            "promising_archive": _archive_snapshot(archives.promising_records()),
            "numerical_evidence_only": True,
        },
    )


def run_adaptive_search(
    seeds: list[PhaseTemplate],
    evaluator: Evaluator,
    config: AdaptiveSearchConfig | None = None,
    *,
    output: str | Path | None = None,
) -> AdaptiveSearchResult:
    """Run adaptive nested rollouts on one shared session and budget."""
    if config is None:
        config = AdaptiveSearchConfig()
    archives = AdaptiveArchives(
        stable_capacity=config.stable_archive_size,
        promising_capacity=config.promising_archive_size,
        promising_per_cluster=config.promising_per_cluster,
    )
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
        search_config=dict(algorithm="adaptive_amcs", **asdict(config)),
        clock=perf_counter,
    ) as session:
        ctx = _Context(session, config)
        try:
            seed_records = session.evaluate_seeds()
        except SearchBudgetExhausted:
            result = session.finish("sdp_budget")
            _write_adaptive(session, result, [], config, archives)
            return AdaptiveSearchResult(result, [], config, archives)
        archives.observe(seed_records)
        seed_best = (
            session.archive.best.evaluation.stable_mu
            if session.archive.best is not None
            else None
        )
        controller = AdaptiveController(config, incumbent_mu=seed_best)

        episodes: list[AdaptiveEpisode] = []
        stop_reason = "iterations"
        for index in range(config.iterations):
            if ctx.stop:
                stop_reason = ctx.stop_reason or "stopped"
                break
            reason = session.budget.stop_reason()
            if reason is not None:
                stop_reason = reason
                break
            # Root evaluation/pruning belong to this episode, including cost
            # and improvements. The dedicated archives observe roots eagerly
            # so a budget exhausted before rollout cannot lose a discovery.
            evaluations_before = session.evaluations
            sdp_calls_before = session.budget.sdp_calls
            incumbent_before = (
                session.archive.best.evaluation.stable_mu
                if session.archive.best is not None else None
            )
            chosen = _choose_adaptive_root(ctx, config, archives)
            if chosen is None:
                stop_reason = ctx.stop_reason or "no_root"
                break
            source_root, root_selection = chosen
            root = source_root
            prune_operation: str | None = None
            if session.rng.random() < controller.prune_probability:
                proposal = _pruning_proposal(ctx, config, archives)
                if proposal.template is not None:
                    candidate = _try_record(
                        ctx,
                        proposal,
                        int(proposal.parameters.get("source_id", proposal.parameters.get("split_id", source_root.id))),
                        f"prune:{proposal.operation}",
                    )
                    if candidate is not None:
                        archives.observe([candidate])
                    if (
                        candidate is not None
                        and candidate.evaluation is not None
                        and candidate.evaluation.status != "invalid"
                    ):
                        root = candidate
                        prune_operation = proposal.operation
                        root_selection = f"prune:{proposal.operation}"
            trace_before = len(session.trace)
            level, depth = controller.level, controller.depth
            prune_probability = controller.prune_probability
            rollout = _rollout(ctx, root, level, depth)
            archives.observe(session.trace[trace_before:])
            incumbent_after = (
                session.archive.best.evaluation.stable_mu
                if session.archive.best is not None
                else None
            )
            improved = incumbent_after is not None and (
                incumbent_before is None
                or incumbent_after > incumbent_before + _MU_TOLERANCE
            )
            stall_before = controller.stall
            decision = controller.after_episode(incumbent_after)

            base = _episode_result(
                index,
                root,
                root_selection,
                rollout,
                ctx,
                evaluations_before,
                sdp_calls_before,
                level=level,
                depth=depth,
            )
            episodes.append(
                AdaptiveEpisode(
                    index=base.index,
                    level=base.level,
                    depth=base.depth,
                    branching=base.branching,
                    root_id=base.root_id,
                    source_root_id=source_root.id,
                    root_selection=base.root_selection,
                    prune_probability=prune_probability,
                    prune_operation=prune_operation,
                    record_ids=base.record_ids,
                    endpoint_id=base.endpoint_id,
                    best_id=base.best_id,
                    endpoint_status=base.endpoint_status,
                    endpoint_stable_mu=base.endpoint_stable_mu,
                    endpoint_margin_gamma=base.endpoint_margin_gamma,
                    endpoint_n_clusters=base.endpoint_n_clusters,
                    best_stable_mu=base.best_stable_mu,
                    evaluations=base.evaluations,
                    sdp_calls=base.sdp_calls,
                    complete=base.complete,
                    incumbent_mu_before=incumbent_before,
                    incumbent_mu_after=incumbent_after,
                    improved=improved,
                    stall_count=stall_before,
                    decision=decision,
                    message=base.message,
                )
            )
            if ctx.stop:
                stop_reason = ctx.stop_reason or "stopped"
                break

        result = session.finish(stop_reason)
        _write_adaptive(session, result, episodes, config, archives)
        return AdaptiveSearchResult(result, episodes, config, archives)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-level", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--stall-threshold", type=int, default=2)
    parser.add_argument("--branching", type=int, default=8)
    parser.add_argument("--prune-probability", type=float, default=0.2)
    parser.add_argument("--prune-increment", type=float, default=0.2)
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
        config = AdaptiveSearchConfig(
            seed=args.seed,
            iterations=args.iterations,
            level=args.level,
            depth=args.depth,
            max_level=args.max_level,
            max_depth=args.max_depth,
            stall_threshold=args.stall_threshold,
            branching=args.branching,
            prune_probability=args.prune_probability,
            prune_increment=args.prune_increment,
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
        counts = [
            m for m in (5, 4) if config.min_clusters <= m <= config.max_clusters
        ]
        seeds = [twisted_seed(m) for m in counts or [config.min_clusters]]
        result = run_adaptive_search(seeds, evaluator, config, output=args.output)
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
                "decisions": Counter(
                    episode.decision for episode in result.episodes
                ),
                "levels": sorted(
                    {episode.level for episode in result.episodes}
                ),
                "depths": sorted(
                    {episode.depth for episode in result.episodes}
                ),
                "pruned_episodes": sum(
                    1
                    for episode in result.episodes
                    if episode.prune_operation is not None
                ),
                "elapsed_seconds": result.elapsed_seconds,
                "stop_reason": result.stop_reason,
                "statuses": dict(statuses),
            }
        )
    )
    print("Numerical candidates only; no rigorous certificate. Results:", args.output)


if __name__ == "__main__":
    main()
