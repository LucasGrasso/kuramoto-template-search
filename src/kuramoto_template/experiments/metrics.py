"""Per-run metrics and aggregation for search benchmarks.

Two signals stay separate throughout:

* ``best_stable_mu`` is the main objective and is only populated by strictly
  stable, numerically screened density evaluations.
* ``best_margin_gamma`` records progress of near-feasible templates at the
  configured target density. It is never compared with ``mu``.

Counts are split between proposal-level and unique/canonical-level views.
``sdp_calls`` always counts real density and margin solver invocations, so a
candidate needing both solves costs two calls even when it produces one status.
"""

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from kuramoto_template.search.infrastructure import SearchResult

from .cache import CacheStats, canonical_key

_INACCURATE_STATUSES = ("optimal_inaccurate", "infeasible_inaccurate")
_PROGRESS_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


@dataclass(eq=False)
class RunMetrics:
    """Measured outcome of one algorithm on one initial template."""

    algorithm: str
    initial: str
    budget: int
    repetition: int
    run_seed: int
    run_id: str
    best_stable_mu: float | None = None
    best_margin_gamma: float | None = None
    best_margin_n_clusters: int | None = None
    best_margin_by_cluster: dict[int, float] = field(default_factory=dict)
    best_density_mu: float | None = None
    stable_templates: int = 0
    near_feasible_templates: int = 0
    unique_templates: int = 0
    evaluations: int = 0
    unique_evaluations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    duplicate_proposals: int = 0
    rejected_proposals: int = 0
    proposals: int = 0
    sdp_calls: int = 0
    density_sdp_calls: int = 0
    margin_sdp_calls: int = 0
    solver_errors: int = 0
    inaccurate_results: int = 0
    invalid_evaluations: int = 0
    numerically_invalid: int = 0
    elapsed_seconds: float = 0.0
    wall_seconds: float = 0.0
    stop_reason: str = ""
    first_stable_call: int | None = None
    best_mu_call: int | None = None
    status_counts: dict[str, int] = field(default_factory=dict)
    unique_status_counts: dict[str, int] = field(default_factory=dict)
    density_status_counts: dict[str, int] = field(default_factory=dict)
    margin_status_counts: dict[str, int] = field(default_factory=dict)
    best_mu_curve: list[dict[str, float | int]] = field(default_factory=list)

    @property
    def found_stable(self) -> bool:
        return self.best_stable_mu is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunMetrics:
        values = dict(data)
        values["best_margin_by_cluster"] = {int(k): float(v) for k, v in values.get("best_margin_by_cluster", {}).items()}
        return cls(**values)


def _unwrap_search(result: object) -> SearchResult:
    search = getattr(result, "search", None)
    if isinstance(search, SearchResult):
        return search
    if isinstance(result, SearchResult):
        return result
    raise TypeError("Expected SearchResult or an object exposing .search")


def _curve(
    trace: Sequence[Any], budget: int
) -> tuple[list[dict[str, float | int]], int | None, int | None]:
    calls = 0
    best: float | None = None
    points: list[dict[str, float | int]] = []
    for record in trace:
        evaluation = record.evaluation
        if evaluation is None:
            continue
        calls += evaluation.sdp_calls
        mu = _finite(evaluation.stable_mu)
        if evaluation.status != "stable" or mu is None:
            continue
        if best is None or mu > best:
            best = mu
            points.append(
                {
                    "sdp_calls": min(calls, budget),
                    "best_mu": best,
                    "record_id": record.id,
                }
            )
    if not points:
        return [], None, None
    return points, int(points[0]["sdp_calls"]), int(points[-1]["sdp_calls"])


def compute_run_metrics(
    result: object,
    *,
    algorithm: str,
    initial: str,
    budget: int,
    repetition: int,
    run_seed: int,
    cache_stats: CacheStats | None = None,
    wall_seconds: float | None = None,
    cache_decimals: int = 12,
) -> RunMetrics:
    """Compute metrics from one completed search result."""
    search = _unwrap_search(result)
    evaluated = [record for record in search.trace if record.evaluation is not None]
    unique = []
    for record in evaluated:
        evaluation = record.evaluation
        if evaluation is not None and not evaluation.cached:
            unique.append(record)

    stable_mus: list[float] = []
    margin_values: list[tuple[float, int]] = []
    density_values: list[float] = []
    stable_keys: set[tuple[int, tuple[tuple[float, float], ...]]] = set()
    near_feasible_keys: set[tuple[int, tuple[tuple[float, float], ...]]] = set()
    unique_keys: set[tuple[int, tuple[tuple[float, float], ...]]] = set()
    density_calls = 0
    margin_calls = 0
    solver_errors = 0
    inaccurate_results = 0
    numerically_invalid = 0
    unique_statuses: Counter[str] = Counter()
    density_statuses: Counter[str] = Counter()
    margin_statuses: Counter[str] = Counter()

    for record in unique:
        evaluation = record.evaluation
        if evaluation is None:  # pragma: no cover - guarded by construction
            continue
        template = evaluation.template
        unique_keys.add(canonical_key(template, decimals=cache_decimals))
        unique_statuses[evaluation.status] += 1

        mu = _finite(evaluation.stable_mu)
        if evaluation.status == "stable" and mu is not None:
            stable_mus.append(mu)
            stable_keys.add(canonical_key(template, decimals=cache_decimals))

        gamma = _finite(evaluation.margin_gamma)
        if gamma is not None:
            margin_values.append((gamma, template.n_clusters))
            if evaluation.status == "near_feasible":
                near_feasible_keys.add(
                    canonical_key(template, decimals=cache_decimals)
                )

        diagnostics = [
            diagnostic
            for diagnostic in (evaluation.density, evaluation.margin)
            if diagnostic is not None
        ]
        if evaluation.sdp_calls > 0:
            if evaluation.density is not None:
                density_calls += 1
                density_statuses[evaluation.density.status] += 1
            if evaluation.margin is not None:
                margin_calls += 1
                margin_statuses[evaluation.margin.status] += 1
        for diagnostic in diagnostics:
            if diagnostic.status == "solver_error":
                solver_errors += 1
            if diagnostic.status in _INACCURATE_STATUSES:
                inaccurate_results += 1
            if not diagnostic.numerically_valid:
                numerically_invalid += 1
            density_mu = _finite(diagnostic.mu)
            if (
                diagnostic.source == "density"
                and density_mu is not None
                and diagnostic.numerically_valid
            ):
                density_values.append(density_mu)

    all_statuses = Counter(
        record.evaluation.status for record in evaluated if record.evaluation is not None
    )
    trace_hits = sum(
        1 for record in evaluated if record.evaluation is not None and record.evaluation.cached
    )
    cache_hits = cache_stats.hits if cache_stats is not None else trace_hits
    cache_misses = cache_stats.misses if cache_stats is not None else len(unique)
    curve, first_stable_call, best_mu_call = _curve(search.trace, budget)
    best_stable_mu = max(stable_mus) if stable_mus else None
    best_margin_by_cluster: dict[int, float] = {}
    for gamma, m in margin_values:
        best_margin_by_cluster[m] = max(best_margin_by_cluster.get(m, -math.inf), gamma)
    # Legacy scalar fields are meaningful only for one cluster-count bucket.
    # Never select a winner by comparing raw gamma across different sizes.
    best_margin_gamma = best_margin_n_clusters = None
    if len(best_margin_by_cluster) == 1:
        best_margin_n_clusters, best_margin_gamma = next(iter(best_margin_by_cluster.items()))

    return RunMetrics(
        algorithm=algorithm,
        initial=initial,
        budget=budget,
        repetition=repetition,
        run_seed=run_seed,
        run_id=search.run_id,
        best_stable_mu=best_stable_mu,
        best_margin_gamma=best_margin_gamma,
        best_margin_n_clusters=best_margin_n_clusters,
        best_margin_by_cluster=best_margin_by_cluster,
        best_density_mu=max(density_values) if density_values else None,
        stable_templates=len(stable_keys),
        near_feasible_templates=len(near_feasible_keys),
        unique_templates=len(unique_keys),
        evaluations=search.evaluations,
        unique_evaluations=len(unique),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        duplicate_proposals=sum(
            1 for record in search.trace if record.disposition == "duplicate"
        ),
        rejected_proposals=sum(
            1 for record in search.trace if record.disposition == "rejected"
        ),
        proposals=search.proposals,
        sdp_calls=search.sdp_calls,
        density_sdp_calls=density_calls,
        margin_sdp_calls=margin_calls,
        solver_errors=solver_errors,
        inaccurate_results=inaccurate_results,
        invalid_evaluations=unique_statuses.get("invalid", 0),
        numerically_invalid=numerically_invalid,
        elapsed_seconds=search.elapsed_seconds,
        wall_seconds=search.elapsed_seconds if wall_seconds is None else wall_seconds,
        stop_reason=search.stop_reason,
        first_stable_call=first_stable_call,
        best_mu_call=best_mu_call,
        status_counts=dict(sorted(all_statuses.items())),
        unique_status_counts=dict(sorted(unique_statuses.items())),
        density_status_counts=dict(sorted(density_statuses.items())),
        margin_status_counts=dict(sorted(margin_statuses.items())),
        best_mu_curve=curve,
    )


@dataclass(frozen=True)
class SummaryStats:
    """Descriptive statistics over independent runs."""

    count: int
    missing: int
    mean: float | None
    median: float | None
    std: float | None
    minimum: float | None
    maximum: float | None
    q25: float | None
    q75: float | None


def summarize_values(values: Iterable[float | None]) -> SummaryStats:
    raw = list(values)
    present = [float(value) for value in raw if value is not None and math.isfinite(value)]
    if not present:
        return SummaryStats(len(raw), len(raw), None, None, None, None, None, None, None)
    array = np.asarray(present, dtype=np.float64)
    q25, q75 = np.quantile(array, (0.25, 0.75))
    return SummaryStats(
        count=len(present),
        missing=len(raw) - len(present),
        mean=float(array.mean()),
        median=float(np.median(array)),
        std=float(array.std()),
        minimum=float(array.min()),
        maximum=float(array.max()),
        q25=float(q25),
        q75=float(q75),
    )


METRIC_FIELDS: tuple[str, ...] = (
    "best_stable_mu",
    "best_margin_gamma",
    "best_density_mu",
    "stable_templates",
    "near_feasible_templates",
    "unique_templates",
    "evaluations",
    "unique_evaluations",
    "cache_hits",
    "cache_misses",
    "duplicate_proposals",
    "rejected_proposals",
    "proposals",
    "sdp_calls",
    "density_sdp_calls",
    "margin_sdp_calls",
    "solver_errors",
    "inaccurate_results",
    "invalid_evaluations",
    "numerically_invalid",
    "elapsed_seconds",
    "wall_seconds",
)


def best_mu_at(curve: Sequence[Mapping[str, float | int]], call_limit: float) -> float | None:
    """Step-function lookup of best stable mu by cumulative SDP calls."""
    best: float | None = None
    for point in curve:
        if float(point["sdp_calls"]) > call_limit:
            break
        best = float(point["best_mu"])
    return best


@dataclass(frozen=True)
class GroupSummary:
    """Aggregated runs for one (algorithm, initial family, budget) cell."""

    algorithm: str
    initial: str
    budget: int
    runs: int
    stable_run_rate: float
    target_run_rate: float
    metrics: dict[str, SummaryStats]
    progress: dict[str, SummaryStats]
    margin_by_cluster: dict[int, SummaryStats] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "initial": self.initial,
            "budget": self.budget,
            "runs": self.runs,
            "stable_run_rate": self.stable_run_rate,
            "target_run_rate": self.target_run_rate,
            "metrics": {name: asdict(stats) for name, stats in self.metrics.items()},
            "progress": {name: asdict(stats) for name, stats in self.progress.items()},
            "margin_by_cluster": {m: asdict(stats) for m, stats in self.margin_by_cluster.items()},
        }


def summarize_runs(
    runs: Sequence[RunMetrics], *, target_mu: float
) -> list[GroupSummary]:
    """Aggregate independent runs by algorithm, initial family, and budget."""
    groups: dict[tuple[str, str, int], list[RunMetrics]] = {}
    for run in runs:
        groups.setdefault((run.algorithm, run.initial, run.budget), []).append(run)

    summaries: list[GroupSummary] = []
    for (algorithm, initial, budget), group in sorted(groups.items()):
        metrics = {
            name: summarize_values(getattr(run, name) for run in group)
            for name in METRIC_FIELDS
        }
        margin_buckets: dict[int, list[float | None]] = {}
        for run in group:
            per_m = run.best_margin_by_cluster or (
                {run.best_margin_n_clusters: run.best_margin_gamma}
                if run.best_margin_n_clusters is not None and run.best_margin_gamma is not None else {}
            )
            for m in per_m:
                margin_buckets.setdefault(m, [])
        for m in margin_buckets:
            margin_buckets[m] = [
                run.best_margin_by_cluster.get(m, run.best_margin_gamma if run.best_margin_n_clusters == m else None)
                for run in group
            ]
        if len(margin_buckets) > 1:
            metrics["best_margin_gamma"] = summarize_values([None] * len(group))
        progress = {
            f"{round(fraction * 100)}pct": summarize_values(
                best_mu_at(run.best_mu_curve, fraction * budget) for run in group
            )
            for fraction in _PROGRESS_FRACTIONS
        }
        stable_count = sum(1 for run in group if run.best_stable_mu is not None)
        target_count = sum(
            1
            for run in group
            if run.best_stable_mu is not None and run.best_stable_mu >= target_mu
        )
        summaries.append(
            GroupSummary(
                algorithm=algorithm,
                initial=initial,
                budget=budget,
                runs=len(group),
                stable_run_rate=stable_count / len(group),
                target_run_rate=target_count / len(group),
                metrics=metrics,
                progress=progress,
                margin_by_cluster={m: summarize_values(v) for m, v in sorted(margin_buckets.items())},
            )
        )
    return summaries


@dataclass(frozen=True)
class PairedComparison:
    """Paired candidate-vs-baseline outcome on matched repetitions."""

    initial: str
    budget: int | None
    baseline: str
    candidate: str
    pairs: int
    candidate_wins: int
    baseline_wins: int
    ties: int
    candidate_only_stable: int
    baseline_only_stable: int
    mean_delta: float | None
    median_delta: float | None


def _paired(
    pairs: Sequence[tuple[RunMetrics, RunMetrics]],
    *,
    initial: str,
    budget: int | None,
    baseline: str,
    candidate: str,
) -> PairedComparison:
    candidate_wins = baseline_wins = ties = 0
    candidate_only = baseline_only = 0
    deltas: list[float] = []
    for baseline_run, candidate_run in pairs:
        a = baseline_run.best_stable_mu
        b = candidate_run.best_stable_mu
        if b is not None and a is None:
            candidate_only += 1
        elif a is not None and b is None:
            baseline_only += 1
        if a is not None and b is not None:
            deltas.append(float(b - a))
        comparison_a = -math.inf if a is None else float(a)
        comparison_b = -math.inf if b is None else float(b)
        if comparison_b > comparison_a:
            candidate_wins += 1
        elif comparison_a > comparison_b:
            baseline_wins += 1
        else:
            ties += 1
    return PairedComparison(
        initial=initial,
        budget=budget,
        baseline=baseline,
        candidate=candidate,
        pairs=len(pairs),
        candidate_wins=candidate_wins,
        baseline_wins=baseline_wins,
        ties=ties,
        candidate_only_stable=candidate_only,
        baseline_only_stable=baseline_only,
        mean_delta=float(np.mean(deltas)) if deltas else None,
        median_delta=float(np.median(deltas)) if deltas else None,
    )


def paired_comparisons(
    runs: Sequence[RunMetrics], *, baseline: str, candidate: str
) -> list[PairedComparison]:
    """Compare two algorithms on matched (initial, budget, repetition) runs."""
    baseline_runs = {
        (run.initial, run.budget, run.repetition): run
        for run in runs
        if run.algorithm == baseline
    }
    candidate_runs = {
        (run.initial, run.budget, run.repetition): run
        for run in runs
        if run.algorithm == candidate
    }
    keys = sorted(set(baseline_runs) & set(candidate_runs))
    if not keys:
        return []

    comparisons: list[PairedComparison] = []
    grouped: dict[tuple[str, int], list[tuple[RunMetrics, RunMetrics]]] = {}
    all_pairs: list[tuple[RunMetrics, RunMetrics]] = []
    for key in keys:
        pair = (baseline_runs[key], candidate_runs[key])
        grouped.setdefault((key[0], key[1]), []).append(pair)
        all_pairs.append(pair)
    for (initial, budget), pairs in sorted(grouped.items()):
        comparisons.append(
            _paired(
                pairs,
                initial=initial,
                budget=budget,
                baseline=baseline,
                candidate=candidate,
            )
        )
    comparisons.append(
        _paired(
            all_pairs,
            initial="all",
            budget=None,
            baseline=baseline,
            candidate=candidate,
        )
    )
    return comparisons
