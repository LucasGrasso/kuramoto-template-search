"""Seeded archive search. CLI: uv run python -m kuramoto_template.search.random_search.

Iterations count proposal attempts, including rejected/duplicate proposals.
Seeds are always evaluated first (even duplicates), outside that proposal budget.
The SDP budget includes every density/fallback call, including exceptions.
Wall-time limits are checked between evaluations, after mandatory seed evaluation;
they cannot interrupt an in-flight conic solve. Output directories must be new.

Future AMCS can reuse sample_mutation, TemplateEvaluator, DiversityArchive and
SearchRecord; nested rollout selection should replace the baseline proposal loop.
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

import numpy as np

from kuramoto_template.mutations import merge_clusters, perturb_phase, resize_clusters, split_cluster
from kuramoto_template.template import PhaseTemplate
from kuramoto_template.sdp_optimizer.validate_input import validate_input
from .evaluator import EvaluationConfig, SolverDiagnostics, TemplateEvaluation, TemplateEvaluator


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


class Evaluator(Protocol):
    config: EvaluationConfig

    def evaluate(self, template: PhaseTemplate, *, max_solver_calls: int | None = None) -> TemplateEvaluation: ...


@dataclass(eq=False)
class ArchiveEntry:
    id: int
    evaluation: TemplateEvaluation


class DiversityArchive:
    """Bounded strata by cluster count/status, with a protected stable incumbent.

    Evict from the most populated removable stratum. Within a stable stratum
    evict lowest density; within a near-feasible stratum evict lowest gamma
    (only equal cluster counts are compared). Otherwise evict the oldest.
    Uniform parent sampling allows nonimproving and unstable paths.
    """

    def __init__(self, size: int):
        self.size = size
        self.entries: list[ArchiveEntry] = []
        self.best: ArchiveEntry | None = None

    def consider(self, entry: ArchiveEntry) -> bool:
        evaluation = entry.evaluation
        if evaluation.status == "stable" and evaluation.stable_mu is not None:
            best_mu = self.best.evaluation.stable_mu if self.best is not None else None
            if best_mu is None or evaluation.stable_mu > best_mu:
                self.best = entry
        self.entries.append(entry)
        while len(self.entries) > self.size:
            removable = [e for e in self.entries if e is not self.best]
            counts = Counter((e.evaluation.template.n_clusters, e.evaluation.status) for e in self.entries)
            stratum = max(((e.evaluation.template.n_clusters, e.evaluation.status) for e in removable), key=counts.__getitem__)
            group = [e for e in removable if (e.evaluation.template.n_clusters, e.evaluation.status) == stratum]
            if stratum[1] == "stable":
                victim = min(group, key=lambda e: e.evaluation.stable_mu if e.evaluation.stable_mu is not None else -np.inf)
            elif stratum[1] == "near_feasible":
                victim = min(group, key=lambda e: e.evaluation.margin_gamma if e.evaluation.margin_gamma is not None else -np.inf)
            else:
                victim = group[0]
            self.entries.remove(victim)
        return entry in self.entries


@dataclass(eq=False)
class MutationProposal:
    operation: str
    parameters: dict[str, float | int]
    template: PhaseTemplate | None = None
    message: str = ""


def sample_mutation(template: PhaseTemplate, rng: np.random.Generator, config: RandomSearchConfig) -> MutationProposal:
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


def _restart(rng: np.random.Generator, config: RandomSearchConfig) -> MutationProposal:
    m = int(rng.integers(config.min_clusters, config.max_clusters + 1))
    kind = int(rng.integers(2))
    template = twisted_seed(m) if kind == 0 else PhaseTemplate(rng.dirichlet(np.ones(m)), rng.uniform(0, 2 * np.pi, m))
    return MutationProposal("restart", {"n_clusters": m, "twisted": kind == 0}, template)


@dataclass(eq=False)
class SearchRecord:
    id: int
    parent_id: int | None
    operation: str
    parameters: dict[str, float | int]
    selection: str
    template: PhaseTemplate | None
    evaluation: TemplateEvaluation | None
    disposition: str
    best_mu: float | None
    improved: bool = False
    in_archive: bool = False
    message: str = ""


@dataclass(eq=False)
class SearchResult:
    best: ArchiveEntry | None
    archive: list[ArchiveEntry]
    trace: list[SearchRecord]
    evaluations: int
    sdp_calls: int
    proposals: int
    elapsed_seconds: float
    stop_reason: str
    run_id: str


def _diagnostic_data(diagnostic: SolverDiagnostics | None) -> dict[str, Any] | None:
    if diagnostic is None:
        return None
    return {name: getattr(diagnostic, name) for name in diagnostic.__dataclass_fields__}


def evaluation_data(evaluation: TemplateEvaluation) -> dict[str, Any]:
    return dict(status=evaluation.status, stable_mu=evaluation.stable_mu, margin_gamma=evaluation.margin_gamma,
                density=_diagnostic_data(evaluation.density), margin=_diagnostic_data(evaluation.margin),
                P_source=evaluation.P_source, sdp_calls=evaluation.sdp_calls,
                elapsed_seconds=evaluation.elapsed_seconds, message=evaluation.message)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None  # Nonfinite diagnostics stay invalid, never emit nonstandard JSON NaN.
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_safe(data), handle, indent=2, allow_nan=False)
        handle.write("\n")


def _template_data(template: PhaseTemplate | None) -> dict[str, Any] | None:
    return dict(x=template.x, phi=template.phi, n_clusters=template.n_clusters) if template is not None else None


def _signature(template: PhaseTemplate) -> tuple[bytes, bytes]:
    # Exact float64 data only: no phase rotation/permutation or rounded merging.
    return template.x.tobytes(), template.phi.tobytes()


def run_random_search(
    seeds: list[PhaseTemplate], evaluator: Evaluator, config: RandomSearchConfig = RandomSearchConfig(),
    *, output: str | Path | None = None,
) -> SearchResult:
    """Run a baseline, optionally persist every record in a new directory.

    JSONL is flushed after each proposal, including rejected actions. result.json
    stores the final archive and best x/phi/P. Records include source-specific
    raw solver matrices, so density and margin metrics are never mixed. Saved
    results can be reused as seeds, but exact interrupted-run resume is deferred.
    """
    if not seeds:
        raise ValueError("At least one initial seed is required")
    if config.max_clusters > evaluator.config.max_clusters:
        raise ValueError("Search max_clusters exceeds evaluator max_clusters")
    if config.max_sdp_calls is not None and config.max_sdp_calls < len(seeds):
        raise ValueError("SDP budget must reserve at least one call per seed")
    for seed in seeds:
        if not config.min_clusters <= seed.n_clusters <= config.max_clusters:
            raise ValueError("Seed cluster count is outside search bounds")
        validate_input(seed.x, seed.phi)
    started = perf_counter()
    rng = np.random.default_rng(config.seed)
    run_id = str(uuid4())
    archive = DiversityArchive(config.archive_size)
    trace: list[SearchRecord] = []
    seen: set[tuple[bytes, bytes]] = set()
    evaluations = sdp_calls = proposals = 0
    destination = Path(output) if output is not None else None
    log = None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=False)
        versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "cvxpy", "clarabel")}
        source_files = sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
        digest = hashlib.sha256()
        for source in source_files:
            digest.update(source.relative_to(Path(__file__).resolve().parents[1]).as_posix().encode())
            digest.update(source.read_bytes())
        _write_json(destination / "metadata.json", dict(
            schema_version=1, run_id=run_id, seed=config.seed, started_at=datetime.now(timezone.utc).isoformat(),
            search_config=asdict(config), evaluation_config=asdict(evaluator.config), versions=versions,
            code_sha256=digest.hexdigest(), seed_templates=[_template_data(s) for s in seeds],
            deduplication="exact float64 arrays; no rounding or symmetry canonicalization",
            numerical_evidence_only=True,
        ))
        log = (destination / "trace.jsonl").open("x", encoding="utf-8")

    def remaining(reserve: int = 0) -> int | None:
        return None if config.max_sdp_calls is None else config.max_sdp_calls - sdp_calls - reserve

    def record(proposal: MutationProposal, parent_id: int | None, selection: str, reserve: int = 0) -> None:
        nonlocal evaluations, sdp_calls
        template = proposal.template
        evaluation = None
        improved = in_archive = False
        disposition = "rejected"
        previous = archive.best
        if template is not None:
            if selection != "seed" and _signature(template) in seen:
                disposition = "duplicate"
            else:
                seen.add(_signature(template))
                evaluation = evaluator.evaluate(template, max_solver_calls=remaining(reserve))
                evaluations += 1
                sdp_calls += evaluation.sdp_calls
                in_archive = archive.consider(ArchiveEntry(len(trace), evaluation))
                improved = archive.best is not previous
                disposition = "evaluated"
        item = SearchRecord(len(trace), parent_id, proposal.operation, proposal.parameters, selection, template,
                            evaluation, disposition, archive.best.evaluation.stable_mu if archive.best else None,
                            improved, in_archive, proposal.message)
        trace.append(item)
        if log is not None:
            data = dict(schema_version=1, run_id=run_id, seed=config.seed, timestamp=datetime.now(timezone.utc).isoformat(),
                        id=item.id, parent_id=parent_id, operation=item.operation, parameters=item.parameters,
                        selection=selection, template=_template_data(template),
                        evaluation=evaluation_data(evaluation) if evaluation is not None else None,
                        disposition=disposition, best_mu=item.best_mu, improved=improved, in_archive=in_archive,
                        evaluations=evaluations, sdp_calls=sdp_calls, message=item.message)
            log.write(json.dumps(_json_safe(data), allow_nan=False) + "\n")
            log.flush()

    stop_reason = "iterations"
    try:
        for index, seed in enumerate(seeds):
            record(MutationProposal("seed", {"index": index}, seed), None, "seed", len(seeds) - index - 1)
        for _ in range(config.iterations):
            calls_left = remaining()
            if calls_left is not None and calls_left <= 0:
                stop_reason = "sdp_budget"
                break
            if config.max_wall_seconds is not None and perf_counter() - started >= config.max_wall_seconds:
                stop_reason = "wall_time"
                break
            proposals += 1
            if rng.random() < config.restart_probability:
                record(_restart(rng, config), None, "restart")
            else:
                parent = archive.entries[int(rng.integers(len(archive.entries)))]
                record(sample_mutation(parent.evaluation.template, rng, config), parent.id, config.parent_selection)
    finally:
        if log is not None:
            log.close()
    result = SearchResult(archive.best, list(archive.entries), trace, evaluations, sdp_calls, proposals,
                          perf_counter() - started, stop_reason, run_id)
    if destination is not None:
        def entry_data(entry: ArchiveEntry) -> dict[str, Any]:
            return dict(id=entry.id, template=_template_data(entry.evaluation.template), evaluation=evaluation_data(entry.evaluation))
        _write_json(destination / "result.json", dict(
            schema_version=1, run_id=run_id, best=entry_data(result.best) if result.best else None,
            archive=[entry_data(e) for e in result.archive], evaluations=evaluations, sdp_calls=sdp_calls,
            proposals=proposals, elapsed_seconds=result.elapsed_seconds, stop_reason=stop_reason,
        ))
    return result


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
