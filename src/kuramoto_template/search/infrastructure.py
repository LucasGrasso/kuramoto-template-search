"""Shared archive, budgets, records and persistence for template search algorithms."""

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Protocol, TextIO
from uuid import uuid4

import numpy as np

from kuramoto_template.template import PhaseTemplate
from kuramoto_template.sdp_optimizer.validate_input import validate_input
from .evaluator import EvaluationConfig, SolverDiagnostics, TemplateEvaluation


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
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("Archive size must be a positive integer")
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


class SearchBudgetExhausted(RuntimeError):
    """An attempted evaluation would exceed the remaining SDP-call budget."""


class SearchBudget:
    """Global call accounting shared by every branch of a search.

    Remaining calls may reserve one call for each unevaluated seed. Wall time
    is checked at algorithm boundaries via stop_reason(), never by interrupting
    an active solver. Mandatory seed evaluation may exceed the wall-time limit.
    """

    def __init__(
        self, max_sdp_calls: int | None = None, max_wall_seconds: float | None = None,
        *, clock: Callable[[], float] | None = None,
    ) -> None:
        if max_sdp_calls is not None and (
            isinstance(max_sdp_calls, bool) or not isinstance(max_sdp_calls, int) or max_sdp_calls < 1
        ):
            raise ValueError("max_sdp_calls must be a positive integer")
        if max_wall_seconds is not None and (not np.isfinite(max_wall_seconds) or max_wall_seconds <= 0):
            raise ValueError("max_wall_seconds must be finite and positive")
        self.max_sdp_calls = max_sdp_calls
        self.max_wall_seconds = max_wall_seconds
        self.clock = clock if clock is not None else perf_counter
        self.started = self.clock()
        self.sdp_calls = 0

    def remaining(self, reserve: int = 0) -> int | None:
        if isinstance(reserve, bool) or not isinstance(reserve, int) or reserve < 0:
            raise ValueError("reserve must be a nonnegative integer")
        return None if self.max_sdp_calls is None else self.max_sdp_calls - self.sdp_calls - reserve

    def stop_reason(self) -> str | None:
        remaining = self.remaining()
        if remaining is not None and remaining <= 0:
            return "sdp_budget"
        if self.max_wall_seconds is not None and self.elapsed_seconds() >= self.max_wall_seconds:
            return "wall_time"
        return None

    def elapsed_seconds(self) -> float:
        return self.clock() - self.started


class SearchSession:
    """One run's RNG, archive, exact deduplication, accounting and persistence.

    Use as a context manager, evaluate_seeds() once, then record proposals with
    parent IDs from earlier records. Call budget.stop_reason() before each
    proposal, including inside future nested rollouts. record() returns the
    complete record so callers can build and compare paths without duplicating
    evaluation logic. It also guards against starting a solve with no calls left.

    All algorithms share one session/budget throughout recursion. No algorithm
    policy lives here: parent selection, restarts and rollout depth belong to
    callers. finish() saves the final snapshot; exceptions still close JSONL,
    preserving already flushed records. Exact interrupted-run resume is deferred.
    """

    def __init__(
        self, seeds: list[PhaseTemplate], evaluator: Evaluator, *, seed: int,
        min_clusters: int = 3, max_clusters: int = 12, archive_size: int = 32,
        max_sdp_calls: int | None = None, max_wall_seconds: float | None = None,
        output: str | Path | None = None, search_config: Mapping[str, Any] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not seeds:
            raise ValueError("At least one initial seed is required")
        for name, value, lower in (("seed", seed, 0), ("min_clusters", min_clusters, 2),
                                   ("max_clusters", max_clusters, 2), ("archive_size", archive_size, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        if max_clusters < min_clusters:
            raise ValueError("max_clusters must be >= min_clusters")
        if max_clusters > evaluator.config.max_clusters:
            raise ValueError("Search max_clusters exceeds evaluator max_clusters")
        if max_sdp_calls is not None and max_sdp_calls < len(seeds):
            raise ValueError("SDP budget must reserve at least one call per seed")
        for template in seeds:
            if not min_clusters <= template.n_clusters <= max_clusters:
                raise ValueError("Seed cluster count is outside search bounds")
            validate_input(template.x, template.phi)
        self.seeds = list(seeds)
        self.evaluator = evaluator
        self.seed = seed
        self.search_config = dict(search_config) if search_config is not None else dict(
            seed=seed, min_clusters=min_clusters, max_clusters=max_clusters,
            archive_size=archive_size, max_sdp_calls=max_sdp_calls, max_wall_seconds=max_wall_seconds,
        )
        self.budget = SearchBudget(max_sdp_calls, max_wall_seconds, clock=clock)
        self.rng = np.random.default_rng(seed)
        self.run_id = str(uuid4())
        self.archive = DiversityArchive(archive_size)
        self.trace: list[SearchRecord] = []
        self.seen: set[tuple[bytes, bytes]] = set()
        self.evaluations = 0
        self.proposals = 0
        self.destination = Path(output) if output is not None else None
        self._log: TextIO | None = None
        self._entered = False
        self._closed = False
        self._seeds_evaluated = False

    def __enter__(self) -> "SearchSession":
        if self._entered or self._closed:
            raise RuntimeError("SearchSession cannot be reopened")
        self._entered = True
        if self.destination is not None:
            self.destination.mkdir(parents=True, exist_ok=False)
            versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "cvxpy", "clarabel")}
            package_root = Path(__file__).resolve().parents[1]
            digest = hashlib.sha256()
            for source in sorted(package_root.rglob("*.py")):
                digest.update(source.relative_to(package_root).as_posix().encode())
                digest.update(source.read_bytes())
            _write_json(self.destination / "metadata.json", dict(
                schema_version=1, run_id=self.run_id, seed=self.seed,
                started_at=datetime.now(timezone.utc).isoformat(), search_config=self.search_config,
                evaluation_config=asdict(self.evaluator.config), versions=versions,
                code_sha256=digest.hexdigest(), seed_templates=[_template_data(s) for s in self.seeds],
                deduplication="exact float64 arrays; no rounding or symmetry canonicalization",
                numerical_evidence_only=True,
            ))
            self._log = (self.destination / "trace.jsonl").open("x", encoding="utf-8")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._log is not None:
            self._log.close()
        self._closed = True

    def _require_open(self) -> None:
        if not self._entered or self._closed:
            raise RuntimeError("Use SearchSession inside a with block")

    def evaluate_seeds(self) -> list[SearchRecord]:
        self._require_open()
        if self._seeds_evaluated:
            raise RuntimeError("Initial seeds have already been evaluated")
        self._seeds_evaluated = True
        records = []
        for index, template in enumerate(self.seeds):
            records.append(self.record(MutationProposal("seed", {"index": index}, template),
                                       None, "seed", reserve=len(self.seeds) - index - 1))
        return records

    def record(
        self, proposal: MutationProposal, parent_id: int | None, selection: str, reserve: int = 0,
    ) -> SearchRecord:
        self._require_open()
        template = proposal.template
        evaluation = None
        improved = in_archive = False
        disposition = "rejected"
        previous = self.archive.best
        if template is not None:
            if selection != "seed" and _signature(template) in self.seen:
                disposition = "duplicate"
            else:
                calls_left = self.budget.remaining(reserve)
                if calls_left is not None and calls_left <= 0:
                    raise SearchBudgetExhausted("No SDP calls remain after reservations")
                self.seen.add(_signature(template))
                evaluation = self.evaluator.evaluate(template, max_solver_calls=calls_left)
                self.evaluations += 1
                self.budget.sdp_calls += evaluation.sdp_calls
                in_archive = self.archive.consider(ArchiveEntry(len(self.trace), evaluation))
                improved = self.archive.best is not previous
                disposition = "evaluated"
        if selection != "seed":
            self.proposals += 1
        best_mu = self.archive.best.evaluation.stable_mu if self.archive.best else None
        item = SearchRecord(len(self.trace), parent_id, proposal.operation, proposal.parameters, selection,
                            template, evaluation, disposition, best_mu, improved, in_archive, proposal.message)
        self.trace.append(item)
        if self._log is not None:
            data = dict(
                schema_version=1, run_id=self.run_id, seed=self.seed, timestamp=datetime.now(timezone.utc).isoformat(),
                id=item.id, parent_id=parent_id, operation=item.operation, parameters=item.parameters,
                selection=selection, template=_template_data(template),
                evaluation=evaluation_data(evaluation) if evaluation is not None else None,
                disposition=disposition, best_mu=best_mu, improved=improved, in_archive=in_archive,
                evaluations=self.evaluations, sdp_calls=self.budget.sdp_calls, message=item.message,
            )
            self._log.write(json.dumps(_json_safe(data), allow_nan=False) + "\n")
            self._log.flush()
        return item

    def finish(self, stop_reason: str) -> SearchResult:
        self._require_open()
        result = SearchResult(
            self.archive.best, list(self.archive.entries), list(self.trace), self.evaluations,
            self.budget.sdp_calls, self.proposals, self.budget.elapsed_seconds(), stop_reason, self.run_id,
        )
        if self.destination is not None:
            def entry_data(entry: ArchiveEntry) -> dict[str, Any]:
                return dict(id=entry.id, template=_template_data(entry.evaluation.template),
                            evaluation=evaluation_data(entry.evaluation))

            _write_json(self.destination / "result.json", dict(
                schema_version=1, run_id=self.run_id, best=entry_data(result.best) if result.best else None,
                archive=[entry_data(e) for e in result.archive], evaluations=result.evaluations,
                sdp_calls=result.sdp_calls, proposals=result.proposals, elapsed_seconds=result.elapsed_seconds,
                stop_reason=stop_reason,
            ))
        return result

