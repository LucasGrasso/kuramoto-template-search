"""Benchmark runner: compare search algorithms at equal SDP-call budgets.

Example
-------
``uv run python -m kuramoto_template.experiments.benchmark --output runs/bench``

Every (algorithm, initial family, budget, repetition) cell is an independent
run. Algorithms receive the same initial template for a matched repetition and
the same SDP-call budget. Each run writes normal search artifacts plus
``benchmark.json``; the benchmark root gets per-run JSONL, aggregate JSON/CSV,
paired comparisons, and a Markdown report.

This module only orchestrates and measures. Search policy stays in
``search.random_search`` and ``search.amcs``; evaluation/caching stays in
``experiments.cache``.
"""

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from kuramoto_template.search.adaptive import (
    AdaptiveSearchConfig,
    run_adaptive_search,
)
from kuramoto_template.search.amcs import NestedSearchConfig, run_nested_search
from kuramoto_template.search.evaluator import EvaluationConfig, TemplateEvaluator
from kuramoto_template.search.random_search import RandomSearchConfig, run_random_search

from .cache import CacheableEvaluator, CachedEvaluator
from .metrics import (
    GroupSummary,
    PairedComparison,
    RunMetrics,
    compute_run_metrics,
    paired_comparisons,
    summarize_runs,
)
from .reporting import json_safe, write_benchmark_report
from .seeds import INITIAL_FAMILIES, build_initial

SUPPORTED_ALGORITHMS = ("random", "nested", "adaptive")
DEFAULT_INITIALS = ("five_twisted", "eight_twisted", "four_perturbed", "random_phases")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Settings shared by every run in one benchmark.

    ``repetitions`` counts independent runs per
    (algorithm, initial family, budget) cell. Budgets are SDP-call limits, not
    proposal counts. Search configuration here is intentionally modest; the
    point is a fair comparison, not a tuned sweep.
    """

    output: Path
    master_seed: int = 20261025
    repetitions: int = 20
    budgets: tuple[int, ...] = (500, 2000, 5000)
    algorithms: tuple[str, ...] = ("random", "nested")
    initials: tuple[str, ...] = DEFAULT_INITIALS
    target_mu: float = 0.693
    min_clusters: int = 3
    max_clusters: int = 12
    archive_size: int = 32
    level: int = 1
    depth: int = 2
    branching: int = 8
    restart_probability: float = 0.1
    exploration_probability: float = 0.2
    adaptive_max_level: int = 3
    adaptive_max_depth: int = 5
    adaptive_stall_threshold: int = 2
    adaptive_prune_probability: float = 0.2
    adaptive_prune_increment: float = 0.2
    cache_decimals: int = 12
    max_wall_seconds: float | None = None
    resume: bool = False
    solver: str = "CLARABEL"
    solver_options: dict[str, Any] = field(default_factory=dict)
    internal_margin: float = 1e-7
    cluster_margin: float = 1e-8
    stability_tolerance: float = 1e-10
    residual_tolerance: float = 1e-7

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", Path(self.output))
        object.__setattr__(self, "budgets", tuple(self.budgets))
        object.__setattr__(self, "algorithms", tuple(self.algorithms))
        object.__setattr__(self, "initials", tuple(self.initials))
        object.__setattr__(self, "solver_options", dict(self.solver_options))

        if not self.output.name:
            raise ValueError("output must name a directory")
        for name, lower in (
            ("master_seed", 0),
            ("repetitions", 1),
            ("min_clusters", 2),
            ("max_clusters", 2),
            ("archive_size", 1),
            ("level", 0),
            ("depth", 0),
            ("branching", 1),
            ("adaptive_max_level", 0),
            ("adaptive_max_depth", 0),
            ("adaptive_stall_threshold", 1),
            ("cache_decimals", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        if self.max_clusters < self.min_clusters:
            raise ValueError("max_clusters must be >= min_clusters")
        if not self.budgets:
            raise ValueError("At least one SDP budget is required")
        for budget in self.budgets:
            if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
                raise ValueError("budgets must contain positive integers")
        if len(set(self.budgets)) != len(self.budgets):
            raise ValueError("budgets must be unique")
        if not self.algorithms:
            raise ValueError("At least one algorithm is required")
        unknown = sorted(set(self.algorithms) - set(SUPPORTED_ALGORITHMS))
        if unknown:
            raise ValueError(f"Unknown algorithms: {', '.join(unknown)}")
        if len(set(self.algorithms)) != len(self.algorithms):
            raise ValueError("algorithms must be unique")
        if not self.initials:
            raise ValueError("At least one initial family is required")
        unknown = sorted(set(self.initials) - set(INITIAL_FAMILIES))
        if unknown:
            raise ValueError(f"Unknown initial families: {', '.join(unknown)}")
        if not np.isfinite(self.target_mu) or not 0 < self.target_mu < 1:
            raise ValueError("target_mu must be finite and in (0, 1)")
        for name in (
            "restart_probability",
            "exploration_probability",
            "adaptive_prune_probability",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not np.isfinite(self.adaptive_prune_increment) or (
            self.adaptive_prune_increment < 0
        ):
            raise ValueError("adaptive_prune_increment must be finite and nonnegative")
        if self.adaptive_max_level < self.level:
            raise ValueError("adaptive_max_level must be >= level")
        if self.adaptive_max_depth < self.depth:
            raise ValueError("adaptive_max_depth must be >= depth")
        for name in (
            "internal_margin",
            "cluster_margin",
            "stability_tolerance",
            "residual_tolerance",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.internal_margin <= 0 or self.cluster_margin <= 0:
            raise ValueError("SDP margins must be positive")
        if not self.solver:
            raise ValueError("solver must be nonempty")
        if self.max_wall_seconds is not None and (
            not np.isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0
        ):
            raise ValueError("max_wall_seconds must be finite and positive")

    def evaluator_config(self) -> EvaluationConfig:
        return EvaluationConfig(
            mu_target=self.target_mu,
            internal_margin=self.internal_margin,
            cluster_margin=self.cluster_margin,
            residual_tolerance=self.residual_tolerance,
            stability_tolerance=self.stability_tolerance,
            solver=self.solver,
            solver_options=dict(self.solver_options),
            max_clusters=self.max_clusters,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output"] = str(self.output)
        data.pop("resume", None)
        return data


@dataclass(frozen=True)
class RunSpec:
    """One independent benchmark run."""

    algorithm: str
    initial: str
    budget: int
    repetition: int
    initial_seed: int
    run_seed: int
    output: Path


@dataclass(eq=False)
class BenchmarkResult:
    config: BenchmarkConfig
    runs: list[RunMetrics]
    summaries: list[GroupSummary]
    comparisons: list[PairedComparison]
    output: Path
    artifacts: dict[str, Path]

    @property
    def best(self) -> RunMetrics | None:
        stable = [run for run in self.runs if run.best_stable_mu is not None]
        if not stable:
            return None
        return max(stable, key=lambda run: run.best_stable_mu or -np.inf)


def derive_seed(*parts: int | str) -> int:
    """Stable 32-bit seed derived from arbitrary parts."""
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def build_run_specs(config: BenchmarkConfig) -> list[RunSpec]:
    """Enumerate the full benchmark matrix deterministically."""
    specs: list[RunSpec] = []
    for algorithm in config.algorithms:
        for initial in config.initials:
            for budget in config.budgets:
                for repetition in range(config.repetitions):
                    initial_seed = derive_seed(
                        config.master_seed, "initial", initial, repetition
                    )
                    run_seed = derive_seed(
                        config.master_seed, "run", initial, repetition, budget
                    )
                    output = (
                        config.output
                        / "runs"
                        / algorithm
                        / initial
                        / f"budget-{budget}"
                        / f"rep-{repetition:03d}"
                    )
                    specs.append(
                        RunSpec(
                            algorithm=algorithm,
                            initial=initial,
                            budget=budget,
                            repetition=repetition,
                            initial_seed=initial_seed,
                            run_seed=run_seed,
                            output=output,
                        )
                    )
    return specs


def _iterations(algorithm: str, budget: int) -> int:
    # Generous proposal caps: the SDP budget is the real stopping rule. Caps
    # keep duplicate-only streaks from spinning forever.
    if algorithm == "random":
        return max(100, budget * 2)
    return max(100, budget)


def _search(
    spec: RunSpec,
    config: BenchmarkConfig,
    evaluator: CacheableEvaluator,
    initial_template: Any,
) -> object:
    if spec.algorithm == "random":
        return run_random_search(
            [initial_template],
            evaluator,
            RandomSearchConfig(
                iterations=_iterations(spec.algorithm, spec.budget),
                seed=spec.run_seed,
                min_clusters=config.min_clusters,
                max_clusters=config.max_clusters,
                archive_size=config.archive_size,
                restart_probability=config.restart_probability,
                max_sdp_calls=spec.budget,
                max_wall_seconds=config.max_wall_seconds,
            ),
            output=spec.output,
        )
    if spec.algorithm == "nested":
        return run_nested_search(
            [initial_template],
            evaluator,
            NestedSearchConfig(
                seed=spec.run_seed,
                min_clusters=config.min_clusters,
                max_clusters=config.max_clusters,
                archive_size=config.archive_size,
                iterations=_iterations(spec.algorithm, spec.budget),
                level=config.level,
                depth=config.depth,
                branching=config.branching,
                restart_probability=config.restart_probability,
                exploration_probability=config.exploration_probability,
                max_sdp_calls=spec.budget,
                max_wall_seconds=config.max_wall_seconds,
            ),
            output=spec.output,
        )
    if spec.algorithm == "adaptive":
        return run_adaptive_search(
            [initial_template],
            evaluator,
            AdaptiveSearchConfig(
                seed=spec.run_seed,
                min_clusters=config.min_clusters,
                max_clusters=config.max_clusters,
                archive_size=config.archive_size,
                iterations=_iterations(spec.algorithm, spec.budget),
                level=config.level,
                depth=config.depth,
                max_level=config.adaptive_max_level,
                max_depth=config.adaptive_max_depth,
                stall_threshold=config.adaptive_stall_threshold,
                branching=config.branching,
                restart_probability=config.restart_probability,
                exploration_probability=config.exploration_probability,
                prune_probability=config.adaptive_prune_probability,
                prune_increment=config.adaptive_prune_increment,
                max_sdp_calls=spec.budget,
                max_wall_seconds=config.max_wall_seconds,
            ),
            output=spec.output,
        )
    raise ValueError(f"Unsupported algorithm: {spec.algorithm}")


def _write_new_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(json_safe(data), handle, indent=2, allow_nan=False)
        handle.write("\n")


def _run_one(
    spec: RunSpec,
    config: BenchmarkConfig,
    evaluator_factory: Callable[[], CacheableEvaluator],
) -> RunMetrics:
    initial_template = build_initial(spec.initial, spec.initial_seed)
    if not config.min_clusters <= initial_template.n_clusters <= config.max_clusters:
        raise ValueError(
            f"Initial family {spec.initial!r} produced {initial_template.n_clusters} "
            "clusters outside the configured search bounds"
        )
    evaluator = CachedEvaluator(evaluator_factory(), decimals=config.cache_decimals)
    started = perf_counter()
    result = _search(spec, config, evaluator, initial_template)
    wall_seconds = perf_counter() - started
    metrics = compute_run_metrics(
        result,
        algorithm=spec.algorithm,
        initial=spec.initial,
        budget=spec.budget,
        repetition=spec.repetition,
        run_seed=spec.run_seed,
        cache_stats=evaluator.stats,
        wall_seconds=wall_seconds,
        cache_decimals=config.cache_decimals,
    )
    _write_new_json(
        spec.output / "benchmark.json",
        {
            "schema_version": 1,
            "spec": {
                "algorithm": spec.algorithm,
                "initial": spec.initial,
                "budget": spec.budget,
                "repetition": spec.repetition,
                "initial_seed": spec.initial_seed,
                "run_seed": spec.run_seed,
            },
            "cache": evaluator.stats.as_dict(),
            "metrics": metrics.to_dict(),
        },
    )
    return metrics


def _verify_existing_config(root: Path, config: BenchmarkConfig) -> None:
    config_path = root / "benchmark_config.json"
    if not config_path.exists():
        raise ValueError(
            f"Cannot resume: {config_path} is missing. Use a new output directory."
        )
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    expected = json_safe(config.to_dict())
    if saved != expected:
        raise ValueError(
            "Cannot resume: benchmark_config.json does not match the requested "
            "configuration. Use the original settings or a new output directory."
        )


def execute_benchmark(
    config: BenchmarkConfig,
    *,
    evaluator_factory: Callable[[], CacheableEvaluator] | None = None,
    progress: Callable[[int, int, RunMetrics], None] | None = None,
) -> BenchmarkResult:
    """Run the full matrix, optionally resuming completed cells."""
    if evaluator_factory is None:
        evaluator_factory = lambda: TemplateEvaluator(config.evaluator_config())

    root = config.output
    if root.exists():
        if not root.is_dir():
            raise FileExistsError(f"Output path is not a directory: {root}")
        if config.resume:
            _verify_existing_config(root, config)
        elif any(root.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {root}. Use a new directory or --resume."
            )
        else:
            _write_new_json(root / "benchmark_config.json", config.to_dict())
    else:
        root.mkdir(parents=True, exist_ok=True)
        _write_new_json(root / "benchmark_config.json", config.to_dict())

    specs = build_run_specs(config)
    runs: list[RunMetrics] = []
    for index, spec in enumerate(specs, start=1):
        marker = spec.output / "benchmark.json"
        if config.resume and marker.exists():
            payload = json.loads(marker.read_text("utf-8"))
            runs.append(RunMetrics.from_dict(payload["metrics"]))
        else:
            if spec.output.exists():
                raise FileExistsError(
                    f"Incomplete run directory: {spec.output}. "
                    "Resume only completed runs or choose a new output directory."
                )
            runs.append(_run_one(spec, config, evaluator_factory))
        if progress is not None:
            progress(index, len(specs), runs[-1])

    summaries = summarize_runs(runs, target_mu=config.target_mu)
    comparisons: list[PairedComparison] = []
    if len(config.algorithms) >= 2:
        baseline = config.algorithms[0]
        for candidate in config.algorithms[1:]:
            comparisons.extend(
                paired_comparisons(runs, baseline=baseline, candidate=candidate)
            )
    artifacts = write_benchmark_report(
        root,
        config_data=config.to_dict(),
        runs=runs,
        summaries=summaries,
        comparisons=comparisons,
    )
    return BenchmarkResult(
        config=config,
        runs=runs,
        summaries=summaries,
        comparisons=comparisons,
        output=root,
        artifacts=artifacts,
    )


def _csv_tuple(text: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value")
    return values


def _csv_ints(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Benchmark directory (never reused)"
    )
    parser.add_argument("--master-seed", type=int, default=20261025)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--budgets", type=_csv_ints, default=(500, 2000, 5000))
    parser.add_argument(
        "--algorithms", type=_csv_tuple, default=("random", "nested")
    )
    parser.add_argument("--initials", type=_csv_tuple, default=DEFAULT_INITIALS)
    parser.add_argument("--target-mu", type=float, default=0.693)
    parser.add_argument("--min-clusters", type=int, default=3)
    parser.add_argument("--max-clusters", type=int, default=12)
    parser.add_argument("--archive-size", type=int, default=32)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--branching", type=int, default=8)
    parser.add_argument("--restart-probability", type=float, default=0.1)
    parser.add_argument("--exploration-probability", type=float, default=0.2)
    parser.add_argument("--adaptive-max-level", type=int, default=3)
    parser.add_argument("--adaptive-max-depth", type=int, default=5)
    parser.add_argument("--adaptive-stall-threshold", type=int, default=2)
    parser.add_argument("--adaptive-prune-probability", type=float, default=0.2)
    parser.add_argument("--adaptive-prune-increment", type=float, default=0.2)
    parser.add_argument("--cache-decimals", type=int, default=12)
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--resume", action="store_true")
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
        config = BenchmarkConfig(
            output=args.output,
            master_seed=args.master_seed,
            repetitions=args.repetitions,
            budgets=args.budgets,
            algorithms=args.algorithms,
            initials=args.initials,
            target_mu=args.target_mu,
            min_clusters=args.min_clusters,
            max_clusters=args.max_clusters,
            archive_size=args.archive_size,
            level=args.level,
            depth=args.depth,
            branching=args.branching,
            restart_probability=args.restart_probability,
            exploration_probability=args.exploration_probability,
            adaptive_max_level=args.adaptive_max_level,
            adaptive_max_depth=args.adaptive_max_depth,
            adaptive_stall_threshold=args.adaptive_stall_threshold,
            adaptive_prune_probability=args.adaptive_prune_probability,
            adaptive_prune_increment=args.adaptive_prune_increment,
            cache_decimals=args.cache_decimals,
            max_wall_seconds=args.max_wall_seconds,
            resume=args.resume,
            solver=args.solver,
            solver_options=args.solver_options,
            internal_margin=args.internal_margin,
            cluster_margin=args.cluster_margin,
            stability_tolerance=args.stability_tolerance,
            residual_tolerance=args.residual_tolerance,
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))

    def show(index: int, total: int, run: RunMetrics) -> None:
        print(
            json.dumps(
                {
                    "progress": f"{index}/{total}",
                    "algorithm": run.algorithm,
                    "initial": run.initial,
                    "budget": run.budget,
                    "repetition": run.repetition,
                    "best_mu": run.best_stable_mu,
                    "sdp_calls": run.sdp_calls,
                    "wall_seconds": round(run.wall_seconds, 3),
                }
            ),
            flush=True,
        )

    try:
        result = execute_benchmark(config, progress=show)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))

    best = result.best
    print(
        json.dumps(
            {
                "runs": len(result.runs),
                "best_mu": best.best_stable_mu if best is not None else None,
                "best_algorithm": best.algorithm if best is not None else None,
                "best_initial": best.initial if best is not None else None,
                "artifacts": {name: str(path) for name, path in result.artifacts.items()},
            }
        )
    )
    print("Numerical candidates only; no rigorous certificate. Results:", result.output)


if __name__ == "__main__":
    main()
