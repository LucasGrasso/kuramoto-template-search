"""JSON, CSV, and Markdown reporting for search benchmarks."""

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import (
    GroupSummary,
    PairedComparison,
    RunMetrics,
    SummaryStats,
)


def json_safe(value: Any) -> Any:
    """Convert NumPy/Path/nonfinite values into standard JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(data), handle, indent=2, allow_nan=False)
        handle.write("\n")


def _fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}g}"


def _mean_sd(stats: SummaryStats) -> str:
    if stats.mean is None:
        return "n/a"
    if stats.std is None:
        return _fmt(stats.mean)
    return f"{_fmt(stats.mean)} +/- {_fmt(stats.std, 3)}"


def _summary_row(summary: GroupSummary) -> dict[str, Any]:
    metrics = summary.metrics
    return {
        "algorithm": summary.algorithm,
        "initial": summary.initial,
        "budget": summary.budget,
        "runs": summary.runs,
        "stable_run_rate": summary.stable_run_rate,
        "target_run_rate": summary.target_run_rate,
        "best_stable_mu_mean": metrics["best_stable_mu"].mean,
        "best_stable_mu_median": metrics["best_stable_mu"].median,
        "best_stable_mu_std": metrics["best_stable_mu"].std,
        "best_stable_mu_max": metrics["best_stable_mu"].maximum,
        "best_margin_gamma_mean": metrics["best_margin_gamma"].mean,
        "best_margin_gamma_max": metrics["best_margin_gamma"].maximum,
        "margin_by_cluster": json.dumps({m: asdict(stats) for m, stats in summary.margin_by_cluster.items()}, sort_keys=True),
        "stable_templates_mean": metrics["stable_templates"].mean,
        "unique_templates_mean": metrics["unique_templates"].mean,
        "sdp_calls_mean": metrics["sdp_calls"].mean,
        "elapsed_seconds_mean": metrics["elapsed_seconds"].mean,
        "wall_seconds_mean": metrics["wall_seconds"].mean,
        "solver_errors_mean": metrics["solver_errors"].mean,
        "inaccurate_results_mean": metrics["inaccurate_results"].mean,
        "cache_hits_mean": metrics["cache_hits"].mean,
        "cache_misses_mean": metrics["cache_misses"].mean,
    }


_CSV_COLUMNS: tuple[str, ...] = (
    "algorithm",
    "initial",
    "budget",
    "runs",
    "stable_run_rate",
    "target_run_rate",
    "best_stable_mu_mean",
    "best_stable_mu_median",
    "best_stable_mu_std",
    "best_stable_mu_max",
    "best_margin_gamma_mean",
    "best_margin_gamma_max",
    "margin_by_cluster",
    "stable_templates_mean",
    "unique_templates_mean",
    "sdp_calls_mean",
    "elapsed_seconds_mean",
    "wall_seconds_mean",
    "solver_errors_mean",
    "inaccurate_results_mean",
    "cache_hits_mean",
    "cache_misses_mean",
)


def _aggregate_table(summaries: Sequence[GroupSummary]) -> list[str]:
    lines = [
        "| algorithm | initial | budget | runs | best mu (mean +/- sd) | best mu max | stable runs | target runs | best gamma | unique templates | SDP calls | wall s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        metrics = summary.metrics
        lines.append(
            "| "
            + " | ".join(
                (
                    summary.algorithm,
                    summary.initial,
                    str(summary.budget),
                    str(summary.runs),
                    _mean_sd(metrics["best_stable_mu"]),
                    _fmt(metrics["best_stable_mu"].maximum),
                    f"{summary.stable_run_rate:.0%}",
                    f"{summary.target_run_rate:.0%}",
                    "; ".join(f"m={m}: {_fmt(stats.maximum)}" for m, stats in summary.margin_by_cluster.items()) or "n/a",
                    _fmt(metrics["unique_templates"].mean),
                    _fmt(metrics["sdp_calls"].mean),
                    _fmt(metrics["wall_seconds"].mean),
                )
            )
            + " |"
        )
    return lines


def _paired_table(comparisons: Sequence[PairedComparison]) -> list[str]:
    lines = [
        "| initial | budget | candidate vs baseline | pairs | wins | losses | ties | candidate-only stable | baseline-only stable | mean delta mu | median delta mu |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for comparison in comparisons:
        lines.append(
            "| "
            + " | ".join(
                (
                    comparison.initial,
                    "all" if comparison.budget is None else str(comparison.budget),
                    f"{comparison.candidate} vs {comparison.baseline}",
                    str(comparison.pairs),
                    str(comparison.candidate_wins),
                    str(comparison.baseline_wins),
                    str(comparison.ties),
                    str(comparison.candidate_only_stable),
                    str(comparison.baseline_only_stable),
                    _fmt(comparison.mean_delta),
                    _fmt(comparison.median_delta),
                )
            )
            + " |"
        )
    return lines


def _progress_table(summaries: Sequence[GroupSummary]) -> list[str]:
    lines = [
        "| algorithm | initial | budget | 25% | 50% | 75% | 100% |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| "
            + " | ".join(
                (
                    summary.algorithm,
                    summary.initial,
                    str(summary.budget),
                    *(
                        _fmt(summary.progress[key].mean)
                        for key in ("25pct", "50pct", "75pct", "100pct")
                    ),
                )
            )
            + " |"
        )
    return lines


def write_benchmark_report(
    output: str | Path,
    *,
    config_data: Mapping[str, Any],
    runs: Sequence[RunMetrics],
    summaries: Sequence[GroupSummary],
    comparisons: Sequence[PairedComparison],
) -> dict[str, Path]:
    """Write machine-readable and human-readable benchmark artifacts."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    runs_path = root / "runs.jsonl"
    with runs_path.open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(json_safe(run.to_dict()), allow_nan=False) + "\n")
    artifacts["runs"] = runs_path

    summary_path = root / "summary.json"
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "numerical_evidence_only": True,
            "config": config_data,
            "runs": len(runs),
            "summaries": [summary.to_dict() for summary in summaries],
        },
    )
    artifacts["summary"] = summary_path

    csv_path = root / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            writer.writerow(_summary_row(summary))
    artifacts["csv"] = csv_path

    paired_path = root / "paired.json"
    _write_json(
        paired_path,
        {
            "schema_version": 1,
            "comparisons": [asdict(comparison) for comparison in comparisons],
        },
    )
    artifacts["paired"] = paired_path

    report_path = root / "report.md"
    lines = [
        "# Kuramoto template search benchmark",
        "",
        "All results are numerical evidence, not mathematical certificates.",
        "`best mu` is the best strictly stable density. `best gamma` is a separate",
        "target-margin signal for near-feasible templates and is never compared",
        "with mu.",
        "",
        "## Configuration",
        "",
    ]
    for key in sorted(config_data):
        lines.append(f"- **{key}**: `{config_data[key]}`")
    lines.extend(
        [
            "",
            "## Aggregate results",
            "",
            *_aggregate_table(summaries),
            "",
            "`stable runs` is the fraction with at least one stable template;",
            "`target runs` is the fraction reaching the configured target mu.",
            "Costs are means per independent run.",
            "",
            "## Paired comparison",
            "",
            *_paired_table(comparisons),
            "",
            "Pairs match the same initial template, budget, and repetition.",
            "`wins` counts candidate > baseline; a stable result beats no stable",
            "result. `mean delta mu` uses only pairs where both found a stable template.",
            "",
            "## Progress toward the budget",
            "",
            *_progress_table(summaries),
            "",
            "Cells are mean best stable mu reached by that fraction of the",
            "SDP-call budget; `n/a` means no stable template by then.",
            "",
            "## Notes",
            "",
            "- Every run receives the same SDP-call budget. Density and margin",
            "  solves both count as one call.",
            "- The benchmark cache canonicalizes global phase rotations and",
            "  cluster permutations. Cache hits cost zero SDP calls and are",
            "  excluded from unique-template and unique-status counts.",
            "- `optimal_inaccurate`, `infeasible_inaccurate`, solver errors, and",
            "  independently rejected candidates are reported but never treated",
            "  as certified outcomes.",
            "- Refinement and certification must re-evaluate candidates with a",
            "  fresh, uncached evaluator.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    artifacts["report"] = report_path
    return artifacts
