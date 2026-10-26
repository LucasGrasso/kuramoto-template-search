"""Benchmark orchestration, persistence, and resume behavior."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ..benchmark import (
    BenchmarkConfig,
    build_run_specs,
    derive_seed,
    execute_benchmark,
)
from .support import FakeEvaluator


class Factory:
    def __init__(self) -> None:
        self.created = 0

    def __call__(self) -> FakeEvaluator:
        self.created += 1
        return FakeEvaluator()


def tiny_config(output: Path, **overrides: Any) -> BenchmarkConfig:
    data: dict[str, Any] = {
        "output": output,
        "master_seed": 11,
        "repetitions": 2,
        "budgets": (12,),
        "algorithms": ("random", "nested"),
        "initials": ("five_twisted", "random_phases"),
        "min_clusters": 3,
        "max_clusters": 12,
        "archive_size": 8,
        "level": 1,
        "depth": 1,
        "branching": 2,
        "cache_decimals": 12,
    }
    data.update(overrides)
    return BenchmarkConfig(**data)


@pytest.mark.parametrize(
    "overrides",
    [
        {"budgets": ()},
        {"budgets": (0,)},
        {"budgets": (12, 12)},
        {"algorithms": ("unknown",)},
        {"initials": ("unknown",)},
        {"repetitions": 0},
        {"level": -1},
        {"depth": -1},
        {"restart_probability": 2.0},
        {"cache_decimals": -1},
        {"target_mu": 1.0},
        {"adaptive_max_level": 0},
        {"adaptive_max_depth": 0},
        {"adaptive_stall_threshold": 0},
        {"adaptive_prune_probability": 1.5},
        {"adaptive_prune_increment": -1.0},
    ],
)
def test_invalid_config_is_rejected(tmp_path: Path, overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        tiny_config(tmp_path / "bench", **overrides)


def test_derive_seed_is_stable_and_distinct() -> None:
    assert derive_seed(1, "initial", "five_twisted", 0) == derive_seed(
        1, "initial", "five_twisted", 0
    )
    assert derive_seed(1, "a", 0) != derive_seed(1, "a", 1)
    assert 0 <= derive_seed("anything") < 2**32


def test_build_run_specs_is_deterministic_and_paired() -> None:
    config = tiny_config(Path("unused"))
    specs = build_run_specs(config)
    assert len(specs) == 2 * 2 * 1 * 2
    assert specs == build_run_specs(config)

    first = next(spec for spec in specs if spec.algorithm == "random")
    paired = next(
        spec
        for spec in specs
        if spec.algorithm == "nested"
        and spec.initial == first.initial
        and spec.budget == first.budget
        and spec.repetition == first.repetition
    )
    assert paired.initial_seed == first.initial_seed
    assert paired.run_seed == first.run_seed
    assert paired.output != first.output


def test_execute_benchmark_writes_runs_and_reports(tmp_path: Path) -> None:
    config = tiny_config(tmp_path / "bench")
    factory = Factory()
    result = execute_benchmark(config, evaluator_factory=factory)

    assert len(result.runs) == 8
    assert factory.created == 8
    assert result.output == config.output
    for name in ("runs", "summary", "csv", "paired", "report"):
        assert result.artifacts[name].exists()
    assert (config.output / "benchmark_config.json").exists()

    for run in result.runs:
        assert run.sdp_calls == run.density_sdp_calls + run.margin_sdp_calls
        assert run.sdp_calls <= run.budget
        directory = (
            config.output
            / "runs"
            / run.algorithm
            / run.initial
            / f"budget-{run.budget}"
            / f"rep-{run.repetition:03d}"
        )
        assert (directory / "metadata.json").exists()
        assert (directory / "result.json").exists()
        assert (directory / "trace.jsonl").exists()
        payload = json.loads((directory / "benchmark.json").read_text("utf-8"))
        assert payload["metrics"]["algorithm"] == run.algorithm

    summary = json.loads(result.artifacts["summary"].read_text("utf-8"))
    assert summary["runs"] == 8
    assert len(summary["summaries"]) == 4


def test_resume_skips_completed_runs(tmp_path: Path) -> None:
    config = tiny_config(tmp_path / "bench")
    factory = Factory()
    first = execute_benchmark(config, evaluator_factory=factory)
    created = factory.created

    resumed = execute_benchmark(
        replace(config, resume=True), evaluator_factory=factory
    )
    assert factory.created == created
    assert len(resumed.runs) == len(first.runs)
    assert [run.run_id for run in resumed.runs] == [run.run_id for run in first.runs]


def test_existing_nonempty_root_requires_resume(tmp_path: Path) -> None:
    config = tiny_config(tmp_path / "bench")
    execute_benchmark(config, evaluator_factory=Factory())
    with pytest.raises(FileExistsError, match="not empty"):
        execute_benchmark(config, evaluator_factory=Factory())


def test_resume_rejects_config_mismatch(tmp_path: Path) -> None:
    config = tiny_config(tmp_path / "bench")
    execute_benchmark(config, evaluator_factory=Factory())
    with pytest.raises(ValueError, match="does not match"):
        execute_benchmark(
            replace(config, resume=True, master_seed=999),
            evaluator_factory=Factory(),
        )


def test_resume_rejects_incomplete_run_directory(tmp_path: Path) -> None:
    config = tiny_config(
        tmp_path / "bench",
        repetitions=1,
        algorithms=("random",),
        initials=("five_twisted",),
    )
    result = execute_benchmark(config, evaluator_factory=Factory())
    marker = (
        config.output
        / "runs"
        / "random"
        / "five_twisted"
        / "budget-12"
        / "rep-000"
        / "benchmark.json"
    )
    marker.unlink()
    assert len(result.runs) == 1
    with pytest.raises(FileExistsError, match="Incomplete"):
        execute_benchmark(
            replace(config, resume=True), evaluator_factory=Factory()
        )


def test_empty_existing_root_is_allowed(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    root.mkdir()
    config = tiny_config(
        root,
        repetitions=1,
        algorithms=("random",),
        initials=("five_twisted",),
    )
    result = execute_benchmark(config, evaluator_factory=Factory())
    assert len(result.runs) == 1


def test_adaptive_algorithm_runs_with_shared_budget(tmp_path: Path) -> None:
    config = tiny_config(
        tmp_path / "bench",
        repetitions=1,
        budgets=(16,),
        algorithms=("adaptive",),
        initials=("five_twisted",),
        level=0,
        depth=1,
        branching=2,
        adaptive_max_level=1,
        adaptive_max_depth=2,
        adaptive_stall_threshold=1,
        adaptive_prune_probability=1.0,
    )
    result = execute_benchmark(config, evaluator_factory=Factory())
    assert len(result.runs) == 1
    run = result.runs[0]
    assert run.algorithm == "adaptive"
    assert 0 < run.sdp_calls <= 16
    assert run.density_sdp_calls + run.margin_sdp_calls == run.sdp_calls
    directory = (
        config.output
        / "runs"
        / "adaptive"
        / "five_twisted"
        / "budget-16"
        / "rep-000"
    )
    assert (directory / "adaptive.json").exists()
    adaptive = json.loads((directory / "adaptive.json").read_text())
    assert adaptive["algorithm"] == "adaptive_amcs"
    assert adaptive["episodes"]
    assert (config.output / "report.md").exists()


@pytest.mark.integration
def test_real_sdp_benchmark_smoke(tmp_path: Path) -> None:
    config = tiny_config(
        tmp_path / "real",
        repetitions=1,
        budgets=(8,),
        algorithms=("random",),
        initials=("five_twisted",),
    )
    result = execute_benchmark(config)
    assert len(result.runs) == 1
    run = result.runs[0]
    assert 0 < run.sdp_calls <= 8
    assert run.density_sdp_calls + run.margin_sdp_calls == run.sdp_calls
    assert (config.output / "report.md").exists()
