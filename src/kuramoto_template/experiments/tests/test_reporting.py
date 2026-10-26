"""Reporting artifacts are written and contain the expected fields."""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..metrics import RunMetrics, paired_comparisons, summarize_runs
from ..reporting import json_safe, write_benchmark_report


def run_metrics(**overrides: Any) -> RunMetrics:
    data: dict[str, Any] = {
        "algorithm": "random",
        "initial": "five_twisted",
        "budget": 50,
        "repetition": 0,
        "run_seed": 1,
        "run_id": "run",
        "best_stable_mu": 0.68,
        "best_margin_gamma": -0.03,
        "sdp_calls": 50,
        "unique_templates": 12,
        "wall_seconds": 0.5,
    }
    data.update(overrides)
    return RunMetrics(**data)


def test_json_safe_normalizes_paths_arrays_and_nonfinite() -> None:
    value = {
        "path": Path("runs") / "x",
        "array": np.array([1.0, 2.0]),
        "scalar": np.float64(1.5),
        "bad": np.inf,
        "tuple": (1, np.int64(2)),
    }
    assert json_safe(value) == {
        "path": str(Path("runs") / "x"),
        "array": [1.0, 2.0],
        "scalar": 1.5,
        "bad": None,
        "tuple": [1, 2],
    }


def test_write_benchmark_report(tmp_path: Path) -> None:
    runs = [
        run_metrics(algorithm="random", repetition=0, best_stable_mu=0.68),
        run_metrics(algorithm="nested", repetition=0, best_stable_mu=0.69),
    ]
    summaries = summarize_runs(runs, target_mu=0.693)
    comparisons = paired_comparisons(runs, baseline="random", candidate="nested")
    config = {"master_seed": 1, "budgets": (50,), "target_mu": 0.693}
    artifacts = write_benchmark_report(
        tmp_path,
        config_data=config,
        runs=runs,
        summaries=summaries,
        comparisons=comparisons,
    )
    assert set(artifacts) == {"runs", "summary", "csv", "paired", "report"}
    for path in artifacts.values():
        assert path.exists()

    rows = (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    assert json.loads(rows[0])["algorithm"] == "random"

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["runs"] == 2
    assert summary["numerical_evidence_only"] is True
    assert len(summary["summaries"]) == 2

    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        csv_rows = list(reader)
    assert reader.fieldnames is not None
    assert "best_stable_mu_mean" in reader.fieldnames
    assert len(csv_rows) == 2
    assert csv_rows[0]["algorithm"] in {"random", "nested"}

    paired = json.loads((tmp_path / "paired.json").read_text(encoding="utf-8"))
    assert paired["comparisons"][-1]["pairs"] == 1

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "numerical evidence" in report
    assert "Aggregate results" in report
    assert "Paired comparison" in report
    assert "random" in report and "nested" in report
