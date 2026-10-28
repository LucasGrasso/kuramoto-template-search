"""Repeat the original 120-solve S7 structural analysis in a new directory."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STEPS = (
    "inspect_artifacts.py", "analyze_sweep.py", "probe_sensitivity.py",
    "extra_probe.py", "final_fusion.py", "invariants_and_lineage.py", "final_stats.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New scratch directory")
    parser.add_argument("--source-run", type=Path, required=True,
                        help="Complete benchmark run with all solve snapshots, generated under runs/")
    parser.add_argument("--reports", action="store_true", help="Also regenerate figures/HTML; requires the reports extra")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    source_run = args.source_run.resolve()
    required = [source_run / f"sweep/{name}" for name in (
        "checkpoint.json", "result.json", "trace.jsonl", "top-limit.json",
    )]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Generate the full benchmark first; missing inputs: " + ", ".join(missing))
    state = json.loads((source_run / "sweep/checkpoint.json").read_text(encoding="utf-8"))
    if any(not (source_run / "sweep" / item["file"]).is_file() for item in state["records"]):
        parser.error("The source run is missing solve snapshots; regenerate the complete benchmark")
    if args.reports:
        import matplotlib  # Fail before spending any solves if the extra is missing.
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / "scripts/s7_analysis", output / "work")
    (output / "outputs").mkdir()
    env = dict(os.environ, KURAMOTO_REPO=str(ROOT), KURAMOTO_SOURCE_RUN=str(source_run),
               KURAMOTO_FINITE_WITNESS=str(ROOT / "research/joint-research-final/finalist-refinement.json"),
               PYTHONDONTWRITEBYTECODE="1")
    for step in STEPS + (("build_report.py",) if args.reports else ()):
        subprocess.run([sys.executable, str(output / "work" / step)], cwd=output, env=env, check=True)
    experiments = json.loads((output / "outputs/experimentos.json").read_text(encoding="utf-8"))
    if experiments["counts"] != {"joint_sdp": 120, "high_precision_audit": 4}:
        raise RuntimeError(f"Unexpected analysis costs: {experiments['counts']}")
    strict = [item for item in experiments["calls"] if item["status"] == "joint_asymptotic_strict"]
    print(json.dumps(dict(counts=experiments["counts"], best_limit_mu=max(item["mu"] for item in strict))))


if __name__ == "__main__":
    main()
