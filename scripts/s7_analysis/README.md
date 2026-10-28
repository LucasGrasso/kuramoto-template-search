# Structural analysis stages

Run through `scripts/reproduce_s7_analysis.py --source-run runs/joint-replay`
from the repository root after generating the complete benchmark. The wrapper copies these scripts into a new scratch `work/` directory and sets
`KURAMOTO_REPO` and `KURAMOTO_SOURCE_RUN`; outputs stay inside that scratch run.

The original order is inspect, analyze, sensitivity (107 joint calls and four
Decimal audits), extra offsets (10 calls), final fusions (three calls),
invariants/lineage, and final statistics. `--reports` adds Matplotlib plotting
and Markdown/HTML generation without solver calls. Git preserves the original
reports, figures, aggregate summaries and winning witness in
`research/s7-analysis/`; full snapshots are regenerated under ignored `runs/`.
The wrapper passes the retained best finite witness as report context through
`KURAMOTO_FINITE_WITNESS`.

The scripts have only been adapted for paths, shared persistence imports,
source provenance and current report setup. Numerical experiments and proposal
ordering are retained. All reported solver costs remain separate from the
original 1,050-proposal sweep and from witness replay.
