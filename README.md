# Kuramoto template search

Phase-cluster templates, mutation operators, and SDP optimization for the homogeneous Kuramoto model.

SDP results are numerical evidence, not mathematical certificates.

## Running the tests

```bash
uv run pytest -v
```

## Random search

```bash
uv run python -m kuramoto_template.search.random_search --seed 42 --iterations 100 --output runs/demo
```

Starts from unrestricted five- and four-cluster twisted seeds. Saves configuration,
per-proposal JSONL diagnostics, and the best numerical candidate to a new directory.
`--target-mu` defaults to `0.693`; use `--help` for margins and budget controls.

## Fixed nested rollouts

```bash
uv run python -m kuramoto_template.search.amcs --seed 42 --iterations 4 \
  --level 1 --depth 2 --branching 4 --max-sdp-calls 64 \
  --output runs/amcs-demo
```

This is the fixed-depth nested-rollout baseline.
Level 0 plays a random mutation sequence; level 1 evaluates each child through a
level-0 rollout and selects by endpoint. Stable candidates are ranked by
`stable_mu`; exploratory candidates compare `gamma` only within one cluster-count
bucket. Every proposal shares the global SDP-call budget and is written to
`trace.jsonl`. `paths.json` stores each episode's root, selected record ids,
endpoint and best record for review against the trace.

## Adaptive nested rollouts

```bash
uv run python -m kuramoto_template.search.adaptive --seed 42 --iterations 40 \
  --level 1 --depth 2 --max-level 3 --max-depth 5 --stall-threshold 2 \
  --branching 8 --prune-probability 0.2 --max-sdp-calls 256 \
  --output runs/adaptive-demo
```

Starts from the fixed-depth schedule and adapts it. When the best strictly
stable `mu` stalls for `--stall-threshold` episodes, the controller alternately
raises search depth and nesting level and increases the pruning rate. Any new
best stable `mu` resets depth, level and pruning to their baseline values. At
the configured caps the schedule returns to the baseline while keeping the
raised pruning rate.

Pruning proposals merge the closest-phase clusters, dissolve the smallest
cluster by distributing its mass proportionally, or undo a recent split. They do
not preserve stability and are never treated as witnesses: every pruned
template is evaluated normally before use. Two archives stay separate: stable
templates ranked by `mu`, and near-feasible templates ranked by `gamma` only
within a cluster-count bucket. `adaptive.json` records every episode's
level/depth/pruning decision plus both archive snapshots.

Adaptation settings are also available to the benchmark as
`--adaptive-max-level`, `--adaptive-max-depth`, `--adaptive-stall-threshold`,
`--adaptive-prune-probability`, and `--adaptive-prune-increment`.

All results remain numerical candidates, not mathematical certificates.

## Benchmarking search algorithms

```bash
uv run python -m kuramoto_template.experiments.benchmark \
  --output runs/benchmark-01 \
  --repetitions 20 --budgets 500,2000,5000 \
  --algorithms random,nested,adaptive --level 1 --depth 2 --branching 8
```

Every `(algorithm, initial family, budget, repetition)` cell is an independent
run. Algorithms receive the same starting template for a matched repetition and
the same SDP-call budget. Density and margin solves both count against that
budget. The default families are the five- and eight-cluster twisted states, a
perturbed four-cluster state, and random Dirichlet/uniform phases; register
literature-specific templates with `experiments.seeds.register_initial`.
With three algorithms, paired comparisons use the first algorithm as the
baseline.

Each run writes the normal `metadata.json`, `result.json`, and `trace.jsonl`
plus `benchmark.json`. The benchmark root contains `runs.jsonl`, `summary.json`,
`summary.csv`, `paired.json`, and `report.md`. Reported metrics include best
stable `mu`, best target-margin `gamma`, stable and near-feasible template
counts, canonical unique templates, density/margin call splits, wall time, and
solver-error/inaccuracy counts. `best mu` and `best gamma` are never mixed into
one ranking.

Evaluations are cached per run after canonicalizing global phase rotations and
cluster permutations. Cache hits cost zero SDP calls and are excluded from
unique-template counts. The cache is only for screening; refinement or
certification must re-evaluate candidates with a fresh, uncached evaluator
(cache hits map `P` back to the requested cluster order but remain quantized
screening results).
Use `--resume` to continue an interrupted matrix; completed run directories are
never reused or overwritten.

A full 20-repetition, three-budget, three-algorithm, four-family matrix is
several hundred runs and can take hours. For a quick workflow check, use
`--repetitions 1 --budgets 64`.

All reported outcomes are numerical evidence, not certificates.
`optimal_inaccurate` results are counted and reported but never treated as
proofs.

## Refining and auditing candidates

```bash
uv run python -m kuramoto_template.experiments.refinement \
  --result runs/bench-adaptive-pilot-256/runs/random/five_twisted/budget-256/rep-000 \
  --output runs/refinement/best-refined.json \
  --level refined
```

Accepts a run directory, `result.json`, or `trace.jsonl`, and writes a new JSON
file (`--force` refines every candidate instead of only triggered ones). Three
levels are available:

- `fast` — the search evaluator's own solve, reused for screening; no second
  solve is paid for.
- `refined` — a fresh, uncached evaluator with tighter margins and residuals,
  plus an independent audit that recomputes `mu`, force residuals, and
  eigenvalue margins from the returned density matrix. A near-feasible candidate
  is audited through its margin solve's matrix.
- `certified` — the refined audit plus a conservative numerical gate: optimal
  solver status, strict stability/target checks, and agreement between reported
  and recomputed objectives. The certified solve requests twice
  `--certification-margin` so the returned matrix has headroom, then requires the
  real margin.

Refinement triggers are stable `mu` above `--mu-threshold` (default `0.69`),
small or uncertain stability eigenvalues, near-zero target margins, and
`optimal_inaccurate` solver statuses. Inaccurate results are kept as candidates
for review, never upgraded to counterexamples or witnesses.

Refinement bypasses the benchmark cache on purpose: hits reuse a quantized
representative and cost zero SDP calls, so they cannot support a fresh audit.
For Clarabel, refined/certified levels explicitly request `tol_gap_abs`,
`tol_gap_rel` and `tol_feas` of `1e-10` with up to 300 iterations; explicit
`--solver-options` override these defaults. The independent audit uses a
separate pairwise/Hessian implementation and checks its actual force and
stability values against the requested thresholds. Arithmetic is still
float64, not arbitrary precision, and the output records that fact.

`certified` is still only a numerical gate. `proof_available` is always `false`
in the output; interval arithmetic and a rigorous equilibrium/stability proof
remain future work. Every result is numerical evidence, not a certificate.

## Numerical and trajectory safeguards

Synchronization uses the smallest containing circular arc, so validity is
independent of cluster order. Templates provably contained in an open semicircle
cannot receive a stable score: nonnegative equilibrium forces would disconnect
the distinct phases. Margin solves can retain nonpositive exploration signals.
Stability screening also checks requested positive margins with a relative
allowance, rather than letting a large absolute residual tolerance erase them.
These checks do not prove exact existence for other ill-conditioned templates.

Gamma is reported separately for each cluster count, never as a winner across
different sizes. `best_margin_by_cluster` stores those buckets; legacy scalar
gamma fields are populated only when a run has one cluster-count bucket.

Adaptive restart and pruning roots enter the dedicated archives immediately,
including at budget exhaustion. Episode costs and improvements include root
creation. Pruning `parent_id` names its actual mutation source;
`source_root_id` retains the independently selected episode root. Undo-split
records include the sampled perturbation index and displacement for exact
replay. Historical traces are not rewritten and may have the earlier defects.
New run metadata records evaluator schema version 2 and Python/platform data.
