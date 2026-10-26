# Kuramoto template search

Phase-cluster templates, mutation operators, and SDP optimization for the homogeneous Kuramoto model.

SDP results are numerical evidence, not mathematical certificates.

## Two-stage asymptotic search

The standalone four-parent search screens limiting split templates, optimizes
first-order corrections with the full effective Hessian, and attempts exact
finite-angle force repair on a spanning tree. It includes the published S7
calibration, separate solver budgets, replayable checkpoints and a small
random/AMCS comparison. See [the API and CLI](src/kuramoto_template/asymptotic_search/README.md)
and [pilot findings](docs/asymptotic-search-findings.md).

```bash
uv run python -m kuramoto_template.asymptotic_search --config examples/asymptotic-small.json --output runs/asym-small
```

## Learned RL policy

Install `uv sync --extra rl` for the optional CPU PyTorch BC/AWAC pipeline.
See [policy training and evaluation](src/kuramoto_template/policy/README.md)
for bounded smoke commands, data/replay safeguards, checkpoints and frozen
held-out comparisons. Historical branch logs train imitation only unless full
sequential context is reconciled. No large research matrix runs by default.

## Gymnasium environment

The optional `environment` package exposes `reset`, `step`, `render`, and
`close`, padded variable-size observations, hierarchical mutation actions,
and exact SDP-call budgets including seed evaluation. Install and run:

```bash
uv sync --extra environment
uv run --extra environment python -m kuramoto_template.environment --seed 42 --episodes 2 --max-sdp-calls 128 --output runs/gym-smoke
```

Import `KuramotoEnv, EnvironmentConfig` from `kuramoto_template.environment`,
or use `gymnasium.make("kuramoto_template.environment:KuramotoTemplate-v0")`.
See [the environment API](src/kuramoto_template/environment/README.md) for
observation/action schemas, rewards, end flags, masks and replay artifacts.
This includes a random policy runner; no learned RL policy is implemented yet.

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

Search defaults to at most 12 clusters. Random and adaptive proposals can create
larger templates, so the cap is a real search restriction rather than a reporting
limit. Raise it explicitly when structure above that size matters, for example
`--max-clusters 20`.

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

Refinement applies the same cluster limit (default 12). When refining a run that
searched with a raised cap, pass the matching `--max-clusters`. An oversized
candidate raises
`... exceed max_clusters=N ... pass --max-clusters M or higher` before any solve;
it is never silently reported as `invalid`.

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

## Current status (numerical)

The `runs/bench-final-*` matrices use 20 independent repetitions for every
`(algorithm, family, budget)` cell, so each budget column holds 80 runs per
algorithm. Budgets count SDP calls, including both density and margin solves.
Mean best strictly stable `mu`, over runs that found a stable template (all 80
for random; a few nested/adaptive runs found none at low budgets):

| SDP budget | random   | nested   | adaptive |
| ---------- | -------- | -------- | -------- |
| 500        | 0.680772 | 0.598494 | 0.635025 |
| 2,000      | 0.681966 | 0.641626 | 0.606130 |
| 5,000      | 0.681966 | 0.675954 | 0.613988 |

- Random search reaches the ten-cluster twisted optimum (`0.681965750`) in every
  run by 2,000 calls. At that budget, 63 of 80 bests are one canonical template
  to 12 decimals. At 5,000 calls, 66 of 80 bests have ten clusters and they
  coincide to `1e-6`; the standard deviation across all bests is `1.5e-8`.
- Paired at 5,000 calls (same starts, 80 pairs): nested wins 5, loses 53, ties
  22 against random; adaptive wins 0, loses 78, ties 2. Extra budget helps nested
  but does not close the gap.
- No run in these matrices reaches `0.6929077` or `0.693`;
  `target_run_rate` is zero everywhere. The best matrix stable `mu` is
  `0.681965858`; the cluster-cap pilot below reaches higher.
- The reported `~1e-9` stability eigenvalues are a `cluster_margin=1e-9` solver
  artifact, not intrinsic marginality; certified re-solves at `2e-6` succeed.

The ten-cluster state is one point in the twisted family. Evaluating
`twisted_seed(m)` with the cluster cap raised to 64 gives:

| `m`  | 8        | 10       | 12       | 16       | 19       | 22       | 25       | 32       | 48       |
| ---- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| `mu` | 0.667893 | 0.681966 | 0.664672 | 0.683847 | 0.683875 | 0.682890 | 0.682140 | 0.678980 | 0.680612 |

The maximum is near `m = 19`, so the default `--max-clusters 12` excludes a
better region. The `runs/bench-clusters-20` pilot (2,000 calls, 20 repetitions,
four families) confirms the cap was binding: 76 of 80 runs reach a 19-cluster
best, 75 of 80 are the identical canonical template, and the overall mean best
`mu` is `0.683870`. A certified audit of that template gives `0.683808332` with
a strict `2e-6` eigenvalue margin, an optimal solve, and only `target_density`
failing.
The best ten-cluster candidate refines to `0.681903635` and likewise fails only
`target_density`.

Both the ten- and nineteen-cluster states are stable fixed points that every run
reproduces, and budget increases converge to them rather than past them. The
current bottleneck is reachable template structure, not search budget. Exceeding
the target requires a non-twisted structural family or direct optimization over
masses and phases; more calls on the same landscape cannot help.
