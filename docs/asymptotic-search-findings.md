# Two-stage asymptotic search: first pilot

Run date: 2026-10-26. The engineering milestone succeeds: S7 passes limiting
screening, admits independently validated first-order corrections, and lifts
to finite numerically strict templates. The pilot finds no improvement over
the comparison target 0.6929077063695864. No result is rigorously certified.

## Reproduction and artifacts

```powershell
uv run python -m kuramoto_template.asymptotic_search.benchmark --output runs/asymptotic-pilot-diverse --seeds 0 1 2 --proposals 15 --full-calls 4
uv run --extra rl pytest -q
```

Use a fresh output directory when rerunning. On this sandbox, uv and pytest
needed cache/temporary directories under the workspace (`UV_CACHE_DIR=runs/uv-cache`
and a fresh `--basetemp runs/...`); these do not change the tests or algorithms.
The complete suite discovers all package test directories: **416 passed**,
including 34 new asymptotic tests. Warnings are two deliberately exercised
inaccurate full-SDP solves and the existing Gymnasium observation-space warnings.
Inaccurate correction solves cannot earn the asymptotic strict classification.

The canonical artifacts are under `runs/asymptotic-pilot-diverse/` (ignored by
Git): `summary.jsonl`, `summary.csv`, `calibration.json`, `findings.md`,
`top-candidate.json`, `top-decimal-audit.json`, nine additional Decimal audits,
and complete per-run checkpoints/results/traces. Earlier smoke/pilot directories
are preliminary runs, not the results summarized here.

Runtime libraries: NumPy 2.5.4, SciPy 1.18.1, CVXPY 1.9.3, Clarabel 0.11.1.
Every result records options, tolerances and source hash; the final pilot hash is
`0e265e82c3f9dacf0805da4a0b4890d0f58cf4afb233ffd8fc878334373658de`.
The seeds repeat the same mathematical S7 calibration; their results are not
nine independent discoveries.

## Calibration and falsification

S7 uses the limiting seed from [Table 2 and Appendix A of the reference](https://arxiv.org/html/2610.21271v1).
Both the analytic Appendix-A correction and the optimized correction satisfy
F0/F1, K and sum(k)=I independently. The complete rescaled m-1 matrix agrees
with the dense Taylor route and converges from finite h. No original SDP or
validator formulation was changed.

| t | Limiting mu0 | Dipole S1 | Dipole S2 | Classification |
|---:|---:|---:|---:|---|
| 0.03 | 0.6877620580656508 | 0.01758286 | 0.01939190 | screened |
| 0.01 | 0.6911969800283011 | 0.00592253 | 0.00650930 | screened |
| 0.003 | 0.6923949582276224 | 0.00178317 | 0.00195751 | screened |
| 0 | 0.6929077063695863 | about -1.1e-16 | about -1.1e-16 | uncertain |
| -0.01 | 0.6946139663943252 | -0.00598338 | -0.00655407 | rejected |

The negative-t control has a tempting density above the reference but fails
before any correction or full finite SDP is attempted. The neutral t=0 seed
is not a stable witness. Uncorrected P0 at finite angle fails independent
validation.

On the analytic correction, force and tree-repair log slopes are 1.996–1.9964
and 1.989–1.990 over h=0.01/0.003/0.001/0.0003, consistent with quadratic
decay. Soft-gap slopes range from 2.11 to 2.48: higher-order terms remain
visible as t approaches zero. The narrower robust-t regression grid verifies
convergence toward slope two; the pilot does not interpret the broader-grid
slopes as pure leading-order measurements.

The LP has a flat optimum on S7 and can choose a density matrix that violates
the necessary filters. The stronger SDP finds another feasible matrix with
the same limiting degree. Regression tests require this distinction.

## Finite result and increased precision

The best local lift is S7 at t=0.003, h=0.0001, using the optimized
minimum-norm correction:

| Diagnostic | Value |
|---|---:|
| Finite density | 0.6923027477669624 |
| Base force infinity norm | 1.4616e-9 |
| Maximum tree density correction | 1.1175e-8 |
| Repaired force infinity norm, float64 | 2.7756e-17 |
| Minimum restoring coefficient | 0.21543698211811932 |
| Full restricted cluster gap | 4.6691e-12 |
| Rescaled full gap | 1.7853e-4 |
| Float64 roundoff gate | 1.3607e-14 |
| Box/support violations | 0 |

The actual finite backbone has winding +1. An independent 60-digit Decimal
audit of the saved float inputs finds force error 3.4433e-17, positive grounded
LDL pivots, and the same density/restoring coefficient. Its smallest pivot is
8.8010e-12. These computations do not enclose an exact equilibrium root and
do not use outward rounding. `proof_available=false` and the rigorous archive
is empty.

## Pilot costs and controls

There are 39 search runs: three diverse searches and nine cells for each
calibration/control algorithm. Every paired cell begins at the same masses,
phases (h=0.003) and support. Random and AMCS have an actual four-full-SDP
budget including initialization. Two-stage construction spends zero full
search calls and separately spends four full audit calls per cell. Equality
of those totals does not make total computation equal; the correction SDP
costs remain additional. No-refinement intentionally pays only initialization.

| Method | Runs | Best finite density | Full search calls | Full audit calls | Additional LP / screen SDP / correction SDP | Tree solves | Run wall seconds |
|---|---:|---:|---:|---:|---|---:|---:|
| no refinement | 9 | 0.6899419861771222 | 9 | 0 | 0 / 0 / 0 | 0 | 0.299 |
| random | 9 | 0.6899419861771222 | 36 | 0 | 0 / 0 / 0 | 0 | 0.441 |
| AMCS | 9 | 0.6899419861771222 | 36 | 0 | 0 / 0 / 0 | 0 | 0.456 |
| two-stage S7 | 9 | 0.6923027477669624 | 0 | 36 | 0 / 0 / 18 | 36 | 1.176 |
| diverse two-stage | 3 | 0.692118335594554 | 0 | 0 | 45 / 39 / 36 | 68 | 3.465 |

The analytic calibration additionally uses 12 tree linear solves and zero
conic calls. Ten Decimal audits are recorded separately with their elapsed
times. Total paid optimization invocations across the search table are 255
conic calls; including calibration there are 116 tree linear solves.
Wall times are small local measurements and exclude interpreter startup;
the artifacts retain the per-run and per-invocation times.

Random/AMCS did not improve their fixed-phase initialization in this small
pilot. Their phase radius is 0.0003 and mass-transfer fraction is 0.001; the
two-stage method explicitly explores smaller h. These settings test this
construction and accounting, not general superiority or equal wall-time
performance. A larger or tuned control search could change the comparison.
The separate `repair_sdp` branch was not merged or benchmarked in this milestone.
RL was not benchmarked because no frozen checkpoint was nominated.

The diverse runs make 45 outer proposals, with unequal splits and multiple
masks across 7/8/9/10/12 clusters. Their 18 Stage-2 attempts yield 17 numerically
strict asymptotic corrections and one uncertain result. Strict counts by
size are 7/6/1/1/2 respectively. All 68 attempted finite lifts from the strict
corrections pass the finite numerical gate. These include nearby split
extensions; none improves the S7 calibration.

Thirty-nine LP outputs receive the heuristic failure label. Subsequent
necessary-screen SDPs yield 24 screened results, seven uncertain results,
seven numerical infeasibility statuses and one independently rejected output.
The numerical infeasibility statuses apply only to the fixed outer point and
configured floors, not to the global family or a certified impossibility claim.

Shortlist regression tests prevent many masks at one size from crowding out
larger sizes. The final policy round-robins sizes first and masks within each
size. Checkpoint tests cancel and resume at all three stage boundaries,
reproduce traces/counts, and verify that completed work is not paid again.
Tests also cover inaccurate solver statuses, box/tangent violations, degenerate
hard blocks, exact cache hits and the monotone finite incumbent.

A next research sweep can increase outer proposals and shortlist budgets or
relax the seed symmetries further. A candidate near or above the reference
would require tighter increased-precision audits followed by interval root and
Hessian certification and graph realization before any lower-bound claim.
