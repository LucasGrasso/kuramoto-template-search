# Joint limiting SDP: research pilot

Date: 2026-10-26. Branch: `asymptotic_search`.

The joint convex model, target-margin mode, weak-mode structural proposals,
resource ledger and resumable outer driver are implemented. **445 tests pass**,
including all 416 previous tests and 29 additive tests. The calibration and
1,050-proposal research sweep completed. No limiting candidate exceeded
0.6930 or the comparison target 0.6929077063695864. No rigorous certificate
was produced.

## Reproduce and inspect

```powershell
uv run python -m kuramoto_template.asymptotic_search.joint_benchmark --output runs/joint-research-final --sweep-proposals 1050 --seed 0
uv run --extra rl pytest -q
```

Use a fresh output directory for another run. The sandbox required
`UV_CACHE_DIR` and a fresh pytest `--basetemp` under `runs/`; these affect only
cache/temporary-file placement. [API, model and CLI details](joint-search.md)
include the smaller example configuration and single-geometry solver CLI.

Canonical artifacts are in `runs/joint-research-final/`: calibration results,
90 paired rows in JSONL/CSV, complete per-cell inputs/results, `sweep/` with
checkpoint/trace/archives/per-solve snapshots and top candidates, summary and
findings. `finalist-refinement.json` and `refine_finalist.py` contain a separately
charged postprocessing experiment. Generated artifacts remain ignored by Git.

The recorded source hash is
`ed53a17af33b100a06e42fd4b381d0cf3008c112738760e2aeb2dbfad30476e6`.
Versions are NumPy 2.5.4, SciPy 1.18.1, CVXPY 1.9.3 and Clarabel 0.11.1.
The original two-stage modules and `repair_sdp` branch are unchanged. The
previous handoff companion files were not present in the checkout; the existing
implementation, prior findings and supplied handoff provided the integration
contracts.

## Calibration and falsification

Fifteen fixed-S7 joint solves cover t=0.03/0.01/0.003/0/-0.01 and guard scales
0.01/0.003/0.001. The positive controls retain their analytic limiting degrees
and positive full second-order gaps, with at least one independently strict
finite lift for each positive t. Neutral calibration uses a zero full-block
floor; t=0 is never labeled strict. The negative-t control maximizes a negative
margin when its original P0 is fixed. The R-independent dipoles reject it even
though its limiting degree exceeds the reference.

Tests compare every affine density/correction coefficient with independent
edge and dense Taylor constructions, test random quotient displacement
convergence, orthonormal gauges, singleton hard blocks, box interpolation,
inward tree edges, solver flags, exact cache keys, partial sensitivities and
Decimal audits. The symmetric-S7 no-go regression checks the supplied two
polynomial inequalities against direct dipole slacks, independently of
first-order correction choices. It is scoped to the original equal-mass,
equal-crossed-density pattern, never used to discard modified outer families.

The prior LP-flat-optimum and negative-t falsification tests still pass.
Native inaccurate results remain uncertain; unused inaccurate secondary
attempts are recorded and cannot replace an independently valid optimal
primary result.

## Main research sweep

The schedule generates 210 proposals at each size 7/8/9/10/12 and 175 for each
of six starting support/anchor families. Proposals comprise 584 restarts,
249 weak-mode topology moves and 217 mass-logit/offset moves. Every guided move
records its actual parent and pre-reoptimization prediction; acceptance always
comes from a fresh joint solve. Two exact cache hits leave **1,048 paid joint
SDPs**. No rounded cache merges are used.

Of these solves, 808 maximize density and 240 maximize target margin.
Targets 0.6930/0.6935/0.6950 receive 90/90/60 solves. Guards 0.01/0.003/0.001
receive 359/359/330. All target-mode outcomes are conditional infeasibility
(239) or a valid negative-margin unstable result (1). They are not impossibility
proofs for a broader family.

| Clusters | Strict limits | Uncertain | Conditional infeasible | Unstable |
|---:|---:|---:|---:|---:|
| 7 | 44 | 35 | 130 | 0 |
| 8 | 98 | 14 | 96 | 1 |
| 9 | 104 | 22 | 84 | 0 |
| 10 | 94 | 27 | 89 | 0 |
| 12 | 76 | 60 | 74 | 0 |
| total | 416 | 158 | 473 | 1 |

These are solved proposals, including nearby descendants, not independent
discoveries of novel families. The best remains a seven-cluster complete-support
case near the known construction. No 8–12-cluster proposal improves it.

The ten-member cluster-first/mask-second shortlist receives 30 finite h trials:
25 are numerically strict, three are unresolved at the numerical gate and two
are unstable. Positive leading-order stability does not guarantee stability at
a chosen finite h. All outcomes are retained separately. Nine limiting Decimal
audits and one finite Decimal audit have positive pivots; these do not prove
exact equilibrium existence.

## Best limit and finite outcomes

The main sweep's best strict limiting density is **0.6929033189148635**, about
4.3875e-6 below the reference and below the initial research target 0.6930.
Its normalized offset variance is one and guard is 0.001.

| Limiting diagnostic | Value |
|---|---:|
| Full quotient gap | 9.9999989e-7 |
| Effective soft gap | 4.8252548e-6 |
| Hard gap | 0.03416980494 |
| Minimum inner restoring | 0.2157446238 |
| F0 infinity norm | 2.4980e-16 |
| F1 infinity norm | 8.8818e-16 |
| K infinity norm | 6.9389e-18 |
| Budget identity error | 0 |
| Full/dense/affine disagreement | 3.3307e-16 |
| Box/cone violations | 0 |

The full gap sits at the configured `1e-6` inner floor. The independently
computed matrix-scale roundoff gate is about 9.33e-14. A 60-digit Decimal audit
of the fixed float inputs has positive pivots, F0 error 2.43e-16, F1 error
7.35e-16 and matrix agreement within 7.49e-17. These are numerical diagnostics,
not interval enclosures.

The main sweep's best finite density is 0.6922885138802887 at h=0.0001.
It is below the earlier pilot's finite best 0.6923027477669624. Thus the main
sweep improved the limiting approximation without improving that finite baseline.

A separate fixed-P0 experiment holds the winning limiting matrix constant.
Its first attempt at exactly the original full margin returned inaccurate and
was rejected. A second experiment requests 80% of the measured positive full
gap to leave numerical headroom. Its accurate primary output is strict; its
inaccurate L1(R) secondary output is **not selected**. R's norm is not improved:
the retained primary has norm 78.84 versus 67.05 before postprocessing.

Nevertheless, this different valid correction lifts at the smaller tested
h=0.00003, yielding the best finite numerical result across these experiments:
**0.6926591963503084**. It exceeds the previous pilot's finite value but is
still below the reference, and is not a new lower-bound claim.

| Postprocessed finite diagnostic | Value |
|---|---:|
| Force infinity norm, float64 | 5.5511e-17 |
| Force infinity norm, 60-digit audit | 5.9666e-17 |
| Minimum restoring coefficient | 0.21574462446 |
| Restricted cluster gap, float64 | 7.1819e-14 |
| Float64 roundoff gate | 1.3596e-14 |
| Rescaled full gap | 3.2931e-5 |
| Smallest grounded Decimal LDL pivot | 1.4663e-13 |
| Box/support violation | 0 |

The finite gap is small, with limited float64 headroom. The Decimal audit has
positive pivots and valid box/symmetry checks for the saved inputs, but neither
it nor the linear force repair encloses an exact equilibrium root. All proof
flags remain false and the rigorous archive is empty.

## Paired controls and resource costs

Thirty outer geometries (five sizes, six families, one seed) are shared by the
joint method and two unchanged sequential controls: LP-then-R and necessary
screen-SDP-then-R. Each has a two-all-conic ceiling and zero full-template SDP
calls. Early failures spend fewer calls. `matched_actual_all_conic` means a row
spent its own ceiling; it does not assert that all methods in that cell spent
equal work. Joint repair-tree restrictions, fixed nonuniform anchors and box
buffers are additional inner restrictions. Old controls use the same support
without those fixed-anchor/buffer restrictions, and optimize the effective-soft
margin rather than the full-block floor. Settings are saved in each cell.

| Method | Cells | Strict limits / finite lifts | Best strict mu0 | Actual conic calls | Wall seconds |
|---|---:|---:|---:|---:|---:|
| joint | 30 | 8 / 8 | 0.69286966467753 | 38 | 0.780 |
| LP then R | 30 | 3 / 3 | 0.47215746539391446 | 34 | 0.212 |
| screen SDP then R | 30 | 6 / 6 | 0.6923949582255815 | 36 | 0.428 |

Only 8/4/6 rows respectively spend the entire two-call ceiling. This small,
unequally spent experiment is not evidence of general algorithmic superiority
or matched wall-time efficiency. The old pilot remains separately reproducible.
No policy training or frozen-RL checkpoint was involved.

| Research component | Joint SDPs | LP / screen / R SDPs | Tree solves | High-precision audits | Full SDPs |
|---|---:|---|---:|---:|---:|
| final calibration | 24 | 0 / 0 / 0 | 27 | 0 | 0 |
| final paired cells | 38 | 30 / 30 / 10 | 17 | 0 | 0 |
| final 1,050-proposal sweep | 1,048 | 0 / 0 / 0 | 30 | 10 | 0 |
| additional fixed-P0 attempts | 3 | 0 / 0 / 0 | 3 | 2 | 0 |

The sweep driver records about 56.78 seconds including persistence. A previous
attempt, `runs/joint-research-1050/`, was interrupted by a transient Windows
sharing violation during atomic checkpoint replacement: 943 completed
proposals, 941 paid joint calls, two cache hits, one pending unsolved proposal,
and about 44.63 seconds. Its calibration and paired cells additionally spent
24+38 joint calls, 30 LPs, 30 screen SDPs, 10 R SDPs and 44 tree solves.
Those costs are not discarded or folded into the successful sweep's budget.
The new driver now retries the same atomic write briefly; tests verify that
this retry performs no new solver work.

Including that interrupted research attempt and the rejected one-call finalist
attempt, these research artifacts account for **2,256 conic invocations**,
**121 tree linear solves**, **12 high-precision audits**, and **zero full-template
SDPs**. Pytest and earlier development/smoke probes are separate engineering
work, outside the comparative research table. No caching or failed attempt is
represented as free optimization work.

## Practical limits and next evidence needed

The larger sweep did not attain its research target. Several high-density
solutions sit close to the requested full-block floor, and minimizing R can be
numerically difficult there. The optional secondary solve is useful on some
calibrations but does not reliably remove the thinning null direction at the
best limit. The rejected secondary attempt is recorded, not called a successful
norm reduction.

An equal-wall-time or fully spent all-conic comparison, more seeds, and explicit
weak-guidance ablation would be needed for performance claims. Changing floors,
guard scale, repair support or model class can change the search outcome.
Conditional infeasibility at the sampled outer points is not an impossibility
result. The symmetric-S7 obstruction is deliberately narrow. A potential record
still needs a limiting improvement, independently strict finite lifting and,
for a theorem, rigorous root/Hessian certification and graph realization.
