# Aggregate-flow, compensated and winding-two pilot

2026-10-26, branch `asymptotic_search`. The bounded pilot completed without a
new record. Its best strict numerical limit is **0.6929007877787521**; its best
verified finite lift is **0.6922130346764885**. Neither new structural family
improved on S7. This is evidence about the sampled geometries and configured
inner restrictions, not an impossibility proof.

The [paper](https://arxiv.org/html/2610.21271v1) distinguishes Theorem 1's
0.6929077063695864 from Proposition 1's concrete example 0.6929050246493944.
Both exceed this pilot's best. The previous joint sweep's limit
0.6929033189148635 and postprocessed finite value 0.6926591963503084 remain
better numerical results. Its different margin/guard settings prevent treating
the two sweeps as a controlled algorithm comparison. The supplied follow-up
value 0.6929072637491324 exceeds the concrete example but stays below the
theorem; it was not independently reproduced here. `informe-s7.html` and the
updated research LaTeX note were not present in this checkout.

## Reproduction and costs

```powershell
uv run python -m kuramoto_template.asymptotic_search.flow_search --config examples/flow-bounded.json --output runs/aggregate-compensated-new
uv run python -m kuramoto_template.asymptotic_search.flow_search --output runs/aggregate-compensated-new --resume
uv run --extra rl pytest -q
```

Canonical evidence is under `runs/aggregate-compensated-pilot/`: raw proposals,
saved-input audits, immutable solutions, checkpoint/result/trace, summary tables,
paired rows and separate top-limit/top-finite JSON. Generated files remain
ignored by Git; earlier runs were preserved. [API and scope](aggregate-flow-search.md)
describe the new helpers and CLI. Full-suite validation: **470 passed**, with
six warnings from native inaccurate statuses and the existing Gym checker.

| Resource | Actual | Cap |
|---|---:|---:|
| Primary joint SDP calls | 662 | 900 |
| Secondary compensation QPs | 60 | 60 |
| Analytic compensated projections | 90 | 150 |
| Finite tree linear solves / finite attempts | 90 / 90 | 90 / 90 |
| High-precision audits | 12 | 12 |

The primary total comprises 30 identical saved-input audits, 25 fresh parent
calibrations and 607 main solves. There were no exact cache hits. All 900 main
proposal slots finished: screening and eight rejected QP proposals explain the
unused primary capacity. The initial run took 280.495 seconds including file
I/O; completed-run resume added no resource events or calls. Solver-ledger
timings, checkpoint overhead and unlike solver types must not be pooled into
an assertion of equal computational cost. Development smoke runs are separate
from these canonical counters.

The recorded source hash is
`0a05bf1cdda790996f41358145d253c5098266619fdbe5d0562196e1b8174148`.
Dependencies: NumPy 2.5.4, SciPy 1.18.1, CVXPY 1.9.3, Clarabel 0.11.1.
The pilot uses physical full margin 1e-5, hard floor 1e-4, guard 0.003,
R entry bound 30, residual tolerance 1e-9 and explicit 1e-8 box buffers.

## Filter and structural families

On the same 30 saved inputs, the necessary aggregate bound rejects 23 (76.7%).
Their deliberately paid unfiltered target solves yield 22 conditional
infeasibilities and one uncertain result, with no strict contradiction. The
other seven are not strict target solutions either: passing is insufficient.
Screening took 0.00413 seconds versus 0.36666 seconds recorded in the unfiltered
solves. A filtered replay would pay seven calls; all 30 actually paid calls
remain in the ledger. These timings do not measure end-to-end speedup.

| Main family | Proposals | Goal-screened | Paid primary | Strict | Best strict mu0 |
|---|---:|---:|---:|---:|---:|
| Independent balanced-flow | 100 | 88 | 12 | 11 | 0.6396989898452423 |
| Legal q-mass window | 100 | 40 | 60 | 0 | none |
| Winding-two skeleton | 100 | 64 | 36 | 7 | 0.41437168829230925 |
| Small asymmetric updates | 300 | 93 | 207 | 204 | 0.6928847787669166 |
| Split / fresh merge to seven | 150 | 0 | 150 | 145 | 0.6929007877787521 |
| Compensated topology | 150 | 0 | 142 | 135 | 0.6929007863119450 |

The 100 legal mass ratios span 1.01282 to 1.30997, below the critical
1.3227888640796643, with five strictly subcapacity flow fractions. Forty are
excluded at their fixed flow; among 60 paid solves, 36 are conditionally
infeasible and 24 uncertain. No edge density above one is used. The uniform
analytic no-go is scoped and tested, but was not used to prune these
heterogeneous families. This negative sample does not eliminate the surviving
mass window.

There are 52 long-cycle and 48 relaxed winding-two starts. Six long and one
relaxed result are strict. Long supports have shortest nonzero-winding cycle
length five; relaxed controls may have four. The simple eight-edge backbone
has winding two in both cases, with finite wrapped-phase validation. This
does not establish cycle-space independence from winding-one cycles. The
best winding-two candidate lifts strictly at all six tested h values, with
best finite density 0.4143683997706362. It is structurally distinct but far
below the S7 density.

## Paired comparisons and finite evidence

The following are descriptive counts on pairs with equal actual primary and
secondary conic counts and both sides strict. Wins use a 1e-8 density threshold;
the repeated historical parents make these neither independent trials nor a
statistical superiority claim.

| A versus B | Matched conic pairs | Both strict | A wins | B wins | Ties |
|---|---:|---:|---:|---:|---:|
| Balanced local versus random | 90 | 87 | 20 | 46 | 21 |
| Split versus fresh seven-cluster merge | 75 | 70 | 22 | 0 | 48 |
| Balanced compensated versus random | 67 | 60 | 0 | 5 | 55 |

Split/merge comparison-only points are deliberately solved even when the goal
bound excludes them. Both sides retain inherited offset/h units and identical
physical margin floors. Split-minus-merge differences range from -7.18e-9 to
8.44e-6. The best nine-cluster split's degree exceeds its fresh seven-cluster
merge by only 1.50e-9. Its final S7-distance proxy is 7.13e-5, consistent with
an S7-like result, without asserting exact twins.

Across balanced compensated proposals, 21 predict a positive full-margin
change above 1e-8, but none predicts a degree gain above that threshold.
Some support openings improve lower-density parents after joint reoptimization;
there is no gain over the best recalibrated S7 parent. Among 60 comparable
both-strict pairs, balanced-minus-random ranges from -0.04565 to 7.69e-13.
Moves that leave the allowed mask unchanged present the same free joint problem
and commonly return exactly the parent optimum. Positive frozen-geometry
predictions are never treated as accepted improvements.

The finite shortlist contains 15 size/mask representatives and an explicit
winding-two representative. Its diversity policy omits the global limiting
winner, source 342: that particular winner has no finite lift or Decimal audit
in this run. Source 392, with limiting mu0=0.6929007857188184, is lifted, but
smaller h trials become numerically uncertain. Therefore the finite maximum
reported here is only over the selected attempts, not a bound on achievable
finite density.

Of 90 finite attempts, 81 are numerically strict and nine uncertain. The best
is source 4, h=0.00003: force infinity norm 5.55e-17, restoring minimum
0.1861263107, cluster gap 6.67e-14 versus a roundoff floor 2.33e-14, and rescaled
gap 3.68e-5. Eleven limiting Decimal audits have positive pivots. The finite
audit recomputes force error 5.43e-17 and positive grounded LDL pivots.

The archive preserves Pareto density/margin/R/slope/h/finite rows and invariant
initial/final S7-distance proxies. Independent starts remain far from S7
(median initial distance 0.815; winding-two 1.036), while the best solution
remains close. No exact equilibrium enclosure, interval Hessian certificate or
graph realization was produced: there are **zero rigorously certified graph
witnesses**. The analytic necessary bounds, strict numerical classifications
and rigorous proof status remain separate.
