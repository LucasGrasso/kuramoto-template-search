# Four-parent asymptotic search

This standalone search uses `phi_a(h) = j_a*pi/2 + h*s_a`, fixed positive
masses, no opposite-parent links, connected limiting graphs within each parent,
and clique diagonals. It does not change the environment or policy action space.
The comparison target `MU_RECORD = 0.6929077063695864` is not an optimality claim.

```powershell
uv run python -m kuramoto_template.asymptotic_search --config examples/asymptotic-small.json --output runs/asym-small
uv run python -m kuramoto_template.asymptotic_search --output runs/asym-small --resume
uv run pytest -q src/kuramoto_template/asymptotic_search/tests
uv run python -m kuramoto_template.asymptotic_search.benchmark --output runs/asym-pilot --seeds 0 1 2 --proposals 15 --full-calls 4
```

New run directories must not exist. All generated data belongs under `runs/`.
If Windows sandboxing restricts the default uv cache or pytest temporary folder,
set `UV_CACHE_DIR` and `--basetemp` to fresh paths under `runs/`.

```python
from kuramoto_template.asymptotic_search import TwoStageConfig, run_two_stage, s7

result = run_two_stage(
    TwoStageConfig(stage1_proposals=3, shortlist_size=3),
    candidates=[s7(.03), s7(.01), s7(.003)],
    output="runs/s7-calibration",
)
print(result.best_finite_numeric_mu, result.call_counts)
```

`families.s7()` retains the raw published scale for analytical regression.
The outer generator creates reproducible unequal splits with 7/8/9/10/12
clusters, perturbs masses and offsets, and varies neighbor masks while protecting
the specified winding backbone. Outer offsets have mass-weighted mean zero and
variance one; the validator accepts any nontrivial scale because scaling `s,R`
and inversely scaling `h` gives the same family. No arbitrary twisted template
is assigned four parent groups automatically.

## Stages and outcomes

* `stage1.screen_candidate` checks all F0/K residuals, internal rank, energy,
  winding and the two necessary dipole inequalities. Groupwise zero-mean solves
  avoid a global pseudoinverse. Neutral slacks are `uncertain`; extra internal
  null directions and trivial splitting are `asymptotic_degenerate`.
* `stage1.optimize_limiting` optimizes the limiting minimum degree for fixed
  masses and offsets. Its LP can return `lp_screen_fail` even though a different
  matrix is feasible. The driver tries the necessary Schur-LMI SDP after an LP
  failure when budget permits. `necessary_screen_infeasible` means an accurate
  solver infeasibility status for that fixed outer point with the configured
  connectivity/energy/backbone floors; it is not a rigorous infeasibility proof.
* `stage2.solve_correction` maximizes the full effective 3x3 margin subject to
  F1, tangent bounds, a finite trial box and bounded R. If margin is positive
  and budget remains, a second paid SDP minimizes R's norm while preserving
  95% of that margin. Both the complete m-1 rescaled matrix and its hard block
  are checked independently using a dense Taylor route. `optimal_inaccurate`
  cannot earn `asymptotic_full_second_order_strict`.
* `finite_lift.lift_candidate` rechecks R, rejects out-of-box bases without
  clipping, and tries up to twelve deterministic spanning trees of allowed
  neighbor edges with usable slack and nonzero sine. Solving the weighted
  incidence equations removes exact finite-angle forces. Changes must lie in
  an explicit O(h²) tube. Full project metrics, a separately rescaled Hessian,
  float64 roundoff headroom and the actual finite backbone determine
  `finite_h_numeric_strict`. Failed, unstable and unresolved outcomes remain
  separate. The benchmark measures force, repair and soft-gap log slopes.

The initial finite grid is 0.01/0.003/0.001/0.0003; extremely small h can fail
the numerical resolution gate. `Numerics` records every threshold and conic
option. Existing full-SDP feasibility standards are unchanged.

`finite_h_sdp_solves=0` disables the optional existing density-SDP fallback.
Positive budgets permit one fresh support-constrained density invocation per
failed finite lift, through `TemplateEvaluator` and `SearchSession`.
`audit_finite=True` also evaluates successful lifts. That API has no local tube,
so these are explicitly `support_only_sdp_fallback` / `support_only_sdp_audit`,
with `family_lift=False`, their distance from P0+hR, and separate diagnostics.
Their matrices never replace a successful local lift in the finite archive.
Margin fallback is disabled in this adapter to keep each full invocation atomic;
`full_margin_sdp` consequently remains zero.

## Accounting and replay

`call_counts` separates Stage-1 LPs, necessary-screen SDPs, both Stage-2 SDPs,
tree linear solves, full density and full margin calls. Failed and interrupted
invocations are paid; each event records elapsed time. The supplementary full
counter agrees with each session's calls and is not added to it a second time.
Tree solves are real linear algebra work, reported separately from conic calls.

`checkpoint.json` contains the stage/cursor, exact proposal arrays, caches,
archives, counts, events, config, source hash and dependency versions. Atomic
checkpoints occur after each outer proposal, correction, and finite h. A
`cancelled` callback or wall-time limit is checked at those boundaries. Resume
rejects different configuration/code/dependency versions. A cancelled boundary
resumes deterministically without paying again; an interrupted in-flight unit
can be retried, with its prior attempts still counted. Wall time is cumulative.

Exact caches include raw masses, offsets, parent labels, limiting matrix, mask
and backbone. They never conflate cluster order or rounded phase data. The
serial RL environment and existing canonical cache are unchanged. The shortlist
round-robins cluster-count/mask strata, sorting density and dipole headroom within
each stratum. Stable finite incumbents improve monotonically. Effective gaps
are diagnostics, never a ranking across cluster counts.

`result.json` and `top-candidate.json` preserve x/s/groups/P0/R/h/phi/P/mask,
the backbone, every numerical diagnostic, and exact solver settings. Trace IDs
connect proposals, screens, corrections and lifts; stage/archive references are
in `archives`. `rigorously_certified` is always empty. `audit.high_precision_audit`
independently recomputes stored float inputs with 60-digit Decimal trig and
grounded LDL pivots. It has no outward rounding or equilibrium root enclosure,
so `proof_available` remains false.

The pilot writes JSONL/CSV summaries, complete per-run snapshots, S7 calibration
and bad controls, scaling slopes, Decimal audits and a findings report. Its
paired controls start with the same x/phi/mask. Full search and full audit costs
are separate: equality of their total call allowances is not equality of total
compute. LP/correction costs and wall time must also be considered.

See [integration details](../../../docs/asymptotic-search-integration.md) and
[pilot findings](../../../docs/asymptotic-search-findings.md).
