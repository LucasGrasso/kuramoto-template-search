# Aggregate-flow screening and compensated search

This additive workflow on `asymptotic_search` reuses `solve_joint_limit`,
`SolverLedger`, the existing tree repair, project finite validator and Decimal
audits. It does not import changes from `repair_sdp` or change RL actions.
See [the measured pilot](aggregate-flow-findings.md) and
[the original joint model](joint-search.md).

## Run and resume

```powershell
uv run python -m kuramoto_template.asymptotic_search.flow_search --config examples/flow-bounded.json --output runs/flow-new
uv run python -m kuramoto_template.asymptotic_search.flow_search --output runs/flow-new --resume
uv run --extra rl pytest -q
```

Use a fresh directory for a new run. The configuration permits 900 primary
joint calls, 60 compensation QPs, 90 tree solves/finite attempts and 12 Decimal
audits. Its 900 main proposal slots comprise 300 independent starts, 300 local
updates, 150 split/merge comparisons and 150 compensated proposals. Screening
can leave primary capacity unused. Thirty saved-input audit calls and fresh
parent calibrations consume that same primary budget before the main phase.
If the historical run directories are absent, the driver uses an analytic S7
parent and records an empty saved-input audit.

The default independent allocation gives 100 generic starts, 100 legal mass
ratios above one and 100 winding-two supports. The q-mass subgroup fixes common
flow to fractions 0.85/0.92/0.97/0.995/0.9995 of capacity; generic and winding
starts let the joint solver select flow. Local coordinate proposals perturb
mass logits or offsets by 1e-4 to 1e-3 and normalize their offset gauge.

## Necessary flow bound and analytic scope

`aggregate_flow.screen_aggregate(x, groups, target, ...)` assumes cardinal
four-parent leading phases and adjacent-parent support. Optional `allowed_mask`
and `leading_phases` explicitly check those assumptions; pass them when calling
the helper outside the typed four-parent model. Opposite support or noncardinal
phases yield `out_of_scope`, never a rejection.

For parent masses M, order-zero equilibrium makes all four oriented cut flows
equal to C. Therefore

```text
Cmax = min_j M[j]*M[j+1]
mu_cap = min_j (M[j] + 2*Cmax/M[j])
reject only if max_j M[j]*(target - M[j]) - 2*Cmax > tolerance
```

The implementation recomputes the excess at 70-digit Decimal precision from
the exact binary64 inputs, with a conservative scale-aware buffer. Equality
and unresolved boundaries pass. The pilot uses the joint residual tolerance
1e-9. Stored diagnostics include M, Cmax, cap, target, tolerance, Decimal excess,
scope and reason. An optional fixed `flow_target` tightens Cmax to that flow.
Passing is necessary, never sufficient for equilibrium or stability.

`uniform_block_rule` is an optional classification helper. Metadata must
explicitly assert uniform parent blocks, complete internals and a singleton
anchor; arrays must also satisfy equal first-three masses, F0/K, equal flows
and nonzero splitting energy. Only under these hypotheses does the analytic
bound C <= u²/sqrt(2), mu <= 0.6512392830509103 apply. `uniform_block_penalty`
implements the closed-form Schur penalty for comparison with dense internal
elimination. Heterogeneous S7 blocks and I=0 are excluded. The bounded driver
stores this diagnostic but does not use it to prune.

`q_mass_ratio > 1` sets masses `(u,u,u,v)`, u=2/(7+q), v=(1+q)/(7+q).
It is separate from `q_cross_density`, which is null in these new families.
All edge densities remain in [0,1]. The mass-only ceiling crosses the theorem
target at q=1.3227888640796643. Starts use strictly subcapacity, heterogeneous
blocks and balanced node fluxes, preserving F0 and K without a paid solve.
This does not assert their stability. Saturated uniform blocks fall under the
narrow no-go only when all its hypotheses hold.

## Winding, compensation and physical margins

`LimitingCandidate.backbone_cycle` now accepts any simple positive-winding
nearest-parent cycle. Its inferred `backbone_winding` preserves old serialized
four-cycle inputs. `topology.cycle_audit` checks signed wrapped finite phase
increments and distinguishes repeated closed walks from simple cycles.
`shortest_nonzero_cycle` exhaustively searches these small supports.

The winding-two family starts at m=8 with eight forward interparent steps.
Connected same-parent blocks introduce shorter winding-one cycles: the long
supports have shortest nonzero cycle length five, versus eight for the bare
ring. Explicit relaxed controls permit four-cycles. Scalar winding does not
establish independence in the graph's cycle space and is never differentiated.
Finite acceptance uses `winding_valid`; the legacy `winding_one` field is true
only for actual winding-one backbones.

`compensated.compensated_move` selects one or two weighted transport rectangles
(4–8 edges) in ker(F0,K). Analytic projection or a paid small QP respects support,
box tangents and a trust radius. The balanced QP combines minimum degree, both
dipole derivatives, three weakest full generalized Hessian modes and an
approximate price from saved degree/full-Hessian dual multipliers. If native
duals are unavailable, the diagnostic labels its active-constraint surrogate.
Predicted full minimum-margin changes are also recomputed after the move.

Every proposed move requires a fresh joint solve; inaccurate QPs cannot propose
a witness. When the allowed mask is unchanged, changing P0 alone leaves the
free joint optimization problem unchanged. Such pairs test the proxy and often
return the same optimum. Opening allowed edges can change that problem.
Balanced/random pairs record predicted and realized changes and are compared
only when their actual primary and secondary conic counts match.

`JointLimitConfig.margin_metric="mass"` enforces H >= gamma B using a physical
mass quotient norm. With M the parent masses, U the existing soft basis and Z
the within-parent basis, its diagonal blocks are

```text
Bsoft = U.T (diag(M) - M M.T) U
Bhard = Z.T (diag(x) - sum_j x_group[j] x_group[j].T/M[j]) Z
```

The joint solver also accepts optional `flow_target`, enforced by four affine
cut-flow equalities and independently rechecked. Existing callers retain the
Euclidean margin default. Both raw and physical spectra are stored; raw
positivity and roundoff gates still apply. Split/merge comparisons preserve
offset and h units and solve each side afresh under identical physical floors.
A close-offset merge is a new optimization problem, not proof of exact twins.
S7 distance is an invariant distribution/edge-feature proxy under relabeling,
parent rotation and reflection, not an isomorphism certificate.

## Evidence and accounting

`result.json` and `checkpoint.json` store configuration, source hash, dependency
versions, proposal plan/seeds, pending unit, parents, cache, archives, resource
events and Pareto rows. Parent and solution arrays are saved in immutable JSON
files. `proposals/` preserves raw inputs, including failed proposals; names and
traces carry separate q-mass/q-cross fields and winding. Resume checks exact
compatibility and does not repay completed boundaries. Paired comparisons bypass
the exact cache deliberately. Projection counts are distinct from conic calls.

`summary.csv`, `summary.jsonl`, `paired.json`, `summary.json` and `findings.md`
are generated automatically. `top-limit.json` and `top-finite.json` are distinct;
no root `top-candidate.json` is required. Strict joint outputs alone update the
limit incumbent. Pareto comparisons stay within size/family/evidence strata and
retain density, physical margin, raw/mass-weighted R norms, minimum R@x slope,
viable h and finite density. Unknown lifts are not verified finite outcomes.
The finite shortlist follows the existing size/mask diversity policy, plus
strict representatives of each new family; it need not contain the global
limiting density winner. Check the actual selected source IDs.

All classifications are numerical. Decimal audits recompute stored floats;
they provide neither outward-rounded root enclosures nor graph realization.
The rigorous archive remains empty and `proof_available` remains false.
