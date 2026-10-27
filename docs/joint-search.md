# Joint limiting SDP and weak-mode search

The implementation is additive on `asymptotic_search`. The original LP,
necessary-screen SDP, correction SDP, two-stage benchmark, evaluator, finite
validator, checkpoint schemas and RL action space are unchanged. No changes
were merged from `repair_sdp`.

## Run it

```powershell
uv run python -m kuramoto_template.asymptotic_search.joint_outer --config examples/joint-small.json --output runs/joint-small
uv run python -m kuramoto_template.asymptotic_search.joint_outer --output runs/joint-small --resume
uv run python -m kuramoto_template.asymptotic_search.joint_benchmark --output runs/joint-research --sweep-proposals 1050 --seed 0
```

Each directory must be new, except when resuming. The benchmark runs S7
calibration before the research sweep. It writes complete paired inputs, which
can also be solved separately:

```powershell
uv run python -m kuramoto_template.asymptotic_search.joint_limit --input runs/joint-research/paired/m7-complete/input.json --output runs/one-joint.json --mode margin --mu-target 0.6935
```

`--config`, `--h-guard`, `--neutral`, `--primary-only` and `--max-joint-calls`
are also available. A saved joint result is a valid input. Results retain their
input ordering. The outer driver stores the canonical ordering and input index
map explicitly; it never attaches a cached matrix to a different ordering.

```python
from kuramoto_template.asymptotic_search.families import s7
from kuramoto_template.asymptotic_search.joint_limit import (
    JointLimitConfig, solve_joint_limit,
)

seed = s7(.01)
result = solve_joint_limit(
    seed.x, seed.s, seed.groups, seed.allowed_mask,
    JointLimitConfig(mode="density", h_guard=.003),
    backbone_cycle=seed.backbone_cycle,
)
print(result.status, result.mu0, result.diagnostics)
```

`fixed_p0=seed.p0` freezes the original limiting matrix for calibration. More
generally, `fixed_p0` is a symmetric matrix with NaN for free entries; JSON uses
null for these entries. Fixed values, forbidden zeros and diagonal ones are
algebraic constants, not rounded solver output. `repair_edges` can specify an
adjacent-parent spanning tree with explicitly inward R and positive base/trial
density. All optional restrictions are persisted.

## Convex model and independent validation

`joint_limit.build_affine_model` assembles coefficient tensors for F0, F1, all
four K values, splitting energy, inner restoring coefficients and the entire
m-1 Hessian. For fixed geometry, its block matrix is

```
[ U.T L_cut(k) U       -U.T G.T Z ]
[ -Z.T G U             Z.T L_in Z ]
```

U is orthonormal on the complement of the four-parent constant; Z has
orthonormal, zero-sum columns within each parent. Singleton groups contribute
no hard columns. Only this quotient Hessian is constrained to be PSD. P is a
symmetric density matrix with box bounds, never a PSD matrix.

Density mode maximizes minimum limiting degree at a fixed positive full-block
margin. Target mode fixes minimum degree and maximizes the unrestricted full
margin. Its negative optima remain unstable exploration signals. Hard and
restoring floors remain positive in either mode. `neutral=True` removes only
the positive full-block floor in density mode for boundary calibration.

The optimized variables obey both `0<=P0<=1` and `0<=P0+h_guard*R<=1`.
Interpolation therefore preserves the base linear path's box for intermediate
h. There are no conditional constraints on optimized P0 and no bilinear or
big-M tangent encoding. Fixed zero/one entries additionally have their valid
linear tangent signs. Opposite links vanish at all orders; internal R and the
diagonal R are zero.

There are explicit conservative inner restrictions: finite guard, R bound,
energy/hard/restoring/full floors, backbone and optional repair tree, and a
default `1e-8` buffer on free densities and neighbor trial endpoints. The buffer
avoids depending on tiny outward solver violations during exact finite lifting.
It changes the feasible inner set and is recorded as `box_buffer`. It is never
implemented by clipping a returned matrix. Decreasing guard scans must be
interpreted together with these other restrictions, especially the endpoint
buffer. Conditional infeasibility is not a theorem about an outer family.

Default full/hard floors are `1e-6` in the stored gauge, restoring floor is
`1e-4`, and outer offsets have mass-weighted variance one. Independent strict
classification additionally requires resolved eigenvalues above both `1e-8`
and a matrix-scale roundoff gate, plus hard/restoring floor headroom. Raw full,
hard and effective spectra and the offset scale are saved. Margin rankings
across sizes or different gauges are not used.

`validate_joint` compares the affine result with the earlier edge-sum full
matrix and an independent dense Taylor construction. It recomputes F0/F1/K,
sum(k)-I, degree, restoring, hard/soft/full gaps, box/cone/fixed-entry/repair
violations, the Schur complement and winding. Primary reported density must
agree with measured degree. An optional second paid SDP minimizes L1(R) while
preserving density or positive target margin. It is independently checked; a
bad or inaccurate secondary output never displaces a valid primary result.
`solve_records.selected` identifies which output was retained.

`optimal_inaccurate` cannot support a strict classification for the selected
solve. Native Clarabel primal/dual residuals and gaps are collected when its
backend exposes them; otherwise they are explicitly unavailable. They are
observability data, not a substitute for independent validation.

## Outer geometry and weak-mode guidance

The six starting families are complete support, anchor zeros, one-cut zeros,
mixed zeros, sparse internal support, and fixed nonuniform anchor densities.
Allowed crossed/anchor densities are independent optimization variables.
Unequal splits cover 7/8/9/10/12 clusters. Fresh restarts and trust-region moves
in mass logits/offsets preserve positive masses and the offset gauge. The
schedule balances cluster count first and family second, and scans guards
0.01/0.003/0.001 and targets 0.6930/0.6935/0.6950.

`weak_modes` computes normalized weakest eigenvectors and ranks topology moves
by the exact fixed-vector quadratic change before reoptimization. The separate
internal-edge partial derivative holds G and cut weights fixed and is the
nonnegative square `(b.T L_in^dagger G eta)^2` per physical edge weight. Neither
quantity predicts the total constrained reoptimized effect. Every proposed
outer point is solved afresh unless its exact input is cached. A positive
predicted gain cannot accept a witness.

Archives distinguish strict, neutral, unstable, uncertain and failed outputs;
the strict density incumbent is protected. Parent pools compare only within a
size/family/gauge stratum. The final shortlist reuses the existing policy that
round-robins sizes before extra masks. Winding protects a specified backbone,
not every cycle of the graph.

## Accounting, replay and proof status

The joint ledger separately caps joint SDPs, tree linear solves, optional full
SDPs and high-precision audits. Old controls retain their distinct LP, screen
and correction counters. Failed/interrupted attempts count before entry, and
every event has wall time. Exact caches do no rounding and retain tiny changes.
Within-parent label sorting precedes gauge reduction so reduction order is
deterministic. Ambiguous tied labels may miss a symmetry hit, but do not merge
unequal data.

Checkpoints use the existing atomic JSON writer, with a new-driver-only short
retry for transient Windows reader sharing violations. A retry repeats file
I/O, never a solver. Enduring errors still propagate. Outer-proposal, joint-solve
and finite-lift boundaries persist RNG, pending geometry, exact cache, archives,
source hash, dependency versions, counters and references to complete immutable
per-solve result files. Resume rejects changed source/settings/versions and
does not repay completed boundaries. Completed-run resume is also zero-cost.

The optional full-template fallback uses the existing SearchSession/evaluator
adapter. Its API has no local tube; it is explicitly support-only, records
`family_lift=False`, and cannot replace a validated local lift as a family
witness. Full-SDP, all-conic and wall-time budgets are distinct comparisons.

Finalists receive independent 60-digit Decimal polynomial assembly/LDL audits
of the limiting Hessian and the existing Decimal finite audit when available.
They interpret saved float inputs, without interval outward rounding or an
equilibrium-root enclosure. `proof_available=False`, certification status is
`not_attempted`, and the rigorous archive is empty. A proved record still needs
root enclosure, interval positivity and graph realization.

The narrow `s7_sym_dipoles(q,w)` regression implements the supplied S7-symmetric
obstruction: for exactly the equal-half masses, original anchor links and four
equal crossed densities, the two R-independent polynomial slacks cannot both
be nonnegative for q>q*. This is not applied to unequal masses, changed anchors,
independent crossed densities or new masks. The reference is [O'Keeffe, Table 2
and Appendix A](https://arxiv.org/html/2610.21271v1).

See [measured results and limitations](joint-search-findings.md).
