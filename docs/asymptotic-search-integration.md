# Integration note: asymptotic template search

The implementation is a new `kuramoto_template.asymptotic_search` package on
the current checkout. `repair_sdp` was inspected read-only; it contains a
continuous finite-template optimizer, but is not required or merged. The
companion handoff file was not present in this checkout.

The reference is O'Keeffe, [Beyond 11/16, v1](https://arxiv.org/html/2610.21271v1),
Table 2 and Appendix A. `s7` implements the limiting Table-2 seed. The optional
`appendix_a_correction` implements the reference first-order coefficients and
provides a solver-independent calibration for the new correction SDP.

## Mathematical normalization

`PhaseTemplate` remains the sole finite-template masses/phases model.
`template_metrics` recomputes degree P@x, restoring coefficients, forces, and
the weighted cluster Laplacian restricted by an orthonormal basis of 1-perp.
The new expansion uses the handoff's sin(phi_b-phi_a) convention; existing
metrics use its negative. Equilibrium residual norms coincide. There is no
factor of two between the edge quadratic form and y.T@L@y.

The new pure math operates on the linear path phi=j*pi/2+h*s. L_in is the
weighted within-parent Laplacian. G maps four parent displacements to g(eta).
Independent zero-mean solves in each parent give G.T L_in^dagger G. The
effective matrix is Q_cut(k) minus that matrix. U4 is Euclidean orthonormal on
the complement of the four-parent constant. The full m-1 rescaled coordinates
are y=C U4 eta+h H z, with H orthonormal and zero-mean within each parent.
Their complete quadratic form and a separate dense Taylor construction agree.
The finite rescaled form uses exact quarter-turn trig identities to avoid
unnecessary cos(pi/2) cancellation; project metrics use the stored wrapped phases.

The two dipole tests are necessary only. Near-zero slacks are inconclusive.
F1, sum(k)=I, all tangent directions, effective and full rescaled gaps are
independently checked before any finite lift. Opposite links, varying masses
with h, quadratic phase paths and additional internal null modes are outside
this derivation. Symmetry, forbidden zeros and clique diagonals are exact
structural constants, never clipped solver entries.

## Existing APIs and accounting

The fixed-template optimizers and evaluator are unmodified. Full fallback or
audit calls use a mask-bound `TemplateEvaluator` inside a normal `SearchSession`:
`evaluate_seeds()` invokes `record()` and pays initialization. Their configured
positive margins and tolerances are persisted. No unsupported tube constraint
is simulated. A support-only solve may jump families and is never labeled a
local lift. Tree corrections do not invoke a full SDP and use existing project
metrics for final validation.

`SolverLedger` supplements existing full-call accounting with explicit LP,
screen SDP, correction SDP and tree-linear counts. It increments before calls,
including failed/interrupted attempts. The optional minimum-norm correction is
a second paid SDP. Exact duplicate outer points cost zero additional conic
calls. Session costs are recorded for reconciliation, not summed twice.

Generated checkpoints and experiment artifacts live outside package source.
Checkpoint compatibility includes all settings, code hash and dependency
versions. Reproduction is deterministic on this environment at saved boundaries;
bitwise conic results across platforms are not promised. The numerical statuses
remain distinct from rigorous certification. A later proof would need an
interval equilibrium enclosure and interval Hessian positivity, followed by a
rational template/network realization argument.
