# Kuramoto S7 joint SDP reproduction

## Scope and setup

This branch retains the original joint limiting SDP benchmark, S7 structural
follow-up, saved numerical witnesses and finite-lift refinement. A separate
`c4_discovery` experiment starts from C4 splits without S7 initialization;
keep its schedule and outputs separate from the historical benchmark. RL/Gym,
random/AMCS, the old full-template optimization/fallback, the old two-stage outer driver
and aggregate-flow/compensated pilots are outside this branch's scope.

Python >=3.14; use `uv sync --locked` and `uv run --locked pytest -q`.
The optional `reports` extra adds Matplotlib; `certification` adds FLINT/Arb
for exact rational and interval certificates. Run `uv run --locked --extra
certification pytest -q` when changing certification code.
Source uses a `src/` layout and the `uv_build` backend.

## Source map

- `src/kuramoto_template/asymptotic_search/`: immutable limiting candidates,
  S7/split families, expansions, joint SDP, guided outer search, finite lifts,
  numerical audits, paired limiting/correction controls and persistence.
- `template.py`: immutable masses and wrapped phases.
- `sdp_optimizer/template_metrics.py`, `validate_input.py`: independent finite
  metrics and circular geometry.
- `scripts/reproduce_best.py`: integrity checks, saved-witness validation and
  fresh limiting/fixed-P0/finite replay.
- `asymptotic_search/c4_discovery.py`, `scripts/refine_c4_result.py`: independent
  C4 splitting, optional C4 symmetry reduction and bounded geometry refinement.
  `scripts/certify_limiting_result.py` certifies explicit rational variants of
  new limiting outputs; complete experimental outputs remain in ignored `runs/`.
- The optional `c4_parametric`, `c4_multisplit`, `c4_recovered` and `c4_session`
  modules support the separately logged C4-only timed research. Standalone
  `c4_finite_probe` controls do not enable the historical full-template fallback.
  `scripts/exact_c4_active_family.py`, `lift_exact_c4.py` and
  `probe_c4_mass_tangent.py` reconstruct and certify exact C4-derived winners.
- `scripts/reproduce_s7_analysis.py`, `scripts/s7_analysis/`: the original
  120-solve structural analysis with portable paths and scratch output.
- `research/`: intentionally versioned reports, figures, configurations,
  aggregate statistics, ledgers and only the three best-result witness files.
  Exact rational certificate companions are limited to these same three best
  results. Non-winning snapshots and embedded candidate matrices stay outside Git.
- `asymptotic_search/interval_certification.py`, `scripts/certify_witnesses.py`:
  exact rational repair, Krawczyk inclusion, interval Cholesky and independent
  certificate verification. Mathematical lifting proof: `docs/interval-certification.md`.

## Mathematical and reproducibility contracts

- Masses are positive float64, sum to one within 1e-10, and are not silently
  normalized by construction. Arrays are copied and read-only.
- Symmetric connection densities obey box, clique and structural constraints.
  Minimum degree is `min(P @ x)`; recompute equilibrium and stability independently.
- Retain within-cluster restoring checks and the intercluster Laplacian
  restricted off the global rotation mode. Solver status alone is insufficient.
- Limiting strictness, finite numerical strictness and rigorous certification
  are separate. Decimal audits evaluate saved float inputs and do not prove
  root existence. Optional interval certificates verify explicitly stored nearby
  rational variants; never transfer their success to the original float arrays.
  Limiting certificates verify the hypotheses of the documented sufficiently-
  small-h lifting lemma, not an explicit finite h. No explicit graph is certified.
- Count every actual conic invocation, including failed and secondary attempts.
  Tree solves and high-precision audits have separate counters. Preserve failed
  and uncertain outputs, actual proposal parentage, seeds, settings and versions.
- Preserve the original schedule, numerical gates, exact cache semantics and
  checkpoint schemas. The unused full-template fallback is disabled; its legacy
  configuration field must be zero and its ledger counters remain zero.
- Preserve historical reports and winning witness values. Curated analysis
  files contain summaries rather than non-winning matrices. New complete runs
  belong in ignored `runs/`. Update `research/manifest.json` deliberately when
  curating evidence.
  Historical source hashes remain historical; resume rejects source changes.
- `.t4_env` contains private material and must remain untracked. Never copy it
  or environment/cache directories into research artifacts.

## Verification

Inspect implementation before trusting tests. Run the complete retained suite,
including tests within `asymptotic_search/`. Solver tests use `integration`.
After changing reproduction code, run the documented witness replay; changes
that affect the sweep schedule also require a fresh full benchmark. Add focused
coverage when behavior changes warrant it. Do not remove numerical checks to
make a saved candidate pass.
