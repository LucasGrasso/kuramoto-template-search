# Kuramoto S7 joint SDP reproduction

## Scope and setup

This branch retains the original joint limiting SDP benchmark, S7 structural
follow-up, saved numerical witnesses and finite-lift refinement. RL/Gym,
random/AMCS, full-template optimization/fallback, the old two-stage outer driver
and aggregate-flow/compensated pilots are outside this branch's scope.

Python >=3.14; use `uv sync --locked` and `uv run --locked pytest -q`.
The optional `reports` extra adds Matplotlib for regenerating the informe.
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
- `scripts/reproduce_s7_analysis.py`, `scripts/s7_analysis/`: the original
  120-solve structural analysis with portable paths and scratch output.
- `research/`: intentionally versioned reports, figures, configurations,
  aggregate statistics, ledgers and only the three best-result witness files.
  Non-winning snapshots and embedded candidate matrices must stay outside Git.

## Mathematical and reproducibility contracts

- Masses are positive float64, sum to one within 1e-10, and are not silently
  normalized by construction. Arrays are copied and read-only.
- Symmetric connection densities obey box, clique and structural constraints.
  Minimum degree is `min(P @ x)`; recompute equilibrium and stability independently.
- Retain within-cluster restoring checks and the intercluster Laplacian
  restricted off the global rotation mode. Solver status alone is insufficient.
- Limiting strictness, finite numerical strictness and rigorous certification
  are separate. Decimal audits evaluate saved float inputs; no root enclosure
  or graph-realization proof is available.
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
