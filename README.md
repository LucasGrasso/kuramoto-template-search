# Kuramoto Template Search by $C_4$ deformations

This branch contains the joint limiting SDP experiment, its S7 structural
follow-up, and the independently checked finite lift. Historical numerical
witnesses for the three best results, reports, configurations and aggregate
statistics are versioned under [`research/`](research/README.md).

A new experiment initialized from C4 splits, without S7 seeds or constants,
certifies a stronger limiting bound, **mu0 >= 0.6929076590503417**, for an
explicit nearby rational variant. See [results and scope](docs/c4-discovery-results.md)
and [reproduction commands](docs/c4-discovery.md). The table below records the
historical experiments.

| Result | Density | Tracked witness |
| --- | ---: | --- |
| Original 1,050-proposal sweep, limit | 0.6929033189148635 | [original limit](research/joint-research-final/sweep/top-limit.json) |
| S7 follow-up, limit | 0.6929072637491324 | [follow-up limit](research/s7-analysis/evidencia/fine-offset-1e-07--0.0001.json) |
| Fixed-P0 refinement, finite h=0.00003 | 0.6926591963503084 | [finite witness and audits](research/joint-research-final/finalist-refinement.json) |

These original snapshots are numerical results. The limiting record and finite
witness are different objects. Separate [interval certificates](research/certification/README.md)
now verify explicitly recorded nearby rational variants. The finite variant
passes Krawczyk inclusion and interval Cholesky. The limiting variants satisfy
exact leading identities and the hypotheses of a small-splitting lifting proof.
The original snapshots' flags remain unchanged; no explicit graph is certified.

## Setup and verification

Python 3.14 or newer and `uv` are required. The lockfile retains the original
NumPy, SciPy, CVXPY and Clarabel versions.

```powershell
uv sync --locked
uv run --locked pytest -q
uv run --locked python scripts/reproduce_best.py --verify-only --output runs/witness-check
uv run --locked python scripts/reproduce_best.py --output runs/best-replay
```

Use a new output directory for each command. Witness verification checks every
tracked research file against its SHA-256 manifest, recomputes the saved
limiting and finite diagnostics, and performs two Decimal audits without
conic solves. Best replay additionally spends three joint SDP calls, three
finite tree solves and two Decimal audits. It checks both numerical densities
against the saved results, with a 1e-9 tolerance for new solves.

## Interval certification

```powershell
uv sync --locked --extra certification
uv run --locked --extra certification python scripts/certify_witnesses.py --verify research/certification
uv run --locked --extra certification python scripts/certify_witnesses.py --output runs/new-certificates
uv run --locked --extra certification pytest -q
```

Certificate verification recomputes exact rational identities and outward Arb
bounds. Rebuilding certificates records every rational repair and does not run
SDPs. See the [method and lifting proof](docs/interval-certification.md) and
[results, scope and rational data](research/certification/README.md).

## Reproduce the experiments

For the separate experiment initialized directly from C4 splits, including a
zero-margin search, see [C4 discovery](docs/c4-discovery.md). It uses no S7 seed
or saved witness and preserves the historical experiment below.

The original benchmark includes S7 calibration, paired LP/SDP-then-R controls,
and the balanced 1,050-proposal joint sweep. Larger sizes, alternative masks,
and weak-mode moves remain because they are part of that experiment.

```powershell
uv run --locked python -m kuramoto_template.asymptotic_search.joint_benchmark --output runs/joint-replay --sweep-proposals 1050 --seed 0
uv run --locked python scripts/reproduce_s7_analysis.py --source-run runs/joint-replay --output runs/s7-replay
```

The benchmark first regenerates all solve snapshots under ignored `runs/`.
The analysis command requires that complete source run and repeats 120 joint
solves and four Decimal audits. Git retains only the original best limit,
improved S7 limit and best finite witness. Other candidates, checkpoints and
embedded trial matrices are excluded; aggregate statistics remain available.

To regenerate its plots, Spanish Markdown report and self-contained HTML:

```powershell
uv sync --locked --extra reports
uv run --locked --extra reports python scripts/reproduce_s7_analysis.py --source-run runs/joint-replay --output runs/s7-report-replay --reports
```

The existing [Spanish report](research/s7-analysis/informe-s7.md) and
[HTML informe](research/s7-analysis/informe-s7.html) are already versioned.
Original reports preserve their historical source hashes and measurements.
Fresh runs record the cleaned source hash; old checkpoints cannot be resumed
under changed source. Numerical differences across platforms or solvers must
be inspected, not relabeled as reproductions.

## Retained implementation

- `asymptotic_search/`: S7 geometry, expansions, joint SDP, original paired
  controls, guided outer search, finite lifting, numerical audits and persistence.
- `template.py`: immutable masses and wrapped phases.
- `sdp_optimizer/template_metrics.py` and `validate_input.py`: independent finite
  metrics and circular geometry checks.
- `scripts/`: best-witness replay and the portable structural-analysis pipeline.

Unused RL/Gym, random/AMCS search, full-template optimizers and fallback,
the old two-stage outer driver, aggregate-flow/compensated workflows and their
specific tests/examples have been removed from this branch. Their committed
history remains available before this cleanup. Other branches are unchanged.
Fresh output stays ignored in `runs/`; curated evidence belongs in `research/`.
See [the joint model](docs/joint-search.md), [original findings](docs/joint-search-findings.md)
and [evidence provenance](research/README.md).
