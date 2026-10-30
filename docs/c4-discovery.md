# Discovery from C4 without an S7 seed

See the [measured results and certification scope](c4-discovery-results.md).

The historical outer search initializes every restart from S7. Its concentration
of high-density results near S7 is therefore conditional on that initialization.
The new `c4_discovery` experiment uses the same limiting SDP and independent
validation, but generates its geometries directly from four equal C4 parents.
It neither calls the S7 constructor nor reads historical witnesses or the S7
constants. The original benchmark, defaults and checkpoint schema are unchanged.

## Reproduce

```powershell
uv sync --locked
uv run --locked python -m kuramoto_template.asymptotic_search.c4_discovery --config examples/c4-discovery.json --output runs/c4-discovery
uv run --locked python -m kuramoto_template.asymptotic_search.c4_discovery --config examples/c4-discovery.json --neutral --output runs/c4-discovery-zero
uv run --locked python -m kuramoto_template.asymptotic_search.c4_discovery --config examples/c4-dihedral.json --output runs/c4-dihedral
uv run --locked python -m kuramoto_template.asymptotic_search.c4_discovery --config examples/c4-dihedral.json --neutral --output runs/c4-dihedral-zero
```

Use a fresh output directory. This driver has no resume mode; interrupted output
is retained and must not be overwritten. `result.json` records configuration,
versions, source hash, actual call counts, all optimizer outcomes and proposals.
`trace.jsonl` and individual solve snapshots preserve failures and uncertain
outputs. `checkpoint.json` is a progress receipt, not a resumable optimizer state.
Exact geometry cache hits are recorded and spend no conic calls. Every local
stage has its own paid-call allocation, including derivative trials and failures.

## Initialization and search

1. Test all 15 nonempty subsets of C4 parents. Each selected parent is split
   into two equal masses with opposite offsets. Parent masses initially remain
   1/4; positive widths are independent uniform draws in [0.6,1.4]. There is
   no preferred seven-cluster partition. Repeating the sweep changes widths.
   The optional `parent_symmetry` configuration also equates masses and widths
   under C4 automorphisms that preserve the selected subset. These orbits are
   computed for every subset; the three-split class then has three geometry
   parameters, rather than five. This additional symmetry restriction is an
   explicit ablation, not part of the independent unequal-split control.
2. Independently sample unequal splits with 6,7,8,9,10,12 clusters. Each added
   cluster chooses a parent uniformly; within-parent fractions follow
   Dirichlet(4). Gaussian offsets are centered within each parent. These are
   separate starts, not descendants of the reflection winners.
3. Select the best independently accepted reflection start in each of the five
   C4 rotation/reflection classes, plus the configured number of best generic
   starts. Optimize their masses and offsets with SLSQP and paid finite
   differences. The symmetric family optimizes all four parent masses and all
   relative split widths. Generic refinement permits independent subcluster
   masses and offsets. Density evaluations re-solve P0 and R with no fixed
   off-diagonal entries. All allowed internal and neighboring densities are free.
4. Refine the overall accepted winner using smaller finite-difference steps and
   independently audit its leading identities and Hessian with Decimal.

Offsets have weighted mean zero and variance one. Every proposed geometry is
stored with its source stage seed and full absolute optimizer parameters. The
inner SDP retains the full quotient Hessian, restoring and hard-mode floors,
guard box, correction bounds, winding check and inward neighbor repair tree.
Neither solver convergence nor outer optimizer success establishes strictness.

`--neutral` requires only a semidefinite full limiting Hessian (gamma=0); it
retains the other positive floors and numerical gates. Independently neutral
results can guide refinement, but they never acquire a strict status for that
reason. `top-search.json` is the best strict-or-neutral density, whereas
`top-limit.json` and `best_mu` retain only numerical strict results. A zero-margin
endpoint alone does not establish a stable finite family: additional null modes
can make higher-order terms decisive. The recorded effective required gamma
is zero even though the legacy, ignored `gamma0` config field remains positive.

To attempt exact rational repair and interval certification of a new winner:

```powershell
uv run --locked --extra certification python scripts/certify_limiting_result.py --input runs/c4-dihedral/top-search.json --output runs/c4-dihedral-certificate
uv run --locked --extra certification python scripts/certify_witnesses.py --verify runs/c4-dihedral-certificate/certificate.json
```

The script verifies at 192,256,320 bits. An indefinite rational variant fails;
its failed receipt remains on disk. A successful limiting certificate proves
the hypotheses of the small-h lifting lemma for that rational variant, with
no explicit finite h or graph realization.

## Refine the discovered geometry at zero margin

The independent SLSQP runs can stall at different nonsmooth optima. A bounded
Nelder-Mead continuation takes a recorded C4 winner and optimizes the same
geometry parameters, without reading S7 data. It retains the initial simplex,
source hashes, paid-call ledger, every inner result and cache hit, and separate
strict and neutral winners.

```powershell
uv run --locked python scripts/refine_c4_result.py --run runs/c4-dihedral --output runs/c4-nelder-zero --gamma 0 --max-calls 1500
uv run --locked python -m kuramoto_template.asymptotic_search.joint_limit --input runs/c4-nelder-zero/top-search.json --config examples/c4-small-margin.json --output runs/c4-small-positive.json --max-joint-calls 1
uv run --locked --extra certification python scripts/certify_limiting_result.py --input runs/c4-small-positive.json --output runs/c4-best-certificate
uv run --locked --extra certification python scripts/certify_witnesses.py --verify runs/c4-best-certificate/certificate.json
```

The final SDP fixes the discovered masses and offsets, and reoptimizes P0 and R
with gamma=1e-10. This is a new solve, not certification of the zero-margin
snapshot. Its numerical status can remain `joint_neutral` because the numerical
strict gate is 1e-8. A successful independent rational/interval proof establishes
strict positivity for the explicitly stored rational variant; the original
numeric status and proof flags stay unchanged.

This tests rediscovery within a C4-based class. Reflection starts still impose
a substantial symmetry assumption; the generic control does not establish a
global search. A winning seven-cluster pattern would be an output of this
experiment, rather than a prescribed seed. Any interval certificate must refer
to separately recorded rational repairs, never to the unchanged float snapshot.
