# Kuramoto Template Search

## Project

Python research package for searching phase-cluster templates in the homogeneous
Kuramoto model. Templates specify cluster mass fractions and phases; semidefinite
programs optimize connection densities and assess equilibrium and stability.
Search methods include random search, nested AMCS, adaptive AMCS, and a learned
mutation policy using behavior cloning (BC) and advantage-weighted actor-critic
(AWAC).

## Stack and setup

- Python 3.14 or newer; dependencies and environments managed with `uv`.
- NumPy and SciPy for numerical calculations.
- CVXPY for SDP models; Clarabel is the default conic solver.
- Optional Gymnasium environment and optional CPU PyTorch policy package.
- Pytest for unit, property, regression, and solver integration tests.

From the repository root:

```bash
uv sync
uv sync --extra environment
uv sync --extra rl
```

The optional extras are selected according to the component being developed.
Dependency declarations live in `pyproject.toml`; `uv.lock` records resolution.
The package uses the `uv_build` backend and a `src/` layout.

## Source map

All package paths below are relative to `src/kuramoto_template/`.

| Path | Responsibility |
| --- | --- |
| `template.py` | Immutable `PhaseTemplate` masses and wrapped phases |
| `mutations.py` | Template splitting, merging, phase perturbation, resizing, and pruning helpers |
| `sdp_optimizer/sdp.py` | Maximum minimum-degree density SDP |
| `sdp_optimizer/stability_sdp.py` | Maximum common stability-margin SDP at a fixed degree target |
| `sdp_optimizer/template_metrics.py` | Recomputed degree, equilibrium, and stability diagnostics |
| `sdp_optimizer/validate_input.py` | Input validation and circular phase geometry |
| `search/evaluator.py` | Numerical screening, evaluation statuses, and SDP-call costs |
| `search/infrastructure.py` | Search sessions, records, shared budgets, and trace persistence |
| `search/random_search.py` | Random-search baseline and twisted seed construction |
| `search/amcs.py`, `search/adaptive.py` | Fixed nested and adaptive AMCS |
| `search/policy_search.py` | Frozen learned-policy search adapter |
| `environment/` | Gymnasium API, observation construction, action decoding, and configuration |
| `policy/` | Equivariant model, action distributions, replay, BC/AWAC learner, dataset import, archive, and training |
| `experiments/` | Seed families, canonicalization/cache, benchmarks, metrics, reporting, refinement, and frozen policy evaluation |

Tests are distributed across `tests/`, `search/tests/`,
`sdp_optimizer/tests/`, and `experiments/tests/` inside the package.

## Mathematical contracts

- `PhaseTemplate.x` contains positive float64 masses summing to one within
  absolute tolerance `1e-10`. Construction does not normalize invalid masses.
  `phi` contains finite phases wrapped into `[0, 2*pi)`. Arrays are copied and
  read-only.
- For fixed masses and phases, both SDPs optimize a symmetric connection matrix
  `P` with entries in `[0, 1]`. Optional structural masks constrain connections;
  clique wiring fixes diagonal entries to one when enabled.
- Cluster degrees are `P @ x`; density `mu` is the minimum cluster degree.
  Equilibrium requires each mass-weighted sine force to vanish.
- Stability checks cover both within-cluster restoring coefficients and the
  mass-weighted intercluster Laplacian restricted to the subspace orthogonal
  to the global phase-rotation mode.
- The density SDP maximizes `mu` under positive stability margins. The margin
  SDP maximizes a common `gamma` at a specified degree target; negative margins
  can provide exploration signals. Gamma comparisons are grouped by cluster
  count rather than ranked across sizes.
- Numerical feasibility and strict stability require residual and eigenvalue
  checks. Solver status alone is insufficient. Float64 refinement and the
  refinement API's `certified` level do not constitute rigorous proofs.

## Environment and policy contracts

- `KuramotoEnv.reset()` returns `(observation, info)`; `step()` returns
  `(observation, reward, terminated, truncated, info)`.
- Observations pad variable-size templates and provide node, operation, pair,
  and diagnostic masks. Relative phase features support rotation invariance
  and permutation equivariance. Policies do not receive the optimized `P`.
- Actions select split, merge, perturb, or resize, with cluster indices and two
  normalized continuous parameters. Action decoding and validity masks live in
  `environment/actions.py`.
- An evaluated child becomes the current state even when it is worse or
  unstable. Structurally rejected actions retain the current state. The best
  stable incumbent is tracked separately.
- Reward is the scaled increase in best stable density. Seed evaluation sets
  the initial incumbent and consumes budget without earning action reward.
- SDP-budget exhaustion terminates an episode; external collection limits
  truncate it. The policy learner treats both as boundaries with no bootstrap
  continuation. Its default discount is one.
- The model uses shared message passing, masked pooling, hierarchical discrete
  choices, and truncated Normal continuous actions in Gym coordinates.
- Historical branching traces are imitation data unless complete serial
  transitions and context can be reconciled. TD replay requires explicit
  transition, reward, budget, end-flag, and provenance validation.
- Training checkpoints contain model and target weights, optimizers, replay,
  archive, random states, and training progress. Training resume requires a
  completed episode boundary and compatible settings.

## Accounting, data, and reproducibility

Count actual density and margin solver invocations, including initialization
and failed solver attempts. Proposal counts and optimizer updates are distinct
from SDP calls. The serial environment disables caching and exact deduplication;
repeated states receive fresh evaluations. Other search methods can use
canonicalized screening caches, whose hits cost zero calls.

Canonicalization handles cluster permutations and global phase rotation with
explicit numerical quantization. Cached matrices must match the requested
cluster ordering. Independent candidate refinement bypasses the cache.

Keep stable-density rankings separate from margin-based exploration. Preserve
actual mutation parentage in traces, use explicit random seeds, and record
configuration, solver options, and numerical tolerances. Frozen comparisons
use matched starting templates and SDP budgets; report training, evaluation,
and fresh audit costs separately. Keep training and held-out provenance distinct.

Generated experiment artifacts belong outside source code. `.venv`, caches,
and generated outputs are ignored by Git. `.t4_env` contains private access
material and must remain untracked; do not copy credentials into artifacts.

## Development and verification

Inspect the implementation before relying on tests as evidence of correctness.
Preserve the mathematical formulation, public API contracts, budget accounting,
and artifact schemas when making changes. Review the relevant tests and add
targeted coverage when behavior changes warrant it.

```bash
uv run --extra rl pytest -q
uv run --extra rl pytest -q src/kuramoto_template/tests/test_policy.py
uv run --extra environment pytest -q src/kuramoto_template/tests/test_environment.py
uv run pytest -q src/kuramoto_template/sdp_optimizer/tests
```

The full suite must discover all package test directories, not only the
top-level `tests/` directory. Optional environment and policy tests need their
corresponding extras. Solver integration tests are marked `integration`.

For detailed interfaces, consult `environment/README.md`, `policy/README.md`,
and `policy/CONTRACT.md`. Module CLIs expose configuration through `--help`.
