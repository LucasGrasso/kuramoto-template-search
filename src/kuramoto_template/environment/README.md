# Gymnasium environment

Install the optional dependency with `uv sync --extra environment`. Import
`KuramotoEnv, EnvironmentConfig` from `kuramoto_template.environment`, or use
`gymnasium.make("kuramoto_template.environment:KuramotoTemplate-v0")`.

```python
from kuramoto_template.environment import KuramotoEnv, EnvironmentConfig

with KuramotoEnv(EnvironmentConfig(max_sdp_calls=128, max_clusters=24)) as env:
    observation, info = env.reset(seed=42)
    terminated = truncated = False
    while not (terminated or truncated):
        action = env.sample_action()  # masked random policy; learned actor later
        observation, reward, terminated, truncated, info = env.step(action)
```

## Task and accounting

Each episode follows a serial sequence of cluster-template mutations. An
evaluated child becomes the current state even if its density is worse, unstable
or invalid. A structurally rejected action leaves the current state unchanged.
The best accepted stable density is retained separately. Reward is
`reward_scale * (best_mu_after - best_mu_before)`, with scale 1000 by default
and a zero incumbent floor until a stable template is found. Reset seed quality
sets the initial incumbent and earns no action reward.

Seed evaluation during `reset()` counts against the same SDP budget as actions;
its cost is in reset `info["sdp_calls"]`. Budget must be at least three, leaving
an action available even when initialization needs both SDPs. Each proposal costs
zero, one or two invocations as reported by the existing evaluator. Repeated
templates are evaluated again: this environment uses an uncached evaluator and
disables `SearchSession` exact-data deduplication so costs do not depend on hidden
visit history. Existing search algorithms keep their default deduplication.

Exhausting the observed finite SDP budget returns `terminated=True`. External
wall-time, proposal-count and consecutive zero-cost limits return
`truncated=True`, preserving the distinction for critic bootstrapping. Wall time
is checked between solves; an active solver is not interrupted. Call `reset()`
after either end flag. Manual `reset()`/`close()` finalizes an unfinished episode
as an externally stopped collection. `close()` is idempotent.

Default initial state is a five-twisted template, clipped to the configured size
bounds. Supply `initial_template=...` or `reset(options={"template": ...})` for
other starting states. No automatic restart schedule or learned policy is added
in this first API. A new seed recreates the environment's sampling stream;
`reset(seed=None)` continues it. The action space is also seeded per episode.

## Fixed observation space

For maximum M clusters, inactive positions are zero-padded. Numerical template
and SDP calculations remain float64; policy observations are float32.

| Field | Shape | Meaning |
|---|---|---|
| `nodes` | `(M, 2)` | Mass and log mass |
| `edges` | `(M, M, 2)` | Sine and cosine of relative phase |
| `node_mask` | `(M,)` | Active cluster positions |
| `operation_mask` | `(4,)` | Structurally available operations |
| `pair_mask` | `(4, M, M)` | Available merge/resize index pairs |
| `global_features` | `(8,)` | Ordered fields listed below |
| `status` | `(6,)` | Current status one-hot |
| `diagnostics` | `(2, 4)` | Density and margin diagnostic rows |
| `diagnostic_mask` | `(2, 4)` | Present, finite diagnostic values |

Global order: best mu, current stable mu, margin gamma, m/M, remaining SDP calls,
maximum SDP calls, proposal fraction, consecutive zero-cost fraction. Absent
scalar signals are zero; status and diagnostic masks distinguish missing values.
Status order is stable, near_feasible, uncertain, infeasible, solver_error,
invalid. Diagnostic columns are mu, gamma, min_restoring,
min_cluster_eigenvalue. Extreme values not representable in float32 are marked
missing. The status and scalar diagnostics are numerical screening signals,
not certificates. In particular, gamma is size-dependent.

Relative-phase features are invariant to global rotation. Node and pair arrays
follow the input permutation, allowing an equivariant actor/critic later. No
canonical sorting or P-matrix features are imposed.

## Hierarchical action space

`Dict(operation=Discrete(4), i=Discrete(M), j=Discrete(M),
parameters=Box(-1,1,shape=(2,),dtype=float32))`.

| Operation | ID | Indices | Parameters p, q |
|---|---:|---|---|
| Split | 0 | i | ratio `.2+.6*(p+1)/2`; signed log-scale separation from q |
| Merge | 1 | i, j | Ignored |
| Perturb | 2 | i | displacement `p*phase_scale`; q ignored |
| Resize | 3 | source i, target j | fraction `resize_fraction*(.01+.99*(p+1)/2)`; q ignored |

Split separation magnitude is log-interpolated from `separation_min` to
`separation_max` using `abs(q)`; q's sign selects direction, with q=0 positive.
Merge uses i<j, respects the minimum cluster count, and masks circular-mean
cancellation. Resize pairs are directed and exclude i=j. Split is masked at M.
Unused arguments are ignored. Out-of-space actions raise without spending a
call. In-space masked actions are logged as zero-cost rejected proposals.
`sample_action()` samples the valid hierarchy using this environment's RNG;
plain `action_space.sample()` can deliberately select masked actions.

## Persistence and inspection

Pass `output=Path(...)` for exclusively created `episode-000000`, etc. Existing
episode directories are never reused. Normal search metadata, `trace.jsonl`,
and final `result.json` are preserved. Additional files are:

- `transitions.jsonl`: action, source ID, attempt record ID, actual successor
  `next_state_id`, best mu before/after, remaining calls, cost, reward, end flags.
  Rejected attempts have an attempt ID but retain their source as next state.
- `events.jsonl`: reset and episode-end records, including early reset, close,
  and wall expiry before an action. A replay importer must consume these events
  rather than assume every stop follows a paid or logged action.

`info["sdp_calls"]` is this reset/step's cost; `total_sdp_calls` is cumulative
spent calls, `remaining_sdp_calls` is the balance, and `state_id` identifies the
actual current trace state. Wall-clock/UUID fields are kept in artifacts rather
than deterministic step info. `last_result` exposes the completed SearchResult.
Optional `render_mode="ansi"` returns a text summary from `render()`.

Random-policy runner:

```text
uv run --extra environment python -m kuramoto_template.environment --seed 42 --episodes 2 --max-sdp-calls 128 --output runs/gym-smoke
```

Run the Gym API and integration tests with `uv run --extra environment pytest`.
Environment tests skip when the extra is absent; the ordinary solver package
does not import Gymnasium. This API supports structured actions for a custom
trainer; it does not promise compatibility with every third-party RL algorithm.
