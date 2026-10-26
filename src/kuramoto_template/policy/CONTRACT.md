# M0: integration contract

The existing KuramotoEnv reset returns (observation, info); step returns
(observation, reward, terminated, truncated, info). Observations are float32
nodes (mass/log mass), relative sine/cosine edges, active-node, operation and
operation-specific pair masks, six-way status, density/margin diagnostics and
availability masks. Eight global fields contain incumbent/current mu, gamma,
m/M, remaining/maximum SDP calls, and decision/zero-cost fractions. The model
also receives the fixed experiment target and active cluster count. No P input.

Actions are the existing Dict(operation,i,j,parameters[2]) in normalized [-1,1]
coordinates: split ratio .2..8 and signed logarithmic separation, unordered
merge i<j, perturb p*phase_scale, directed resize fraction .01..1 times
resize_fraction. Inactive continuous parameters contribute no likelihood.
The policy uses truncated Normal densities directly in these normalized
coordinates; no physical-coordinate density or post-sampling clipping is used.

Every proposal passes through SearchSession.record. Initialization spends the
same budget; invalid structural actions can spend zero calls but count a
decision. The serial environment disables caching/deduplication, so repeat
states pay fresh solves. A mutated child remains current even if unstable.
SDP-budget exhaustion terminates; other collection limits truncate. V1 treats
both as hard episode boundaries, with zero TD continuation across either.
Discount is 1; primary reward is the environment's telescoping incumbent gain.

Search trace rows contain attempt IDs, actual parent IDs, template and optional
evaluation; siblings are not sequential. Old prune records may use source_id
or split_id instead of parent_id. Undo-split is compound and cannot be imitated
as one of the four actions. Historical cached P may use another cluster order.
The canonical key removes permutation/global rotation approximately at its
quantization; reflection is not currently part of that key. We group leakage
checks at six decimals as an explicit near-duplicate approximation.

Legacy branch logs supply reconstructed BC/auxiliary samples only. Clean Gym
transition logs additionally contain action, source/attempt/successor IDs,
before/after incumbent and budget, reward, cost and end flags; events.jsonl
records initialization and external stops. Only fully reconciled episodes can
enter TD replay. Stored stability is re-screened against current analytic,
mass, residual and positive-margin rules. Unknowns remain unknown. Float64
screening/refinement does not establish arbitrary-precision or rigorous proof.

Minimal integration change: extract the environment's existing observation
construction into a pure build_observation function, reused unchanged by the
environment and historical importer. This avoids paid resets just to encode
old states. Environment transition/reward semantics are unchanged.

Frozen search returns SearchResult and inference timing; the evaluation CLI
uses the existing metrics/paired/reporting APIs. It does not rewrite the old
benchmark harness. Serial random uses the same resets, budgets, evaluator and
no-cache schedule. Archive random and direct twisted enumeration are optional
stronger comparisons; M4's tiny run is an integration result, not a claim of
outperforming them. Training and fresh refinement calls are separate costs.
