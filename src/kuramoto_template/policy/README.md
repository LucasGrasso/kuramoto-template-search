# Learned mutation policy (M1–M4)

Install with `uv sync --extra rl`. The solver/environment packages remain usable
without PyTorch. The initial implementation runs on CPU with one Torch thread;
it does not silently launch a GPU or a large training experiment.

## Reproduce a small, bounded experiment

From the project directory, use new output directories:

```powershell
uv run --extra rl python -m kuramoto_template.policy.train --output runs/rl-smoke/train --training-calls 64 --episode-calls 16 --warmup-calls 16 --bc-steps 10 --batch-size 8 --max-updates 32 --hidden 32 --history runs/bench-final-500/runs/random/five_twisted --history-limit 300
uv run --extra rl python -m kuramoto_template.experiments.evaluate_policy --checkpoint runs/rl-smoke/train/checkpoint.pt --bc-checkpoint runs/rl-smoke/train/bc.pt --output runs/rl-smoke/evaluation --methods random,bc,awac --families eight_twisted,twisted_19 --budgets 16 --repetitions 1
```

This caps fresh online training at 64 solver calls and testing at 96 calls
(three methods, two families, 16 each), plus separately reported fresh audits
of the best results. The imported historical runs' original solver cost is
reported separately, even when only a prefix is imported. This tiny experiment
tests the pipeline, not search superiority or a new mathematical construction.
The m=19 family requires max_clusters>=19; use separate experiments with caps
12 and 24 for later research comparisons. No 500/2000/5000 matrix is launched
automatically. Default training families are five_twisted, four_perturbed and
random_phases; eight_twisted, twisted_16 and twisted_19 are held out.

## Model and action semantics

`encoding.py` batches variable-size padded observations without sorting clusters.
`network.py` uses shared message/update MLPs, masked pooling, equivariant node
and pair heads, and invariant context. Unordered merge scores are symmetrized;
resize scores remain directed. Padding is excluded from aggregation/softmax.
The encoder sees status and diagnostic availability separately from values,
remaining call/decision context, active m, objective threshold .6929077 and
the actual fixed SDP margin target .693. It has no P or absolute-phase input.

`actions.py` factors operation, index/pair, and active continuous parameters.
Continuous components use a Normal truncated to [-1,1], sampled by inverse CDF
and scored with its exact normalizing constant. Locations stay in [-.8,.8],
scales in [.05,.5], so the truncation normalizer is well-conditioned. Numerical
CDF endpoint protection is applied before inversion; actions are not clipped
after sampling. Densities refer to normalized Gym coordinates, not physical
angles/masses; no physical-coordinate Jacobian is needed for this objective.
Ignored Gym arguments are canonicalized and contribute no density. The output
decodes through the existing four operations with their existing domains.

`sample_action` returns a Gym action and recomputable joint log probability.
`mode_action` is deterministic: argmax discrete choices and continuous location
(the truncated Normal's mode). `entropy` supplies a Monte Carlo joint entropy
estimate. A genuinely empty valid-action mask raises NoValidAction rather than
producing NaNs; the current environment always offers perturb/resize.

Optional `--reflection` augments BC states/actions coherently. Split separation
and phase displacement change sign; masses and index order do not. Split q=0
is excluded from reflection because the existing decoder maps both signed
zeros to positive separation_min. This is an augmentation, not a claim that
the network is explicitly reflection-equivariant.

## Data contracts and leakage safeguards

See CONTRACT.md for the actual environment/session/log inventory. The one
environment integration change extracts its observation builder as a pure
function. The live environment uses exactly that same function; no solver,
mutation, reward, cache or AMCS implementation was replaced.

`dataset.py` reads source metadata and preserves eligible branch actions and
state labels. It resolves actual source_id/split_id rather than blindly using
old pruning parent_id, then replays the mutation and checks its child. Compound
undo-split and dissolve cannot be represented by one of the four actions and
are excluded from actor imitation; their state/provenance data remain retained.
Legacy random-search phase_scale is a Gaussian sampling standard deviation,
not the bounded Gym displacement: physical parameters are inverted through
the current decoder instead of copying old sampler settings.

Stored stability is re-screened using current analytic and numerical rules.
Missing, cached, malformed or nonfinite matrices do not establish stable mu.
An inaccurate status alone never makes a trusted label or certificate. Masks
preserve unknown/solver-error labels rather than replacing them with zero.

Legacy branches NEVER enter TD replay. Their BC observations use explicitly
synthetic local incumbent/budget context; no chronological future return is
invented. Complete serial Gym logs can enter TD only when IDs, decoded states,
reward, budget, original attempt costs, evaluation settings and final events
reconcile. Reset/close events are consumed; rejected attempts are not assumed
to be successor states. Historical TD is not re-appended during resume.
The reward scale, minimum size, zero-cost limit, decision/call ratio, wall-time
setting and physical action decoder must match the current task; otherwise the
episode remains BC/state data only. Unknown evaluator schema versions are not TD.

Whole runs are assigned validation via a seeded stable hash, with explicitly
held-out families excluded from training. Parent/child canonical groups at six
decimals are removed from training if present in validation/held-out data.
This is a documented near-duplicate quantization, not a complete metric-space
distance guarantee. Known held-out seed constructions are also excluded.
Fresh online replay also excludes reserved validation/held-out canonical states.
Evaluation rejects training families by default and reports canonical overlap
for every visited test state. `--allow-training-families` explicitly requests
in-distribution diagnostics. Overlap does not count as novel discovery.

`--history-limit 300` is an explicit row cap logged in the manifest;
`--history-limit 0` imports all rows. Use all eligible data for a later research
run; do not confuse a capped smoke dataset with the entire corpus.

## Training and checkpoints

`train.py` first collects clean random episodes, performs a short weighted BC
warm start, then runs online AWAC. BC trains the encoder and actor jointly;
without history it uses positive immediate gains for bounded weighting and
retains lower-weight non-improving actions. Historical BC uses only validated,
reconstructable actions. This supervised signal alone does not discover
long-horizon paths; a synthetic two-step regression verifies the TD learner can
prefer a temporarily worse action.

`learner.py` trains the encoder/twin critics with Huber TD loss, target networks
and detached target actions. The actor uses detached advantages relative to a
policy-sampled value baseline, safe exponentiation and bounded weights. Actor
updates do not backpropagate through critics or target networks. Both end flags
stop continuation in v1. With discount=1, episode reward telescopes to final
best-density gain. Setting --discount below 1 explicitly changes that objective.
Gradient clipping and finite-loss checks are enforced. Q ranges, TD errors,
targets, advantage weights and replay negative log likelihood are logged.

Optional --auxiliary-weight enables masked outcome-class, accepted-stability,
conditional-mu and tanh(m*gamma) heads on clean replay. These are auxiliary
losses, not additive rewards or proof labels. Gamma never ranks across sizes.

Historical actions are separate BC data unless a full serial episode is safe
for TD. No arbitrary 50% historical critic batch is manufactured. Fresh replay
and any usable historical TD share the replay sampler. No unvalidated simulator
model is substituted for paid SDP outcomes.

After warmup, random actions are paced using actual paid action calls rather
than proposal counts; realized fractions are logged. Warmup calls also appear
in the reported all-training random fraction, so a tiny experiment can exceed
20%. Zero-cost actions still consume the environment's decision limit. Seeds
consume training calls but are excluded from action-allocation fractions.

Reset families and canonical/descriptor-diverse archive parents form a logged
mixture. The archive stores stable and exploratory states and keeps quality
within m/mass/phase-gap descriptor cells; it is not a certificate or a new
evaluator. There is no novelty reward in v1. No REFINE action or PPO is added.

Artifacts: config.json, dataset_manifest.json, historical_dataset.pt (when
supplied), metrics.jsonl, full episode trace/transition/event files,
archive.json, bc.pt, checkpoint.pt and summary.json. Checkpoints contain the
encoder/actor/critics/targets, all optimizers, RNGs, replay contents and sampler
position, archive, source digest and training counters. They use tensor and
primitive state compatible with weights-only loading. BC checkpoints are
inference-only if made mid-episode. Resumption requires an episode-boundary
checkpoint, unchanged settings/data, and a nondecreasing total call cap:

```powershell
uv run --extra rl python -m kuramoto_template.policy.train --resume runs/rl-smoke/train/checkpoint.pt --output runs/rl-resumed --training-calls 128 --episode-calls 16 --warmup-calls 16 --bc-steps 10 --batch-size 8 --max-updates 32 --hidden 32 --history runs/bench-final-500/runs/random/five_twisted --history-limit 300
```

All other original arguments must match. Continuation from the saved boundary
is deterministic on the tested CPU configuration; a previous shortened final
episode need not match a hypothetical run that never shortened that episode.
Cross-platform conic/Torch numerical identity is not promised.

## Frozen evaluation and remaining research

`search/policy_search.py` is a frozen adapter returning the existing SearchResult
and separate inference timing. It uses the Gym serial task and existing
SearchSession; neither inference nor the model calls SDPs directly. The
evaluation CLI reuses the existing per-run metrics, paired comparisons and
report generation without rewriting the benchmark harness.

Methods: random (same serial schedule), bc, awac, optional archive_random
(existing strongest archive search), and twisted (direct 3..M enumeration).
The last can finish below its allocation; actual cost is reported. Checkpoint
paths and hashes are stored. Headline best mu is freshly checked with an
uncached tighter solve at the SAME feasibility margins and independent audit;
the conservative minimum of screening/fresh density is used. Screening curves
and best non-twisted diagnostics are labeled as such. Fresh audit cost is
separate from frozen search. Every method gets the same instances, cap and
numerical settings; serial methods also share the exact episode schedule.

The tiny run makes no performance claim and no confidence-interval inference.
M5 requires adequate paired repetitions at 500/2000/5000 calls, caps 12 and 24
as separate conditions, archive-random and enumeration baselines, dataset/
curriculum/exploration/reflection/auxiliary ablations, confidence intervals,
and independently checked non-twisted improvements. Finding twisted_seed(19)
is a baseline result. Existing refinement is float64 numerical verification,
not arbitrary precision or rigorous certification; proof_available stays false.
