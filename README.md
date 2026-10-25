# Kuramoto template search

Phase-cluster templates, mutation operators, and SDP optimization for the homogeneous Kuramoto model.

SDP results are numerical evidence, not mathematical certificates.

## Running the tests

```bash
uv run pytest -v
```

## Random search

```bash
uv run python -m kuramoto_template.search.random_search --seed 42 --iterations 100 --output runs/demo
```

Starts from unrestricted five- and four-cluster twisted seeds. Saves configuration,
per-proposal JSONL diagnostics, and the best numerical candidate to a new directory.
`--target-mu` defaults to `0.693`; use `--help` for margins and budget controls.

## Fixed nested rollouts

```bash
uv run python -m kuramoto_template.search.amcs --seed 42 --iterations 4 \
  --level 1 --depth 2 --branching 4 --max-sdp-calls 64 \
  --output runs/amcs-demo
```

This is the fixed-depth nested-rollout baseline, not the adaptive AMCS driver yet.
Level 0 plays a random mutation sequence; level 1 evaluates each child through a
level-0 rollout and selects by endpoint. Stable candidates are ranked by
`stable_mu`; exploratory candidates compare `gamma` only within one cluster-count
bucket. Every proposal shares the global SDP-call budget and is written to
`trace.jsonl`. `paths.json` stores each episode's root, selected record ids,
endpoint and best record for review against the trace.

Adaptive depth/nesting, pruning, and learned policies are intentionally deferred.
All results remain numerical candidates, not mathematical certificates.
