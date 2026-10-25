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
