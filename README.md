# Kuramoto template search

## Template representation

`PhaseTemplate(x, phi)` stores only candidate cluster masses and phases.
It copies inputs to read-only float64 arrays, requires positive masses summing
to one within absolute tolerance `1e-10` (without normalization), and wraps
phases to `[0, 2π)`. Equality uses object identity. Synchronized configurations
are permitted in this representation and rejected by the SDP evaluator.

The pure operations in `kuramoto_template.mutations` are `split_cluster`,
`merge_clusters`, `perturb_phase`, and `resize_clusters`. Split children replace
their parent in order and preserve the unwrapped mass-weighted phase mean.
Merge keeps the smaller index, removes the larger, and uses a mass-weighted
circular mean; near antipodal cancellation raises `ValueError`. Resize leaves
the source strictly above the optional `min_mass` (default zero).
All parameters are explicit; callers can sample with `np.random.default_rng(seed)`.

SDP results describe asymptotic edge densities and are float64 numerical
evidence, not rigorous stability certificates or finite simple graphs.
The next stage is a separate fixed-target-degree maximum-stability-margin SDP,
followed by a seeded search baseline that records every evaluation.

## Running the tests

```bash
uv run pytest -v
```
