# Joint asymptotic experiment

The retained workflow is the joint limiting SDP benchmark and its S7
structural follow-up. See the [repository reproduction instructions](../../../README.md),
[joint model](../../../docs/joint-search.md),
[historical findings](../../../docs/joint-search-findings.md), and
[tracked evidence](../../../research/README.md).

```powershell
uv run --locked python -m kuramoto_template.asymptotic_search --config examples/joint-small.json --output runs/joint-small
uv run --locked python -m kuramoto_template.asymptotic_search --output runs/joint-small --resume
```

The earlier two-stage outer CLI was removed. The LP, necessary-screen SDP and
correction SDP remain as controls used by `joint_benchmark`; their independent
screen/correction checks are also used by finite lifting. `persistence.py`
contains the shared JSON writer, source hash and structural shortlist.
