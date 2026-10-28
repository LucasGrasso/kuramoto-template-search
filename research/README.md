# Tracked research evidence

Git contains only the witnesses for the three best reported results:

| Result | Density | Witness |
| --- | ---: | --- |
| Original sweep's best limit | 0.6929033189148635 | [top-limit.json](joint-research-final/sweep/top-limit.json) |
| Improved S7 limiting result | 0.6929072637491324 | [fine-offset-1e-07--0.0001.json](s7-analysis/evidencia/fine-offset-1e-07--0.0001.json) |
| Best finite result, h=0.00003 | 0.6926591963503084 | [finalist-refinement.json](joint-research-final/finalist-refinement.json) |

The finite witness includes its required limiting correction and the retained
finite matrix. Other h trials have scalar summaries only. The improved S7
limit has no recorded finite lift of its own.

Reports, the Spanish Markdown/HTML informe, figures, scalar tables, proposal
trace, configurations, ledgers and original provenance summaries are retained.
Analysis JSON files contain aggregates and diagnostics; non-winning candidate
geometries, connection matrices and full dual matrices have been omitted.
`manifest.json` records SHA-256 and byte size for the curated files.
`cleanup-verification.json` records the full reproductions performed before
pruning and the subsequent verification of the winning witnesses.

## Reproduction

Best-result replay needs only the tracked witnesses:

```powershell
uv run --locked python scripts/reproduce_best.py --verify-only --output runs/witness-check
uv run --locked python scripts/reproduce_best.py --output runs/best-replay
```

To reproduce the complete experiment and all figures, first regenerate the
full dataset outside Git, then pass it explicitly to the analysis:

```powershell
uv run --locked python -m kuramoto_template.asymptotic_search.joint_benchmark --output runs/joint-replay --sweep-proposals 1050 --seed 0
uv run --locked python scripts/reproduce_s7_analysis.py --source-run runs/joint-replay --output runs/s7-replay
```

The analysis spends 120 joint solves and four Decimal audits. Add `--reports`
and use the `reports` extra to regenerate the plots and Markdown/HTML report.
The retained finite witness supplies the report's finite-result context; its
original costs are separate from these 120 solves. Use new output directories.

The original reports describe the complete historical experiment, whose
unselected snapshots and full analysis data are intentionally excluded from
Git. Original raw files remain locally in ignored `runs/` and the source
Documents/Codex task output. A fresh checkout can regenerate them using the
commands above. Historical absolute paths and source hashes are provenance.

All witnesses are numerical. Decimal audits do not enclose an exact equilibrium
or certify a graph realization. Proof flags remain false.
