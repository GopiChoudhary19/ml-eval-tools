# ml-eval-tools

Small command-line utilities for deciding whether an ML change is a real improvement or just noise.
Requires Python 3, numpy, pandas and scikit-learn (`pip install -r requirements.txt`).

| Script | What it does |
|---|---|
| `paired_delta.py` | Paired per-row test of candidate vs baseline predictions on the same holdout rows (t-statistic, rows better/worse). Exit 0 = better, 1 = not better. |
| `oof_blend.py` | Checks whether an ensemble is worth it: member scores, pairwise correlation, equal-weight blend, and greedy blend weights fit on one half and scored on the other. |
| `exp_log.py` | A tab-separated experiment ledger: one change per row, delta vs parent, keep/kill verdicts. |
| `overlap_check.py` | Detects overlapping sliding windows in sequence data before you cross-validate, since random-row splits leak there. |
| `draw_check.py` | Compares two scripts after normalising seeds and reports whether a new run is a real change or only a reseed; optionally compares prediction CSVs and estimates selection inflation from public scores. |
| `comp_metrics.py` | Shared metric registry (AUC, AP, log loss, RMSE, accuracy, F1, ...) used by the scripts above. |

## Examples

```bash
# is B better than A on the same rows?
python paired_delta.py cand.csv baseline.csv --truth truth.csv --target y --metric auc

# is a blend worth shipping?
python oof_blend.py --truth truth.csv --target y --metric auc --preds a.csv b.csv c.csv

# log an experiment
python exp_log.py init MyTask --metric auc --direction max
python exp_log.py add MyTask --id E1 --change "baseline" --holdout 0.731 --verdict keep

# reseed or real change?
python draw_check.py new_solution.py old_solution.py --csv new.csv old.csv --pub 0.73 0.74 0.72
```

## Tests

```bash
pytest -q
```

Each tool is tested end to end on small synthetic data (a known improvement, a reseed vs a real change,
sliding windows vs independent rows, a ledger round trip); CI runs them on every push.

## License

MIT
