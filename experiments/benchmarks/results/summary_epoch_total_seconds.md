# SPTnet training-speed benchmark (epoch_total_seconds)

Steady-state per-epoch `epoch_total_seconds` (warmup epochs dropped: 1; 10000 hierarchical bootstrap resamples). train_seconds = compute only; epoch_total_seconds = full epoch incl. plotting.

| Config | Runs | Epochs | Mean s/epoch | 95% CI |
|---|---|---|---|---|
| new_fresh_good_runs | 8 | 24 | 116.26 | [115.62, 116.80] |
| old_good_8_runs | 8 | 24 | 511.71 | [505.52, 518.56] |

## Headline speedup
**old_good_8_runs / new_fresh_good_runs = 4.40x** (95% CI [4.34, 4.46]).
